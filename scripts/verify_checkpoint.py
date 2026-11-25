#!/usr/bin/env python3
"""Verify a checkpoint BEFORE spending GPU hours evaluating it.

Three checks, in order of how much time they save:

 1. Files present and the quantization config says what you think it says.
 2. Memory actually dropped (roughly half for 8-bit weights, a quarter for
    4-bit). If it did not, vLLM may have loaded it in a fallback path -- e.g.
    FP8 on an Ampere card silently becomes weight-only W8A16 Marlin, which is a
    different experiment.
 3. 100 GSM8K prompts. If accuracy collapses to near zero, the checkpoint is
    BROKEN -- wrong chat template, missing tokenizer files, bad calibration --
    not "bad at quantization". Rebuild, do not report it.

    python scripts/verify_checkpoint.py llama31-8b__w-fp8__kv-bf16
    python scripts/verify_checkpoint.py llama31-8b__w-fp8__kv-bf16 --n 100
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import common

# Expected VRAM for weights, as a fraction of BF16. Loose bounds: the embedding
# and lm_head stay unquantized, so 4-bit lands well above a literal quarter.
EXPECTED_WEIGHT_FRACTION = {
    "bf16": (0.95, 1.05),
    "fp8": (0.45, 0.65),
    "int8": (0.45, 0.65),
    "int4gptq": (0.25, 0.45),
    "int4awq": (0.25, 0.45),
}


def check_files(model_path: pathlib.Path) -> dict:
    out: dict = {"path": str(model_path), "exists": model_path.exists()}
    if not model_path.exists():
        return out
    files = {f.name for f in model_path.iterdir() if f.is_file()}
    out["has_config"] = "config.json" in files
    out["has_tokenizer"] = any(
        f.startswith("tokenizer") for f in files
    )
    out["has_chat_template"] = (
        "chat_template.jinja" in files
        or "tokenizer_config.json" in files
    )
    out["size_gb"] = round(
        sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1e9, 2
    )
    cfg_path = model_path / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        qc = cfg.get("quantization_config") or cfg.get("compression_config")
        out["quantization_config"] = qc
        if qc is None:
            out["WARNING"] = (
                "no quantization_config in config.json -- this may be an "
                "unquantized copy"
            )
    return out


def kv_cache_tokens_from_log(log_text: str, block_size: int = 16) -> int | None:
    """KV cache capacity IN TOKENS from a vLLM startup log.

    vLLM's wording moves between versions, so several patterns are tried. The
    important subtlety: "# GPU blocks: N" is in BLOCKS, not tokens -- one block
    holds `block_size` (default 16) tokens. Returning that number as if it were
    tokens understates capacity by 16x, and KV-cache capacity is one of the
    headline numbers in this study, so the conversion is explicit here.
    """
    token_patterns = [
        r"GPU KV cache size[:=]\s*([\d,_]+)\s*tokens",
        r"KV cache size[:=]\s*([\d,_]+)\s*tokens",
    ]
    for pat in token_patterns:
        m = re.search(pat, log_text, re.I)
        if m:
            return int(m.group(1).replace(",", "").replace("_", ""))

    # Older/other wording reports blocks. Pick up the real block size from the
    # log if vLLM printed it, otherwise assume vLLM's default of 16.
    bs = re.search(r"block[_ ]size[:=]\s*(\d+)", log_text, re.I)
    if bs:
        block_size = int(bs.group(1))
    m = re.search(r"#\s*GPU blocks:\s*([\d,_]+)", log_text, re.I)
    if m:
        blocks = int(m.group(1).replace(",", "").replace("_", ""))
        return blocks * block_size
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=100, help="GSM8K prompts for the sanity check")
    ap.add_argument("--skip-eval", action="store_true", help="files + memory only")
    ap.add_argument("--bf16-size-gb", type=float, default=None,
                    help="size of the BF16 checkpoint, for the memory ratio check")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    meta = common.parse_config_name(args.config)
    model_path = pathlib.Path(cfg["model_path"])

    print(f"=== verifying {args.config} ===\n")

    # The BF16 baseline's model_path is a Hugging Face repo id, not a local
    # directory, so the file checks do not apply to it. Without this the
    # baseline fails check 1 with "checkpoint does not exist", which is wrong
    # and sends you rebuilding something that was never meant to be built.
    is_local = model_path.is_absolute() or model_path.exists()
    if not is_local:
        print(f"--- 1. files ---\n  {model_path} is a Hugging Face repo id, "
              f"not a local checkpoint.")
        print("  Nothing to verify on disk; skipping the file and memory checks.")
        if args.skip_eval:
            return 0
        files = {"exists": True, "size_gb": None}
    else:
        print("--- 1. files ---")
        files = check_files(model_path)
        for k, v in files.items():
            print(f"  {k}: {v}")
        if not files["exists"]:
            print("\nFAIL: checkpoint does not exist. Build it first.")
            return 1
    if meta["weights"] != "bf16" and files.get("quantization_config") is None:
        print("\nFAIL: no quantization config found. Do not evaluate this.")
        return 1

    print("\n--- 2. memory ---")
    lo, hi = EXPECTED_WEIGHT_FRACTION.get(meta["weights"], (0.0, 10.0))
    if not is_local:
        print("  skipped (remote model)")
    elif args.bf16_size_gb:
        frac = files["size_gb"] / args.bf16_size_gb
        verdict = "OK" if lo <= frac <= hi else "SUSPICIOUS"
        print(f"  size {files['size_gb']} GB vs BF16 {args.bf16_size_gb} GB "
              f"= {frac:.2f}x (expected {lo}-{hi}) -> {verdict}")
        if verdict == "SUSPICIOUS":
            print("  A checkpoint that did not shrink was probably not quantized.")
    else:
        print(f"  size {files['size_gb']} GB (pass --bf16-size-gb to check the ratio)")

    ok, msg = common.check_supported(meta["weights"])
    print(f"  hardware: {msg}")

    if args.skip_eval:
        return 0

    print(f"\n--- 3. {args.n}-prompt GSM8K sanity check ---")
    print("  This runs lm_eval. It takes a few minutes. If it reports near-zero")
    print("  accuracy, STOP: the checkpoint is broken, not 'bad at quantization'.")
    print()
    model_args = ",".join(
        f"{k}={v}" for k, v in cfg["eval_model_args"].items()
    )
    cmd = (
        f"{common.lm_eval_cmd()} --model vllm "
        f"--model_args {model_args} "
        f"--tasks gsm8k --num_fewshot 5 --batch_size auto "
        f"--gen_kwargs temperature=0 --log_samples "
        f"--limit {args.n} "
        f"--output_path results/quality/_verify/{args.config}"
    )
    print(f"  {cmd}\n")
    import subprocess

    rc = subprocess.call(cmd, shell=True)
    if rc != 0:
        print("\nlm_eval failed. Read the error before rebuilding anything.")
        return rc

    print("\n--- what to do with the number ---")
    print("  >= ~0.5x the BF16 GSM8K score : fine, proceed to the full eval")
    print("  near zero                     : broken. Read 20 raw outputs by hand")
    print("                                  (they are in the _verify samples file),")
    print("                                  check the chat template and tokenizer")
    print("                                  files, then rebuild.")
    print("  somewhere in between          : read 20 outputs anyway before trusting it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
