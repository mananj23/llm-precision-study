#!/usr/bin/env python3
"""Generate fake lm-eval and vllm-bench output, in the REAL file formats.

    python tests/make_synthetic.py            # writes into results/_synthetic
    python tests/make_synthetic.py --inplace  # writes into results/ (careful)

Why this exists: the whole analysis and router pipeline can be debugged now, on
a laptop, instead of at 2am on a rented H100. The generator writes
`samples_*.jsonl` files with the same keys lm-eval emits and benchmark JSON
with the same keys vllm bench serve emits, so collect_logs.py and
collect_bench.py parse synthetic data through exactly the same code path they
will use on real data.

The numbers it invents are NOT predictions. They are shaped so that every
branch of the analysis gets exercised: one configuration with a real
regression, one that is genuinely indistinguishable from BF16, a KV setting
that degrades only at long context, and enough prompts where the cheap model
beats the expensive one that the router has something to find. If the real
results look like this, that is a coincidence.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

CONFIGS = [
    "llama31-8b__w-bf16__kv-bf16",
    "llama31-8b__w-fp8__kv-bf16",
    "llama31-8b__w-bf16__kv-fp8",
    "llama31-8b__w-fp8__kv-fp8",
    "llama31-8b__w-int8__kv-bf16",
    "llama31-8b__w-int4gptq__kv-bf16",
    "llama31-8b__w-int4awq__kv-bf16",
]

# Invented quality penalties, in probability of flipping a correct answer.
# Chosen to exercise the analysis, not to predict anything.
PENALTY = {
    "w-bf16__kv-bf16": 0.000,
    "w-fp8__kv-bf16": 0.004,   # should land inside the noise -> "no detectable difference"
    "w-bf16__kv-fp8": 0.006,
    "w-fp8__kv-fp8": 0.010,
    "w-int8__kv-bf16": 0.018,
    "w-int4gptq__kv-bf16": 0.055,  # clearly detectable
    "w-int4awq__kv-bf16": 0.040,
}
# Extra penalty applied only on the long-context task, and only for FP8 KV --
# so plot_longcontext has a real effect to find.
KV_LONGCTX_EXTRA = 0.05

TASKS = {
    "gsm8k": dict(n=600, base_acc=0.78, len_lo=60, len_hi=220,
                  metric="exact_match,strict-match"),
    "mmlu": dict(n=500, base_acc=0.66, len_lo=80, len_hi=400, metric="acc,none"),
    # Real LongBench prompts run to tens of thousands of tokens. The synthetic
    # ones stop at 8k because the generator writes prompt text at its TRUE
    # length (input_len is derived from the text, exactly as it is for real
    # lm-eval output), and 30k-token prompts would make this fixture hundreds
    # of megabytes. The length SPREAD is what the bucketing code needs, and 500
    # to 8000 gives five populated buckets.
    "longbench": dict(n=200, base_acc=0.52, len_lo=500, len_hi=8000,
                      metric="acc,none"),
}

# Invented throughput, tokens/s, at concurrency 32, short shape.
THROUGHPUT = {
    "w-bf16__kv-bf16": 2100,
    "w-fp8__kv-bf16": 3050,
    "w-bf16__kv-fp8": 2250,
    "w-fp8__kv-fp8": 3300,
    "w-int8__kv-bf16": 2850,
    "w-int4gptq__kv-bf16": 3500,
    "w-int4awq__kv-bf16": 3450,
}
KV_TOKENS = {
    "w-bf16__kv-bf16": 131_072, "w-fp8__kv-bf16": 262_144,
    "w-bf16__kv-fp8": 262_144, "w-fp8__kv-fp8": 458_752,
    "w-int8__kv-bf16": 245_760, "w-int4gptq__kv-bf16": 360_448,
    "w-int4awq__kv-bf16": 360_448,
}


def key(config: str) -> str:
    return config.split("__", 1)[-1]


def make_quality(out_root: pathlib.Path, seed: int = 20261116) -> None:
    rng = np.random.default_rng(seed)
    run_id = "20261116T120000"

    for task, spec in TASKS.items():
        n = spec["n"]
        # Shared latent difficulty: this is what makes the paired bootstrap
        # worth using, and what the unpaired version wastes.
        difficulty = rng.beta(2, 2, size=n)
        lengths = rng.integers(spec["len_lo"], spec["len_hi"], size=n)
        p_base = np.clip(spec["base_acc"] + 0.35 * (0.5 - difficulty), 0.02, 0.99)
        base_correct = (rng.random(n) < p_base).astype(float)

        for config in CONFIGS:
            k = key(config)
            pen = PENALTY[k]
            if task == "longbench" and k.endswith("kv-fp8"):
                # degrade with length, not uniformly
                frac = (lengths - spec["len_lo"]) / (spec["len_hi"] - spec["len_lo"])
                pen = pen + KV_LONGCTX_EXTRA * frac
            pen = np.broadcast_to(np.asarray(pen, float), (n,))

            # Quantization damage is made LENGTH-DEPENDENT, not uniform random:
            # longer prompts take up to 3x the penalty. This is what gives the
            # router something real to find, so that the router code is tested
            # on the "there is signal" branch as well as the "there is none"
            # branch. Whether real quantization damage actually correlates with
            # prompt length is an open question -- that is what Step 7 measures.
            lf = (lengths - spec["len_lo"]) / max(spec["len_hi"] - spec["len_lo"], 1)
            pen = pen * (0.4 + 2.6 * lf)

            correct = base_correct.copy()
            flip_down = (rng.random(n) < pen) & (base_correct == 1)
            correct[flip_down] = 0.0
            # A few prompts the quantized model gets right and BF16 does not.
            # Without these there is no headroom above the expensive baseline
            # and the router can only lose.
            if k != "w-bf16__kv-bf16":
                flip_up = (rng.random(n) < 0.025) & (base_correct == 0)
                correct[flip_up] = 1.0

            out_dir = out_root / "quality" / config / task / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"samples_{task}_2026-11-16T12-00-00.jsonl"
            with path.open("w") as fh:
                for i in range(n):
                    # Write the prompt at its TRUE word count. collect_logs.py
                    # derives input_len from the text, so capping the text here
                    # would collapse every prompt into one length bucket and
                    # silently break the long-context figure.
                    words = max(4, int(lengths[i] * 0.75))
                    prompt = (f"[{task} #{i}] " +
                              " ".join(rng.choice(VOCAB, size=words)))
                    rec = {
                        "doc_id": int(i),
                        "doc": {"question": prompt},
                        "target": "42",
                        "arguments": [[prompt, {"until": ["\n\n"]}]],
                        "resps": [[f"answer {i}"]],
                        "filtered_resps": [f"answer {i}"],
                        spec["metric"]: float(correct[i]),
                    }
                    fh.write(json.dumps(rec) + "\n")
        print(f"  {task}: {n} prompts x {len(CONFIGS)} configs")


VOCAB = np.array(
    ("the a of and to in that is was for with as on by it at from "
     "model token cache memory context prompt answer compute latency "
     "question passage document reasoning step result because").split()
)


def make_system(out_root: pathlib.Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    shapes = {"short": (512, 128), "medium": (4096, 256), "long": (16384, 256)}
    for config in CONFIGS:
        k = key(config)
        d = out_root / "system" / config
        d.mkdir(parents=True, exist_ok=True)
        base = THROUGHPUT[k]
        (d / "serve_20261116T120000.log").write_text(
            "INFO ... Starting vLLM ...\n"
            f"INFO ... GPU KV cache size: {KV_TOKENS[k]:,} tokens\n"
            "INFO ... Application startup complete.\n"
        )
        for shape, (in_len, out_len) in shapes.items():
            for conc in (1, 8, 32, 64):
                # throughput scales sublinearly with concurrency and drops at
                # long context; FP8 KV suffers less at long context because the
                # cache holds more.
                scale = conc ** 0.72 / 32 ** 0.72
                ctx_pen = {"short": 1.0, "medium": 0.78, "long": 0.52}[shape]
                if shape == "long" and k.endswith("kv-fp8"):
                    ctx_pen *= 1.25
                # Long shape at high concurrency may not fit a BF16 KV cache.
                if shape == "long" and conc == 64 and k.endswith("kv-bf16"):
                    (d / f"{shape}_c{conc}_seed0.FAILED.json").write_text(
                        json.dumps({"config": config, "shape": shape,
                                    "concurrency": conc, "seed": 0,
                                    "failed": True,
                                    "reason": "KV cache cannot fit 64 x 16k"})
                    )
                    continue
                for seed_i in (0, 1, 2):
                    tput = base * scale * ctx_pen * (1 + rng.normal(0, 0.025))
                    ttft = (18 + 0.004 * in_len) * (1 + 0.06 * conc) * (
                        1 + rng.normal(0, 0.04))
                    tpot = 1000.0 / (tput / max(conc, 1)) * (1 + rng.normal(0, 0.03))
                    payload = {
                        "date": "20261116-120000",
                        "backend": "vllm",
                        "model_id": config,
                        "num_prompts": conc * 8,
                        "max_concurrency": conc,
                        "request_throughput": tput / out_len,
                        "output_throughput": tput,
                        "total_token_throughput": tput * (1 + in_len / out_len),
                        "mean_ttft_ms": ttft,
                        "median_ttft_ms": ttft * 0.97,
                        "p99_ttft_ms": ttft * 1.6,
                        "mean_tpot_ms": tpot,
                        "median_tpot_ms": tpot * 0.98,
                        "median_itl_ms": tpot * 0.95,
                        "median_e2el_ms": ttft + tpot * out_len,
                        "duration": conc * 8 * out_len / tput,
                        "completed": conc * 8,
                        "total_input_tokens": conc * 8 * in_len,
                        "total_output_tokens": conc * 8 * out_len,
                    }
                    (d / f"{shape}_c{conc}_seed{seed_i}.json").write_text(
                        json.dumps(payload, indent=1)
                    )
    print(f"  system: {len(CONFIGS)} configs x 3 shapes x 4 concurrencies x 3 seeds")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inplace", action="store_true",
                    help="write into results/ instead of results/_synthetic")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    out = ROOT / "results" if args.inplace else ROOT / "results" / "_synthetic"
    if args.clean and out.exists() and not args.inplace:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"writing synthetic data to {out.relative_to(ROOT)}")
    print("THIS IS FAKE DATA. No number produced from it belongs in the report.")
    make_quality(out)
    make_system(out)
    print("\nnow run:")
    rel = out.relative_to(ROOT)
    print(f"  python scripts/collect_logs.py  --quality-root {rel}/quality "
          f"--out {rel}/logs/per_prompt.parquet")
    print(f"  python scripts/collect_bench.py --system-root {rel}/system "
          f"--out {rel}/logs/system.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
