#!/usr/bin/env python3
"""Generate one YAML per precision configuration in configs/models/.

Run once; re-run only if you change the naming scheme. Keeping this as a
generator (rather than seven hand-written files) means the naming convention
cannot drift, and scripts/analysis read the names back out of the filenames.

    python scripts/gen_configs.py
"""
from __future__ import annotations

import pathlib
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "models"

FAMILIES = {
    "llama31-8b": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "phase": 1,
        "gated": True,
    },
    # Phase 2 replication. The point of a second family is to test whether a
    # finding is a property of quantization or a quirk of one checkpoint, so
    # the second family should differ in more than name: Qwen2.5-7B has a
    # different tokenizer, different pretraining mix, and 28 layers with 4 KV
    # heads against Llama's 32 and 8 -- which changes the KV-cache arithmetic
    # and so the memory story as well as the quality story.
    #
    # It is also NOT gated, which removes the access-request step that cost a
    # week on Llama.
    "qwen25-7b": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "phase": 2,
        "gated": False,
    },
    "mistral7b": {
        "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "phase": 3,
        "gated": True,
    },
}

# weights key -> (quantize script, needs calibration, vllm load note)
WEIGHTS = {
    "bf16": (None, False, "The original model, loaded with --dtype bfloat16."),
    "fp8": (
        "scripts/quantize_fp8.py",
        False,
        "llm-compressor FP8_DYNAMIC (RTN, per-channel weights, dynamic per-token "
        "activations). Needs compute capability >= 8.9 (Ada/Hopper) to run as "
        "true W8A8; on older cards vLLM falls back to FP8-Marlin W8A16, which is "
        "a DIFFERENT experiment -- check and say which you got.",
    ),
    "int8": (
        "scripts/quantize_int8.py",
        True,
        "llm-compressor SmoothQuant(0.8) + GPTQ W8A8. Needs compute capability "
        "> 7.5. NOT supported on Blackwell (cc >= 10.0) -- if you rent a B200 or "
        "RTX 6000 Blackwell, INT8 drops out of the plan and you say so.",
    ),
    "int4gptq": (
        "scripts/quantize_int4_gptq.py",
        True,
        "llm-compressor GPTQ W4A16, group size 128.",
    ),
    "int4awq": (
        "scripts/quantize_int4_awq.py",
        True,
        "llm-compressor AWQ W4A16, group size 128.",
    ),
}

KV = {
    "bf16": ("auto", "Unquantized KV cache (vLLM default, follows model dtype)."),
    "fp8": (
        "fp8",
        "FP8 KV cache. This is a RUNTIME FLAG, not a checkpoint. Default scales "
        "are 1.0 with no calibration; calibrated per-head scales via "
        "llm-compressor are an optional extra, and a different configuration if "
        "you use them.",
    ),
}

# Phase-1 matrix for Llama. w-bf16/kv-fp8 is deliberately included: without it
# you cannot separate the effect of the FP8 KV cache from the effect of FP8
# weights, because every other FP8-KV row also has quantized weights.
PHASE1 = [
    ("bf16", "bf16"),
    ("fp8", "bf16"),
    ("bf16", "fp8"),
    ("fp8", "fp8"),
    ("int8", "bf16"),
    ("int4gptq", "bf16"),
    ("int4awq", "bf16"),
]
# The replication family gets the SAME grid that produced the findings on
# Llama, minus AWQ. A two-cell second family (bf16 and fp8 weights only) was
# the original plan and it is not enough: the composition result needs the
# isolating KV cell and the combined cell, or there is nothing to replicate.
#
#   (bf16, bf16)  baseline
#   (fp8,  bf16)  weights perturbed alone        \
#   (bf16, fp8)   KV cache perturbed alone        > the composition test
#   (fp8,  fp8)   both                           /
#   (int8, bf16)  the significant MMLU effect
#   (int4gptq, bf16)  the GSM8K-vs-MMLU contrast
#
# AWQ is excluded: it has not built successfully on either family under
# llm-compressor 0.16, and GPTQ already covers W4A16.
REPLICATION = [
    ("bf16", "bf16"),
    ("fp8", "bf16"),
    ("bf16", "fp8"),
    ("fp8", "fp8"),
    ("int8", "bf16"),
    ("int4gptq", "bf16"),
]

PHASE2 = [("bf16", "bf16"), ("fp8", "bf16")]


def render(family: str, w: str, kv: str) -> tuple[str, str]:
    fam = FAMILIES[family]
    name = f"{family}__w-{w}__kv-{kv}"
    script, needs_cal, wnote = WEIGHTS[w]
    kv_dtype, kvnote = KV[kv]

    checkpoint = (
        f"${{CHECKPOINT_DIR}}/{family}__w-{w}" if script else fam["base_model"]
    )
    quant_block = (
        textwrap.dedent(
            f"""\
            quantize:
              script: {script}
              needs_calibration: {str(needs_cal).lower()}
              calibration_config: {"configs/calibration.yaml" if needs_cal else "null"}
              output_dir: ${{CHECKPOINT_DIR}}/{family}__w-{w}
            """
        )
        if script
        else "quantize: null   # baseline, nothing to build\n"
    )

    body = f"""\
# {name}
# Generated by scripts/gen_configs.py -- edit the generator, not this file.

name: {name}
family: {family}
phase: {fam["phase"]}
base_model: {fam["base_model"]}
weights: {w}
kv_cache: {kv}

# Path passed to vLLM. ${{CHECKPOINT_DIR}} is expanded by the scripts.
model_path: {checkpoint}

# Notes that must end up in the report:
weights_note: >
  {wnote}
kv_note: >
  {kvnote}

serve:
  dtype: bfloat16
  # 20480 fits on a 48 GB budget (one L40S, or 2x L4 with TP=2), which keeps
  # the 16k-token benchmark shape runnable. Drop to 8192 on a single 24 GB
  # card and change the long shape in configs/bench_grid.yaml to match.
  max_model_len: 20480
  gpu_memory_utilization: 0.90
  kv_cache_dtype: {kv_dtype}
  # Number of GPUs the model is split across; ${{TP}} comes from the TP
  # environment variable (default 1). The 2x L4 node needs TP=2: an 8B model
  # in BF16 is ~15 GiB and will not leave a usable KV cache on one 24 GB card.
  tensor_parallel_size: ${{TP}}
  port: 8000
  # Added for benchmark servers only; see configs/bench_grid.yaml.
  benchmark_extra_args:
    - --no-enable-prefix-caching

{quant_block}
# lm-eval model_args fragment for this configuration. The scripts build the
# full string; this is the part that is specific to the precision setting.
eval_model_args:
  pretrained: {checkpoint}
  max_model_len: 20480
  kv_cache_dtype: {kv_dtype}
  add_bos_token: true
  gpu_memory_utilization: 0.90
  tensor_parallel_size: ${{TP}}
"""
    return name, body


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for family, matrix in (("llama31-8b", PHASE1), ("qwen25-7b", REPLICATION),
                           ("mistral7b", PHASE2)):
        for w, kv in matrix:
            name, body = render(family, w, kv)
            (OUT / f"{name}.yaml").write_text(body)
            written.append(name)
    print(f"wrote {len(written)} configs to {OUT.relative_to(ROOT)}:")
    for n in written:
        print("  ", n)


if __name__ == "__main__":
    main()
