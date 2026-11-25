#!/usr/bin/env python3
"""INT8 W8A8 via SmoothQuant + GPTQ. Needs the frozen calibration set.

Recipe from the vLLM / llm-compressor reference example (checked 2026-11-16):
    [SmoothQuantModifier(smoothing_strength=0.8),
     GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])]

SmoothQuant shifts the quantization difficulty from activations (which have
outlier channels in Llama-class models) into the weights, where GPTQ can
handle it. Running GPTQ W8A8 without SmoothQuant is a different, worse recipe.

NOT supported on Blackwell (compute capability >= 10.0).

    python scripts/quantize_int8.py llama31-8b__w-int8__kv-bf16
"""
import quantize_common


def build(tokenizer):
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]
    return recipe, {}


if __name__ == "__main__":
    a = quantize_common.parse_args(__doc__)
    quantize_common.run_recipe(a.config, build, needs_calibration=True, dry_run=a.dry_run)
