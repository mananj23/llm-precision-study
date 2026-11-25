#!/usr/bin/env python3
"""FP8 W8A8 (dynamic). No calibration data needed.

Recipe from the vLLM / llm-compressor reference example (checked 2026-11-16):
    QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

FP8_DYNAMIC = static per-channel weight scales + dynamic per-token activation
scales. Because activations are scaled at runtime, there is nothing to
calibrate -- which is why this is the first recipe to build and the one most
likely to be in the 6 Dec note.

    python scripts/quantize_fp8.py llama31-8b__w-fp8__kv-bf16
"""
import quantize_common


def build(tokenizer):
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )
    return recipe, {}


if __name__ == "__main__":
    a = quantize_common.parse_args(__doc__)
    quantize_common.run_recipe(a.config, build, needs_calibration=False, dry_run=a.dry_run)
