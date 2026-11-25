#!/usr/bin/env python3
"""INT4 W4A16 via GPTQ. Needs the frozen calibration set.

W4A16: 4-bit weights, 16-bit activations. Weight-only, so the win is memory and
memory bandwidth, not compute. INT4 needs compute capability > 8.0.

The scheme is spelled out explicitly in scripts/w4a16_config.py rather than
passed as scheme="W4A16", so that GPTQ and AWQ provably differ ONLY in the
algorithm. See that file for why this matters.

    python scripts/quantize_int4_gptq.py llama31-8b__w-int4gptq__kv-bf16
    python scripts/quantize_int4_gptq.py llama31-8b__w-int4gptq__kv-bf16 --asym

Verify the API on the day you run this -- modifier signatures move between
llm-compressor releases:
    python -c "from llmcompressor.modifiers.quantization import GPTQModifier; help(GPTQModifier)"
"""
import quantize_common
import w4a16_config


def make_builder(symmetric: bool):
    def build(tokenizer):
        from llmcompressor.modifiers.quantization import GPTQModifier

        recipe = GPTQModifier(
            targets="Linear",
            config_groups=w4a16_config.w4a16_config_groups(symmetric=symmetric),
            ignore=w4a16_config.IGNORE,
        )
        return recipe, {}

    return build


if __name__ == "__main__":
    ap = quantize_common.build_parser(__doc__)
    ap.add_argument("--asym", action="store_true",
                    help="asymmetric weights. Default is symmetric, which is "
                         "what matches the AWQ run for the controlled "
                         "comparison. Only use this for a clearly labelled "
                         "secondary row.")
    a = ap.parse_args()
    print(f"[scheme] {w4a16_config.describe(symmetric=not a.asym)}")
    quantize_common.run_recipe(
        a.config, make_builder(symmetric=not a.asym),
        needs_calibration=True, dry_run=a.dry_run,
    )
