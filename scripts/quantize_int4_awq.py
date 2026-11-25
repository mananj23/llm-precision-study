#!/usr/bin/env python3
"""INT4 W4A16 via AWQ. Needs the frozen calibration set.

AWQ and GPTQ are the head-to-head pair at 4 bits: AWQ protects salient weight
channels identified from activation statistics, GPTQ minimises layerwise output
error. Same bit width, same group size, same symmetry, same calibration set --
so a difference between them is a difference between the two algorithms. That
is the one genuinely controlled comparison in this project; protect it, and see
scripts/w4a16_config.py for the trap that nearly broke it.

API NOTE (checked 2026-11-16). In current llm-compressor the AWQModifier is
transform-only: it rescales activation channels and does NOT quantize. It must
be paired with a modifier that performs the quantization:

    recipe = [AWQModifier(), QuantizationModifier(...)]

In releases up to roughly 0.10 the AWQModifier inherited QuantizationMixin and
did both, so a single modifier was enough. This script detects which behaviour
your installed version has instead of guessing. Confirm with:
    python -c "from llmcompressor.modifiers.awq import AWQModifier; print(AWQModifier.__mro__)"

    python scripts/quantize_int4_awq.py llama31-8b__w-int4awq__kv-bf16
    python scripts/quantize_int4_awq.py llama31-8b__w-int4awq__kv-bf16 --asym

If the recipe is still failing after two days, fall back to AutoAWQ or a
published checkpoint, and state in the report who built it and how.
"""
import quantize_common
import w4a16_config


def awq_is_transform_only() -> bool:
    """True when AWQModifier needs a separate quantizing modifier alongside it.

    Decided by inspecting the installed class rather than by version string,
    because the version at which this changed is not something to hard-code.
    """
    from llmcompressor.modifiers.awq import AWQModifier

    names = {c.__name__ for c in AWQModifier.__mro__}
    return "QuantizationMixin" not in names


def make_builder(symmetric: bool):
    def build(tokenizer):
        from llmcompressor.modifiers.awq import AWQModifier

        groups = w4a16_config.w4a16_config_groups(symmetric=symmetric)

        if awq_is_transform_only():
            from llmcompressor.modifiers.quantization import QuantizationModifier

            print("[awq] transform-only AWQModifier detected: pairing it with "
                  "QuantizationModifier (current llm-compressor layout)")
            recipe = [
                AWQModifier(),
                QuantizationModifier(
                    targets=["Linear"],
                    config_groups=groups,
                    ignore=w4a16_config.IGNORE,
                ),
            ]
        else:
            print("[awq] AWQModifier quantizes directly (older llm-compressor "
                  "layout): using it alone")
            recipe = [
                AWQModifier(
                    targets=["Linear"],
                    config_groups=groups,
                    ignore=w4a16_config.IGNORE,
                ),
            ]
        return recipe, {}

    return build


if __name__ == "__main__":
    ap = quantize_common.build_parser(__doc__)
    ap.add_argument("--asym", action="store_true",
                    help="asymmetric weights, which is what the AWQ reference "
                         "example and most published AWQ checkpoints use. The "
                         "DEFAULT here is symmetric, to match the GPTQ run. "
                         "Mixing the two silently is how the controlled "
                         "comparison gets lost.")
    a = ap.parse_args()
    print(f"[scheme] {w4a16_config.describe(symmetric=not a.asym)}")
    quantize_common.run_recipe(
        a.config, make_builder(symmetric=not a.asym),
        needs_calibration=True, dry_run=a.dry_run,
    )
