#!/usr/bin/env python3
"""ONE explicit 4-bit weight config, shared by GPTQ and AWQ.

Why this file exists
--------------------
The llm-compressor reference examples do not use the same 4-bit scheme for the
two algorithms: the GPTQ example passes `scheme="W4A16"` (symmetric) and the
AWQ example passes `scheme="W4A16_ASYM"` (asymmetric). Taking each example's
default would mean the GPTQ-vs-AWQ comparison differs in TWO ways at once --
the algorithm and the zero-point -- and no difference between them could be
attributed to the algorithm.

Asymmetric quantization keeps a per-group zero point, so it fits a skewed
weight distribution better and usually scores higher; it is also what most
published AWQ checkpoints ship. So both settings are worth having, but they
must be chosen deliberately rather than inherited from whichever example page
was copied:

    symmetric=True   both algorithms matched -> THE controlled comparison.
                     This is the pair that goes in the results table.
    symmetric=False  each algorithm as it is usually shipped. Run it as a
                     secondary row if there is time, and label it clearly.

Everything else -- 4 bits, group size 128, group strategy, lm_head excluded --
is held fixed across both.
"""
from __future__ import annotations

GROUP_SIZE = 128
NUM_BITS = 4
IGNORE = ["lm_head"]


def w4a16_config_groups(symmetric: bool = True, actorder: str | None = "weight"):
    """Explicit config_groups for a 4-bit weight-only scheme.

    Spelled out rather than relying on the string scheme names, so that the
    checkpoint's settings are visible in this repository and cannot change
    underneath the project when a library default moves.
    """
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationScheme,
        QuantizationStrategy,
        QuantizationType,
    )

    return {
        "config_group": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=NUM_BITS,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                group_size=GROUP_SIZE,
                symmetric=symmetric,
                dynamic=False,
                actorder=actorder,
            ),
        ),
    }


def describe(symmetric: bool) -> str:
    return (
        f"W{NUM_BITS}A16, group size {GROUP_SIZE}, "
        f"{'symmetric' if symmetric else 'asymmetric'}, lm_head excluded"
    )
