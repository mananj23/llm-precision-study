#!/usr/bin/env python3
"""Guards against the three mistakes that would silently invalidate the project.

    python tests/test_guards.py

1. Leakage: a router feature that is only known after the model has answered.
2. Unpaired intervals: using independent standard errors instead of a paired
   bootstrap, which hides real regressions.
3. Unpaired prompts: comparing two configurations on different prompt sets.

These are not unit tests of convenience. Each one corresponds to a specific way
the headline result could be wrong in a way nobody would notice from the plots.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "router"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))

import dataset as D  # noqa: E402
from bootstrap import compare_to_baseline, paired_bootstrap_diff, unpaired_bootstrap_diff  # noqa: E402

FAILURES: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        FAILURES.append(name)
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        FAILURES.append(name)


# ---------------------------------------------------------------- 1. leakage --
def test_leakage_guard_rejects_outcome_features():
    for bad in ("correct", "output_len", "output_text"):
        try:
            D.assert_no_leakage(["input_len", bad])
        except D.LeakageError:
            continue
        raise AssertionError(f"leakage guard let {bad!r} through")


def test_real_features_pass_the_guard():
    df = pd.DataFrame({
        "task": ["gsm8k"] * 60 + ["mmlu"] * 60,
        "prompt_text": [f"question about numbers {i}" for i in range(120)],
        "input_len": np.random.default_rng(0).integers(50, 500, 120),
    })
    X, names = D.make_features(df, use_task_flag=True, n_text_components=4)
    assert X.shape[0] == 120, "wrong row count"
    assert not (set(names) & D.POST_HOC_COLUMNS), "a post-hoc column reached the features"


# ------------------------------------------------------------- 2. pairing ----
def test_paired_detects_what_unpaired_misses():
    """The reason the project specifies a paired bootstrap at all."""
    rng = np.random.default_rng(1)
    n = 1500
    difficulty = rng.random(n)
    base = (rng.random(n) < 1 - difficulty).astype(float)
    flip = rng.random(n) < 0.03
    quant = np.where(flip & (base == 1), 0.0, base)

    p = paired_bootstrap_diff(quant, base, n_resamples=4000)
    u = unpaired_bootstrap_diff(quant, base, n_resamples=4000)
    pw, uw = p["ci_hi"] - p["ci_lo"], u["ci_hi"] - u["ci_lo"]
    assert pw < uw, f"paired interval ({pw:.4f}) not tighter than unpaired ({uw:.4f})"
    assert not p["contains_zero"], "paired bootstrap failed to detect a real regression"
    assert u["contains_zero"], (
        "unpaired bootstrap unexpectedly detected it; the demonstration in the "
        "report depends on it NOT doing so"
    )


def test_identical_inputs_give_zero_difference():
    x = np.random.default_rng(2).integers(0, 2, 500).astype(float)
    r = paired_bootstrap_diff(x, x.copy(), n_resamples=1000)
    assert r["diff"] == 0.0, "identical inputs must give zero difference"
    assert r["ci_lo"] == 0.0 and r["ci_hi"] == 0.0, "zero difference must have a zero interval"
    assert r["contains_zero"], "zero difference must read as no detectable difference"


def test_mismatched_shapes_rejected():
    try:
        paired_bootstrap_diff(np.zeros(10), np.zeros(11))
    except ValueError:
        return
    raise AssertionError("arrays of different length must be rejected, not broadcast")


# ------------------------------------------------- 3. prompts must be paired --
def test_compare_intersects_on_doc_id():
    """A config evaluated on fewer prompts must be compared only on the overlap."""
    rows = []
    for cfg, n in (("fam__w-bf16__kv-bf16", 100), ("fam__w-fp8__kv-bf16", 60)):
        for i in range(n):
            rows.append({"family": "fam", "task": "gsm8k", "config": cfg,
                         "doc_id": i, "correct": float(i % 2)})
    df = pd.DataFrame(rows)
    res = compare_to_baseline(df, "fam__w-bf16__kv-bf16", n_resamples=500)
    row = res[res["config"] == "fam__w-fp8__kv-bf16"].iloc[0]
    assert row["n_prompts"] == 60, (
        f"expected the 60 shared prompts, got {row['n_prompts']}"
    )
    assert row["prompts_dropped_unpaired"] == 40, "unpaired prompts not reported"


def test_router_label_definition():
    """sufficient = cheap correct OR expensive also wrong."""
    pp = pd.DataFrame({
        "config": ["c"] * 4 + ["e"] * 4,
        "family": ["f"] * 8, "weights": ["w"] * 8, "kv_cache": ["k"] * 8,
        "task": ["t"] * 8, "doc_id": [0, 1, 2, 3] * 2,
        "run_id": ["r"] * 8, "metric_name": ["m"] * 8,
        "correct": [1.0, 1.0, 0.0, 0.0,   # cheap:     y y n n
                    1.0, 0.0, 1.0, 0.0],  # expensive: y n y n
        "input_len": [10] * 8, "output_len": [5] * 8,
        "prompt_text": ["p"] * 8, "output_text": ["o"] * 8,
    })
    d = D.build_pair_dataset(pp, "c", "e")
    got = d.sort_values("doc_id")["sufficient"].tolist()
    # doc0 cheap right -> 1 ; doc1 cheap right -> 1 ;
    # doc2 cheap wrong, expensive right -> 0 ; doc3 both wrong -> 1
    assert got == [1, 1, 0, 1], f"label definition wrong: {got}"


if __name__ == "__main__":
    print("guards against the three silent-failure modes:")
    check("leakage guard rejects outcome features", test_leakage_guard_rejects_outcome_features)
    check("routing-time features pass the guard", test_real_features_pass_the_guard)
    check("paired bootstrap detects what unpaired misses", test_paired_detects_what_unpaired_misses)
    check("identical inputs -> zero difference", test_identical_inputs_give_zero_difference)
    check("mismatched array lengths rejected", test_mismatched_shapes_rejected)
    check("comparison intersects on doc_id", test_compare_intersects_on_doc_id)
    check("router label definition", test_router_label_definition)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} GUARD(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("all guards passed")
