#!/usr/bin/env python3
"""Paired bootstrap over per-prompt correctness.

Why paired, in one paragraph you can put in the report:

    Two configurations are scored on the SAME prompts. Prompt difficulty is the
    largest source of variance in an accuracy estimate, and it is shared between
    the two runs, so it cancels when you take a per-prompt difference. Computing
    two accuracies with independent standard errors and subtracting throws that
    cancellation away and gives an interval that is far too wide -- wide enough
    that a real 1-point regression looks like noise. Resampling PROMPTS (not
    runs) and recomputing the difference on each resample keeps the pairing and
    gives the correct, much tighter interval.

The unit of resampling is the prompt. Everything here is vectorised, so 10,000
resamples over a few thousand prompts takes well under a second.

    python analysis/bootstrap.py --help
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 20261116,
) -> dict:
    """Interval on mean(a) - mean(b), paired elementwise.

    a, b: per-prompt scores for the two configurations, SAME prompts, SAME order.
    Returns point estimate, CI bounds, and whether the interval contains zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    n = a.size
    if n == 0:
        raise ValueError("no paired prompts")

    diff = a - b
    point = float(diff.mean())

    rng = np.random.default_rng(seed)
    # Resample prompt INDICES, then average the per-prompt differences. This is
    # exactly equivalent to recomputing both accuracies on the resampled prompts
    # and subtracting, but makes the pairing impossible to get wrong.
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = diff[idx].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])

    return {
        "n_prompts": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_level": ci,
        "n_resamples": int(n_resamples),
        # The headline judgement. If this is True you must write "no detectable
        # difference" and you must NOT rank this configuration against the
        # baseline.
        "contains_zero": bool(lo <= 0.0 <= hi),
        "bootstrap_se": float(boot.std(ddof=1)),
    }


def unpaired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000,
    ci: float = 0.95, seed: int = 20261116,
) -> dict:
    """Deliberately WRONG-for-this-purpose version, kept for one figure.

    Resamples the two sets independently, i.e. pretends they were scored on
    different prompts. Use it once, in the report, to show how much wider the
    interval becomes when you discard the pairing -- then never again.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ba = a[rng.integers(0, a.size, size=(n_resamples, a.size))].mean(axis=1)
    bb = b[rng.integers(0, b.size, size=(n_resamples, b.size))].mean(axis=1)
    boot = ba - bb
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return {"diff": float(a.mean() - b.mean()), "ci_lo": float(lo),
            "ci_hi": float(hi), "contains_zero": bool(lo <= 0 <= hi)}


def compare_to_baseline(
    df: pd.DataFrame,
    baseline_config: str,
    *,
    group_cols: tuple[str, ...] = ("family", "task"),
    config_col: str = "config",
    doc_col: str = "doc_id",
    score_col: str = "correct",
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 20261116,
) -> pd.DataFrame:
    """Every configuration vs the baseline, per group, paired on doc_id.

    Prompts are intersected on doc_id within each group, so a configuration
    evaluated on fewer prompts is compared only on the prompts both ran. The
    number actually used is reported as n_prompts -- check it.
    """
    out = []
    for keys, g in df.groupby(list(group_cols), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = g[g[config_col] == baseline_config]
        if base.empty:
            print(f"  [skip] {keys}: no baseline {baseline_config!r} in this group")
            continue
        base_s = base.set_index(doc_col)[score_col]
        base_s = base_s[~base_s.index.duplicated(keep="last")]

        for cfg, cg in g.groupby(config_col, sort=False):
            cfg_s = cg.set_index(doc_col)[score_col]
            cfg_s = cfg_s[~cfg_s.index.duplicated(keep="last")]
            shared = base_s.index.intersection(cfg_s.index)
            if len(shared) == 0:
                continue
            dropped = max(len(base_s), len(cfg_s)) - len(shared)
            if cfg == baseline_config:
                res = {
                    "n_prompts": len(shared), "mean_a": float(base_s.loc[shared].mean()),
                    "mean_b": float(base_s.loc[shared].mean()), "diff": 0.0,
                    "ci_lo": 0.0, "ci_hi": 0.0, "ci_level": ci,
                    "n_resamples": 0, "contains_zero": True, "bootstrap_se": 0.0,
                }
            else:
                res = paired_bootstrap_diff(
                    cfg_s.loc[shared].to_numpy(),
                    base_s.loc[shared].to_numpy(),
                    n_resamples=n_resamples, ci=ci, seed=seed,
                )
            row = dict(zip(group_cols, keys))
            row[config_col] = cfg
            row["baseline"] = baseline_config
            row["prompts_dropped_unpaired"] = int(dropped)
            row.update(res)
            out.append(row)

    res_df = pd.DataFrame(out)
    if not res_df.empty:
        res_df = res_df.rename(
            columns={"mean_a": "accuracy", "mean_b": "baseline_accuracy",
                     "diff": "delta_vs_baseline"}
        )
    return res_df


def format_interval(row, pct: bool = True, digits: int = 1) -> str:
    """'-0.8 [-1.9, +0.3]' or 'no detectable difference' -- report-ready."""
    scale = 100.0 if pct else 1.0
    d, lo, hi = row["delta_vs_baseline"] * scale, row["ci_lo"] * scale, row["ci_hi"] * scale
    body = f"{d:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"
    return body + (" (n.d.)" if row["contains_zero"] else "")


if __name__ == "__main__":
    # Self-check: a known-size effect must be recovered, and the paired interval
    # must be tighter than the unpaired one on correlated data.
    rng = np.random.default_rng(0)
    n = 2000
    difficulty = rng.random(n)                 # shared prompt difficulty
    base = (rng.random(n) < 1 - difficulty).astype(float)
    # quantized version: same prompts, flips 2% of the prompts the baseline got
    # right. The realised effect is therefore -0.02 * baseline_accuracy, NOT
    # -0.02 -- which is the kind of slip the assertion below exists to catch.
    flip = rng.random(n) < 0.02
    quant = np.where(flip & (base == 1), 0.0, base)
    true_effect = -0.02 * base.mean()

    p = paired_bootstrap_diff(quant, base)
    u = unpaired_bootstrap_diff(quant, base)
    print("paired  :", {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()})
    print("unpaired:", {k: round(v, 4) if isinstance(v, float) else v for k, v in u.items()})
    pw = p["ci_hi"] - p["ci_lo"]
    uw = u["ci_hi"] - u["ci_lo"]
    print(f"\npaired interval is {uw / pw:.1f}x tighter than unpaired "
          f"({pw * 100:.2f} vs {uw * 100:.2f} points)")
    assert pw < uw, "paired interval should be tighter"
    assert p["ci_lo"] <= true_effect <= p["ci_hi"], (
        f"true effect {true_effect:.4f} outside paired interval "
        f"[{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]"
    )
    assert not p["contains_zero"], "a real 1-point effect should be detectable"
    assert u["contains_zero"], (
        "the unpaired interval is expected to MISS this real effect -- that is "
        "the whole reason the project uses a paired bootstrap"
    )
    print(f"self-check OK (true effect {true_effect * 100:+.2f} points, recovered)")
