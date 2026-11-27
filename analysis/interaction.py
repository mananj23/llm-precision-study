#!/usr/bin/env python3
"""Is quantization damage compositional?

The matrix contains the cell almost nobody builds: w-bf16 + kv-fp8, which
quantizes the KV cache and NOTHING else. With it, three configurations can be
compared against one baseline on the same prompts:

    W  = FP8 weights, BF16 KV      (weights perturbed alone)
    K  = BF16 weights, FP8 KV      (KV cache perturbed alone)
    C  = FP8 weights, FP8 KV       (both)

and the question becomes whether C's behaviour follows from W's and K's.

Two folk claims are in circulation and they contradict each other:

  "Quantization stacks ... each look fine in isolation but together can break
   in non-obvious ways."                              -> damage compounds
  "KV cache quantization stacks with everything else. Mostly independent of
   weight quantization choices."                      -> damage is independent

Both are testable against per-prompt outcomes, and on Llama-3.1-8B both appear
to be wrong. What this module measures:

  rate_C vs rate_indep   rate_indep = 1-(1-rate_W)(1-rate_K). Compounding
                         predicts rate_C >= rate_indep; independence predicts
                         equality.
  jaccard(W, K)          do the two perturbations damage the SAME prompts?
  novel_fraction         share of C's damaged prompts that are damaged by
                         NEITHER W nor K alone. Independence and compounding
                         both predict this is near zero.

Every quantity gets a bootstrap interval over prompts, because a point
estimate of a set-overlap statistic says nothing about whether it would
survive a different sample of prompts.

IMPORTANT -- the control this rests on:
    These numbers are only interpretable against the run-to-run noise floor.
    Two identical runs of the baseline disagree on some prompts (vLLM's
    continuous batching is not bitwise deterministic across runs), and that
    disagreement rate is the floor for every number here. Pass --floor with
    the measured baseline-vs-baseline damage rate; the report then states
    each estimate net of it. Without --floor the module says so loudly rather
    than letting the numbers be read as if the floor were zero.

Usage:
    python analysis/interaction.py --task gsm8k
    python analysis/interaction.py --task mmlu --floor 0.004
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def load(parquet: pathlib.Path, task: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    # MMLU and LongBench arrive as ~57 and ~21 subtasks. doc_id restarts at 0
    # in every subtask, so the pairing key has to carry the subtask name or
    # rows from different subjects collide.
    df["group"] = df["task"].str.replace(r"^(mmlu|longbench)_.*", r"\1", regex=True)
    df["key"] = df["task"] + ":" + df["doc_id"].astype(str)
    return df[df["group"] == task]


def aligned(df: pd.DataFrame, configs: dict[str, str]) -> pd.DataFrame:
    """One row per prompt, one column per configuration, prompts present in all."""
    cols = {}
    for label, cfg in configs.items():
        sub = df[df["config"] == cfg]
        if sub.empty:
            raise SystemExit(
                f"configuration not found in the data: {cfg}\n"
                f"available: {sorted(df['config'].unique())}"
            )
        cols[label] = sub.groupby("key")["correct"].first()
    out = pd.concat(cols, axis=1).dropna()
    if out.empty:
        raise SystemExit("no prompts are shared by all four configurations")
    return out


def damage_sets(j: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks: baseline CORRECT and the configuration WRONG.

    Damage only, deliberately. A flip from wrong to right is not evidence that
    quantization preserved anything -- counting both directions mixes damage
    with luck, and the composition question is about damage.
    """
    b = j["B"].to_numpy() == 1
    return {
        "W": b & (j["W"].to_numpy() == 0),
        "K": b & (j["K"].to_numpy() == 0),
        "C": b & (j["C"].to_numpy() == 0),
    }


def statistics_from(masks: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    w, k, c = masks["W"][idx], masks["K"][idx], masks["C"][idx]
    n = len(idx)
    rw, rk, rc = w.mean(), k.mean(), c.mean()
    union = w | k
    inter = w & k
    novel = c & ~union
    return {
        "n": n,
        "rate_W": rw,
        "rate_K": rk,
        "rate_C": rc,
        "rate_indep": 1.0 - (1.0 - rw) * (1.0 - rk),
        # the headline contrast: observed minus what independence predicts
        "excess_over_indep": rc - (1.0 - (1.0 - rw) * (1.0 - rk)),
        "jaccard_WK": inter.sum() / union.sum() if union.sum() else np.nan,
        "novel_fraction": novel.sum() / c.sum() if c.sum() else np.nan,
        "novel_rate": novel.mean(),
    }


def bootstrap(masks: dict[str, np.ndarray], n_resamples: int, seed: int) -> dict:
    n = len(masks["W"])
    point = statistics_from(masks, np.arange(n))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in point if k != "n"}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        s = statistics_from(masks, idx)
        for k in draws:
            draws[k].append(s[k])

    out = {"n_prompts": n, "n_resamples": n_resamples}
    for k, vals in draws.items():
        a = np.asarray(vals, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            out[k] = {"point": float("nan")}
            continue
        out[k] = {
            "point": float(point[k]),
            "ci_lo": float(np.percentile(a, 2.5)),
            "ci_hi": float(np.percentile(a, 97.5)),
            "se": float(a.std(ddof=1)),
        }
    return out


def fmt(v: dict, pct: bool = True) -> str:
    if "ci_lo" not in v:
        return "n/a"
    s = 100.0 if pct else 1.0
    u = " pp" if pct else ""
    return (f"{v['point'] * s:+.2f}{u}  [{v['ci_lo'] * s:+.2f}, "
            f"{v['ci_hi'] * s:+.2f}]")


def report(res: dict, task: str, floor: float | None) -> None:
    n = res["n_prompts"]
    print(f"\n=== is damage compositional?  task={task}  n={n}  "
          f"({res['n_resamples']} resamples) ===\n")

    print("  damage rates (baseline correct -> configuration wrong)")
    for label, name in [("rate_W", "FP8 weights only "),
                        ("rate_K", "FP8 KV cache only"),
                        ("rate_C", "both            ")]:
        v = res[label]
        print(f"    {name}  {v['point'] * 100:6.2f}%  "
              f"[{v['ci_lo'] * 100:.2f}, {v['ci_hi'] * 100:.2f}]")
    v = res["rate_indep"]
    print(f"    if independent      {v['point'] * 100:6.2f}%  "
          f"[{v['ci_lo'] * 100:.2f}, {v['ci_hi'] * 100:.2f}]")

    print("\n  the three tests")
    ex = res["excess_over_indep"]
    print(f"    observed - independence   {fmt(ex)}")
    if ex["ci_hi"] < 0:
        print("      -> SUB-independent: stacking damages FEWER prompts than")
        print("         two independent perturbations would. Compounding is")
        print("         not what happens.")
    elif ex["ci_lo"] > 0:
        print("      -> SUPER-independent: damage compounds, as the folk")
        print("         guidance says.")
    else:
        print("      -> consistent with independence; interval contains zero.")

    jc = res["jaccard_WK"]
    print(f"    Jaccard(weights, KV)      {fmt(jc, pct=False)}")
    if jc["ci_hi"] < 0.5:
        print("      -> the two perturbations damage LARGELY DIFFERENT prompts.")

    nf = res["novel_fraction"]
    print(f"    damaged by neither alone  {fmt(nf)}")
    if floor is not None:
        # The floor is a rate; express it as a share of C's damage so it is
        # comparable to novel_fraction.
        share = floor / res["rate_C"]["point"] if res["rate_C"]["point"] else float("nan")
        print(f"      noise floor is {floor * 100:.2f}% of prompts, i.e. about "
              f"{share * 100:.1f}% of the")
        print(f"      damage attributed to the combination. Net of it, the novel")
        print(f"      share is roughly {(nf['point'] - share) * 100:.1f}%.")
        if nf["ci_lo"] <= share:
            print("      !!! the interval reaches the floor: this is NOT separable")
            print("          from run-to-run noise. Do not claim it.")
    else:
        print("      !!! NO NOISE FLOOR SUPPLIED. Two identical baseline runs")
        print("          disagree on some prompts, and that rate is the floor")
        print("          for this number. Measure it and pass --floor, or this")
        print("          figure cannot be interpreted.")

    print("\n  reading: independence and compounding both predict the novel")
    print("  share is near zero -- C's damage should be built from W's and K's.")
    print("  A large novel share with low Jaccard means the stacked")
    print("  configuration is its own model, not the sum of two perturbations.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results/logs/per_prompt.parquet"))
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--family", default="llama31-8b")
    ap.add_argument("--floor", type=float, default=None,
                    help="measured baseline-vs-baseline DAMAGE rate, as a "
                         "fraction. Required to interpret novel_fraction.")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    f = args.family
    configs = {
        "B": f"{f}__w-bf16__kv-bf16",
        "W": f"{f}__w-fp8__kv-bf16",
        "K": f"{f}__w-bf16__kv-fp8",
        "C": f"{f}__w-fp8__kv-fp8",
    }

    df = load(pathlib.Path(args.results), args.task)
    j = aligned(df, configs)
    res = bootstrap(damage_sets(j), args.n_resamples, args.seed)
    res["task"] = args.task
    res["family"] = args.family
    res["configs"] = configs
    res["noise_floor"] = args.floor
    report(res, args.task, args.floor)

    if args.json_out:
        p = pathlib.Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2, default=float))
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
