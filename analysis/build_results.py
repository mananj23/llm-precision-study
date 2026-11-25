#!/usr/bin/env python3
"""Build the ONE tidy results file that the report's table and plots come from.

    python analysis/build_results.py

Inputs (produced by scripts/collect_logs.py and scripts/collect_bench.py):
    results/logs/per_prompt.parquet
    results/logs/system.parquet

Outputs:
    analysis/results.parquet      one row per (family, config, task)
    analysis/results.csv          same, for eyeballing
    analysis/system_summary.csv   median-over-seeds system numbers

Never edit the outputs by hand. If a number is wrong, fix it here and re-run,
so the table always regenerates from the raw files with one command.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import common  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bootstrap import compare_to_baseline  # noqa: E402

ROOT = common.ROOT


def summarise_system(sys_df: pd.DataFrame) -> pd.DataFrame:
    """Median across the three seeds; keep the spread so it can be reported."""
    keys = ["family", "config", "weights", "kv_cache", "shape", "concurrency"]
    num = [c for c in sys_df.columns
           if sys_df[c].dtype.kind == "f" and c not in ("seed",)]
    agg = sys_df.groupby(keys)[num].median()
    spread = (
        sys_df.groupby(keys)["output_throughput_tok_s"]
        .agg(tput_min="min", tput_max="max", n_seeds="count")
    )
    out = agg.join(spread).reset_index()
    # Relative spread across repeats IS the system noise floor. If it is large,
    # say so instead of reporting a confident ranking.
    out["tput_spread_pct"] = (
        100 * (out["tput_max"] - out["tput_min"]) / out["output_throughput_tok_s"]
    )
    out["kv_cache_tokens"] = (
        sys_df.groupby(keys)["kv_cache_tokens"].first().reset_index(drop=True)
        if "kv_cache_tokens" in sys_df else np.nan
    )
    return out


def add_cost(sys_sum: pd.DataFrame, price_per_hour: float | None) -> pd.DataFrame:
    """cost per million output tokens = price/hr / (tok/s * 3600) * 1e6."""
    df = sys_sum.copy()
    if price_per_hour is None or not np.isfinite(price_per_hour):
        df["cost_per_m_output_tokens"] = np.nan
        return df
    tps = df["output_throughput_tok_s"].replace(0, np.nan)
    df["cost_per_m_output_tokens"] = price_per_hour / (tps * 3600.0) * 1e6
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-prompt", default="results/logs/per_prompt.parquet")
    ap.add_argument("--system", default="results/logs/system.parquet")
    ap.add_argument("--baseline", default=None,
                    help="baseline config; default = <family>__w-bf16__kv-bf16")
    ap.add_argument("--outdir", default="analysis")
    ap.add_argument("--price-per-hour", type=float, default=None,
                    help="override configs/bench_grid.yaml. Use it for a price "
                         "sensitivity check: if the Pareto frontier changes when "
                         "the GPU price changes, say so in the report.")
    args = ap.parse_args()

    bench_cfg = common.load_yaml("configs/bench_grid.yaml")
    eval_cfg = common.load_yaml("configs/eval.yaml")
    price = args.price_per_hour or bench_cfg["cost"].get("gpu_price_per_hour")
    ref_conc = bench_cfg["reference_concurrency"]

    pp_path = ROOT / args.per_prompt
    if not pp_path.exists():
        print(f"missing {pp_path}\nrun: python scripts/collect_logs.py")
        return 1
    pp = pd.read_parquet(pp_path)
    print(f"per-prompt rows: {len(pp):,}  configs: {pp['config'].nunique()}  "
          f"tasks: {pp['task'].nunique()}")

    # ---------------------------------------------------- quality + intervals --
    frames = []
    for family, g in pp.groupby("family"):
        baseline = args.baseline or f"{family}__w-bf16__kv-bf16"
        if baseline not in set(g["config"]):
            print(f"\n[{family}] NO BASELINE {baseline!r}. Every number in this "
                  f"project is a difference from BF16; run the baseline first.")
            continue
        print(f"\n[{family}] baseline = {baseline}")
        res = compare_to_baseline(
            g, baseline,
            group_cols=("family", "task"),
            n_resamples=eval_cfg["bootstrap"]["n_resamples"],
            ci=eval_cfg["bootstrap"]["ci"],
            seed=eval_cfg["bootstrap"]["seed"],
        )
        frames.append(res)
    if not frames:
        return 1
    quality = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------- system -----
    sys_path = ROOT / args.system
    if sys_path.exists():
        sys_df = pd.read_parquet(sys_path)
        sys_sum = add_cost(summarise_system(sys_df), price)
        ref = sys_sum[
            (sys_sum["concurrency"] == ref_conc) & (sys_sum["shape"] == "short")
        ]
        keep = ["config", "output_throughput_tok_s", "ttft_ms_median",
                "tpot_ms_median", "cost_per_m_output_tokens", "kv_cache_tokens",
                "tput_spread_pct"]
        merged = quality.merge(ref[keep], on="config", how="left")
        if ref.empty:
            print(f"\nNo system rows at concurrency={ref_conc}, shape=short. "
                  "Quality table will have empty cost columns.")
    else:
        print(f"\nno {sys_path} yet -- quality table only "
              "(run scripts/collect_bench.py after the benchmarks)")
        sys_sum = pd.DataFrame()
        merged = quality.copy()
        for c in ("output_throughput_tok_s", "ttft_ms_median", "tpot_ms_median",
                  "cost_per_m_output_tokens", "kv_cache_tokens", "tput_spread_pct"):
            merged[c] = np.nan

    # ----------------------------------------- cost per correct answer --------
    # Puts quality and price in one number: what you pay for an answer that is
    # actually right. A configuration that is cheap but wrong looks bad here,
    # which is the point.
    merged["cost_per_correct_answer"] = np.where(
        merged["accuracy"] > 0,
        merged["cost_per_m_output_tokens"] / merged["accuracy"],
        np.nan,
    )

    merged["delta_pct_points"] = merged["delta_vs_baseline"] * 100
    merged["ci_lo_pct_points"] = merged["ci_lo"] * 100
    merged["ci_hi_pct_points"] = merged["ci_hi"] * 100
    merged["verdict"] = np.where(
        merged["contains_zero"], "no detectable difference",
        np.where(merged["delta_vs_baseline"] < 0, "worse than BF16", "better than BF16"),
    )

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(outdir / "results.parquet", index=False)
    merged.to_csv(outdir / "results.csv", index=False)
    if not sys_sum.empty:
        sys_sum.to_csv(outdir / "system_summary.csv", index=False)

    # ------------------------------------------------------------- report -----
    print("\n" + "=" * 78)
    print("CHANGE FROM BF16, percentage points (95% paired bootstrap interval)")
    print("=" * 78)
    for family, g in merged.groupby("family"):
        print(f"\n{family}")
        piv = g.pivot_table(index="config", columns="task",
                            values="delta_pct_points", aggfunc="first")
        lo = g.pivot_table(index="config", columns="task",
                           values="ci_lo_pct_points", aggfunc="first")
        hi = g.pivot_table(index="config", columns="task",
                           values="ci_hi_pct_points", aggfunc="first")
        nd = g.pivot_table(index="config", columns="task",
                           values="contains_zero", aggfunc="first")
        cells = piv.copy().astype(object)
        for r in piv.index:
            for c in piv.columns:
                if pd.isna(piv.loc[r, c]):
                    cells.loc[r, c] = "--"
                else:
                    s = f"{piv.loc[r, c]:+.1f} [{lo.loc[r, c]:+.1f},{hi.loc[r, c]:+.1f}]"
                    if nd.loc[r, c]:
                        s += " n.d."
                    cells.loc[r, c] = s
        print(cells.to_string())

    nd_count = int(merged["contains_zero"].sum()) - merged["config"].str.contains(
        "w-bf16__kv-bf16").sum()
    print(f"\n{nd_count} of {len(merged)} comparisons show NO DETECTABLE DIFFERENCE.")
    print("Do not rank configurations whose intervals overlap. Write "
          "'no detectable difference' and mean it.")
    print(f"\nwrote {outdir / 'results.parquet'} and results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
