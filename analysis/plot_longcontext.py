#!/usr/bin/env python3
"""Second figure: long-context accuracy against input length, per KV setting.

    python analysis/plot_longcontext.py

This is the figure that answers the KV-cache question. An FP8 KV cache stores
more tokens in the same memory, which should raise throughput and let longer
contexts fit -- but the cache is where long-context accuracy lives, so if
quality degrades anywhere it should degrade here, and more at longer inputs.

Prompts are bucketed by input length (from the per-prompt log), and accuracy is
plotted per bucket with a bootstrap interval. Buckets with fewer than
--min-bucket prompts are dropped rather than plotted as a noisy point -- an
accuracy computed on nine prompts is not a measurement.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plotstyle  # noqa: E402
from plotstyle import apply_style, color_for, finish, short_name  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_EDGES = [0, 1000, 2000, 4000, 8000, 16000, 32000, 1_000_000]


def bucket_accuracy(g: pd.DataFrame, edges, n_boot=2000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = g.copy()
    g["bucket"] = pd.cut(g["input_len"], bins=edges, right=False)
    rows = []
    for b, bg in g.groupby("bucket", observed=True):
        v = bg["correct"].to_numpy(float)
        if v.size == 0:
            continue
        boot = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "bucket": b, "center": (b.left + min(b.right, edges[-2])) / 2,
            "n": v.size, "acc": v.mean(), "lo": lo, "hi": hi,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-prompt", default="results/logs/per_prompt.parquet")
    ap.add_argument("--task", default="longbench")
    ap.add_argument("--family", default=None)
    ap.add_argument("--min-bucket", type=int, default=25)
    ap.add_argument("--all-configs", action="store_true",
                    help="plot every configuration. Default is the four that "
                         "isolate the KV-cache question, because seven "
                         "translucent bands on one axis are unreadable and this "
                         "figure exists to answer one question.")
    ap.add_argument("--out", default="report/fig_longcontext.png")
    args = ap.parse_args()

    path = ROOT / args.per_prompt
    if not path.exists():
        print(f"missing {path}\nrun: python scripts/collect_logs.py")
        return 1
    df = pd.read_parquet(path)
    family = args.family or df["family"].iloc[0]
    df = df[(df["family"] == family) & (df["task"] == args.task)]
    if df.empty:
        print(f"no rows for task={args.task!r} family={family!r}. "
              f"available tasks: {sorted(pd.read_parquet(path)['task'].unique())}")
        return 1

    # The 2x2 that isolates the KV cache: weights held fixed, KV varied.
    KV_CONTRAST = ("w-bf16__kv-bf16", "w-bf16__kv-fp8",
                   "w-fp8__kv-bf16", "w-fp8__kv-fp8")
    if not args.all_configs:
        keep = df["config"].str.split("__", n=1).str[-1].isin(KV_CONTRAST)
        if keep.any():
            df = df[keep]
            print(f"  plotting the KV contrast only ({df['config'].nunique()} "
                  "configs); pass --all-configs for the rest")

    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    end_points = []
    for cfg, g in df.groupby("config"):
        bk = bucket_accuracy(g, DEFAULT_EDGES)
        bk = bk[bk["n"] >= args.min_bucket]
        if bk.empty:
            print(f"  {short_name(cfg)}: every bucket under {args.min_bucket} "
                  "prompts, skipped")
            continue
        c = color_for(cfg)
        ax.plot(bk["center"], bk["acc"] * 100, "-o", color=c, markersize=8,
                markeredgecolor=plotstyle.SURFACE, markeredgewidth=2.0,
                label=short_name(cfg), zorder=3)
        ax.fill_between(bk["center"], bk["lo"] * 100, bk["hi"] * 100,
                        color=c, alpha=0.14, linewidth=0, zorder=2)
        last = bk.iloc[-1]
        end_points.append((last["center"], last["acc"] * 100, short_name(cfg)))

    # End labels, staggered: several configurations usually converge at the
    # longest bucket, so a fixed offset stacks their names on one another.
    for rank, (ex, ey, name) in enumerate(sorted(end_points, key=lambda t: -t[1])):
        # rank 0 is the highest-scoring line, so it must be pushed UP, not down.
        dy = ((len(end_points) - 1) / 2 - rank) * 14
        ax.annotate(name, (ex, ey), textcoords="offset points",
                    xytext=(16, dy), fontsize=8.5,
                    color=plotstyle.INK_SECONDARY, va="center",
                    arrowprops=dict(arrowstyle="-", color=plotstyle.AXIS,
                                    linewidth=0.8, shrinkA=1, shrinkB=1))

    ax.set_xscale("log")
    ax.set_xlabel("input length  (tokens, log scale)")
    ax.set_ylabel("accuracy  (%)")
    finish(ax, f"Long-context accuracy by KV-cache setting — {args.task}",
           f"shaded = 95% bootstrap interval · buckets with < {args.min_bucket} "
           "prompts dropped")
    ax.legend(loc="lower left", fontsize=9, ncols=2)
    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    print("\nWhat to check before claiming anything from this figure:")
    print("  - Do the FP8-KV and BF16-KV bands OVERLAP at every length? Then you")
    print("    have not detected a long-context cost, whatever the lines do.")
    print("  - Does any gap WIDEN with length? That is the claim worth making,")
    print("    and only if the intervals separate at the long end.")
    print("  - Were long prompts truncated to max_model_len? Truncation looks")
    print("    exactly like a quality drop. Check before you write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
