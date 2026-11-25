#!/usr/bin/env python3
"""The headline figure: quality change from BF16 against cost.

    python analysis/plot_pareto.py

x = dollars per million output tokens (at the reference concurrency)
y = accuracy change from BF16, in percentage points, with the 95% paired
    bootstrap interval as a vertical bar
one point per configuration, one panel for the average and one per task.

The per-task panels are the point of the figure, not decoration: if the
frontier is the same in every panel, a single fixed precision wins and the
router in Step 7 has nothing to exploit. If the frontier differs by task, that
difference IS the router's motivation. Say which one you see.

Pareto frontier = the points no other point beats on BOTH axes (cheaper AND
at least as accurate). Drawn as a connecting line; everything off it is
dominated.
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

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices on the frontier: minimise x (cost), maximise y (quality)."""
    order = np.argsort(x)
    front, best_y = [], -np.inf
    for i in order:
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            continue
        if y[i] > best_y:
            front.append(i)
            best_y = y[i]
    return np.array(front, dtype=int)


def label_offsets(x: np.ndarray, y: np.ndarray) -> list[tuple[int, int]]:
    """Fan out labels for points that sit close together.

    Points are often bunched: several quantized settings land at almost the
    same cost. A fixed offset then stacks the names on top of each other, which
    is the most common way an otherwise correct figure becomes unreadable.
    Points within 6% of the x-range of their neighbour take successive slots in
    a rotation of offsets instead.
    """
    n = len(x)
    offs: list[tuple[int, int]] = [(10, 6)] * n
    if n == 0:
        return offs
    span = (np.nanmax(x) - np.nanmin(x)) or 1.0
    # A configuration name is ~90pt wide against an axis ~500pt wide, so two
    # labels collide when their points are within roughly a fifth of the range
    # -- far wider than the points themselves look. Alternate above/below with
    # growing offsets so consecutive labels can never share a line.
    rotation = [(11, 8), (11, -12), (11, 23), (11, -27)]
    order = np.argsort(x)
    slot = 0
    for j, i in enumerate(order):
        if j > 0:
            close = abs(x[i] - x[order[j - 1]]) < 0.22 * span
            slot = slot + 1 if close else 0
        offs[i] = rotation[slot % len(rotation)]
    return offs


def panel(ax, d: pd.DataFrame, title: str, subtitle: str | None = None) -> None:
    d = d.dropna(subset=["cost_per_m_output_tokens", "delta_pct_points"])
    if d.empty:
        ax.text(0.5, 0.5, "no cost data yet\n(run the benchmarks)",
                ha="center", va="center", transform=ax.transAxes,
                color=plotstyle.MUTED, fontsize=9)
        finish(ax, title, subtitle)
        return

    x = d["cost_per_m_output_tokens"].to_numpy()
    y = d["delta_pct_points"].to_numpy()
    lo = d["ci_lo_pct_points"].to_numpy()
    hi = d["ci_hi_pct_points"].to_numpy()

    # Zero line: BF16's own level. Anything whose interval crosses it is "no
    # detectable difference" from the baseline.
    ax.axhline(0, color=plotstyle.AXIS, linewidth=1.2, zorder=1)

    fi = pareto_front(x, y)
    if len(fi) > 1:
        ax.plot(x[fi], y[fi], color=plotstyle.MUTED, linewidth=2.0,
                linestyle="--", zorder=2, label="Pareto frontier")

    offsets = label_offsets(x, y)
    for i, (_, row) in enumerate(d.iterrows()):
        c = color_for(row["config"])
        on_front = i in set(fi.tolist())
        ax.errorbar(
            x[i], y[i], yerr=[[y[i] - lo[i]], [hi[i] - y[i]]],
            fmt="o", color=c, ecolor=c, elinewidth=2.0, capsize=4,
            markersize=11 if on_front else 8,
            markeredgecolor=plotstyle.SURFACE, markeredgewidth=2.0,
            zorder=4 if on_front else 3,
        )
        ax.annotate(
            short_name(row["config"]),
            (x[i], y[i]), textcoords="offset points", xytext=offsets[i],
            fontsize=8.5,
            color=plotstyle.INK if on_front else plotstyle.INK_SECONDARY,
            fontweight="bold" if on_front else "normal",
            ha="left" if offsets[i][0] > 0 else "right",
            va="center",
            # Surface-coloured halo: labels stay readable where they cross an
            # error bar or the frontier line.
            path_effects=[pe.withStroke(linewidth=3.0,
                                        foreground=plotstyle.SURFACE)],
            zorder=6,
        )

    ax.set_xlabel("cost per million output tokens  ($)")
    ax.set_ylabel("accuracy change from BF16  (points)")
    finish(ax, title, subtitle)
    if len(fi) > 1:
        ax.legend(loc="lower right", fontsize=8.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="analysis/results.parquet")
    ap.add_argument("--family", default=None)
    ap.add_argument("--out", default="report/fig_pareto.png")
    args = ap.parse_args()

    path = ROOT / args.results
    if not path.exists():
        print(f"missing {path}\nrun: python analysis/build_results.py")
        return 1
    df = pd.read_parquet(path)
    family = args.family or df["family"].iloc[0]
    df = df[df["family"] == family]
    if df.empty:
        print(f"no rows for family {family!r}")
        return 1

    apply_style()
    tasks = sorted(df["task"].unique())
    n = len(tasks) + 1
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.6 * nrows),
                             squeeze=False)
    flat = [a for r in axes for a in r]

    # Average panel: mean over tasks, per configuration. The interval here is
    # the mean of the per-task intervals, which is indicative only -- the
    # defensible intervals are the per-task ones. Say that in the caption.
    avg = (df.groupby("config", as_index=False)
             .agg(delta_pct_points=("delta_pct_points", "mean"),
                  ci_lo_pct_points=("ci_lo_pct_points", "mean"),
                  ci_hi_pct_points=("ci_hi_pct_points", "mean"),
                  cost_per_m_output_tokens=("cost_per_m_output_tokens", "first")))
    panel(flat[0], avg, "Average over tasks",
          "interval = mean of per-task intervals (indicative)")

    for ax, task in zip(flat[1:], tasks):
        panel(ax, df[df["task"] == task], task,
              "95% paired bootstrap on the change from BF16")
    for ax in flat[len(tasks) + 1:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Quality cost of precision: {family}",
        x=0.01, ha="left", fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    print("\nRead your own plot critically before writing about it:")
    nd = df[df["contains_zero"] & ~df["config"].str.contains("w-bf16__kv-bf16")]
    if len(nd):
        print(f"  {len(nd)} (config, task) pairs show NO detectable difference "
              "from BF16. Do not claim an ordering for these:")
        for _, r in nd.iterrows():
            print(f"    {short_name(r['config'])} on {r['task']}: "
                  f"{r['delta_pct_points']:+.1f} "
                  f"[{r['ci_lo_pct_points']:+.1f}, {r['ci_hi_pct_points']:+.1f}]")
    fronts = {t: {short_name(c) for c in
                  df[df["task"] == t].iloc[pareto_front(
                      df[df["task"] == t]["cost_per_m_output_tokens"].to_numpy(),
                      df[df["task"] == t]["delta_pct_points"].to_numpy())]["config"]}
              for t in tasks
              if df[df["task"] == t]["cost_per_m_output_tokens"].notna().any()}
    if len(fronts) > 1:
        same = len({frozenset(v) for v in fronts.values()}) == 1
        print(f"\n  Frontier identical across tasks: {same}")
        print("    True  -> one fixed precision wins everywhere; the router has "
              "little to exploit, and you should say so.")
        print("    False -> the best setting depends on the task. THAT is the "
              "router's motivation; quote it in the report.")
        for t, f in fronts.items():
            print(f"    {t}: {sorted(f)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
