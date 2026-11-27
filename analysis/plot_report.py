#!/usr/bin/env python3
"""The report's three figures, built from report/tables.json.

    python3 analysis/plot_report.py

Figures read the SAME json the report text quotes, so a figure can never
disagree with a sentence. Every series keeps the colour plotstyle assigns to
that configuration, and every point is directly labelled, so identity never
rests on colour alone.

fig1_pareto.png   throughput against accuracy, with measured error on both axes
fig2_churn.png    aggregate accuracy change vs fraction of answers that changed
fig3_composition.png  where the damage goes when two quantizations are combined
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plotstyle  # noqa: E402
from plotstyle import apply_style, color_for, finish, short_name  # noqa: E402

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
BASE = "llama31-8b__w-bf16__kv-bf16"

HALO = [pe.withStroke(linewidth=3, foreground=plotstyle.SURFACE)]


def load() -> dict:
    p = REPORT / "tables.json"
    if not p.exists():
        raise SystemExit("run analysis/report_tables.py first")
    return json.loads(p.read_text())


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices on the frontier: maximise x (throughput), maximise y (accuracy)."""
    order = np.argsort(-x)
    front, best = [], -np.inf
    for i in order:
        if y[i] >= best:
            front.append(i)
            best = y[i]
    return np.array(front)[np.argsort(x[np.array(front)])]


def fig_pareto(T: dict) -> None:
    peak = {r["config"]: r for r in T["throughput"]["peak"]}
    acc = {r["config"]: r for r in T["accuracy"]["mmlu"]["rows"]}
    cfgs = [c for c in peak if c in acc]

    x = np.array([peak[c]["tput"] for c in cfgs])
    xerr = np.array([peak[c].get("tput_sd") or 0.0 for c in cfgs])
    y = np.array([acc[c]["delta_pp"] for c in cfgs])
    ylo = np.array([acc[c]["delta_pp"] - acc[c]["ci_lo_pp"] for c in cfgs])
    yhi = np.array([acc[c]["ci_hi_pp"] - acc[c]["delta_pp"] for c in cfgs])

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.axhline(0, color=plotstyle.AXIS, lw=1.2, zorder=1)

    front = pareto_front(x, y)
    ax.plot(x[front], y[front], color=plotstyle.MUTED, lw=1.6,
            ls="--", zorder=2, label="Pareto frontier")

    for i, c in enumerate(cfgs):
        col = color_for(c)
        ax.errorbar(x[i], y[i], yerr=[[ylo[i]], [yhi[i]]], xerr=xerr[i],
                    fmt="o", ms=9, color=col, ecolor=col, elinewidth=1.8,
                    capsize=3, capthick=1.8, zorder=4,
                    markeredgecolor=plotstyle.SURFACE, markeredgewidth=2)
        dy = 10 if y[i] >= -1.0 else -20
        ax.annotate(short_name(c), (x[i], y[i]), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=9,
                    color=plotstyle.INK, path_effects=HALO, zorder=5)

    ax.set_xlabel("peak output throughput (tokens/s, mean of 3 seeds)")
    ax.set_ylabel("MMLU accuracy change from BF16 (pp)")
    finish(ax, "Throughput against accuracy",
           "Llama-3.1-8B-Instruct, one L40S. Vertical bars: 95% paired bootstrap "
           "interval, n=14,042. Horizontal bars: SD across seeds.")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORT / "fig1_pareto.png", bbox_inches="tight")
    plt.close(fig)


def fig_churn(T: dict) -> None:
    """The point of the study: the score barely moves, the answers do."""
    blocks = [("gsm8k_strict", "GSM8K (strict-match), n=1,319"),
              ("mmlu", "MMLU, n=14,042"),
              ("ifeval", "IFEval, n=541")]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))

    for ax, (key, sub) in zip(axes, blocks):
        rows = [r for r in T["accuracy"][key]["rows"] if r["config"] != BASE]
        names = [short_name(r["config"]) for r in rows]
        yy = np.arange(len(rows))[::-1]

        churn = np.array([r["disagreement_pct"] for r in rows])
        delta = np.array([abs(r["delta_pp"]) for r in rows])

        for i, r in enumerate(rows):
            col = color_for(r["config"])
            # Churn bar: how many answers changed at all.
            ax.barh(yy[i], churn[i], height=0.52, color=col, alpha=0.30,
                    edgecolor=plotstyle.SURFACE, linewidth=2, zorder=2)
            # Net movement: what a score table would show.
            ax.barh(yy[i], delta[i], height=0.52, color=col,
                    edgecolor=plotstyle.SURFACE, linewidth=2, zorder=3)
            ax.text(churn[i] + 0.4, yy[i], f"{churn[i]:.1f}%", va="center",
                    fontsize=9, color=plotstyle.INK_SECONDARY, path_effects=HALO)

        ax.set_yticks(yy, names, fontsize=9)
        ax.set_xlabel("percent of prompts")
        ax.set_xlim(0, max(churn) * 1.28)
        finish(ax, sub)

    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=plotstyle.MUTED, alpha=0.30,
                       label="answers that changed from BF16"),
        mpatches.Patch(facecolor=plotstyle.MUTED,
                       label="net accuracy change (|Δ|)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.975),
               fontsize=9, ncol=2)
    fig.suptitle("A near-unchanged score hides a large change in behaviour",
                 x=0.012, y=0.985, ha="left", va="top",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(REPORT / "fig2_churn.png", bbox_inches="tight")
    plt.close(fig)


def fig_composition(T: dict) -> None:
    """Combining FP8 weights with an FP8 KV cache does not add the damage."""
    blocks = [("gsm8k_strict", "GSM8K (strict-match)"), ("mmlu", "MMLU"),
              ("ifeval", "IFEval")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    cW = color_for("llama31-8b__w-fp8__kv-bf16")
    cK = color_for("llama31-8b__w-bf16__kv-fp8")
    cC = color_for("llama31-8b__w-fp8__kv-fp8")

    for ax, (key, sub) in zip(axes, blocks):
        c = T["composition"][key]
        labels = ["FP8 W\nalone", "FP8 KV\nalone", "either\n(union)",
                  "independent\nprediction", "both\n(measured)"]
        vals = [c["damage_W_pct"], c["damage_K_pct"],
                100 * (c["n_damage_W"] + c["n_damage_K"] - c["n_overlap_WK"])
                / c["n_baseline_correct"],
                c["damage_independent_pct"], c["damage_C_pct"]]
        cols = [cW, cK, plotstyle.MUTED, plotstyle.AXIS, cC]
        xx = np.arange(len(vals))

        for i in range(len(vals)):
            hatch = "///" if i == 3 else None
            ax.bar(xx[i], vals[i], width=0.66, color=cols[i], hatch=hatch,
                   edgecolor=plotstyle.SURFACE, linewidth=2, zorder=3)
            ax.text(xx[i], vals[i] + 0.25, f"{vals[i]:.1f}%", ha="center",
                    fontsize=9, color=plotstyle.INK_SECONDARY, path_effects=HALO)

        ax.set_xticks(xx, labels, fontsize=8.5)
        ax.tick_params(axis="x", pad=2)
        ax.set_ylabel("prompts damaged (% of baseline-correct)")
        ax.set_ylim(0, max(vals) * 1.22)
        finish(ax, sub,
               f"{c['novel_damage_pct']:.0f}% of the combined damage is new")

    fig.suptitle("Framing A: on the damage set, the combination looks sub-additive",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(REPORT / "fig3_composition.png", bbox_inches="tight")
    plt.close(fig)



def fig_margin(T: dict) -> None:
    """Where quantization damage actually falls, by baseline confidence."""
    mp = REPORT / "margins.json"
    if not mp.exists():
        print("skip fig4: run analysis/margins.py first")
        return
    M = json.loads(mp.read_text())
    bins = M["bins"]

    cW = color_for("llama31-8b__w-fp8__kv-bf16")
    cK = color_for("llama31-8b__w-bf16__kv-fp8")
    cC = color_for("llama31-8b__w-fp8__kv-fp8")

    labels = []
    for i, b in enumerate(bins):
        lo, hi = b["lo"], b["hi"]
        if not np.isfinite(lo):
            labels.append(f"< {hi:.2f}")
        elif not np.isfinite(hi):
            labels.append(f"> {lo:.2f}")
        else:
            labels.append(f"{lo:.2f}-{hi:.2f}")

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.5), gridspec_kw={"width_ratios": [1.65, 1]})

    xx = np.arange(len(bins))
    w = 0.26
    for off, key, col, name in ((-w, "dW", cW, "FP8 weights"),
                                (0.0, "dK", cK, "FP8 KV cache"),
                                (w, "dC", cC, "both")):
        vals = [100 * b[key] for b in bins]
        ax.bar(xx + off, vals, width=w, color=col, label=name,
               edgecolor=plotstyle.SURFACE, linewidth=1.5, zorder=3)
    for i, b in enumerate(bins):
        top = 100 * max(b["dW"], b["dK"], b["dC"])
        if top >= 0.3:
            ax.text(xx[i], top + 0.7, f"{top:.1f}%", ha="center", fontsize=8.5,
                    color=plotstyle.INK_SECONDARY, path_effects=HALO)
    ax.set_xticks(xx, labels, fontsize=8.5)
    ax.set_xlabel("baseline confidence margin (nats), equal-count bins")
    ax.set_ylabel("prompts damaged (%)")
    ax.legend(fontsize=9, loc="upper right")
    finish(ax, "Damage falls almost entirely on near-ties",
           "MMLU, prompts the BF16 baseline answered correctly (n = 9,583)")

    # Cumulative share of total damage.
    conc = M["concentration"]
    qs = sorted(conc, key=float)
    xs = [100 * float(q) for q in qs]
    for key, col, name in (("W", cW, "FP8 weights"), ("K", cK, "FP8 KV cache"),
                           ("C", cC, "both")):
        ys = [100 * conc[q][key] for q in qs]
        ax2.plot([0] + xs, [0] + ys, "o-", color=col, label=name, zorder=3)
    ax2.plot([0, 100], [0, 100], ls=":", color=plotstyle.AXIS, zorder=2,
             label="if damage were uniform")
    ax2.set_xlabel("least-confident % of prompts")
    ax2.set_ylabel("share of all damage captured (%)")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 103)
    ax2.legend(fontsize=8.5, loc="lower right")
    finish(ax2, "Concentration",
           f"the least-confident 25% carry {100*conc['0.25']['W']:.0f}% of weight damage")

    fig.tight_layout()
    fig.savefig(REPORT / "fig4_margin.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_style()
    T = load()
    fig_pareto(T)
    fig_churn(T)
    fig_composition(T)
    fig_margin(T)
    for n in ("fig1_pareto.png", "fig2_churn.png", "fig3_composition.png",
              "fig4_margin.png"):
        p = REPORT / n
        if not p.exists():
            continue
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
