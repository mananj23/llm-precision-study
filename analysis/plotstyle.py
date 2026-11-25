#!/usr/bin/env python3
"""Shared matplotlib style and a fixed, colourblind-checked config palette.

The eight hues below are assigned to configurations in FIXED ORDER and never
cycled, so a configuration keeps its colour across every figure in the report.
Adjacent pairs clear a CVD separation check (worst adjacent pair dE 9.1 under
protanopia), and every point is also directly labelled, so identity never rests
on colour alone.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed categorical order. Index = slot, never re-sorted by value or rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Configuration -> slot. Fixed here so the colour follows the ENTITY, not its
# position in whatever subset a given figure happens to plot.
CONFIG_ORDER = [
    "w-bf16__kv-bf16",
    "w-fp8__kv-bf16",
    "w-bf16__kv-fp8",
    "w-fp8__kv-fp8",
    "w-int8__kv-bf16",
    "w-int4gptq__kv-bf16",
    "w-int4awq__kv-bf16",
]

PRETTY = {
    "w-bf16__kv-bf16": "BF16",
    "w-fp8__kv-bf16": "FP8 W8A8",
    "w-bf16__kv-fp8": "BF16 + FP8 KV",
    "w-fp8__kv-fp8": "FP8 + FP8 KV",
    "w-int8__kv-bf16": "INT8 W8A8",
    "w-int4gptq__kv-bf16": "INT4 GPTQ",
    "w-int4awq__kv-bf16": "INT4 AWQ",
}


def short_name(config: str) -> str:
    """llama31-8b__w-fp8__kv-bf16 -> 'FP8 W8A8'"""
    key = config.split("__", 1)[-1] if "__" in config else config
    return PRETTY.get(key, key)


def color_for(config: str) -> str:
    key = config.split("__", 1)[-1] if "__" in config else config
    try:
        return SERIES[CONFIG_ORDER.index(key)]
    except ValueError:
        return SERIES[-1]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "figure.dpi": 140,
    })


def finish(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", pad=16 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                fontsize=9, color=MUTED, ha="left", va="bottom")
