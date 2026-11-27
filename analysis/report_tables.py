"""Produce every number the technical report quotes, from the parquet files.

Run:  python3 analysis/report_tables.py
Writes: report/tables.json  and  report/tables.md

Nothing in the report is typed by hand. If a number in report.md is not in
tables.json, it is not supported by the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import paired_bootstrap_diff  # noqa: E402
from load import (  # noqa: E402
    format_miss, load_gsm8k, load_ifeval, load_mmlu, load_system,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report"
BASE = "llama31-8b__w-bf16__kv-bf16"

# Display order and short labels.
CONFIGS = [
    ("llama31-8b__w-bf16__kv-bf16", "BF16 / BF16", "bf16", "bf16"),
    ("llama31-8b__w-fp8__kv-bf16", "FP8 / BF16", "fp8", "bf16"),
    ("llama31-8b__w-bf16__kv-fp8", "BF16 / FP8", "bf16", "fp8"),
    ("llama31-8b__w-fp8__kv-fp8", "FP8 / FP8", "fp8", "fp8"),
    ("llama31-8b__w-int8__kv-bf16", "INT8 W8A8 / BF16", "int8", "bf16"),
    ("llama31-8b__w-int4gptq__kv-bf16", "INT4 GPTQ / BF16", "int4gptq", "bf16"),
]
LABEL = {c: lab for c, lab, _, _ in CONFIGS}
ORDER = [c for c, _, _, _ in CONFIGS]

SEED = 20261116
NBOOT = 10_000


def _first_run(df: pd.DataFrame) -> pd.DataFrame:
    keep = df.groupby("config").run_id.min()
    return df[df.run_id == df.config.map(keep)]


def paired(df: pd.DataFrame, uid: str, cfg: str) -> np.ndarray:
    s = df[df.config == cfg].set_index(uid).correct.sort_index()
    return s


def accuracy_block(df: pd.DataFrame, uid: str, name: str) -> dict:
    """Accuracy per config plus paired bootstrap vs BF16."""
    base = paired(df, uid, BASE)
    rows = []
    for cfg in ORDER:
        if cfg not in set(df.config):
            continue
        s = paired(df, uid, cfg)
        shared = base.index.intersection(s.index)
        a, b = s.loc[shared].to_numpy(), base.loc[shared].to_numpy()
        r = paired_bootstrap_diff(a, b, n_resamples=NBOOT, seed=SEED)
        # Per-prompt disagreement with the baseline: how many answers changed,
        # regardless of whether the score moved.
        flips = float((a != b).mean())
        rows.append(
            {
                "config": cfg,
                "label": LABEL[cfg],
                "n": int(len(shared)),
                "accuracy": round(100 * float(a.mean()), 2),
                "delta_pp": round(100 * r["diff"], 2),
                "ci_lo_pp": round(100 * r["ci_lo"], 2),
                "ci_hi_pp": round(100 * r["ci_hi"], 2),
                "contains_zero": r["contains_zero"],
                "disagreement_pct": round(100 * flips, 2),
            }
        )
    return {"metric": name, "rows": rows}


def noise_floor(g: pd.DataFrame) -> dict:
    """BF16 evaluated twice, 17 h apart, same config and same seed."""
    b = g[(g.config == BASE) & (g["filter"] == "strict-match")]
    runs = sorted(b.run_id.unique())
    if len(runs) < 2:
        return {"available": False}
    r1 = b[b.run_id == runs[0]].set_index("doc_id").sort_index()
    r2 = b[b.run_id == runs[1]].set_index("doc_id").sort_index()
    shared = r1.index.intersection(r2.index)
    c1 = r1.loc[shared].correct.to_numpy()
    c2 = r2.loc[shared].correct.to_numpy()
    t1 = r1.loc[shared].output_text.astype(str).to_numpy()
    t2 = r2.loc[shared].output_text.astype(str).to_numpy()
    return {
        "available": True,
        "run_a": runs[0],
        "run_b": runs[1],
        "n": int(len(shared)),
        "acc_a": round(100 * float(c1.mean()), 4),
        "acc_b": round(100 * float(c2.mean()), 4),
        # The two quantities any claim in this study has to clear.
        "correctness_disagreement_pct": round(100 * float((c1 != c2).mean()), 6),
        "text_disagreement_pct": round(100 * float((t1 != t2).mean()), 6),
    }


def filter_sensitivity(g: pd.DataFrame) -> dict:
    """The same configuration, scored under both GSM8K extraction filters.

    This is the finding that most constrains how published quantization
    accuracy deltas can be read.
    """
    g = _first_run(g)
    out = []
    for cfg in ORDER:
        row = {"config": cfg, "label": LABEL[cfg]}
        for filt in ("strict-match", "flexible-extract"):
            sub = g[g["filter"] == filt]
            base = sub[sub.config == BASE].set_index("doc_id").correct.sort_index()
            s = sub[sub.config == cfg].set_index("doc_id").correct.sort_index()
            shared = base.index.intersection(s.index)
            r = paired_bootstrap_diff(
                s.loc[shared].to_numpy(), base.loc[shared].to_numpy(),
                n_resamples=NBOOT, seed=SEED,
            )
            key = "strict" if filt == "strict-match" else "flexible"
            row[f"acc_{key}"] = round(100 * float(s.loc[shared].mean()), 2)
            row[f"delta_{key}_pp"] = round(100 * r["diff"], 2)
            row[f"ci_{key}"] = [round(100 * r["ci_lo"], 2), round(100 * r["ci_hi"], 2)]
            row[f"zero_{key}"] = r["contains_zero"]
        row["sign_flips"] = bool(
            np.sign(row["delta_strict_pp"]) * np.sign(row["delta_flexible_pp"]) < 0
            and not row["zero_strict"]
            and not row["zero_flexible"]
        )
        out.append(row)
    return {"rows": out}


def composition(df: pd.DataFrame, uid: str, name: str) -> dict:
    """Does damage from weight quantization and KV quantization compose?

    The 2x2: BF16/BF16, FP8/BF16 (weights only), BF16/FP8 (KV only),
    FP8/FP8 (both). "Damage" = baseline right, variant wrong.
    """
    cells = {
        "base": BASE,
        "W": "llama31-8b__w-fp8__kv-bf16",
        "K": "llama31-8b__w-bf16__kv-fp8",
        "C": "llama31-8b__w-fp8__kv-fp8",
    }
    series = {k: paired(df, uid, v) for k, v in cells.items()}
    shared = series["base"].index
    for s in series.values():
        shared = shared.intersection(s.index)
    arr = {k: s.loc[shared].to_numpy().astype(bool) for k, s in series.items()}
    base = arr["base"]

    dmg = {k: (base & ~arr[k]) for k in ("W", "K", "C")}
    n = int(base.sum())  # denominator: prompts the baseline got right

    rW, rK, rC = (dmg[k].sum() / n for k in ("W", "K", "C"))
    r_indep = 1 - (1 - rW) * (1 - rK)
    union = dmg["W"] | dmg["K"]
    inter = dmg["W"] & dmg["K"]
    jaccard = float(inter.sum() / union.sum()) if union.sum() else float("nan")
    novel = dmg["C"] & ~union
    novel_frac = float(novel.sum() / dmg["C"].sum()) if dmg["C"].sum() else float("nan")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(shared), size=(NBOOT, len(shared)))

    def boot(fn):
        vals = []
        for i in idx:
            b = base[i]
            if b.sum() == 0:
                continue
            d = {k: (b & ~arr[k][i]) for k in ("W", "K", "C")}
            v = fn(b.sum(), d)
            if v is not None and np.isfinite(v):
                vals.append(v)
            if len(vals) >= 2000:  # enough for a stable interval; keeps runtime sane
                break
        return (
            round(100 * float(np.percentile(vals, 2.5)), 2),
            round(100 * float(np.percentile(vals, 97.5)), 2),
        )

    ci_excess = boot(
        lambda m, d: d["C"].sum() / m - (1 - (1 - d["W"].sum() / m) * (1 - d["K"].sum() / m))
    )
    ci_jac = boot(
        lambda m, d: (d["W"] & d["K"]).sum() / (d["W"] | d["K"]).sum()
        if (d["W"] | d["K"]).sum() else None
    )
    ci_novel = boot(
        lambda m, d: (d["C"] & ~(d["W"] | d["K"])).sum() / d["C"].sum()
        if d["C"].sum() else None
    )

    return {
        "metric": name,
        "n_baseline_correct": n,
        "damage_W_pct": round(100 * rW, 2),
        "damage_K_pct": round(100 * rK, 2),
        "damage_C_pct": round(100 * rC, 2),
        "damage_independent_pct": round(100 * r_indep, 2),
        "excess_over_independent_pp": round(100 * (rC - r_indep), 2),
        "excess_ci_pp": list(ci_excess),
        "jaccard_WK": round(jaccard, 3),
        "jaccard_ci": [round(x / 100, 3) for x in ci_jac],
        "novel_damage_pct": round(100 * novel_frac, 2),
        "novel_ci_pct": list(ci_novel),
        "n_damage_W": int(dmg["W"].sum()),
        "n_damage_K": int(dmg["K"].sum()),
        "n_damage_C": int(dmg["C"].sum()),
        "n_overlap_WK": int(inter.sum()),
        "n_novel": int(novel.sum()),
    }


def throughput() -> dict:
    s = load_system()
    s = s[s.config.isin(ORDER)]
    agg = (
        s.groupby(["config", "shape", "concurrency"])
        .agg(
            tput=("output_throughput_tok_s", "mean"),
            tput_sd=("output_throughput_tok_s", "std"),
            ttft=("ttft_ms_median", "mean"),
            tpot=("tpot_ms_median", "mean"),
            kv=("kv_cache_tokens", "max"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    agg["label"] = agg.config.map(LABEL)
    peak = (
        agg.sort_values("tput")
        .groupby("config")
        .tail(1)[["config", "label", "shape", "concurrency", "tput", "tput_sd"]]
    )
    base_peak = float(peak[peak.config == BASE].tput.iloc[0])
    peak["speedup"] = (peak.tput / base_peak).round(3)
    kv = agg.groupby("config").kv.max()
    return {
        "cells": len(s),
        "configs": int(s.config.nunique()),
        "shapes": sorted(s["shape"].unique().tolist()),
        "concurrencies": sorted(int(c) for c in s.concurrency.unique()),
        "seeds_per_cell": int(s.groupby(["config", "shape", "concurrency"]).seed.nunique().max()),
        "kv_cache_tokens": {c: int(v) for c, v in kv.items()},
        "peak": peak.sort_values("tput", ascending=False).round(2).to_dict("records"),
        "grid": agg.round(3).to_dict("records"),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    g_all, checks = load_gsm8k()
    g = _first_run(g_all)
    g_strict = g[g["filter"] == "strict-match"].assign(uid=lambda d: d.doc_id.astype(str))
    g_flex = g[g["filter"] == "flexible-extract"].assign(uid=lambda d: d.doc_id.astype(str))
    m = load_mmlu()
    ife = load_ifeval()
    fm = format_miss()

    res = {
        "provenance": {
            "n_boot": NBOOT,
            "seed": SEED,
            "filter_checks": [
                {"config": c.config, "run_id": c.run_id, "confirmed": c.ok,
                 "recovered": {k: round(v, 4) for k, v in c.recovered.items()},
                 "lm_eval_printed": c.printed}
                for c in checks
            ],
            "all_filters_confirmed": all(c.ok for c in checks),
        },
        "noise_floor": noise_floor(g_all),
        "accuracy": {
            "gsm8k_strict": accuracy_block(g_strict, "uid", "GSM8K exact_match (strict-match)"),
            "gsm8k_flexible": accuracy_block(g_flex, "uid", "GSM8K exact_match (flexible-extract)"),
            "mmlu": accuracy_block(m, "uid", "MMLU acc"),
            "ifeval": accuracy_block(ife, "uid", "IFEval prompt_level_strict_acc"),
            "format_miss": accuracy_block(
                fm, "uid",
                "GSM8K format-miss (flexible correct, strict wrong)"),
        },
        "filter_sensitivity": filter_sensitivity(g_all),
        "composition": {
            "gsm8k_strict": composition(g_strict, "uid", "GSM8K (strict-match)"),
            "mmlu": composition(m, "uid", "MMLU"),
            "ifeval": composition(ife, "uid", "IFEval"),
        },
        "throughput": throughput(),
    }

    (OUT / "tables.json").write_text(json.dumps(res, indent=2, default=str))

    # Human-readable companion.
    L = []
    nf = res["noise_floor"]
    L.append("# Measured tables\n")
    L.append(f"All filter recoveries confirmed against lm-eval output: "
             f"**{res['provenance']['all_filters_confirmed']}**\n")
    L.append("## Noise floor (BF16 evaluated twice)\n")
    L.append(f"- runs `{nf['run_a']}` and `{nf['run_b']}`, n = {nf['n']}")
    L.append(f"- accuracy {nf['acc_a']}% vs {nf['acc_b']}%")
    L.append(f"- per-prompt correctness disagreement **{nf['correctness_disagreement_pct']}%**")
    L.append(f"- per-prompt extracted-answer disagreement **{nf['text_disagreement_pct']}%**\n")

    for key, block in res["accuracy"].items():
        L.append(f"## {block['metric']}\n")
        L.append("| config | n | acc % | Δ vs BF16 (pp) | 95% CI | resolved | answers changed % |")
        L.append("|---|---:|---:|---:|---|---|---:|")
        for r in block["rows"]:
            L.append(
                f"| {r['label']} | {r['n']} | {r['accuracy']} | {r['delta_pp']:+} | "
                f"[{r['ci_lo_pp']:+}, {r['ci_hi_pp']:+}] | "
                f"{'no' if r['contains_zero'] else 'yes'} | {r['disagreement_pct']} |"
            )
        L.append("")

    L.append("## Filter sensitivity (same runs, two GSM8K extraction filters)\n")
    L.append("| config | strict Δ (pp) | 95% CI | flexible Δ (pp) | 95% CI | sign flips |")
    L.append("|---|---:|---|---:|---|---|")
    for r in res["filter_sensitivity"]["rows"]:
        L.append(
            f"| {r['label']} | {r['delta_strict_pp']:+} | {r['ci_strict']} | "
            f"{r['delta_flexible_pp']:+} | {r['ci_flexible']} | "
            f"{'**YES**' if r['sign_flips'] else 'no'} |"
        )
    L.append("")

    for key, c in res["composition"].items():
        L.append(f"## Composition -- {c['metric']}\n")
        L.append(f"- prompts the baseline answered correctly: {c['n_baseline_correct']}")
        L.append(f"- damaged by weights only (FP8 W): {c['damage_W_pct']}% ({c['n_damage_W']})")
        L.append(f"- damaged by KV only (FP8 KV): {c['damage_K_pct']}% ({c['n_damage_K']})")
        L.append(f"- damaged by both together: {c['damage_C_pct']}% ({c['n_damage_C']})")
        L.append(f"- independence prediction: {c['damage_independent_pct']}%")
        L.append(f"- excess over independence: {c['excess_over_independent_pp']:+} pp, CI {c['excess_ci_pp']}")
        L.append(f"- Jaccard overlap of the two damage sets: {c['jaccard_WK']}, CI {c['jaccard_ci']}")
        L.append(f"- combined damage in neither single-component set: {c['novel_damage_pct']}%, CI {c['novel_ci_pct']}\n")

    tp = res["throughput"]
    L.append("## Throughput\n")
    L.append(f"- {tp['cells']} measured cells: {tp['configs']} configs x "
             f"{len(tp['shapes'])} shapes x {len(tp['concurrencies'])} concurrencies x "
             f"{tp['seeds_per_cell']} seeds")
    L.append(f"- shapes {tp['shapes']}, concurrencies {tp['concurrencies']}\n")
    L.append("| config | peak output tok/s | shape | conc | speedup vs BF16 | KV cache tokens |")
    L.append("|---|---:|---|---:|---:|---:|")
    for r in tp["peak"]:
        L.append(f"| {r['label']} | {r['tput']} | {r['shape']} | {r['concurrency']} | "
                 f"{r['speedup']}x | {tp['kv_cache_tokens'][r['config']]} |")
    L.append("")

    (OUT / "tables.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
