#!/usr/bin/env python3
"""The 2x2 interaction on the SYMMETRIC score scale.

Why this exists
---------------
`composition()` in report_tables.py measures non-compositionality on a DAMAGE
SET: it conditions on the prompts the BF16 baseline answers correctly, counts
only the ones a variant gets wrong, and compares the combination against an
independence model over the two components.

That framing has a selection effect built into it. Conditioning on
baseline-correct selects prompts where the baseline may have been lucky, and
counting only losses ignores the recoveries that run the other way -- and the
recoveries are comparable in size to the losses (section 4.4). Section 4.6 showed
the damage concentrates at the model's decision boundary, which is exactly where
regression to the mean operates. So a sub-additive result on the damage set is
consistent with no interaction at all.

This module runs the test without that conditioning. For a 2x2 factorial in
(weight precision) x (KV precision), the interaction contrast is

    I = score_C - score_W - score_K + score_base

computed per prompt and averaged, with a paired bootstrap over prompts. I = 0
means the two effects are additive: the combination is exactly what the parts
predict. It is symmetric -- gains and losses both count -- it needs no
conditioning, and it works unchanged for a continuous score such as LongBench's
F1/ROUGE, where a binary damage model does not apply at all.

Run:  python3 analysis/interaction_contrast.py
Writes: report/interaction_contrast.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import load_gsm8k, load_ifeval, load_mmlu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PER_PROMPT = ROOT / "results" / "logs" / "per_prompt.parquet"

BASE = "llama31-8b__w-bf16__kv-bf16"
W = "llama31-8b__w-fp8__kv-bf16"
K = "llama31-8b__w-bf16__kv-fp8"
C = "llama31-8b__w-fp8__kv-fp8"

NBOOT = 10_000
SEED = 20261116


def longbench() -> pd.DataFrame:
    d = pd.read_parquet(PER_PROMPT)
    lb = d[d.task.str.contains("longbench", case=False, na=False)].copy()
    lb["uid"] = lb.task + ":" + lb.doc_id.astype(str)
    return lb


def tasks():
    g, _ = load_gsm8k()
    keep = g.groupby("config").run_id.min()
    g = g[g.run_id == g.config.map(keep)]
    gs = g[g["filter"] == "strict-match"].copy()
    gs["uid"] = gs.doc_id.astype(str)
    yield "GSM8K (strict-match)", gs
    yield "MMLU", load_mmlu()
    yield "IFEval", load_ifeval()
    yield "LongBench", longbench()


def analyse(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["score"] = df.correct.astype(float)
    P = {c: df[df.config == c].set_index("uid").score.sort_index()
         for c in (BASE, W, K, C)}
    idx = P[BASE].index
    for v in P.values():
        idx = idx.intersection(v.index)
    A = {k: v.loc[idx].to_numpy() for k, v in P.items()}
    b, w, k, c = A[BASE], A[W], A[K], A[C]

    inter = c - w - k + b
    rng = np.random.default_rng(SEED)
    B = rng.integers(0, len(idx), size=(NBOOT, len(idx)))
    boot = inter[B].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Churn: any change at all, and typical size of the change.
    def churn(x):
        return float((x != b).mean()), float(np.abs(x - b).mean())

    cw, mw = churn(w)
    ck, mk = churn(k)
    cc, mc = churn(c)

    return {
        "n": int(len(idx)),
        "base": round(100 * float(b.mean()), 2),
        "delta_W": round(100 * float((w - b).mean()), 2),
        "delta_K": round(100 * float((k - b).mean()), 2),
        "delta_C": round(100 * float((c - b).mean()), 2),
        "additive_prediction": round(100 * float((w + k - b).mean()), 2),
        "interaction_pp": round(100 * float(inter.mean()), 3),
        "ci_pp": [round(100 * float(lo), 3), round(100 * float(hi), 3)],
        "resolved": bool(lo * hi > 0),
        "churn_W_pct": round(100 * cw, 2), "mean_abs_delta_W": round(100 * mw, 2),
        "churn_K_pct": round(100 * ck, 2), "mean_abs_delta_K": round(100 * mk, 2),
        "churn_C_pct": round(100 * cc, 2), "mean_abs_delta_C": round(100 * mc, 2),
    }


def main() -> None:
    out = {}
    print(f"{'task':22s} {'n':>6s} {'dW':>7s} {'dK':>7s} {'dC':>7s} "
          f"{'additive':>9s} {'interaction':>12s}  95% CI             resolved")
    print("-" * 104)
    for name, df in tasks():
        r = analyse(df)
        out[name] = r
        print(f"{name:22s} {r['n']:6d} {r['delta_W']:+6.2f} {r['delta_K']:+6.2f} "
              f"{r['delta_C']:+6.2f} {r['additive_prediction']:8.2f}% "
              f"{r['interaction_pp']:+11.3f}  "
              f"[{r['ci_pp'][0]:+6.2f},{r['ci_pp'][1]:+6.2f}]   "
              f"{'YES' if r['resolved'] else 'no'}")

    print("\nchurn (any score change from baseline) and mean |change|:")
    print(f"{'task':22s} {'W':>18s} {'KV':>18s} {'both':>18s}")
    for name, r in out.items():
        print(f"{name:22s} "
              f"{r['churn_W_pct']:7.2f}% |d|{r['mean_abs_delta_W']:5.2f} "
              f"{r['churn_K_pct']:7.2f}% |d|{r['mean_abs_delta_K']:5.2f} "
              f"{r['churn_C_pct']:7.2f}% |d|{r['mean_abs_delta_C']:5.2f}")

    p = ROOT / "report" / "interaction_contrast.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p.relative_to(ROOT)}")
    n_res = sum(r["resolved"] for r in out.values())
    print(f"\n{n_res}/{len(out)} tasks show a resolved interaction on the "
          f"symmetric score scale.")


if __name__ == "__main__":
    main()
