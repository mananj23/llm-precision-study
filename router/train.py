#!/usr/bin/env python3
"""Train and evaluate the router, honestly.

    python router/train.py --cheap llama31-8b__w-fp8__kv-bf16 \
                           --expensive llama31-8b__w-bf16__kv-bf16

Two evaluation protocols, both reported:

  by-prompt   K-fold split on prompts. Optimistic: train and test contain the
              same tasks, so the model can learn "GSM8K is easy for FP8".
  held-out-task  train on all tasks but one, test on the one left out. This is
              the number that tells you whether the router GENERALISES. If it
              collapses here, say so -- a router that only works on tasks it
              has already seen is not a router, it is a lookup table.

Models are deliberately simple and in this order: logistic regression first,
gradient boosting second. A neural router is not justified until both are
beaten, and on a few thousand prompts it will not be.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE.parent / "analysis"))

import baselines as B  # noqa: E402
import common  # noqa: E402
import dataset as D  # noqa: E402

ROOT = common.ROOT


def get_models(seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic": make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        ),
        "gbt": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=seed,
        ),
    }


def cross_val_scores(X, y, groups, model, n_splits=5, seed=0) -> np.ndarray:
    """Out-of-fold P(sufficient). Split by PROMPT, never by row."""
    from sklearn.model_selection import GroupKFold

    scores = np.full(len(y), np.nan)
    n_splits = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            scores[te] = float(y[tr].mean())
            continue
        m = model
        m.fit(X[tr], y[tr])
        scores[te] = m.predict_proba(X[te])[:, 1]
    return scores


def heldout_task_scores(X, y, tasks, model) -> tuple[np.ndarray, dict]:
    """Train on all tasks but one; predict the held-out one. Repeat."""
    scores = np.full(len(y), np.nan)
    per_task = {}
    from sklearn.metrics import roc_auc_score

    for t in np.unique(tasks):
        te = tasks == t
        tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        m = model
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        scores[te] = p
        try:
            per_task[str(t)] = float(roc_auc_score(y[te], p))
        except ValueError:
            per_task[str(t)] = float("nan")  # one class only in this task
    return scores, per_task


def evaluate(df: pd.DataFrame, scores: np.ndarray, label: str) -> dict:
    fixed = B.fixed_baselines(df)
    rnd = B.random_mix_curve(df)
    orc = B.oracle_curve(df)
    rtr = B.router_curve(df, scores)
    vs = B.beats_random_at_matched_cost(rtr, rnd)
    text = B.summarise(vs, orc, fixed)

    from sklearn.metrics import roc_auc_score

    try:
        auc = float(roc_auc_score(df["sufficient"], scores))
    except ValueError:
        auc = float("nan")

    print(f"\n--- {label} ---")
    print(f"AUC (predicting 'cheap is sufficient'): {auc:.3f}")
    if auc < 0.55:
        print("  At this AUC the router has essentially no signal. The correct")
        print("  write-up is that per-request routing did not work here, not a")
        print("  search for a threshold that happens to look good.")
    print(text)
    return {"label": label, "auc": auc, "summary": text,
            "fixed": fixed, "random": rnd, "oracle": orc, "router": vs}


def plot(results: list[dict], out: pathlib.Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import plotstyle
    from plotstyle import apply_style, finish

    apply_style()
    # Pick the x unit so the axis shows readable numbers rather than 0.00040.
    peak = max(float(np.nanmax(r["random"]["cost"])) for r in results)
    scale, unit = ((1e3, "1,000 prompts") if peak * 1e3 >= 0.05
                   else (1e6, "1M prompts"))

    fig, axes = plt.subplots(1, len(results), figsize=(7.2 * len(results), 5.2),
                             squeeze=False)
    for ax, res in zip(axes[0], results):
        rnd, orc, rtr, fixed = res["random"], res["oracle"], res["router"], res["fixed"]

        ax.fill_between(rnd["cost"] * scale, rnd["accuracy_lo"] * 100,
                        rnd["accuracy_hi"] * 100, color=plotstyle.MUTED,
                        alpha=0.15, linewidth=0, zorder=1)
        ax.plot(rnd["cost"] * scale, rnd["accuracy"] * 100, color=plotstyle.MUTED,
                linewidth=2.0, label="random mix (the bar to clear)", zorder=2)
        ax.plot(orc["cost"] * scale, orc["accuracy"] * 100,
                color=plotstyle.SERIES[2], linewidth=2.0, linestyle="--",
                label="oracle (upper bound)", zorder=3)
        ax.plot(rtr["cost"] * scale, rtr["accuracy"] * 100,
                color=plotstyle.SERIES[0], linewidth=2.4,
                label="router", zorder=4)

        for _, r in fixed.iterrows():
            c = (plotstyle.SERIES[7] if r["method"] == "always_expensive"
                 else plotstyle.SERIES[1])
            ax.plot(r["cost"] * scale, r["accuracy"] * 100, "o", color=c,
                    markersize=11, markeredgecolor=plotstyle.SURFACE,
                    markeredgewidth=2.0, zorder=5)
            ax.annotate(r["method"].replace("_", " "),
                        (r["cost"] * scale, r["accuracy"] * 100),
                        textcoords="offset points", xytext=(8, -4),
                        fontsize=8.5, color=plotstyle.INK_SECONDARY)

        ax.set_xlabel(f"average cost per {unit}  ($)")
        ax.set_ylabel("accuracy  (%)")
        finish(ax, res["label"], f"AUC {res['auc']:.3f}")
        ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-prompt", default="results/logs/per_prompt.parquet")
    ap.add_argument("--results", default="analysis/results.parquet",
                    help="where the per-config costs come from")
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--expensive", required=True)
    ap.add_argument("--model", default="gbt", choices=["logistic", "gbt", "both"])
    ap.add_argument("--no-task-flag", action="store_true",
                    help="drop the task identity feature (the honest setting)")
    ap.add_argument("--confidence-cost-share", type=float, default=0.0,
                    help="fraction of a cheap run paid on EVERY prompt if the "
                         "router uses the cheap model's confidence")
    ap.add_argument("--out", default="report/fig_router.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = ROOT / args.per_prompt
    if not path.exists():
        print(f"missing {path}\nrun: python scripts/collect_logs.py")
        return 1
    pp = pd.read_parquet(path)

    print(f"building routing dataset: {args.cheap} vs {args.expensive}")
    costs = D.load_costs(ROOT / args.results)
    df = D.build_pair_dataset(pp, args.cheap, args.expensive, costs,
                              args.confidence_cost_share)
    if not np.isfinite(df["cost_cheap"]).any():
        print("\nNo per-token costs available. Using output token counts as a")
        print("cost proxy so the shape of the curves is still visible -- but do")
        print("NOT put dollar figures from this run in the report.")
        df["cost_cheap"] = df["cheap_output_len"] / 1e6
        df["cost_exp"] = df["exp_output_len"] / 1e6

    X, names = D.make_features(df, use_task_flag=not args.no_task_flag, seed=args.seed)
    print(f"  {X.shape[1]} features (task flag: {not args.no_task_flag})")
    y = df["sufficient"].to_numpy(int)
    groups = df["doc_id"].astype(str) + "|" + df["task"].astype(str)
    tasks = df["task"].to_numpy()

    model_names = ["logistic", "gbt"] if args.model == "both" else [args.model]
    results, report = [], {}
    for mn in model_names:
        models = get_models(args.seed)
        s_cv = cross_val_scores(X, y, groups.to_numpy(), models[mn], seed=args.seed)
        results.append(evaluate(df, s_cv, f"{mn} · split by prompt"))

        if len(np.unique(tasks)) > 1:
            models = get_models(args.seed)
            s_ho, per_task = heldout_task_scores(X, y, tasks, models[mn])
            keep = ~np.isnan(s_ho)
            res = evaluate(df[keep].reset_index(drop=True), s_ho[keep],
                           f"{mn} · held-out task")
            res["per_task_auc"] = per_task
            print("  per-task AUC when that task was held out:")
            for t, a in per_task.items():
                print(f"    {t}: {a:.3f}")
            results.append(res)
        else:
            print("\nOnly one task present: the held-out-task evaluation needs at "
                  "least two. Run more tasks before claiming the router generalises.")

    plot(results, ROOT / args.out,
         f"Router vs baselines — cheap={args.cheap.split('__', 1)[-1]} "
         f"vs {args.expensive.split('__', 1)[-1]}")

    report = {
        "cheap": args.cheap, "expensive": args.expensive,
        "task_flag_used": not args.no_task_flag,
        "confidence_cost_share": args.confidence_cost_share,
        "n_prompts": int(len(df)),
        "results": [{"label": r["label"], "auc": r["auc"], "summary": r["summary"],
                     "per_task_auc": r.get("per_task_auc")} for r in results],
    }
    out_json = ROOT / "report" / "router_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_json}")

    if not args.no_task_flag:
        print("\nNow run it again with --no-task-flag and report BOTH numbers.")
        print("Real traffic does not arrive labelled with its benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
