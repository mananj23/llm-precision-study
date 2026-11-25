#!/usr/bin/env python3
"""The baselines the router has to beat. This file is the honesty of Step 7.

A router that beats "always expensive" on cost and "always cheap" on quality
has demonstrated nothing: any mixture of the two does that. The question is
whether choosing PER PROMPT beats choosing AT RANDOM at the same average cost.
If it does not, the router does not work, and that is a publishable answer.

Five curves, all on one plot:

  always_expensive   one point. The quality ceiling of a fixed setting.
  always_cheap       one point. The cost floor.
  random_mix         the honest benchmark: send a random fraction f to cheap.
                     Expected accuracy is exactly the straight line between the
                     two points above. The router must sit ABOVE this line.
  router             sweep the decision threshold; each threshold is a point.
  oracle             perfect foresight. The upper bound on what ANY router
                     could achieve, so the gap tells you how much room exists
                     even in principle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _outcome(df: pd.DataFrame, to_cheap: np.ndarray) -> tuple[float, float]:
    """(accuracy, mean cost) when `to_cheap` picks the cheap configuration."""
    correct = np.where(to_cheap, df["cheap_correct"], df["exp_correct"])
    cost = np.where(to_cheap, df["cost_cheap"], df["cost_exp"])
    return float(np.mean(correct)), float(np.mean(cost))


def fixed_baselines(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, mask in (
        ("always_expensive", np.zeros(len(df), bool)),
        ("always_cheap", np.ones(len(df), bool)),
    ):
        acc, cost = _outcome(df, mask)
        rows.append({"method": name, "frac_cheap": float(mask.mean()),
                     "accuracy": acc, "cost": cost})
    return pd.DataFrame(rows)


def random_mix_curve(df: pd.DataFrame, n_points: int = 21,
                     n_draws: int = 200, seed: int = 11) -> pd.DataFrame:
    """Route a uniformly random fraction f to cheap, for f in [0, 1].

    Simulated rather than computed in closed form so that the curve carries the
    same sampling noise as the router's -- otherwise a comparison at matched
    cost quietly favours the router.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    rows = []
    for f in np.linspace(0, 1, n_points):
        accs, costs = [], []
        for _ in range(n_draws):
            mask = rng.random(n) < f
            a, c = _outcome(df, mask)
            accs.append(a)
            costs.append(c)
        rows.append({
            "method": "random_mix", "frac_cheap": float(f),
            "accuracy": float(np.mean(accs)), "cost": float(np.mean(costs)),
            "accuracy_lo": float(np.percentile(accs, 2.5)),
            "accuracy_hi": float(np.percentile(accs, 97.5)),
        })
    return pd.DataFrame(rows)


def oracle_curve(df: pd.DataFrame, n_points: int = 41) -> pd.DataFrame:
    """Upper bound: perfect foresight, sweeping the cost budget.

    For each prompt, routing it cheap saves (cost_exp - cost_cheap) and costs
    (exp_correct - cheap_correct) in accuracy -- which is NEGATIVE for prompts
    the cheap model gets right and the expensive one does not. The oracle sends
    prompts cheap in increasing order of accuracy-loss-per-dollar-saved, so the
    first ones it moves are free wins.
    """
    loss = (df["exp_correct"] - df["cheap_correct"]).to_numpy(float)
    saving = (df["cost_exp"] - df["cost_cheap"]).to_numpy(float)
    saving = np.where(np.isfinite(saving) & (saving > 0), saving, 1.0)
    order = np.argsort(loss / saving, kind="stable")

    rows = []
    n = len(df)
    for k in np.unique(np.linspace(0, n, n_points).astype(int)):
        mask = np.zeros(n, bool)
        mask[order[:k]] = True
        acc, cost = _outcome(df, mask)
        rows.append({"method": "oracle", "frac_cheap": k / n,
                     "accuracy": acc, "cost": cost})
    return pd.DataFrame(rows)


def router_curve(df: pd.DataFrame, scores: np.ndarray,
                 n_points: int = 41) -> pd.DataFrame:
    """Sweep the threshold: route cheap when P(sufficient) >= t."""
    rows = []
    for t in np.linspace(0.0, 1.0, n_points):
        mask = scores >= t
        acc, cost = _outcome(df, mask)
        rows.append({"method": "router", "threshold": float(t),
                     "frac_cheap": float(mask.mean()),
                     "accuracy": acc, "cost": cost})
    return pd.DataFrame(rows)


def beats_random_at_matched_cost(
    router: pd.DataFrame, random: pd.DataFrame
) -> pd.DataFrame:
    """The headline comparison, done properly.

    For every router threshold, interpolate the random-mix curve at the SAME
    average cost and report the difference. Comparing a router point to a
    random point at a different cost is the most common way this analysis is
    got wrong.
    """
    r = random.sort_values("cost")
    out = router.copy()
    out["random_accuracy_at_same_cost"] = np.interp(
        out["cost"], r["cost"], r["accuracy"],
        left=np.nan, right=np.nan,
    )
    out["gain_over_random"] = out["accuracy"] - out["random_accuracy_at_same_cost"]
    return out


def _money(x: float) -> str:
    """Readable cost, whatever the scale. Avoids printing '0.000'."""
    if not np.isfinite(x):
        return "n/a"
    for scale, unit in ((1e3, "per 1k prompts"), (1e6, "per 1M prompts")):
        v = x * scale
        if v >= 0.01:
            return f"${v:,.3g} {unit}"
    return f"${x * 1e6:.3g} per 1M prompts"


def summarise(router_vs_random: pd.DataFrame, oracle: pd.DataFrame,
              fixed: pd.DataFrame) -> str:
    """The paragraph that goes in the report. Written from the numbers, not hope.

    Two operating points are reported deliberately:

      reference  the threshold nearest to routing half the prompts cheap,
                 chosen WITHOUT looking at the outcome.
      best       the threshold with the largest gain over random.

    The second is selected on the same data it is evaluated on, so it is
    optimistically biased and is labelled as such. Quoting only the best point
    is the standard way an offline routing result is oversold; the reference
    point is the one to lead with.
    """
    valid = router_vs_random.dropna(subset=["gain_over_random"])
    valid = valid[(valid["frac_cheap"] > 0.02) & (valid["frac_cheap"] < 0.98)]
    if valid.empty:
        return ("The router never operated strictly between the two fixed "
                "settings, so there is nothing to compare against random "
                "mixing. Report that and stop.")

    best = valid.loc[valid["gain_over_random"].idxmax()]
    ref = valid.iloc[(valid["frac_cheap"] - 0.5).abs().argmin()]

    exp_acc = float(fixed.loc[fixed["method"] == "always_expensive", "accuracy"].iloc[0])
    exp_cost = float(fixed.loc[fixed["method"] == "always_expensive", "cost"].iloc[0])
    o = oracle.sort_values("cost")
    orc_at = float(np.interp(ref["cost"], o["cost"], o["accuracy"]))

    gain_pts = ref["gain_over_random"] * 100
    verdict = (
        f"beats random mixing by {gain_pts:+.1f} points" if gain_pts > 0.5 else
        f"does NOT beat random mixing ({gain_pts:+.1f} points)" if gain_pts <= 0.0
        else f"is within noise of random mixing ({gain_pts:+.1f} points)"
    )
    cost_cut = (1 - ref["cost"] / exp_cost) * 100 if exp_cost else float("nan")

    return (
        f"At the reference operating point (threshold chosen in advance to "
        f"route about half the prompts cheap, not tuned on the outcome), the "
        f"router sends {ref['frac_cheap']:.0%} of prompts to the cheap "
        f"configuration, reaching {ref['accuracy'] * 100:.1f}% accuracy at "
        f"{_money(ref['cost'])} ({cost_cut:.0f}% below always-expensive, which "
        f"scores {exp_acc * 100:.1f}%). A random mix at the SAME average cost "
        f"reaches {ref['random_accuracy_at_same_cost'] * 100:.1f}%, so the "
        f"router {verdict}. An oracle at that cost "
        f"would reach {orc_at * 100:.1f}%, leaving "
        f"{(orc_at - ref['accuracy']) * 100:.1f} points of headroom that no "
        f"router built from these features captured. The best threshold found "
        f"anywhere on the sweep gains {best['gain_over_random'] * 100:+.1f} "
        f"points at {best['frac_cheap']:.0%} routed cheap, but that threshold "
        f"was selected on the data it is scored on and is optimistic. This is "
        f"an OFFLINE simulation replayed over logged benchmark outputs; it "
        f"does not show that the router would work on real traffic or in a "
        f"live serving system."
    )
