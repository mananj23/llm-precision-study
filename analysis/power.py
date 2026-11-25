#!/usr/bin/env python3
"""How many prompts do you need to detect a quantization regression?

    python analysis/power.py                 # self-check + worked example
    python analysis/power.py --from-data     # using your own measured discordance

PRIOR WORK -- read this before claiming anything is new here
------------------------------------------------------------
The machinery below is NOT novel, and the report must say so plainly.

  Miller, "Adding Error Bars to Evals: A Statistical Approach to Language
  Model Evaluations", arXiv:2411.00640 (2024). Recommends power analysis to
  decide whether an eval can test a hypothesis at all, and inference on
  question-level paired differences rather than population summaries -- i.e.
  the argument behind bootstrap.py.

  Kotawala, "Resolution Diagnostics for Paired LLM Evaluation",
  arXiv:2605.30315, ICML 2026. Derives exactly this framework -- minimum
  detectable effect at the current N, required paired N*, and a resolution
  ratio q = N/N* -- and applies it to public leaderboards, finding many
  displayed rankings unresolved. If you read one paper before writing this
  section, read that one, and cite it as the source of the framework.

  Bowyer, Aitchison & Ivanova, "Don't use the CLT in LLM evals with fewer
  than a few hundred datapoints", arXiv:2503.01747. The normal approximation
  used here UNDERSTATES uncertainty below a few hundred prompts. See the
  small-n warning in mde() and treat HumanEval-sized results as indicative
  only.

What is left for this project, then
-----------------------------------
Not the method. The application: nobody has asked whether the specific effects
the QUANTIZATION literature reports -- 1 to 3 accuracy points for FP8, INT8 and
INT4 -- are resolvable on the benchmarks they are reported on, using
discordance rates measured from real quantized checkpoints rather than assumed.
That is a narrow contribution, and the report should present it as one.

The quantization literature reports effects of 1-3 accuracy points, almost
always as point estimates on standard benchmarks. On GSM8K's ~1.3k prompts a
1-point regression is at or below the detection threshold, so a published
"1% drop" and a published "no change" can be the same underlying truth
measured twice.

The statistics
--------------
For paired binary outcomes the per-prompt difference takes values in
{-1, 0, +1}. Write

    p10 = P(baseline correct, quantized wrong)     "losses"
    p01 = P(baseline wrong, quantized correct)     "wins"
    r   = p10 + p01                                 DISCORDANCE
    d   = p10 - p01                                 the effect

Only discordant prompts carry information; the ones both configurations get
right (or both wrong) contribute nothing to the difference. So

    Var(per-prompt difference) = r - d^2
    SE(d_hat)                  = sqrt((r - d^2) / n)

and a 95% interval excludes zero when |d| > 1.96 * SE. Solving for d:

    MDE = z * sqrt(r / (n + z^2))        (z = 1.96)

The key consequence: **required sample size scales with the discordance rate,
not with accuracy.** Two configurations that agree on 95% of prompts need far
fewer prompts to separate than two that disagree constantly — which is exactly
backwards from the intuition that "closer means harder".

This also explains why the paired bootstrap is ~7x tighter than the unpaired
one (see bootstrap.py): the unpaired estimator's variance is governed by
p(1-p) ~ 0.2, the paired one's by r ~ 0.05. The ratio of standard errors is
sqrt(0.2/0.05) = 2, and larger still when the configurations agree more.

Everything below is validated against simulation in `self_check()`.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

try:
    from scipy.stats import norm
    _Z = norm.ppf
except Exception:  # scipy is in requirements, but do not die without it
    def _Z(q):  # noqa: D103
        from math import sqrt
        # Acklam's inverse-normal approximation, plenty accurate here
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        pl, ph = 0.02425, 1 - 0.02425
        if q < pl:
            x = sqrt(-2 * np.log(q))
            return (((((c[0]*x+c[1])*x+c[2])*x+c[3])*x+c[4])*x+c[5]) / \
                   ((((d[0]*x+d[1])*x+d[2])*x+d[3])*x+1)
        if q > ph:
            x = sqrt(-2 * np.log(1 - q))
            return -(((((c[0]*x+c[1])*x+c[2])*x+c[3])*x+c[4])*x+c[5]) / \
                     ((((d[0]*x+d[1])*x+d[2])*x+d[3])*x+1)
        x = q - 0.5
        rr = x * x
        return (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*x / \
               (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1)

# Standard benchmark sizes, for marking on plots and tables.
BENCHMARK_SIZES = {
    "GSM8K": 1319,
    "MMLU (full)": 14042,
    "MMLU (10 subjects)": 1400,
    "HumanEval": 164,
    "IFEval": 541,
    "LongBench (subset)": 800,
}


# ------------------------------------------------------------------ core ----
SMALL_N = 300   # below this the normal approximation is not trustworthy
_warned_small_n = set()


def mde(n: int, discordance: float, ci: float = 0.95,
        warn_small: bool = True) -> float:
    """Smallest effect whose paired interval excludes zero, at this n.

    Returns an absolute accuracy difference (0.015 = 1.5 points).

    CAUTION below n ~ 300. This is a normal approximation, and Bowyer et al.
    (arXiv:2503.01747) show CLT-based methods badly understate uncertainty on
    small benchmarks -- so the true MDE there is LARGER than this returns, and
    the real situation is worse than the number suggests, not better. For
    HumanEval-sized sets treat the output as a lower bound and say so.
    """
    z = _Z(1 - (1 - ci) / 2)
    if warn_small and n < SMALL_N and n not in _warned_small_n:
        _warned_small_n.add(n)
        print(f"  [warning] n={n} is below {SMALL_N}: the normal approximation "
              f"understates uncertainty here (Bowyer et al. 2025). Treat this "
              f"MDE as a LOWER bound.")
    return float(z * np.sqrt(discordance / (n + z * z)))


def power_at(n: int, effect: float, discordance: float,
             ci: float = 0.95) -> float:
    """P(the interval excludes zero) when the true effect is `effect`."""
    if discordance <= abs(effect):
        discordance = abs(effect) + 1e-9      # |d| <= r always holds
    z = _Z(1 - (1 - ci) / 2)
    se = np.sqrt((discordance - effect ** 2) / n)
    if se <= 0:
        return 1.0
    # two-sided: detect in either tail
    lam = abs(effect) / se
    from math import erf, sqrt
    cdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))  # noqa: E731
    return float(cdf(lam - z) + cdf(-lam - z))


def n_required(effect: float, discordance: float, power: float = 0.80,
               ci: float = 0.95) -> int:
    """Prompts needed to detect `effect` with this power."""
    z_a = _Z(1 - (1 - ci) / 2)
    z_b = _Z(power)
    if abs(effect) <= 0:
        return 10 ** 9
    return int(np.ceil((z_a + z_b) ** 2 * (discordance - effect ** 2)
                       / effect ** 2))


def discordance_from_pair(a: np.ndarray, b: np.ndarray) -> dict:
    """Measured discordance between two configurations, from per-prompt data."""
    a = np.asarray(a, float) > 0.5
    b = np.asarray(b, float) > 0.5
    losses = float(np.mean(b & ~a))    # baseline right, this one wrong
    wins = float(np.mean(a & ~b))
    return {
        "n": int(a.size),
        "losses": losses,
        "wins": wins,
        "discordance": losses + wins,
        "effect": wins - losses,
        "agreement": float(np.mean(a == b)),
    }


# -------------------------------------------------------------- validation --
def self_check(verbose: bool = True) -> bool:
    """Check the analytic formulas against simulation. Must pass before use."""
    rng = np.random.default_rng(42)
    ok = True

    for n, r, d in [(1319, 0.08, 0.015), (500, 0.05, 0.02), (5000, 0.12, 0.01)]:
        p10 = (r + d) / 2          # losses
        p01 = (r - d) / 2          # wins
        if p01 < 0:
            continue
        trials, detected = 600, 0
        for _ in range(trials):
            u = rng.random(n)
            diff = np.where(u < p10, -1.0, np.where(u < p10 + p01, 1.0, 0.0))
            # 95% normal interval on the mean difference
            se = diff.std(ddof=1) / np.sqrt(n)
            lo, hi = diff.mean() - 1.96 * se, diff.mean() + 1.96 * se
            if not (lo <= 0 <= hi):
                detected += 1
        empirical = detected / trials
        predicted = power_at(n, -d, r)
        delta = abs(empirical - predicted)
        good = delta < 0.06
        ok &= good
        if verbose:
            print(f"  n={n:>5} r={r:.2f} d={d * 100:+.1f}pts -> "
                  f"power predicted {predicted:.3f}, simulated {empirical:.3f} "
                  f"{'OK' if good else 'MISMATCH'}")
    return bool(ok)


# ------------------------------------------------------------------ report --
def literature_table(discordance: float = 0.08) -> str:
    """Can standard benchmarks resolve the effects the literature reports?"""
    lines = [
        f"Assumed discordance rate: {discordance:.0%} of prompts flip either way.",
        "",
        f"{'Benchmark':<22}{'n':>7}{'MDE':>9}   {'1 pt?':<9}{'2 pts?':<9}{'3 pts?':<9}",
        "-" * 68,
    ]
    small = []
    for name, n in BENCHMARK_SIZES.items():
        m = mde(n, discordance, warn_small=False)
        cells = []
        for eff in (0.01, 0.02, 0.03):
            p = power_at(n, -eff, discordance)
            cells.append(f"{p:.0%}")
        mark = " *" if n < SMALL_N else ""
        if n < SMALL_N:
            small.append(name)
        lines.append(f"{name:<22}{n:>7}{m * 100:>8.2f}p{mark:<2} "
                     f"{cells[0]:<9}{cells[1]:<9}{cells[2]:<9}")
    lines += [
        "",
        "MDE = smallest effect whose 95% paired interval excludes zero.",
        "Percentages are the power to detect an effect of that size.",
    ]
    if small:
        lines += [
            f"* {', '.join(small)}: under {SMALL_N} prompts the normal "
            "approximation understates",
            "  uncertainty (Bowyer et al., arXiv:2503.01747), so the true MDE is "
            "LARGER than shown.",
            "  Report these as lower bounds, not estimates.",
        ]
    lines += [
        "",
        "Framework follows Kotawala, arXiv:2605.30315 (ICML 2026); see also",
        "Miller, arXiv:2411.00640. Cite them -- the method is theirs.",
    ]
    return "\n".join(lines)


def plot(out: pathlib.Path, discordances=(0.04, 0.08, 0.15)) -> None:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import plotstyle
    from plotstyle import apply_style, finish

    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ns = np.unique(np.logspace(2, 4.4, 160).astype(int))

    for i, r in enumerate(discordances):
        ax.plot(ns, [mde(n, r) * 100 for n in ns],
                color=plotstyle.SERIES[i], linewidth=2.2,
                label=f"{r:.0%} of prompts flip")

    for eff, style in ((1.0, ":"), (3.0, "--")):
        ax.axhline(eff, color=plotstyle.MUTED, linewidth=1.2, linestyle=style)
        ax.annotate(f"{eff:.0f}-point effect", (ns[0] * 1.05, eff),
                    fontsize=8.5, color=plotstyle.INK_SECONDARY,
                    va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Label the benchmark markers inside the axes, whatever the data range.
    ymin, ymax = ax.get_ylim()
    label_y = ymax * 0.92
    for name, n in (("HumanEval", 164), ("IFEval", 541), ("GSM8K", 1319),
                    ("MMLU full", 14042)):
        ax.axvline(n, color=plotstyle.AXIS, linewidth=1.0, zorder=1)
        ax.annotate(f" {name} ", (n, label_y), rotation=90, fontsize=8,
                    color=plotstyle.INK_SECONDARY, va="top", ha="center",
                    bbox=dict(boxstyle="square,pad=0.15", fc=plotstyle.SURFACE,
                              ec="none"))
    ax.set_ylim(ymin, ymax)

    # Plain numbers on a log axis; "6 x 10^0 accuracy points" reads as noise.
    from matplotlib.ticker import FuncFormatter, LogLocator
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{int(v):,}" if v >= 1 else ""))

    ax.set_xlabel("prompts evaluated")
    ax.set_ylabel("minimum detectable effect  (accuracy points)")
    finish(ax, "What size regression can a benchmark actually resolve?",
           "95% paired interval excludes zero · lower is better")
    # Lower left: the only quadrant the decreasing curves and the benchmark
    # labels both leave free.
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def from_data(per_prompt_path: pathlib.Path, baseline_suffix="w-bf16__kv-bf16"):
    """Measured discordance per (task, config) -- the honest version."""
    import pandas as pd

    df = pd.read_parquet(per_prompt_path)
    rows = []
    for (fam, task), g in df.groupby(["family", "task"]):
        base_cfg = f"{fam}__{baseline_suffix}"
        base = g[g["config"] == base_cfg].set_index("doc_id")["correct"]
        if base.empty:
            continue
        base = base[~base.index.duplicated(keep="last")]
        for cfg, cg in g.groupby("config"):
            if cfg == base_cfg:
                continue
            s = cg.set_index("doc_id")["correct"]
            s = s[~s.index.duplicated(keep="last")]
            shared = base.index.intersection(s.index)
            if len(shared) < 20:
                continue
            st = discordance_from_pair(s.loc[shared].to_numpy(),
                                       base.loc[shared].to_numpy())
            st.update({"family": fam, "task": task, "config": cfg,
                       "mde": mde(st["n"], st["discordance"]),
                       "n_for_1pt": n_required(0.01, st["discordance"]),
                       "detected": abs(st["effect"]) > mde(st["n"],
                                                           st["discordance"])})
            rows.append(st)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-data", action="store_true",
                    help="use measured discordance from results/logs/per_prompt.parquet")
    ap.add_argument("--per-prompt", default="results/logs/per_prompt.parquet")
    ap.add_argument("--discordance", type=float, default=0.08)
    ap.add_argument("--out", default="report/fig_power.png")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]

    print("=== validating the formulas against simulation ===")
    if not self_check():
        print("SELF-CHECK FAILED -- do not use these numbers")
        return 1
    print("  all OK\n")

    if args.from_data:
        p = root / args.per_prompt
        if not p.exists():
            print(f"missing {p}; run scripts/collect_logs.py first")
            return 1
        tbl = from_data(p)
        if tbl.empty:
            print("no paired data yet")
            return 1
        cols = ["task", "config", "n", "discordance", "effect", "mde",
                "detected", "n_for_1pt"]
        print(tbl[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:.4f}"))
        out_csv = root / "analysis" / "power.csv"
        tbl.to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")
        med = float(tbl["discordance"].median())
        print(f"\nmedian measured discordance: {med:.1%}")
        print(literature_table(med))
        plot(root / args.out, (round(med / 2, 3), round(med, 3), round(med * 2, 3)))
    else:
        print("=== worked example (assumed discordance; rerun with "
              "--from-data once you have results) ===\n")
        print(literature_table(args.discordance))
        print()
        for eff in (0.01, 0.02, 0.03):
            print(f"  to detect {eff * 100:.0f} point(s) with 80% power you need "
                  f"{n_required(eff, args.discordance):,} prompts")
        plot(root / args.out)

    print("\nThe sentence this supports, for the report:")
    print("  Effects of the size this literature reports sit close to the")
    print("  resolution limit of the benchmarks it reports them on. A point")
    print("  estimate without an interval cannot distinguish a real 1-point")
    print("  regression from no regression at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
