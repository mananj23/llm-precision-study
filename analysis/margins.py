#!/usr/bin/env python3
"""Does the combination REPAIR prompts, or just reshuffle marginal ones?

    python3 analysis/margins.py

Section 4.5 of the report finds that combining FP8 weights with an FP8 KV cache
repairs ~52-58% of the damage each component causes alone, while introducing a
comparable volume of new damage. The report states two readings and says the
experiment cannot separate them:

  (a) STRUCTURE. Each perturbation moves the model in a different direction;
      the combination lands somewhere else again, so component damage carries
      no information about combined damage.

  (b) MARGINALITY. Near the decision boundary the outcome is close to a coin
      flip, and any large-enough perturbation reshuffles those prompts. The
      "repairs" would then be regression to the mean, not recovery.

MMLU is multiple-choice and was evaluated with --log_samples, so lm-eval wrote
the per-choice log-likelihoods for every document. That gives a per-prompt
CONFIDENCE MARGIN -- the gap between the model's best and second-best choice --
and the two readings make different predictions about it:

  under (b), flips should be concentrated at small baseline margin, and once
  you condition on margin the weight-flip and KV-flip events should look
  independent, and the "novel" damage should mostly vanish.

  under (a), substantial flipping and substantial novel damage should survive
  at margins large enough that noise cannot explain them.

This script measures that. It reads the raw lm-eval sample files, which are
gitignored (they are large), so it must be run on the machine that holds them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUALITY = ROOT / "results" / "quality"

BASE = "llama31-8b__w-bf16__kv-bf16"
CELLS = {
    "base": BASE,
    "W": "llama31-8b__w-fp8__kv-bf16",
    "K": "llama31-8b__w-bf16__kv-fp8",
    "C": "llama31-8b__w-fp8__kv-fp8",
}


def _open(p: pathlib.Path):
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p, "r")


def sample_files(config: str, task: str = "mmlu") -> list[pathlib.Path]:
    d = QUALITY / config / task
    if not d.exists():
        return []
    runs = sorted([p for p in d.iterdir() if p.is_dir()])
    # Earliest run FIRST, matching analysis/load.py's choice -- but skip run
    # directories that hold no sample files at all. Aborted runs leave a
    # directory behind, and taking one blindly would silently yield nothing.
    for run in runs:
        out: list[pathlib.Path] = []
        for pat in ("**/samples_*.jsonl", "**/samples_*.jsonl.gz"):
            out.extend(run.glob(pat))
        if out:
            return sorted(out)
    return []


def margin_from_record(rec: dict) -> tuple[float | None, int | None, int | None]:
    """(margin, predicted_index, gold_index) from one lm-eval sample record.

    For a multiple-choice task, filtered_resps is one entry per choice, each
    either [loglikelihood, is_greedy] or a bare float. The margin is the gap
    between the best and second-best log-likelihood, in nats.
    """
    resps = rec.get("filtered_resps") or rec.get("resps")
    if not resps:
        return None, None, None
    lls = []
    for r in resps:
        v = r
        while isinstance(v, (list, tuple)) and v:
            v = v[0]
        # lm-eval 0.4.13 serialises these as STRINGS, e.g. ["-2.3289", "False"],
        # not as floats. Accept either; reject anything that is not a number.
        if isinstance(v, bool):
            return None, None, None
        if isinstance(v, (int, float)):
            lls.append(float(v))
        elif isinstance(v, str):
            try:
                lls.append(float(v.strip()))
            except ValueError:
                return None, None, None
        else:
            return None, None, None
    if len(lls) < 2:
        return None, None, None
    a = np.asarray(lls, dtype=float)
    order = np.argsort(-a)
    margin = float(a[order[0]] - a[order[1]])

    gold = rec.get("target")
    if isinstance(gold, str):
        gold = {"A": 0, "B": 1, "C": 2, "D": 3}.get(gold.strip().upper())
    if not isinstance(gold, int):
        gold = None
    return margin, int(order[0]), gold


def load_config(config: str) -> dict[str, dict]:
    """uid -> {margin, pred, gold, correct} for one configuration."""
    out: dict[str, dict] = {}
    files = sample_files(config)
    if not files:
        return out
    for f in files:
        with _open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task = rec.get("doc_hash") and None
                task = f.name
                # Subtask name is in the filename: samples_mmlu_anatomy_<ts>.jsonl
                stem = f.name.split("samples_", 1)[-1]
                sub = stem.rsplit("_20", 1)[0]
                uid = f"{sub}:{rec.get('doc_id')}"
                margin, pred, gold = margin_from_record(rec)
                acc = rec.get("acc")
                if acc is None and pred is not None and gold is not None:
                    acc = float(pred == gold)
                out[uid] = {
                    "margin": margin,
                    "pred": pred,
                    "gold": gold,
                    "correct": None if acc is None else float(acc),
                }
    return out


def describe_schema() -> None:
    files = sample_files(BASE)
    if not files:
        print(f"No MMLU sample files under {QUALITY / BASE / 'mmlu'}.")
        print("They are gitignored -- run this on the machine that produced them.")
        sys.exit(2)
    with _open(files[0]) as fh:
        rec = json.loads(fh.readline())
    print(f"sample files found: {len(files)}  (e.g. {files[0].name})")
    print(f"record keys: {sorted(rec.keys())}")
    m, p, g = margin_from_record(rec)
    print(f"parsed first record -> margin={m}, pred={p}, gold={g}, acc={rec.get('acc')}")
    if m is None:
        print("\nCould not read per-choice log-likelihoods from this record.")
        print("filtered_resps was:", json.dumps(rec.get("filtered_resps"))[:400])
        sys.exit(3)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--schema-only", action="store_true")
    args = ap.parse_args()

    describe_schema()
    if args.schema_only:
        return

    data = {}
    for key, cfg in CELLS.items():
        d = load_config(cfg)
        if not d:
            print(f"missing sample files for {cfg}")
            sys.exit(2)
        data[key] = d
        print(f"{key:4s} {cfg:34s} {len(d)} documents")

    uids = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"\nshared documents: {len(uids)}\n")

    margin = np.array([data["base"][u]["margin"] for u in uids], dtype=float)
    corr = {k: np.array([data[k][u]["correct"] for u in uids], dtype=float) for k in CELLS}
    ok = np.isfinite(margin) & np.all([np.isfinite(v) for v in corr.values()], axis=0)
    margin, uids = margin[ok], [u for u, m in zip(uids, ok) if m]
    corr = {k: v[ok].astype(bool) for k, v in corr.items()}
    print(f"usable documents: {ok.sum()}")

    b = corr["base"]
    dW, dK, dC = (b & ~corr[k] for k in ("W", "K", "C"))
    U = dW | dK
    repaired = U & ~dC
    novel = dC & ~U

    # Quantile bins of baseline margin, over baseline-correct prompts.
    sel = b
    qs = np.quantile(margin[sel], np.linspace(0, 1, args.bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf

    print("\nBaseline-correct prompts, binned by baseline confidence margin (nats).")
    print("If the reshuffling were only marginality, every column would collapse")
    print("toward zero in the high-margin bins.\n")
    hdr = f"{'margin bin':>22} {'n':>6} {'dmg W':>7} {'dmg K':>7} {'dmg C':>7} {'repaired':>9} {'novel':>7}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for i in range(args.bins):
        lo, hi = qs[i], qs[i + 1]
        m = sel & (margin >= lo) & (margin < hi)
        n = int(m.sum())
        if n == 0:
            continue
        r = {
            "lo": float(lo), "hi": float(hi), "n": n,
            "dW": float(dW[m].sum() / n), "dK": float(dK[m].sum() / n),
            "dC": float(dC[m].sum() / n),
            "repaired": float(repaired[m].sum() / max(U[m].sum(), 1)),
            "novel": float(novel[m].sum() / max(dC[m].sum(), 1)),
            "n_U": int(U[m].sum()), "n_C": int(dC[m].sum()),
        }
        rows.append(r)
        lo_s = "-inf" if not math.isfinite(lo) else f"{lo:.2f}"
        hi_s = "inf" if not math.isfinite(hi) else f"{hi:.2f}"
        print(f"{lo_s:>10} .. {hi_s:>8} {n:>6} {100*r['dW']:>6.1f}% {100*r['dK']:>6.1f}% "
              f"{100*r['dC']:>6.1f}% {100*r['repaired']:>8.1f}% {100*r['novel']:>6.1f}%")

    # The decisive test: restrict to CONFIDENT prompts and re-run the
    # composition arithmetic there. If structure survives at high margin,
    # marginality cannot be the whole story.
    print("\n" + "=" * 72)
    for q in (0.5, 0.75):
        thr = float(np.quantile(margin[sel], q))
        m = sel & (margin >= thr)
        n = int(m.sum())
        rW, rK, rC = (x[m].sum() / n for x in (dW, dK, dC))
        indep = 1 - (1 - rW) * (1 - rK)
        uu, cc = U[m], dC[m]
        jac = float((dW[m] & dK[m]).sum() / max(uu.sum(), 1))
        print(f"\nTop {100*(1-q):.0f}% most confident baseline-correct prompts "
              f"(margin >= {thr:.2f} nats, n = {n}):")
        print(f"  damaged by W {100*rW:.2f}%   by K {100*rK:.2f}%   by both {100*rC:.2f}%")
        print(f"  independence prediction {100*indep:.2f}%  ->  excess {100*(rC-indep):+.2f} pp")
        print(f"  Jaccard(W,K) {jac:.3f}")
        print(f"  single-component failures repaired by the combination: "
              f"{int((uu & ~cc).sum())} of {int(uu.sum())} "
              f"({100*(uu & ~cc).sum()/max(uu.sum(),1):.1f}%)")
        print(f"  combination failures that are new: "
              f"{int((cc & ~uu).sum())} of {int(cc.sum())} "
              f"({100*(cc & ~uu).sum()/max(cc.sum(),1):.1f}%)")

    # ---- how concentrated is the damage? -----------------------------------
    # The first thing to establish is WHERE quantization does its damage. If
    # almost all of it sits among prompts the model was nearly indifferent
    # about, then the churn headline in section 4.4 needs that qualification.
    rng = np.random.default_rng(20261116)

    def share_below(d, q):
        thr = float(np.quantile(margin[sel], q))
        lowm = sel & (margin < thr)
        tot = d[sel].sum()
        return (float(d[lowm].sum() / tot) if tot else float("nan")), thr

    print("\n" + "=" * 72)
    print("CONCENTRATION -- what share of all damage sits at low confidence?\n")
    for q in (0.1, 0.25, 0.5):
        print(f"  bottom {100*q:>4.0f}% of baseline-correct prompts by margin:", end="")
        parts = []
        for nm, d in (("W", dW), ("K", dK), ("C", dC)):
            s, thr = share_below(d, q)
            parts.append(f"{nm} {100*s:5.1f}%")
        print("   " + "   ".join(parts) + f"   (margin < {thr:.2f} nats)")

    # Highest margin at which ANY of the three still damages >= 0.5% of prompts.
    ceiling = None
    for i in range(args.bins):
        lo_b, hi_b = qs[i], qs[i + 1]
        m = sel & (margin >= lo_b) & (margin < hi_b)
        if m.sum() and max(dW[m].mean(), dK[m].mean(), dC[m].mean()) >= 0.005:
            ceiling = float(hi_b)
    if ceiling is not None and math.isfinite(ceiling):
        above = sel & (margin >= ceiling)
        print(f"\n  above {ceiling:.2f} nats (n = {int(above.sum())}, "
              f"{100*above.sum()/sel.sum():.0f}% of baseline-correct prompts):")
        print(f"    damaged by W {100*dW[above].mean():.3f}%   "
              f"K {100*dK[above].mean():.3f}%   C {100*dC[above].mean():.3f}%")

    # ---- the discriminator, evaluated where damage still exists ------------
    # CAUTION: "repaired" and "novel" above rise with margin even under pure
    # marginality, because their denominators shrink -- they are descriptive,
    # not decisive. And a threshold set so high that BOTH components damage
    # nothing makes the ratio 0/0, which is uninformative rather than
    # reassuring. So evaluate the ratio over the widest high-margin band in
    # which the larger component still damages at least `floor` of prompts.
    FLOOR = 0.002
    band = None
    for q in (0.75, 0.5, 0.25):
        thr = float(np.quantile(margin[sel], q))
        m = sel & (margin >= thr)
        if m.sum() and max(dW[m].mean(), dK[m].mean()) >= FLOOR:
            band = (q, thr, m)
            break

    print("\n" + "=" * 72)
    if band is None:
        print("DISCRIMINATOR -- not evaluable")
        print(f"  Above every threshold tried, neither component damages as much")
        print(f"  as {100*FLOOR:.1f}% of prompts, so the ratio has no denominator.")
        print("  That is itself the finding: component damage does not reach")
        print("  confident prompts at all. Report the CONCENTRATION block.")
        disc = {"evaluable": False, "floor": FLOOR}
    else:
        q, thr, m = band
        nt = int(m.sum())
        rW, rK, rC = (x[m].sum() / nt for x in (dW, dK, dC))
        ratio = rC / max(rW, rK)
        boot = []
        idx_b = np.flatnonzero(m)
        for _ in range(4000):
            i = rng.choice(idx_b, size=nt, replace=True)
            mx = max(dW[i].mean(), dK[i].mean())
            if mx > 0:
                boot.append(dC[i].mean() / mx)
        lo_c, hi_c = ((float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
                      if boot else (float("nan"),) * 2)
        print(f"DISCRIMINATOR -- top {100*(1-q):.0f}% by margin "
              f"(>= {thr:.2f} nats, n = {nt})")
        print(f"  damaged by W {100*rW:.2f}%   K {100*rK:.2f}%   C {100*rC:.2f}%")
        print(f"  combination / larger component = {ratio:.2f}  "
              f"95% CI [{lo_c:.2f}, {hi_c:.2f}]")
        print("  interval covering 1 -> MARGINALITY (b): the combination is no")
        print("     worse than a component once the near-ties are excluded.")
        print("  interval above 1    -> STRUCTURE (a): the combination carries")
        print("     damage the components do not, where noise cannot explain it.")
        disc = {"evaluable": True, "quantile": q, "threshold": thr, "n": nt,
                "rate_W": rW, "rate_K": rK, "rate_C": rC,
                "ratio": ratio, "ci": [lo_c, hi_c], "floor": FLOOR}

    # ---- durable export ----------------------------------------------------
    # The raw lm-eval sample files are gigabytes and gitignored, so they die
    # with the machine. Everything section 4.6 needs is four columns, which
    # compress to well under a megabyte and DO belong in the repository.
    try:
        import pandas as pd

        tidy = pd.DataFrame({
            "uid": uids,
            "margin_base": margin,
            **{f"correct_{k}": corr[k].astype(int) for k in CELLS},
        })
        pq = ROOT / "results" / "logs" / "mmlu_margins.parquet"
        pq.parent.mkdir(parents=True, exist_ok=True)
        tidy.to_parquet(pq, index=False)
        print(f"\nwrote {pq.relative_to(ROOT)}  ({pq.stat().st_size // 1024} KB, "
              f"{len(tidy)} rows) -- COMMIT THIS: it is what survives the box")
    except Exception as e:  # never let the export break the analysis
        print(f"\n[warn] could not write the margins parquet: {e}")

    out = ROOT / "report" / "margins.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"bins": rows, "n_usable": int(ok.sum()), "discriminator": disc,
         "concentration": {
             str(q): {nm: share_below(d, q)[0]
                      for nm, d in (("W", dW), ("K", dK), ("C", dC))}
             for q in (0.1, 0.25, 0.5)},
         "damage_ceiling_nats": ceiling},
        indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
