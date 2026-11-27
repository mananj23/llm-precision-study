#!/usr/bin/env python3
"""Fold every lm-eval sample file into ONE tidy per-prompt table.

    python scripts/collect_logs.py
    python scripts/collect_logs.py --out results/logs/per_prompt.parquet

Output schema, one row per (model, config, task, prompt):

    family        llama31-8b
    config        llama31-8b__w-fp8__kv-bf16
    weights       fp8
    kv_cache      fp8
    task          gsm8k
    doc_id        int, stable across configs -- this is what pairs the bootstrap
    run_id        20261130T141005
    correct       0.0/1.0
    metric_name   exact_match
    input_len     tokens in the prompt (approx, whitespace fallback)
    output_len    tokens generated (approx)
    prompt_text   the prompt, truncated -- routing features come from this
    output_text   the model's filtered response

This file is the input to BOTH analysis/ (paired bootstrap) and router/
(per-prompt labels). If it is wrong, everything downstream is wrong, so it is
the one place with real validation: run `python tests/test_collect.py`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import pandas as pd

import common

ROOT = common.ROOT

# lm-eval names the correctness field differently per task. Order matters:
# the first match wins, so put the strict variants first.
METRIC_KEYS = [
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match",
    "pass@1",
    "prompt_level_strict_acc",
    "prompt_level_loose_acc",
    "acc_norm",
    "acc",
    "score",
    "f1",
]


# Fields that look like a metric but are not. Without this the guessing
# fallback below can latch onto a sampling parameter and happily report it as
# per-prompt correctness, which would corrupt every number in the paper.
NOT_METRICS = {
    "doc_id", "idx", "seed", "temperature", "top_p", "top_k", "epoch",
    "repeats", "num_fewshot", "input_len", "output_len", "elapsed",
}

_GUESSED: set[str] = set()


def pick_metric(rec: dict) -> tuple[str, float] | None:
    for key in METRIC_KEYS:
        if key in rec:
            val = rec[key]
            if isinstance(val, (int, float, bool)):
                return key, float(val)
    # Fall back: any top-level float in [0, 1] that is not an id or a knob.
    # `correct` is the dependent variable of the whole study, so a guess here
    # gets announced rather than made quietly.
    for key, val in rec.items():
        if key in NOT_METRICS or not isinstance(val, (int, float)):
            continue
        if isinstance(val, bool) or 0.0 <= float(val) <= 1.0:
            if key not in _GUESSED:
                _GUESSED.add(key)
                print(f"    !!! no known metric key; GUESSING '{key}' is "
                      f"correctness. Verify this before using any result, and "
                      f"add the real key to METRIC_KEYS in collect_logs.py.")
            return key, float(val)
    return None


# lm-eval names sample files samples_<task>_<timestamp>.jsonl. Reading the task
# from the filename rather than the directory matters for task GROUPS: `--tasks
# mmlu` writes 57 subtask files into one directory, and labelling them all
# "mmlu" collides their doc_ids and silently breaks the pairing.
SAMPLE_RE = re.compile(r"^samples_(?P<task>.+?)_\d{4}-\d{2}-\d{2}T.*\.jsonl$")
SAMPLE_RE_LOOSE = re.compile(r"^samples_(?P<task>.+?)_[\dT:\.\-]+\.jsonl$")


def task_from_filename(path: pathlib.Path, fallback: str) -> str:
    for rx in (SAMPLE_RE, SAMPLE_RE_LOOSE):
        m = rx.match(path.name)
        if m:
            return m.group("task")
    return fallback


def approx_tokens(text: str) -> int:
    """Rough token count without loading a tokenizer.

    ~0.75 words per token for English. Good enough for a routing FEATURE and
    for plotting accuracy against input length. If you use it for anything that
    goes in the results table, swap in the real tokenizer -- there is a hook in
    exact_token_lengths() below.
    """
    if not text:
        return 0
    return max(1, int(len(re.findall(r"\S+", text)) / 0.75))


def extract_prompt(rec: dict) -> str:
    args = rec.get("arguments")
    # lm-eval has used several shapes for `arguments` across versions.
    if isinstance(args, list) and args:
        first = args[0]
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        if isinstance(first, dict):
            for k in ("arg_0", "context", "prompt"):
                if k in first:
                    return str(first[k])
    if isinstance(args, dict):
        for k in ("arg_0", "context", "prompt"):
            if k in args:
                return str(args[k])
    doc = rec.get("doc")
    if isinstance(doc, dict):
        for k in ("question", "input", "prompt", "text"):
            if k in doc:
                return str(doc[k])
    return ""


# lm-eval writes ONE SAMPLE LINE PER (doc, filter). GSM8K ships two filters,
# strict-match and flexible-extract, so a 1319-question task produces 2638
# lines and every doc_id appears twice. Collapsing that blindly is how you end
# up comparing one config's strict-match against another's flexible-extract --
# which flipped the sign of the INT4 result by 3.3 points before this was
# caught. The filter has to be a column, and comparisons have to fix it.
FILTER_KEYS = ("filter", "filter_name", "filter_key")

# Preference order when reducing to one row per doc. Strict first: it is the
# conservative reading and the one usually quoted for GSM8K.
FILTER_PREFERENCE = (
    "strict-match",
    "strict_match",
    "flexible-extract",
    "flexible_extract",
    "none",
)


def extract_filter(rec: dict) -> str:
    for k in FILTER_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    return "none"


def choose_filter(available: set[str]) -> str:
    for f in FILTER_PREFERENCE:
        if f in available:
            return f
    return sorted(available)[0]


def extract_response(rec: dict) -> str:
    for key in ("filtered_resps", "resps"):
        val = rec.get(key)
        if isinstance(val, list) and val:
            inner = val[0]
            if isinstance(inner, (list, tuple)) and inner:
                return str(inner[0])
            return str(inner)
        if isinstance(val, str):
            return val
    return ""


def parse_sample_file(path: pathlib.Path, config: str, task: str, run_id: str) -> list[dict]:
    meta = common.parse_config_name(config)
    rows: list[dict] = []
    skipped = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            got = pick_metric(rec)
            if got is None:
                skipped += 1
                continue
            metric_name, correct = got
            prompt = extract_prompt(rec)
            resp = extract_response(rec)
            rows.append(
                {
                    "family": meta["family"],
                    "config": config,
                    "weights": meta["weights"],
                    "kv_cache": meta["kv_cache"],
                    "task": task,
                    "doc_id": int(rec.get("doc_id", len(rows))),
                    "run_id": run_id,
                    "metric_name": metric_name,
                    "filter_name": extract_filter(rec),
                    "correct": correct,
                    "input_len": approx_tokens(prompt),
                    "output_len": approx_tokens(resp),
                    # The prompt text is kept because the ROUTER needs features
                    # computable before the model answers. Truncated to keep the
                    # file manageable; the first 2000 characters are plenty for
                    # length and n-gram features.
                    "prompt_text": prompt[:2000],
                    "output_text": resp[:4000],
                }
            )
    if skipped:
        print(f"    (skipped {skipped} lines with no usable metric in {path.name})")
    return rows


def discover(quality_root: pathlib.Path) -> list[dict]:
    """results/quality/<config>/<task>/<run_id>/**/samples_*.jsonl"""
    rows: list[dict] = []
    if not quality_root.exists():
        return rows
    for cfg_dir in sorted(quality_root.iterdir()):
        if not cfg_dir.is_dir() or cfg_dir.name.startswith("_"):
            continue
        for task_dir in sorted(cfg_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for run_dir in sorted(task_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                files = sorted(run_dir.rglob("samples_*.jsonl"))
                if not files:
                    continue
                print(f"  {cfg_dir.name} / {task_dir.name} / {run_dir.name}: "
                      f"{len(files)} sample file(s)")
                for f in files:
                    task = task_from_filename(f, task_dir.name)
                    if task != task_dir.name:
                        print(f"      {f.name} -> task={task}")
                    rows.extend(
                        parse_sample_file(f, cfg_dir.name, task, run_dir.name)
                    )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quality-root", default="results/quality")
    ap.add_argument("--out", default="results/logs/per_prompt.parquet")
    ap.add_argument("--keep", choices=["latest", "all"], default="latest",
                    help="latest: one run per (config, task). all: keep reruns, "
                         "which is what you want for estimating the noise floor.")
    ap.add_argument("--filter", default=None,
                    help="lm-eval filter to keep when a task has several "
                         "(e.g. strict-match, flexible-extract). Default: the "
                         "strict one. The SAME filter is used for every config, "
                         "which is what makes the comparison valid.")
    ap.add_argument("--all-filters", action="store_true",
                    help="keep every filter's rows. Downstream analysis assumes "
                         "one row per prompt, so this is for inspection only.")
    args = ap.parse_args()

    root = pathlib.Path(args.quality_root)
    if not root.is_absolute():
        root = ROOT / root
    print(f"scanning {root}")
    rows = discover(root)
    if not rows:
        print("\nNo sample files found. Either no evals have run yet, or you "
              "forgot --log_samples (run_quality.sh always passes it).")
        return 1

    df = pd.DataFrame(rows)

    if args.keep == "latest":
        latest = (
            df.groupby(["config", "task"])["run_id"].max().rename("keep_run").reset_index()
        )
        before = len(df)
        df = df.merge(latest, on=["config", "task"])
        df = df[df["run_id"] == df["keep_run"]].drop(columns=["keep_run"])
        if before != len(df):
            print(f"\nkept latest run per (config, task): {before} -> {len(df)} rows")
            print("(use --keep all to include reruns for the noise floor)")

    # ---- one row per (config, task, doc_id) -----------------------------------
    # Everything downstream -- the paired bootstrap, the discordance counts, the
    # router labels -- assumes one outcome per prompt per config. With two
    # filters per doc it would silently double-count, and comparing across
    # configs could pair different filters against each other.
    if "filter_name" in df.columns and df["filter_name"].nunique() > 1:
        print("\n--- filters ---")
        for task, sub in df.groupby("task"):
            avail = set(sub["filter_name"].dropna().unique())
            if len(avail) <= 1:
                continue
            chosen = choose_filter(avail)
            dropped = sorted(avail - {chosen})
            print(f"  {task}: filters {sorted(avail)} -> keeping '{chosen}'")
            print(f"    (dropped {dropped}; rerun with --filter to pick another,")
            print(f"     or --all-filters to keep every row)")
            if not args.all_filters:
                keep = (df["task"] != task) | (df["filter_name"] == (args.filter or chosen))
                df = df[keep]

    # ---- integrity checks: these catch the errors that silently ruin Step 4 ---
    print("\n--- integrity ---")
    counts = df.groupby(["task", "config"])["doc_id"].nunique().unstack()
    print(counts.to_string())
    for task, row in counts.iterrows():
        vals = row.dropna().unique()
        if len(vals) > 1:
            print(f"  WARNING: {task} has different prompt counts per config: "
                  f"{dict(row.dropna())}")
            print("  A paired bootstrap needs the SAME prompts in both runs. The "
                  "analysis will intersect on doc_id, but check why they differ.")
    dupes = df.duplicated(subset=["config", "task", "doc_id", "run_id"]).sum()
    if dupes:
        print(f"  WARNING: {dupes} duplicate (config, task, doc_id, run_id) rows")
        print("  The paired bootstrap pairs on doc_id, so duplicates would")
        print("  double-count prompts. Usually a task GROUP collapsed into one")
        print("  task label; check the '-> task=' lines above.")

    # The paired bootstrap and the discordance arithmetic (p10, p01) assume a
    # BINARY per-prompt outcome. A continuous metric would not error, it would
    # just produce numbers that mean something else.
    non_binary = df.loc[~df["correct"].isin([0.0, 1.0]), "metric_name"].unique()
    if len(non_binary):
        print(f"  WARNING: non-binary correctness values from metric(s): "
              f"{list(non_binary)}")
        print("  Discordance (p10/p01) and the paired bootstrap over flips are")
        print("  defined for 0/1 outcomes. Report those tasks separately.")

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {len(df):,} rows -> {out}")
    print(df.groupby(["task"])["correct"].agg(["count", "mean"]).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
