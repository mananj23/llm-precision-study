"""Load per-prompt results with lm-eval filter recovery, and validate.

Why this module exists
----------------------
`scripts/collect_logs.py` wrote one row per (document, lm-eval filter) but did
not record WHICH filter a row came from. For GSM8K, lm-eval applies two filters
-- `strict-match` and `flexible-extract` -- and both report under the metric
name `exact_match`. The result is 2 x 1319 = 2638 rows per config with no column
distinguishing them.

Mixing the two filters is not a cosmetic problem. `flexible-extract` scores
~7 points higher than `strict-match` on this model, so any dedupe that silently
keeps one row per document can pick strict-match for one config and
flexible-extract for another, which fabricates differences between precisions
that do not exist.

Recovery
--------
lm-eval writes its sample file filter-major: all documents under the first
filter, then all documents under the second. This module relies on that, and
then PROVES it per (config, task) by checking the recovered per-block accuracy
against the accuracy table lm-eval itself printed in stdout.log. If a block does
not match a printed filter value to within 1e-4, the loader raises. No number
reaches the report without that check passing.

MMLU is unaffected: it reports a single `acc` metric under one filter, exactly
14042 rows per config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PER_PROMPT = REPO / "results" / "logs" / "per_prompt.parquet"
SYSTEM = REPO / "results" / "logs" / "system.parquet"
QUALITY = REPO / "results" / "quality"

TOL = 1e-4

# Row counts we require. A short smoke run must not be mistaken for a real pass.
GSM8K_N = 1319
MMLU_N = 14042

# The filter we report for GSM8K. strict-match requires the model to emit the
# answer in the requested format; flexible-extract digs a number out of free
# text. strict-match is the conservative choice and the one the pre-registration
# named, so it is the primary. Both are carried through so either can be shown.
PRIMARY_GSM8K_FILTER = "strict-match"

# lm-eval's printed table, e.g.
# |gsm8k|      3|flexible-extract|     5|exact_match|^  |0.7756|^  |0.0115|
_TABLE_ROW = re.compile(
    r"\|\s*(?P<filter>strict-match|flexible-extract|none)\s*\|"
    r"[^|]*\|[^|]*\|[^|]*\|\s*(?P<value>0\.\d+|1\.0+)\s*\|"
)


@dataclass(frozen=True)
class FilterCheck:
    config: str
    task: str
    run_id: str
    recovered: dict[str, float]
    printed: dict[str, float]
    ok: bool

    def report(self) -> str:
        mark = "OK  " if self.ok else "FAIL"
        parts = []
        for name in sorted(self.recovered):
            got = self.recovered[name]
            exp = self.printed.get(name)
            exp_s = "n/a" if exp is None else f"{exp:.4f}"
            parts.append(f"{name}={got:.4f} (lm-eval {exp_s})")
        return f"[{mark}] {self.config} {self.task} {self.run_id}: " + "; ".join(parts)


def _printed_values(config: str, task: str, run_id: str) -> dict[str, float]:
    """Filter -> accuracy, as lm-eval printed it for this exact run."""
    d = QUALITY / config / task / run_id
    log = d / "stdout.log"
    if not log.exists():
        return {}
    text = log.read_text(errors="replace")
    out: dict[str, float] = {}
    for m in _TABLE_ROW.finditer(text):
        out[m.group("filter")] = float(m.group("value"))
    return out


def _split_filters(block: pd.DataFrame, n_docs: int) -> dict[str, pd.DataFrame]:
    """Split one (config, task, run) frame into filter-major blocks.

    Returns {filter_name: frame}. Raises if the shape is not a clean multiple
    of n_docs or if document ids do not repeat identically across blocks.
    """
    if len(block) % n_docs != 0:
        raise ValueError(f"{len(block)} rows is not a multiple of {n_docs} documents")
    k = len(block) // n_docs
    if k == 1:
        return {"none": block.reset_index(drop=True)}
    if k != 2:
        raise ValueError(f"expected 1 or 2 filters, inferred {k}")

    first = block.iloc[:n_docs].reset_index(drop=True)
    second = block.iloc[n_docs:].reset_index(drop=True)
    if not first.doc_id.equals(second.doc_id):
        raise ValueError("document ids differ across filter blocks; not filter-major")

    # strict-match can never score above flexible-extract: it is the same
    # extraction with an added format requirement. Use that to label the blocks,
    # then the lm-eval table confirms it independently.
    a, b = first.correct.mean(), second.correct.mean()
    if a <= b:
        return {"strict-match": first, "flexible-extract": second}
    return {"flexible-extract": first, "strict-match": second}


def load_gsm8k(validate: bool = True) -> tuple[pd.DataFrame, list[FilterCheck]]:
    """Per-document GSM8K outcomes with a real `filter` column.

    Returns (frame, checks). The frame has one row per
    (config, run_id, filter, doc_id).
    """
    raw = pd.read_parquet(PER_PROMPT)
    raw = raw[raw.task == "gsm8k"]

    frames, checks = [], []
    for (config, run_id), block in raw.groupby(["config", "run_id"], sort=True):
        if len(block) < GSM8K_N:  # smoke run
            continue
        blocks = _split_filters(block, GSM8K_N)
        recovered = {name: df.correct.mean() for name, df in blocks.items()}
        printed = _printed_values(config, "gsm8k", run_id)
        ok = bool(printed) and all(
            abs(recovered[n] - printed[n]) < TOL for n in recovered if n in printed
        ) and set(recovered) <= set(printed)
        checks.append(FilterCheck(config, "gsm8k", run_id, recovered, printed, ok))
        for name, df in blocks.items():
            df = df.copy()
            df["filter"] = name
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    if validate:
        bad = [c for c in checks if not c.ok]
        if bad:
            raise AssertionError(
                "filter recovery not confirmed against lm-eval output:\n"
                + "\n".join(c.report() for c in bad)
            )
    return out, checks


def load_mmlu() -> pd.DataFrame:
    """Per-document MMLU outcomes, subtasks stacked, one row per document."""
    raw = pd.read_parquet(PER_PROMPT)
    raw = raw[raw.task.str.startswith("mmlu_")].copy()
    counts = raw.groupby("config").size()
    wrong = counts[counts != MMLU_N]
    if len(wrong):
        raise AssertionError(f"unexpected MMLU row counts: {wrong.to_dict()}")
    # A document is identified by (subtask, doc_id); doc_id restarts per subtask.
    raw["uid"] = raw.task + ":" + raw.doc_id.astype(str)
    raw["filter"] = "none"
    return raw


IFEVAL_N = 541
IFEVAL_METRIC = "prompt_level_strict_acc"


def load_ifeval() -> pd.DataFrame:
    """Per-prompt IFEval outcomes. One filter, one metric, one row per prompt."""
    raw = pd.read_parquet(PER_PROMPT)
    raw = raw[(raw.task == "ifeval") & (raw.metric_name == IFEVAL_METRIC)].copy()
    counts = raw.groupby("config").size()
    wrong = counts[counts != IFEVAL_N]
    if len(wrong):
        raise AssertionError(f"unexpected IFEval row counts: {wrong.to_dict()}")
    raw["uid"] = raw.doc_id.astype(str)
    raw["filter"] = "none"
    return raw


def format_miss(config: str | None = None) -> pd.DataFrame:
    """Per-prompt indicator: GSM8K answer correct under flexible-extract but
    NOT under strict-match.

    This isolates whether the model reasoned to the right answer and then
    failed to emit it in the form its 5-shot exemplars demonstrate. It is a
    measure of in-context format copying, and it is deliberately distinct from
    IFEval, which measures compliance with an instruction given in prose.
    """
    g, _ = load_gsm8k()
    keep = g.groupby("config").run_id.min()
    g = g[g.run_id == g.config.map(keep)]
    rows = []
    for cfg, sub in g.groupby("config"):
        s = sub[sub["filter"] == "strict-match"].set_index("doc_id").correct.sort_index()
        f = sub[sub["filter"] == "flexible-extract"].set_index("doc_id").correct.sort_index()
        i = s.index.intersection(f.index)
        rows.append(
            pd.DataFrame({
                "config": cfg,
                "doc_id": i,
                "uid": i.astype(str),
                "correct": ((f.loc[i] == 1) & (s.loc[i] == 0)).astype(float).to_numpy(),
            })
        )
    out = pd.concat(rows, ignore_index=True)
    return out[out.config == config] if config else out


def primary(task: str) -> pd.DataFrame:
    """The single frame the report uses for a task: one row per document."""
    if task == "gsm8k":
        df, _ = load_gsm8k()
        df = df[df["filter"] == PRIMARY_GSM8K_FILTER].copy()
        df["uid"] = df.doc_id.astype(str)
        # Keep only the first full run per config so the noise-floor pair does
        # not double-count; the repeat run is analysed separately.
        keep = df.groupby("config").run_id.min()
        df = df[df.run_id == df.config.map(keep)]
        return df
    if task == "mmlu":
        return load_mmlu()
    raise ValueError(task)


def load_system() -> pd.DataFrame:
    return pd.read_parquet(SYSTEM)


if __name__ == "__main__":
    _, checks = load_gsm8k(validate=False)
    for c in checks:
        print(c.report())
    n_bad = sum(not c.ok for c in checks)
    print(f"\n{len(checks) - n_bad}/{len(checks)} GSM8K runs confirmed against lm-eval")
    m = load_mmlu()
    print(f"MMLU: {len(m)} rows, {m.config.nunique()} configs, {m.task.nunique()} subtasks -- shape OK")
