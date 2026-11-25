#!/usr/bin/env python3
"""Turn the per-prompt logs into a routing dataset.

The router decides, per request and BEFORE the model answers, whether the cheap
configuration will do. Three things have to be right or the whole of Step 7 is
worthless:

1. THE LABEL. For a (cheap, expensive) pair, the cheap configuration is
   "sufficient" when it is correct, OR when the expensive one is also wrong --
   because in that case nothing was lost by being cheap. Defining the label as
   "cheap is correct" instead would punish the router for prompts no
   configuration can answer.

2. FEATURES KNOWN AT ROUTING TIME. Prompt length and prompt text are known.
   The output length is NOT -- it only exists after the model has answered, so
   using it is leakage that will make the router look far better than it is.
   This module physically refuses to build a feature from an output column.

3. COST. Every prompt's cost is its own token count times the per-token price
   of the configuration that answered it. If the router consults the cheap
   model for a confidence signal, that run is PAID FOR whether or not its
   answer is used, and must be added. There is a flag for it below.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import common  # noqa: E402

ROOT = common.ROOT

# Columns that exist only after the model has answered. Using any of these as a
# feature is leakage. The guard below is not decoration -- it is the single
# easiest way to accidentally publish a router that does not work.
POST_HOC_COLUMNS = {"correct", "output_len", "output_text", "metric_name"}


class LeakageError(RuntimeError):
    pass


def assert_no_leakage(feature_names) -> None:
    bad = sorted(set(feature_names) & POST_HOC_COLUMNS)
    if bad:
        raise LeakageError(
            f"features {bad} are only known AFTER the model answers. A router "
            "cannot see them at routing time. Remove them."
        )


def build_pair_dataset(
    per_prompt: pd.DataFrame,
    cheap_config: str,
    expensive_config: str,
    cost_per_m_tokens: dict[str, float] | None = None,
    confidence_signal_cost_share: float = 0.0,
) -> pd.DataFrame:
    """One row per prompt both configurations ran.

    confidence_signal_cost_share: if the router uses the cheap model's own
    output (e.g. first-token entropy) as a feature, it pays for a cheap run on
    EVERY prompt, including the ones it then sends to the expensive model. Pass
    the fraction of a full cheap run that costs (1.0 = a full generation).
    """
    need = {cheap_config, expensive_config}
    have = set(per_prompt["config"].unique())
    missing = need - have
    if missing:
        raise ValueError(f"missing configs in the logs: {sorted(missing)}")

    cheap = per_prompt[per_prompt["config"] == cheap_config]
    exp = per_prompt[per_prompt["config"] == expensive_config]

    key = ["task", "doc_id"]
    merged = cheap.merge(
        exp, on=key, suffixes=("_cheap", "_exp"), how="inner", validate="1:1"
    )
    dropped = max(len(cheap), len(exp)) - len(merged)
    if dropped:
        print(f"  {dropped} prompts dropped: not present in both configurations")

    df = pd.DataFrame({
        "task": merged["task"],
        "doc_id": merged["doc_id"],
        # ---- known at routing time ----
        "prompt_text": merged["prompt_text_cheap"],
        "input_len": merged["input_len_cheap"],
        # ---- outcomes (labels and cost accounting only, NEVER features) ----
        "cheap_correct": merged["correct_cheap"],
        "exp_correct": merged["correct_exp"],
        "cheap_output_len": merged["output_len_cheap"],
        "exp_output_len": merged["output_len_exp"],
    })

    # ---- the label -----------------------------------------------------------
    df["sufficient"] = (
        (df["cheap_correct"] > 0.5) | (df["exp_correct"] < 0.5)
    ).astype(int)

    # ---- per-prompt cost -----------------------------------------------------
    cpm = cost_per_m_tokens or {}
    c_cheap = cpm.get(cheap_config, np.nan)
    c_exp = cpm.get(expensive_config, np.nan)
    df["cost_cheap"] = df["cheap_output_len"] / 1e6 * c_cheap
    df["cost_exp"] = df["exp_output_len"] / 1e6 * c_exp
    if confidence_signal_cost_share > 0:
        # Paid on every prompt, including those routed to the expensive model.
        overhead = df["cheap_output_len"] / 1e6 * c_cheap * confidence_signal_cost_share
        df["cost_cheap"] = df["cost_cheap"] + overhead
        df["cost_exp"] = df["cost_exp"] + overhead
        print(f"  confidence-signal overhead charged on every prompt "
              f"({confidence_signal_cost_share:.0%} of a cheap run)")

    df.attrs["cheap_config"] = cheap_config
    df.attrs["expensive_config"] = expensive_config

    print(f"  {len(df):,} paired prompts across {df['task'].nunique()} tasks")
    print(f"  cheap accuracy      {df['cheap_correct'].mean():.4f}")
    print(f"  expensive accuracy  {df['exp_correct'].mean():.4f}")
    print(f"  label 'sufficient'  {df['sufficient'].mean():.4f} of prompts")
    gain = ((df["cheap_correct"] > 0.5) & (df["exp_correct"] < 0.5)).mean()
    print(f"  cheap right where expensive wrong: {gain:.4f} "
          "<- this is the entire headroom above the expensive baseline")
    return df


def make_features(
    df: pd.DataFrame,
    use_task_flag: bool = True,
    n_text_components: int = 24,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Routing-time features only.

    use_task_flag: including the task identity makes routing much easier, but
    real traffic does not arrive labelled with the benchmark it came from. Run
    BOTH ways and report both; the no-flag number is the honest one.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    blocks, names = [], []

    length = df["input_len"].to_numpy(float).reshape(-1, 1)
    blocks.append(np.hstack([length, np.log1p(length)]))
    names += ["input_len", "log_input_len"]

    if use_task_flag:
        tasks = sorted(df["task"].unique())
        oh = np.stack([(df["task"] == t).to_numpy(float) for t in tasks], axis=1)
        blocks.append(oh)
        names += [f"task={t}" for t in tasks]

    if n_text_components > 0:
        text = df["prompt_text"].fillna("").astype(str)
        tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_features=20000
        )
        X = tfidf.fit_transform(text)
        k = min(n_text_components, max(1, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=k, random_state=seed)
        blocks.append(svd.fit_transform(X))
        names += [f"text_svd_{i}" for i in range(k)]

    assert_no_leakage(names)
    return np.hstack(blocks), names


def load_costs(results_path: pathlib.Path | None = None) -> dict[str, float]:
    """cost per million output tokens, per config, from the results table."""
    results_path = results_path or (ROOT / "analysis" / "results.parquet")
    if not results_path.exists():
        print(f"  no {results_path}; costs unavailable, using token counts only")
        return {}
    r = pd.read_parquet(results_path)
    col = "cost_per_m_output_tokens"
    if col not in r or r[col].isna().all():
        print("  no cost column populated; fill gpu_price_per_hour in "
              "configs/bench_grid.yaml and rebuild")
        return {}
    return r.groupby("config")[col].first().dropna().to_dict()
