#!/usr/bin/env python3
"""Fold the raw vllm-bench JSON files into one tidy system table.

    python scripts/collect_bench.py

Output: results/logs/system.parquet, one row per
(config, shape, concurrency, seed), with the median across seeds computed
later in analysis/build_results.py -- not here, so that the spread stays
visible and you can report it.

vLLM's result JSON field names have changed between releases. This reads
defensively: it maps several known spellings onto canonical names and tells
you which fields it could not find rather than silently writing NaN.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import pandas as pd

import common

# canonical -> candidate keys in vllm bench serve output, best first
FIELD_MAP = {
    "ttft_ms_median": ["median_ttft_ms", "p50_ttft_ms"],
    "ttft_ms_mean": ["mean_ttft_ms"],
    "ttft_ms_p99": ["p99_ttft_ms"],
    "tpot_ms_median": ["median_tpot_ms", "p50_tpot_ms"],
    "tpot_ms_mean": ["mean_tpot_ms"],
    "itl_ms_median": ["median_itl_ms", "p50_itl_ms"],
    "e2el_ms_median": ["median_e2el_ms", "p50_e2el_ms"],
    "output_throughput_tok_s": ["output_throughput"],
    "total_throughput_tok_s": ["total_token_throughput"],
    "request_throughput_rps": ["request_throughput"],
    "duration_s": ["duration"],
    "completed": ["completed"],
    "total_input_tokens": ["total_input_tokens"],
    "total_output_tokens": ["total_output_tokens"],
}

NAME_RE = re.compile(r"^(?P<shape>\w+?)_c(?P<conc>\d+)_seed(?P<seed>\d+)\.json$")


def read_one(path: pathlib.Path, config: str) -> dict | None:
    m = NAME_RE.match(path.name)
    if not m:
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"  unreadable: {path}")
        return None

    meta = common.parse_config_name(config)
    row = {
        "config": config,
        "family": meta["family"],
        "weights": meta["weights"],
        "kv_cache": meta["kv_cache"],
        "shape": m.group("shape"),
        "concurrency": int(m.group("conc")),
        "seed": int(m.group("seed")),
        "source_file": str(path.relative_to(common.ROOT)),
    }
    missing = []
    for canon, candidates in FIELD_MAP.items():
        for c in candidates:
            if c in raw:
                row[canon] = raw[c]
                break
        else:
            row[canon] = float("nan")
            missing.append(canon)
    row["_missing_fields"] = ";".join(missing)
    return row


def read_server_log_kv(config_dir: pathlib.Path) -> int | None:
    """KV cache capacity from the newest serve log in this config's folder."""
    logs = sorted(config_dir.glob("serve_*.log"))
    if not logs:
        return None
    sys.path.insert(0, str(common.ROOT / "scripts"))
    from verify_checkpoint import kv_cache_tokens_from_log

    return kv_cache_tokens_from_log(logs[-1].read_text(errors="ignore"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system-root", default="results/system")
    ap.add_argument("--out", default="results/logs/system.parquet")
    args = ap.parse_args()

    root = pathlib.Path(args.system_root)
    if not root.is_absolute():
        root = common.ROOT / root

    if not root.exists():
        print(f"No benchmark directory yet: {root}")
        return 1

    rows, failures = [], []
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        kv_tokens = read_server_log_kv(cfg_dir)
        n = 0
        for f in sorted(cfg_dir.glob("*.json")):
            # Matches both ".FAILED.json" (a run that died) and
            # ".WARMUP_FAILED.json" (a cell skipped because the warm-up died).
            # Getting this suffix wrong silently discards the KV-cache cap,
            # which is one of the actual findings.
            if f.name.endswith("FAILED.json"):
                failures.append(json.loads(f.read_text()))
                continue
            if f.name == "provenance.json":
                continue
            row = read_one(f, cfg_dir.name)
            if row:
                row["kv_cache_tokens"] = kv_tokens
                rows.append(row)
                n += 1
        print(f"  {cfg_dir.name}: {n} runs, kv_cache_tokens={kv_tokens}")

    # Failures are a result: a config that cannot serve 16k at concurrency 64
    # has a smaller KV cache, and that belongs in the paper. Persist them even
    # when there are no successful rows at all.
    fail_out = common.ROOT / "results" / "logs" / "system_failures.json"
    fail_out.parent.mkdir(parents=True, exist_ok=True)
    fail_out.write_text(json.dumps(failures, indent=2))

    if not rows:
        print("\nNo benchmark results found yet.")
        return 1

    df = pd.DataFrame(rows)
    missing = df.loc[df["_missing_fields"] != "", "_missing_fields"].unique()
    if len(missing):
        print("\nWARNING: these canonical fields were not found in the vLLM JSON:")
        for m in missing:
            print(f"  {m}")
        print("vLLM renamed them between versions. Open one raw JSON, find the")
        print("new key, and add it to FIELD_MAP at the top of this script.")

    if failures:
        print(f"\n{len(failures)} runs failed (recorded, not silently dropped):")
        for f in failures:
            stage = f.get("stage", "run")
            print(f"  {f.get('config','?')} {f.get('shape','?')} "
                  f"conc={f.get('concurrency','?')} ({stage})")
        print("If these are the 16k shapes, that is the KV-cache cap and it is a")
        print("RESULT. Put it in the report, do not quietly omit the row.")

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = common.ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
