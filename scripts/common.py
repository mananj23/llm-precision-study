#!/usr/bin/env python3
"""Shared helpers: config loading, naming, provenance stamping.

Every script that produces an artefact writes a provenance record next to it.
If you cannot say which library versions and which GPU produced a number, the
number is not evidence.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = pathlib.Path(
    os.environ.get("CHECKPOINT_DIR", ROOT / "results" / "checkpoints")
)

# How many GPUs vLLM splits the model across. 1 for a single card; 2 for the
# 2x L4 node, where 8B weights do not fit on one 24 GB card with any useful KV
# cache. Set TP in the environment so it can change without regenerating every
# config file:  export TP=2
TENSOR_PARALLEL_SIZE = int(os.environ.get("TP", "1"))


def load_yaml(path: str | pathlib.Path) -> dict:
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open() as fh:
        return yaml.safe_load(fh)


def load_config(name: str) -> dict:
    """Load a model config by name, path, or filename, expanding paths."""
    cand = pathlib.Path(name)
    if cand.suffix == ".yaml" and cand.exists():
        cfg = load_yaml(cand)
    else:
        cfg = load_yaml(ROOT / "configs" / "models" / f"{name}.yaml")
    return expand(cfg)


def expand(obj):
    """Expand ${CHECKPOINT_DIR} and ${TP} recursively."""
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    if isinstance(obj, str):
        out = obj.replace("${CHECKPOINT_DIR}", str(CHECKPOINT_DIR))
        out = out.replace("${TP}", str(TENSOR_PARALLEL_SIZE))
        # A field that is now a bare integer should come back as one, not "2".
        if out.isdigit():
            return int(out)
        return out
    return obj


def parse_config_name(name: str) -> dict:
    """llama31-8b__w-fp8__kv-bf16 -> {family, weights, kv_cache}."""
    stem = pathlib.Path(name).stem
    parts = stem.split("__")
    if len(parts) != 3:
        raise ValueError(f"config name must be family__w-X__kv-Y, got {stem!r}")
    family, w, kv = parts
    return {
        "config": stem,
        "family": family,
        "weights": w.removeprefix("w-"),
        "kv_cache": kv.removeprefix("kv-"),
    }


def run_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=60, check=False
        ).stdout.strip()
    except Exception:
        return None


def provenance(extra: dict | None = None) -> dict:
    """Everything needed to reproduce or distrust a result."""
    rec = {
        "recorded_at": _dt.datetime.now().astimezone().isoformat(),
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_cmd(["git", "status", "--porcelain"])),
        "hostname": _cmd(["hostname"]),
        "python": sys.version.split()[0],
        "gpu": _cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    for pkg in ("vllm", "lm_eval", "llmcompressor", "torch", "transformers"):
        try:
            mod = __import__(pkg)
            rec[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            rec[f"{pkg}_version"] = None
    if extra:
        rec.update(extra)
    return rec


def write_provenance(path: str | pathlib.Path, extra: dict | None = None) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(provenance(extra), indent=2))
    print(f"[provenance] {p}")


def lm_eval_cmd() -> str:
    """The harness entrypoint that actually exists on this machine.

    Spelled three ways across releases: `lm-eval` (hyphen, current, which
    auto-inserts its `run` subcommand for the legacy flag-only form),
    `lm_eval` (older console script), and the module form. Detect rather
    than guess -- a wrong guess fails at the worst moment.
    """
    import shutil

    for exe in ("lm-eval", "lm_eval"):
        if shutil.which(exe):
            return exe
    return f"{sys.executable} -m lm_eval"


def compute_capability() -> float | None:
    out = _cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if not out:
        return None
    try:
        return float(out.splitlines()[0].strip())
    except ValueError:
        return None


def check_supported(weights: str) -> tuple[bool, str]:
    """Hardware gate, from the vLLM supported-hardware table (checked 2026-11-16).

    Returns (ok, message). Call this BEFORE spending an hour building a
    checkpoint the card cannot run.
    """
    cc = compute_capability()
    if cc is None:
        return True, "no nvidia-smi; cannot check compute capability (assuming ok)"
    if weights == "fp8":
        if cc >= 8.9:
            return True, f"cc {cc} supports FP8 W8A8"
        if cc >= 7.5:
            return (
                False,
                f"cc {cc}: FP8 will LOAD but run as weight-only W8A16 via FP8-Marlin, "
                "not true W8A8. That is a different experiment -- either move to an "
                "Ada/Hopper card or report it as W8A16 and say so.",
            )
        return False, f"cc {cc}: FP8 not supported"
    if weights == "int8":
        if cc >= 10.0:
            return (
                False,
                f"cc {cc} is Blackwell: llm-compressor INT8 W8A8 is NOT supported. "
                "Drop INT8 from the matrix and say why, or use an Ada/Hopper card.",
            )
        if cc > 7.5:
            return True, f"cc {cc} supports INT8 W8A8"
        return False, f"cc {cc}: INT8 W8A8 needs cc > 7.5"
    if weights in ("int4gptq", "int4awq"):
        if cc >= 8.0:
            return True, f"cc {cc} supports 4-bit weight-only with Marlin kernels"
        return True, f"cc {cc}: 4-bit should load, Marlin kernels need cc >= 8.0"
    return True, "bf16: no constraint"
