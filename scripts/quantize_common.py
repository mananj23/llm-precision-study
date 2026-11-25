#!/usr/bin/env python3
"""Shared machinery for every quantization recipe.

Run these in the llm-compressor environment (.venv-compressor), NOT the vLLM one.

The recipes themselves follow the llm-compressor reference examples exactly;
only the model name and output path change. Do not write your own quantizer --
a subtly wrong one will produce a checkpoint that scores 40% instead of 0%,
which is far harder to notice than an obvious failure.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import common


def build_calibration_dataset(tokenizer, cal_cfg: dict):
    """Load the ONE frozen calibration set. Identical for every method."""
    from datasets import load_dataset

    ds = load_dataset(cal_cfg["dataset"], split=cal_cfg["split"])
    ds = ds.shuffle(seed=cal_cfg["shuffle_seed"]).select(range(cal_cfg["num_samples"]))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=cal_cfg["max_seq_length"],
            truncation=True,
            add_special_tokens=cal_cfg["add_special_tokens"],
        )

    return ds.map(tokenize, remove_columns=ds.column_names)


def build_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", help="config name, e.g. llama31-8b__w-fp8__kv-bf16")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, build nothing")
    return ap


def parse_args(description: str) -> argparse.Namespace:
    return build_parser(description).parse_args()


def run_recipe(cfg_name: str, recipe_builder, needs_calibration: bool, dry_run: bool):
    """recipe_builder(tokenizer) -> (recipe, extra_oneshot_kwargs)."""
    cfg = common.load_config(cfg_name)
    meta = common.parse_config_name(cfg_name)
    out_dir = pathlib.Path(cfg["quantize"]["output_dir"])

    ok, msg = common.check_supported(meta["weights"])
    print(f"[hardware] {msg}")
    if not ok:
        print("[hardware] REFUSING to build a checkpoint this card cannot run as "
              "intended. Fix the card or cut this configuration from the plan.")
        if not dry_run:
            raise SystemExit(2)

    print(f"[plan] {cfg['base_model']} -> {out_dir}")
    print(f"[plan] weights={meta['weights']} calibration={needs_calibration}")
    if dry_run:
        print("[dry-run] stopping before any model is loaded.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot

    t0 = time.time()
    model_id = cfg["base_model"]
    print(f"[load] {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    recipe, extra = recipe_builder(tokenizer)

    kwargs = dict(model=model, recipe=recipe)
    cal_cfg = None
    if needs_calibration:
        cal_cfg = common.load_yaml("configs/calibration.yaml")
        print(f"[calibration] {cal_cfg['dataset']} "
              f"n={cal_cfg['num_samples']} seed={cal_cfg['shuffle_seed']}")
        kwargs["dataset"] = build_calibration_dataset(tokenizer, cal_cfg)
        kwargs["max_seq_length"] = cal_cfg["max_seq_length"]
        kwargs["num_calibration_samples"] = cal_cfg["num_samples"]
    kwargs.update(extra)

    print("[oneshot] running...")
    oneshot(**kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    build_seconds = time.time() - t0

    size_bytes = sum(
        f.stat().st_size for f in out_dir.rglob("*") if f.is_file()
    )
    record = {
        "config": cfg_name,
        "base_model": model_id,
        "weights": meta["weights"],
        "output_dir": str(out_dir),
        "build_seconds": round(build_seconds, 1),
        "size_gb": round(size_bytes / 1e9, 2),
        "recipe_repr": repr(recipe),
        "calibration": cal_cfg,
    }
    # Absolute, via common.ROOT. These scripts are launched with cwd=scripts/
    # by run_all.sh, so a relative path here silently writes the build record
    # into scripts/results/checkpoints/ where nothing looks for it.
    common.write_provenance(
        common.ROOT / "results" / "checkpoints" / f"{cfg_name}.build.json",
        record,
    )
    print(json.dumps(record, indent=2, default=str))
    print(f"\n[done] {out_dir}  ({record['size_gb']} GB, {build_seconds / 60:.1f} min)")
    print("[next] VERIFY IT BEFORE USING IT:")
    print(f"       python scripts/verify_checkpoint.py {cfg_name}")
