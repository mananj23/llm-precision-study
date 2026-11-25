# Runbook — the 29-hour H100 window

One shot. The clock runs from container creation to deletion, at roughly
₹379/hour. This is the page to keep open; everything else in the repo is
reference.

**The single rule: if the GPU is not computing, the container should be
stopped.** Reading an error, waiting on a reply, eating, sleeping — stop it.
That one habit is the difference between finishing in 22 hours and running out
at 29.

---

## Before you create the container (free, do it now)

- [ ] SSH public key ready (`~/.ssh/project.pub` — the public one, never the other)
- [ ] **New** HF token created and the old leaked one revoked
- [ ] `meta-llama/Llama-3.1-8B-Instruct` access granted (check: the model page
      shows the files instead of a request form)
- [ ] **GitHub repo created, and `git remote -v` prints a URL.** This is the
      one that will quietly cost you everything: `run_all.sh` pushes results
      after every phase, and when the node is deleted the only surviving copy
      is the one on GitHub. Verify with an actual push before you create the
      container:
      ```bash
      git remote add origin https://github.com/<you>/llm-precision-study.git
      git push -u origin main    # must succeed, not just print something
      ```
- [ ] A GitHub personal access token or SSH key set up on the GPU node too, so
      the pushes from `run_all.sh` do not stop to ask for a password
- [ ] You have a clear 3–4 hour block ahead of you

Create the container only when all of these are ticked.

### The time budget, measured

| Stage | Pessimistic | Likely |
|---|---|---|
| Setup, install, model download | 1.5 h | 1 h |
| Session 1 preflight | 0.5 h | 0.3 h |
| Build 4 quantized checkpoints | 3 h | 2 h |
| Quality: 4 tasks × 7 configs + noise-floor rerun | 11 h | 7 h |
| Benchmark grid, 7 configs | 10 h | 4 h |
| Collection, figures, writing | 1 h | 1 h |
| **Total** | **27 h** | **15.3 h** |

The 29 hours fit, with about 2 hours of slack in the pessimistic column. The
slack is thin, which is why the ordering below matters: quality evals run
task-major (GSM8K across all seven configs first), so the headline result
exists after the first pass and everything after it only adds.

**Container settings:** H100 SXM, **1× GPU**, image **PyTorch v2.8**
(Python 3.12, CUDA 12.9.1, Ubuntu 24.04). Not the vLLM image — it ships v0.15.1,
which is far behind and may lack `vllm bench serve`. Attach persistent storage
of ~150 GB if offered.

---

## Session 1 — prove it works (~1 hour, ~₹380)

This session buys certainty, not results. Do not skip it.

```bash
# 1. shell in, then:
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
```

**Stop immediately if** you don't see one H100 with compute capability **9.0**.
Wrong card means wrong experiment.

```bash
# 2. repo + environment
git clone https://github.com/<you>/llm-precision-study.git
cd llm-precision-study

export TP=1
export HF_HOME=/workspace/hf_cache          # or wherever your persistent disk is
export CHECKPOINT_DIR=/workspace/checkpoints
export RATE_PER_HOUR=379
cat >> ~/.bashrc <<'EOF'
export TP=1
export HF_HOME=/workspace/hf_cache
export CHECKPOINT_DIR=/workspace/checkpoints
export RATE_PER_HOUR=379
EOF

# 3. install (10-15 min)
python -m venv .venv-vllm && . .venv-vllm/bin/activate
pip install --upgrade pip
pip install -r requirements-vllm.txt
pip freeze > requirements-vllm.lock.txt     # pin NOW, while it works
hf auth login

# 4. the preflight — this is the point of session 1
./scripts/session1.sh 2>&1 | tee results/session1.log
```

`session1.sh` runs nine checks ending in a real 20-prompt evaluation, and
prints **GO** or **NO-GO**.

- **GO** → continue to session 2.
- **NO-GO** → **stop the container**, send me `results/session1.log`, fix it,
  restart. Do not debug with the meter running.

```bash
git add -A && git commit -m "session 1: environment verified" && git push
```

---

## Session 2 — the run (start it, then leave)

```bash
tmux new -s all
. .venv-vllm/bin/activate
./scripts/run_all.sh 2>&1 | tee results/run_all.log
```

**Watch until the first evaluation is visibly producing output** — perhaps
15 minutes. Then `Ctrl-B`, then `D` to detach, and leave it alone.

`run_all.sh` is resumable, tolerant of failures, and ordered by importance:
baseline → FP8 → everything else. It pushes to git after each phase.

### Checking in

```bash
tmux attach -t all          # look, then Ctrl-B D again
grep FAILED results/run_all_status.tsv
column -t -s$'\t' results/run_all_status.tsv | tail -20
```

Nothing in `FAILED` → leave it running. Something there → send it to me.

---

## The two decisions

**At ~15 hours.** Check `run_all_status.tsv`.

| What you see | Do |
|---|---|
| Checkpoints built, GSM8K pass done, MMLU running | on track, continue |
| GSM8K pass not finished | `Ctrl-C`, then `LIMIT=500 ./scripts/run_all.sh`. It resumes; finished evals are skipped. A 500-prompt GSM8K has a worse MDE, and you say so |
| Still fighting setup or checkpoint builds | drop the two INT4 configs: comment them out of `CONFIGS` in `run_all.sh` and restart it. Four precisions well measured beats seven half-measured |

Because evals are task-major, you never have to delete a task to save time —
just stop before the pass you don't have time for. GSM8K then MMLU is already
a complete, publishable comparison; MMLU is the one with enough prompts
(n≈14k) to resolve a 1-point effect at all.

**At ~22 hours.** If benchmarks haven't started, shrink the grid with
environment variables — do not edit the script under time pressure:

```bash
# two seeds instead of three, and a tighter token budget per run
BENCH_SEEDS="0 1" PROMPT_TOKEN_BUDGET=300000 ./scripts/run_all.sh
```

`run_bench.sh` prints its plan and a time estimate before spending any GPU
time, so you can see the effect before committing. Report the reduced grid in
the method section.

---

## Finishing — in this order

```bash
# 1. collect everything
python scripts/collect_logs.py --keep all
python scripts/collect_bench.py

# 2. push
git add -A && git commit -m "final results" && git push

# 3. VERIFY THE PUSH WORKED — open GitHub in a browser and look at results/
#    Do not skip this. A failed push you did not notice means the GPU hours
#    are gone for nothing.

# 4. only now
#    stop / delete the container
```

Everything after this is CPU work on your laptop: `make all`, `make power`,
`make router`, and the report. No clock running.

---

## If something breaks

**Stop the container first**, then diagnose. Useful context to send me:

```bash
tail -50 results/run_all.log
grep FAILED results/run_all_status.tsv
tail -40 results/eval_<config>_<task>.log
nvidia-smi
```

Known failure modes and what they mean:

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'common'` | PYTHONPATH | already patched; re-pull the repo |
| Eval scores ~0 | broken checkpoint, not quantization damage | rebuild; check chat template and tokenizer files |
| `vllm bench serve` not found | vLLM too old | `pip install -U vllm`; tell me |
| CUDA OOM at long context | KV cache | lower `max_model_len` in `configs/models/*.yaml` |
| Task name not found | harness version | `lm-eval --tasks list \| grep <name>`, fix `configs/eval.yaml` |
| AWQ recipe fails | API moved | skip it; `run_all.sh` continues without it |

---

## What "done" means

You have enough when `results/logs/per_prompt.parquet` contains the BF16
baseline and FP8 on at least two tasks. **That is the 6 December note**, and
`run_all.sh` reaches it in roughly the first five hours.

Everything after that improves the final report but is not required for the
deadline. If hours 6–29 went badly, you still have the thing that mattered.

---

## Cost tracker

Write the time down each session. It is five seconds of work and it is a
number the report needs.

| Session | Started | Stopped | Hours | Running total | ₹ |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |

Also record, once, for `configs/bench_grid.yaml`:

- GPU: `1x NVIDIA H100 SXM 80GB`
- Price: `₹379.05/hr`
- Date: the day you ran it
- Provider: E2E Networks, Delhi
