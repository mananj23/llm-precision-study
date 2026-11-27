#!/usr/bin/env bash
# Run the whole data-collection matrix unattended.
#
#   tmux new -s all
#   . .venv-vllm/bin/activate
#   export TP=2                 # 2x L4 node; 1 on a single card
#   ./scripts/run_all.sh 2>&1 | tee results/run_all.log
#   # Ctrl-B then D to detach, then go to sleep.
#
# Designed for a rented GPU with a hard deadline. Three properties matter:
#
#   RESUMABLE   Every step checks whether its output already exists and skips
#               it. If the node dies at hour 30, rerun this and it picks up
#               where it stopped. Never start from scratch.
#   TOLERANT    A failing step is recorded and the run continues. One broken
#               recipe must not cost you the other six configurations.
#   ORDERED     Work is ordered by how much it matters, not by convenience.
#               If the window is cut short, what you have is the important
#               part: the BF16 baseline and FP8, which is the 6 Dec note.
#
# It never deletes anything and never terminates the node. Stopping the node
# is a decision you make, after checking the results are pushed.
set -uo pipefail

cd "$(dirname "$0")/.."
# scripts/ holds common.py; the heredocs below import it after this cd,
# so the repo root alone is not enough. Without this every config lookup
# dies with ModuleNotFoundError: No module named 'common'.
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
mkdir -p results results/logs results/quality results/system results/checkpoints
STATUS="results/run_all_status.tsv"
touch "$STATUS"

FAMILY="${FAMILY:-llama31-8b}"
LIMIT="${LIMIT:-}"            # e.g. LIMIT=500 to cut every eval to 500 prompts
TP="${TP:-1}"
export TP

# Priority order. Everything about this list is deliberate.
BASELINE="${FAMILY}__w-bf16__kv-bf16"
CONFIGS=(
  "${FAMILY}__w-bf16__kv-bf16"     # the baseline: nothing means anything without it
  "${FAMILY}__w-fp8__kv-bf16"      # the 6 Dec note
  "${FAMILY}__w-bf16__kv-fp8"      # isolates the KV cache
  "${FAMILY}__w-fp8__kv-fp8"
  "${FAMILY}__w-int8__kv-bf16"
  "${FAMILY}__w-int4gptq__kv-bf16"
  "${FAMILY}__w-int4awq__kv-bf16"
)
# humaneval excluded: needs a container.
# Override without editing this file:  TASKS="gsm8k mmlu" ./scripts/run_all.sh
read -r -a TASKS <<< "${TASKS:-gsm8k mmlu ifeval longbench}"

# Some tasks only need a subset of the matrix. LongBench is ~21 subtasks at long
# context, so a full seven-config sweep costs most of a day -- and the question
# LongBench answers is specifically about the KV CACHE, which is isolated by the
# 2x2 of {bf16, fp8} weights x {bf16, fp8} KV. INT8 and INT4 add nothing to that
# contrast, so they are left out by default rather than paid for.
# Override with LONGBENCH_ALL=1 to sweep every config.
configs_for_task() {  # task -> the configs to evaluate it on
  case "$1" in
    longbench)
      if [ "${LONGBENCH_ALL:-0}" = "1" ]; then
        printf '%s\n' "${CONFIGS[@]}"
      else
        printf '%s\n' \
          "${FAMILY}__w-bf16__kv-bf16" \
          "${FAMILY}__w-bf16__kv-fp8" \
          "${FAMILY}__w-fp8__kv-bf16" \
          "${FAMILY}__w-fp8__kv-fp8"
      fi
      ;;
    *) printf '%s\n' "${CONFIGS[@]}" ;;
  esac
}

# ------------------------------------------------------------------ helpers --
ts() { date +%H:%M:%S; }

note() {  # note <step> <result>
  printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS"
  echo "[$(ts)] $1 -> $2"
}

already_evaluated() {  # config task
  # find, not a glob: ** needs globstar, which is off by default in bash, so
  # the pattern would quietly match the wrong depth and re-run finished work.
  [ -n "$(find "results/quality/$1/$2" -name 'samples_*.jsonl' -print -quit \
          2>/dev/null)" ]
}

needs_checkpoint() {  # config -- true unless the WEIGHTS are bf16
  case "$1" in
    *__w-bf16__*) return 1 ;;   # kv-fp8 is a runtime flag, not a checkpoint
    *) return 0 ;;
  esac
}

checkpoint_exists() {  # config
  local d
  d=$(python3 -c "
import sys; sys.path.insert(0,'scripts')
import common
c = common.load_config('$1')
print(c['model_path'])
" 2>/dev/null) || return 1
  [ -f "$d/config.json" ]
}

push() {
  # Report honestly. A silent push failure is the worst outcome available:
  # you believe the results are safe, delete the node, and they are not.
  git add -A >/dev/null 2>&1
  if ! git commit -q -m "run_all: $1" >/dev/null 2>&1; then
    echo "[$(ts)] nothing new to commit ($1)"
    return 0
  fi
  if git push -q 2>/tmp/push_err; then
    echo "[$(ts)] pushed: $1"
  else
    echo "[$(ts)] !!! PUSH FAILED for '$1' -- results exist only on this node"
    sed 's/^/        /' /tmp/push_err | head -5
    note "push:$1" "FAILED -- fix the remote before deleting the node"
  fi
}

banner() { echo; echo "================ $* ================"; echo; }

# --------------------------------------------------------------- phase 0 ----
banner "PHASE 0  environment"
bash scripts/env_check.sh 2>&1 | tail -25
note "env_check" "done"

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
echo "GPUs visible: ${NGPU:-0}   TP=$TP"
if [ "${NGPU:-0}" -gt 0 ] && [ "$TP" -gt "${NGPU:-0}" ]; then
  note "env_check" "FATAL: TP=$TP > $NGPU GPUs"
  echo "Set TP=$NGPU and rerun."
  exit 2
fi

# --------------------------------------------------------------- phase 1 ----
# Build every checkpoint first. They are the long pole and they are what you
# lose if the node dies, so get them onto disk (and to Hugging Face, if you
# set HF_PUSH_REPO) before spending hours on evaluations.
# vLLM and llm-compressor pin INCOMPATIBLE versions of compressed-tensors
# (vLLM 0.34 wants ==0.17.0; llmcompressor 0.15 pulls 0.20.1), and vLLM's own
# docs say to keep them in separate environments. So quantization runs in
# .venv-compressor and everything else in the active (vLLM) environment.
# Without this, installing both in one venv silently breaks whichever loses.
COMPRESSOR_PY="$PWD/.venv-compressor/bin/python"
if [ ! -x "$COMPRESSOR_PY" ]; then
  COMPRESSOR_PY="$(command -v python3)"
  echo "[$(ts)] WARNING: no .venv-compressor; building checkpoints with $COMPRESSOR_PY"
  echo "           If llmcompressor is not installed there, every build fails."
  echo "           Create it:  python3 -m venv .venv-compressor &&"
  echo "                       .venv-compressor/bin/pip install -r requirements-compressor.txt"
fi

banner "PHASE 1  build checkpoints"
for cfg in "${CONFIGS[@]}"; do
  needs_checkpoint "$cfg" || continue          # bf16 weights: nothing to build
  if checkpoint_exists "$cfg"; then
    note "build:$cfg" "skip (exists)"
    continue
  fi
  script=$(python3 -c "
import sys; sys.path.insert(0,'scripts')
import common
c = common.load_config('$cfg')
q = c.get('quantize')
print(q['script'] if q else '')
" 2>/dev/null)
  [ -z "$script" ] && continue

  echo "[$(ts)] building $cfg via $script  [python: $COMPRESSOR_PY]"
  if (cd scripts && "$COMPRESSOR_PY" "$(basename "$script")" "$cfg"); then
    note "build:$cfg" "ok"
    push "built $cfg"
  else
    note "build:$cfg" "FAILED"
    echo "  continuing; this configuration will be skipped in later phases"
  fi
done

banner "PHASE 2  verify checkpoints"
for cfg in "${CONFIGS[@]}"; do
  needs_checkpoint "$cfg" || continue
  checkpoint_exists "$cfg" || { note "verify:$cfg" "skip (no checkpoint)"; continue; }
  if python3 scripts/verify_checkpoint.py "$cfg" --n 100 >"results/verify_${cfg}.log" 2>&1; then
    note "verify:$cfg" "ok"
  else
    note "verify:$cfg" "FAILED -- read results/verify_${cfg}.log before trusting it"
  fi
done
push "checkpoint verification"

# --------------------------------------------------------------- phase 3 ----
# Evaluations, ordered so that the first configuration finished is the baseline
# and the second is FP8. GSM8K runs first within each configuration for the
# same reason.
banner "PHASE 3  quality evaluations"
EXTRA=()
[ -n "$LIMIT" ] && EXTRA=(--limit "$LIMIT")

# TASK-major, not config-major. This ordering is the difference between
# finishing with a publishable result and finishing with nothing.
#
# Config-major (all four tasks for config 1, then config 2, ...) means that if
# the rental runs out at hour 25 you have four tasks on four configs and NO
# complete comparison: the paired bootstrap needs the same task across all
# seven precisions. Task-major finishes GSM8K across every config first, so
# the headline figure exists after the first pass and every later pass only
# adds to it.
for task in "${TASKS[@]}"; do
  echo
  mapfile -t TASK_CONFIGS < <(configs_for_task "$task")
  echo "[$(ts)] ===== task pass: $task  (${#TASK_CONFIGS[@]} configs) ====="
  if [ "${#TASK_CONFIGS[@]}" -ne "${#CONFIGS[@]}" ]; then
    echo "[$(ts)]   subset: ${TASK_CONFIGS[*]}"
  fi
  for cfg in "${TASK_CONFIGS[@]}"; do
    if needs_checkpoint "$cfg" && ! checkpoint_exists "$cfg"; then
      note "eval:$cfg:$task" "skip (no checkpoint)"
      continue
    fi
    if already_evaluated "$cfg" "$task"; then
      note "eval:$cfg:$task" "skip (done)"
      continue
    fi
    echo "[$(ts)] eval $cfg / $task"
    if ./scripts/run_quality.sh "$cfg" "$task" "${EXTRA[@]+"${EXTRA[@]}"}" \
         >"results/eval_${cfg}_${task}.log" 2>&1; then
      note "eval:$cfg:$task" "ok"
    else
      note "eval:$cfg:$task" "FAILED -- see results/eval_${cfg}_${task}.log"
    fi
  done

  # The noise floor: two identical runs of the baseline differ by some amount,
  # and no later claim may be smaller than it. Do this right after the GSM8K
  # pass, while GSM8K is the thing being written up.
  # Each run_quality.sh call writes into its own timestamped run_id directory,
  # so "already rerun" means two or more run directories hold sample files.
  n_runs=$(find "results/quality/$BASELINE/gsm8k" -name 'samples_*.jsonl' \
             -printf '%h\n' 2>/dev/null | sort -u | wc -l)
  if [ "$task" = "gsm8k" ] && [ "${n_runs:-0}" -ge 1 ] && [ "${n_runs:-0}" -lt 2 ]; then
    echo "[$(ts)] baseline GSM8K rerun (noise floor)"
    ./scripts/run_quality.sh "$BASELINE" gsm8k "${EXTRA[@]+"${EXTRA[@]}"}" \
      >"results/eval_${BASELINE}_gsm8k_rerun.log" 2>&1 \
      && note "eval:$BASELINE:gsm8k:rerun" "ok" \
      || note "eval:$BASELINE:gsm8k:rerun" "FAILED"
  fi

  python3 scripts/collect_logs.py --keep all >/dev/null 2>&1
  push "quality: $task across all configs"
done

# --------------------------------------------------------------- phase 4 ----
# Benchmarks last. They need a server to themselves, so each configuration
# starts one, waits for it, measures, and tears it down.
# SKIP_BENCH=1 runs quality only. Useful for a replication grid, where the
# scientific question is whether the ACCURACY structure reproduces on another
# model family; the throughput grid costs ~6 h and answers a hardware question
# that the first family already answered.
banner "PHASE 4  system benchmarks"
if [ "${SKIP_BENCH:-0}" = "1" ]; then
  echo "[$(ts)] SKIP_BENCH=1 -- skipping all system benchmarks"
  note "bench" "skip (SKIP_BENCH=1)"
fi
for cfg in "${CONFIGS[@]}"; do
  if [ "${SKIP_BENCH:-0}" = "1" ]; then break; fi
  if needs_checkpoint "$cfg" && ! checkpoint_exists "$cfg"; then
    note "bench:$cfg" "skip (no checkpoint)"
    continue
  fi
  # A provenance.json alone is not a completed benchmark; look for real results.
  if [ -n "$(find "results/system/$cfg" -name '*_c*_seed*.json' -print -quit 2>/dev/null)" ]; then
    note "bench:$cfg" "skip (done)"
    continue
  fi

  # Make sure nothing is squatting on the GPU or the port from a previous
  # iteration. A leftover vLLM holds all the VRAM and every later config fails.
  pkill -f "vllm serve" 2>/dev/null && { echo "[$(ts)] killed a stale vLLM"; sleep 15; }

  echo "[$(ts)] starting benchmark server for $cfg"
  ./scripts/serve.sh "$cfg" --bench >"results/serve_${cfg}.log" 2>&1 &
  SERVER_PID=$!

  # Wait up to 20 minutes for readiness (first run downloads the model).
  ready=0
  for _ in $(seq 1 240); do
    sleep 5
    curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
  done

  if [ "$ready" = 1 ]; then
    ./scripts/run_bench.sh "$cfg" >"results/bench_${cfg}.log" 2>&1 \
      && note "bench:$cfg" "ok" || note "bench:$cfg" "FAILED"
  else
    note "bench:$cfg" "FAILED (server never became ready)"
  fi

  # serve.sh ends in `exec vllm serve ... | tee`, so $SERVER_PID is the shell
  # running the pipeline, not vllm itself. Killing it can leave vLLM alive and
  # holding every byte of VRAM, which makes every later config fail. Kill the
  # process group and then the process by name, and verify VRAM actually freed.
  kill "$SERVER_PID" 2>/dev/null
  pkill -f "vllm serve" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  for _ in 1 2 3 4 5 6; do
    sleep 5
    pgrep -f "vllm serve" >/dev/null || break
  done
  if pgrep -f "vllm serve" >/dev/null; then
    echo "[$(ts)] vLLM will not die; SIGKILL"
    pkill -9 -f "vllm serve" 2>/dev/null
    sleep 10
  fi
  sleep 10                      # let VRAM actually free before the next server
  push "benchmarks for $cfg"
done

# --------------------------------------------------------------- phase 5 ----
banner "PHASE 5  collect"
python3 scripts/collect_logs.py --keep all
python3 scripts/collect_bench.py || true
push "final collection"

banner "SUMMARY"
column -t -s$'\t' "$STATUS" 2>/dev/null || cat "$STATUS"
echo
echo "FAILED steps:"
grep -c FAILED "$STATUS" 2>/dev/null || echo 0
grep FAILED "$STATUS" 2>/dev/null || echo "  none"
echo
echo "Everything is pushed. Before you delete the node:"
echo "  1. git log --oneline | head     (confirm your commits are there)"
echo "  2. check results/ on GitHub in a browser"
echo "  3. only then delete the node"
