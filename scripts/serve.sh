#!/usr/bin/env bash
# Start a vLLM server for one configuration.
#
#   ./scripts/serve.sh llama31-8b__w-fp8__kv-bf16            # eval server
#   ./scripts/serve.sh llama31-8b__w-fp8__kv-bf16 --bench    # benchmark server
#
# --bench adds --no-enable-prefix-caching, because repeating a benchmark
# against a server with a warm prefix cache reuses prompts and inflates
# throughput. Quality evals keep the default (that is how it would be served).
#
# ALWAYS run this inside tmux.
set -euo pipefail

CONFIG="${1:?usage: serve.sh <config-name> [--bench]}"
MODE="${2:-}"
cd "$(dirname "$0")/.."
# scripts/ holds common.py; the heredocs below import it after this cd,
# so the repo root alone is not enough. Without this every config lookup
# dies with ModuleNotFoundError: No module named 'common'.
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PWD/results/checkpoints}"
export CHECKPOINT_DIR

read -r MODEL_PATH MAX_LEN GPU_UTIL KV_DTYPE PORT TP_SIZE <<<"$(
  python3 - "$CONFIG" <<'PY'
import sys, common
c = common.load_config(sys.argv[1])
s = c["serve"]
print(c["model_path"], s["max_model_len"], s["gpu_memory_utilization"],
      s["kv_cache_dtype"], s["port"], s.get("tensor_parallel_size", 1))
PY
)"

# If the config lookup failed, `read` quietly leaves every variable empty and
# vLLM dies much later with something unrelated. Fail here, loudly, and show
# the actual Python error.
if [ -z "${MODEL_PATH:-}" ] || [ -z "${TP_SIZE:-}" ]; then
  echo "FAILED to read config '$CONFIG'. The error was:"
  python3 - "$CONFIG" <<'PY' || true
import sys, common
c = common.load_config(sys.argv[1])
print(c["model_path"], c["serve"])
PY
  echo
  echo "Did you run 'python3 scripts/gen_configs.py' first?"
  exit 2
fi

# Refuse to start with a tensor-parallel size the machine cannot satisfy: vLLM
# fails late and confusingly, after loading weights.
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  if [ "$TP_SIZE" -gt "$NGPU" ]; then
    echo "REFUSING: tensor_parallel_size=$TP_SIZE but this machine has $NGPU GPU(s)."
    echo "Set TP to match:  export TP=$NGPU"
    exit 2
  fi
  echo "GPUs: $NGPU   tensor_parallel_size: $TP_SIZE"
fi

ARGS=(
  --dtype bfloat16
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --tensor-parallel-size "$TP_SIZE"
  --port "$PORT"
  --seed 1234
)
[ "$KV_DTYPE" != "auto" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
[ "$MODE" = "--bench" ] && ARGS+=(--no-enable-prefix-caching)

LOG="results/system/${CONFIG}/serve_$(date +%Y%m%dT%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

echo "config : $CONFIG"
echo "model  : $MODEL_PATH"
echo "mode   : ${MODE:-eval}"
echo "log    : $LOG"
echo
echo "vllm serve $MODEL_PATH ${ARGS[*]}"
echo
echo "AFTER IT REPORTS READY, grep the log for the KV cache line and record it:"
echo "    grep -i 'KV cache' $LOG"
echo "That number (how many tokens fit) is what explains throughput differences."
echo

exec vllm serve "$MODEL_PATH" "${ARGS[@]}" 2>&1 | tee "$LOG"
