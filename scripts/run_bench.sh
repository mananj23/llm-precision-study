#!/usr/bin/env bash
# System benchmark grid for ONE configuration (Step 5).
#
#   # terminal 1 (inside tmux):
#   ./scripts/serve.sh llama31-8b__w-fp8__kv-bf16 --bench
#   # terminal 2, once it reports ready:
#   ./scripts/run_bench.sh llama31-8b__w-fp8__kv-bf16
#
# Rules this script enforces so the numbers mean something:
#   - one warm-up pass per shape, discarded
#   - three repeats with different seeds, all kept (median reported later)
#   - nothing else on the GPU (it refuses to start if another process is on it)
#   - raw JSON per run, never edited
set -euo pipefail

CONFIG="${1:?usage: run_bench.sh <config-name>}"
cd "$(dirname "$0")/.."
# scripts/ holds common.py; the heredocs below import it after this cd,
# so the repo root alone is not enough. Without this every config lookup
# dies with ModuleNotFoundError: No module named 'common'.
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"

PORT=8000
MODEL_PATH="$(python3 -c "import common;print(common.load_config('$CONFIG')['model_path'])")"
TP_SIZE="$(python3 -c "import common;print(common.load_config('$CONFIG')['serve'].get('tensor_parallel_size',1))")"

if [ -z "$MODEL_PATH" ]; then
  echo "FAILED to read config '$CONFIG'. Run: python3 scripts/gen_configs.py"
  exit 2
fi

# --- refuse to share the GPU -------------------------------------------------
# One vLLM server with tensor parallelism shows up as one compute process PER
# GPU, so the old test (NPROC > 1) refused to benchmark on any multi-GPU node.
# What we actually care about is a SECOND job, i.e. more processes than this
# server's own workers, or distinct PIDs beyond the expected count.
if command -v nvidia-smi >/dev/null 2>&1; then
  NPROC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if [ "${NPROC:-0}" -gt "$TP_SIZE" ]; then
    echo "REFUSING: $NPROC processes are using the GPU(s), expected at most $TP_SIZE."
    echo "A benchmark that shares the GPU measures nothing. Stop the other job."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 2
  fi
fi

# --- server must be up -------------------------------------------------------
if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null; then
  echo "No vLLM server on port ${PORT}. Start it first:"
  echo "    ./scripts/serve.sh $CONFIG --bench"
  exit 2
fi

OUTDIR="results/system/${CONFIG}"
mkdir -p "$OUTDIR"
python3 -c "import common; common.write_provenance('$OUTDIR/provenance.json', {'config':'$CONFIG'})"

CONCURRENCIES=(1 8 32 64)
SHAPES=("512 128 short" "4096 256 medium" "16384 256 long")
SEEDS=(${BENCH_SEEDS:-0 1 2})
NUM_PROMPTS_PER_CONC=8   # requests per concurrency level, scaled below

# A token budget, not a fixed request count. 8 requests per concurrency level
# is fine at 512 input tokens and ruinous at 16384: conc=64 would be 512
# requests x 16.6k tokens = 8.5M tokens, about 25 minutes for ONE cell, and
# roughly 3 hours for the long shape alone, per config. Seven configs would
# not fit in a 29-hour rental. Capping prompt tokens per run keeps the long
# shape honest (still >= 2x concurrency requests, so the steady-state window
# dominates) and brings the long shape to well under an hour per config.
PROMPT_TOKEN_BUDGET="${PROMPT_TOKEN_BUDGET:-600000}"

prompt_count() {  # concurrency input_len -> number of requests
  local conc="$1" in_len="$2" n budget_n
  n=$(( conc * NUM_PROMPTS_PER_CONC ))
  budget_n=$(( PROMPT_TOKEN_BUDGET / in_len ))
  [ "$n" -gt "$budget_n" ] && n="$budget_n"
  # floors: enough requests to reach steady state, and never a trivial run
  [ "$n" -lt $(( 2 * conc )) ] && n=$(( 2 * conc ))
  [ "$n" -lt 32 ] && n=32
  echo "$n"
}

# Print the plan and a rough time estimate before burning any GPU time, so a
# grid that cannot fit the rental window is visible in the first 10 seconds.
echo "benchmark plan for $CONFIG   (seeds: ${SEEDS[*]}, budget: ${PROMPT_TOKEN_BUDGET} prompt tokens/run)"
TOTAL_TOK=0
for shape in "${SHAPES[@]}"; do
  read -r IN_LEN OUT_LEN LABEL <<<"$shape"
  line="  ${LABEL}  (in=${IN_LEN} out=${OUT_LEN}):"
  for conc in "${CONCURRENCIES[@]}"; do
    n=$(prompt_count "$conc" "$IN_LEN")
    line="$line  c${conc}:n=${n}"
    TOTAL_TOK=$(( TOTAL_TOK + (n * (IN_LEN + OUT_LEN)) * (${#SEEDS[@]} + 1) ))
  done
  echo "$line"
done
echo "  total tokens to push (incl. warm-ups): ~$(( TOTAL_TOK / 1000000 ))M"
echo "  at a conservative 5k tok/s that is ~$(( TOTAL_TOK / 5000 / 60 )) min for this config"
echo

for shape in "${SHAPES[@]}"; do
  read -r IN_LEN OUT_LEN LABEL <<<"$shape"
  for conc in "${CONCURRENCIES[@]}"; do

    N=$(prompt_count "$conc" "$IN_LEN")

    # ---- warm-up, discarded ----
    echo "### warm-up  ${LABEL} conc=${conc}"
    if ! vllm bench serve \
      --backend vllm --model "$MODEL_PATH" \
      --endpoint /v1/completions \
      --host localhost --port "$PORT" \
      --dataset-name random \
      --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
      --num-prompts "$conc" --max-concurrency "$conc" \
      --ignore-eos --seed 99 \
      >/dev/null 2>&1; then
      # A warm-up that fails at this shape means the real runs will fail too,
      # three times over, each after its own timeout. Skip the cell instead.
      echo "  warm-up failed -- KV cache almost certainly will not hold"
      echo "  ${LABEL} at concurrency ${conc}. Recording the cap and skipping."
      echo "{\"config\":\"$CONFIG\",\"shape\":\"$LABEL\",\"concurrency\":$conc,\"failed\":true,\"stage\":\"warmup\"}" \
        > "$OUTDIR/${LABEL}_c${conc}.WARMUP_FAILED.json"
      continue
    fi

    for seed in "${SEEDS[@]}"; do
      NAME="${LABEL}_c${conc}_seed${seed}.json"
      echo "### ${LABEL} conc=${conc} seed=${seed} n=${N} -> ${NAME}"

      if ! vllm bench serve \
          --backend vllm --model "$MODEL_PATH" \
          --endpoint /v1/completions \
          --host localhost --port "$PORT" \
          --dataset-name random \
          --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
          --num-prompts "$N" --max-concurrency "$conc" \
          --ignore-eos --seed "$seed" \
          --percentile-metrics ttft,tpot,itl,e2el \
          --save-result --save-detailed \
          --result-dir "$OUTDIR" --result-filename "$NAME" \
          2>&1 | tee "$OUTDIR/${NAME%.json}.log"; then
        echo "  FAILED: ${LABEL} conc=${conc}. If this is the 16k shape, the KV"
        echo "  cache probably does not fit. That IS a result -- record the cap"
        echo "  in notes/ and move on."
        echo "{\"config\":\"$CONFIG\",\"shape\":\"$LABEL\",\"concurrency\":$conc,\"seed\":$seed,\"failed\":true}" \
          > "$OUTDIR/${NAME%.json}.FAILED.json"
      fi
    done
  done
done

echo
echo "done. raw JSON in $OUTDIR -- never edit these files."
echo "record the KV cache capacity from the server log now:"
echo "    grep -i 'KV cache' $OUTDIR/serve_*.log"
