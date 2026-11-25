#!/usr/bin/env bash
# One reproducible quality run: one config, one task.
#
#   ./scripts/run_quality.sh llama31-8b__w-bf16__kv-bf16 gsm8k
#   ./scripts/run_quality.sh llama31-8b__w-fp8__kv-bf16  gsm8k --limit 200
#
# Everything that defines the run comes from configs/eval.yaml and the config
# file. Nothing is typed by hand, so a rerun is byte-identical in its settings.
#
# Results land in results/quality/<config>/<task>/<run_id>/ and are NEVER
# overwritten: each run gets its own timestamped folder, so a rerun gives you a
# second measurement you can use to estimate the noise floor.
set -euo pipefail

CONFIG="${1:?usage: run_quality.sh <config-name> <task> [extra lm_eval args]}"
TASK="${2:?usage: run_quality.sh <config-name> <task> [extra lm_eval args]}"
shift 2
cd "$(dirname "$0")/.."
# scripts/ holds common.py; the heredocs below import it after this cd,
# so the repo root alone is not enough. Without this every config lookup
# dies with ModuleNotFoundError: No module named 'common'.
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PWD/results/checkpoints}"
export CHECKPOINT_DIR

RUN_ID="$(date +%Y%m%dT%H%M%S)"
OUT="results/quality/${CONFIG}/${TASK}/${RUN_ID}"
mkdir -p "$OUT"

# Build the lm_eval invocation from the configs.
mapfile -t PARTS < <(
  python3 - "$CONFIG" "$TASK" <<'PY'
import sys, common
cfg = common.load_config(sys.argv[1])
ev = common.load_yaml("configs/eval.yaml")
task = sys.argv[2]
if task not in ev["tasks"]:
    raise SystemExit(f"unknown task {task!r}; known: {list(ev['tasks'])}")
t = ev["tasks"][task]
ma = dict(cfg["eval_model_args"])
ma["seed"] = ev["decoding"]["seed"]
print(",".join(f"{k}={v}" for k, v in ma.items()))
print(t["harness_task"])
print(t["num_fewshot"])
print(ev["run"]["batch_size"])
print(int(bool(t.get("requires_code_execution"))))
PY
)
# Under `set -u` an empty PARTS dies with an unreadable "unbound variable".
# The cause is always the python above failing, so say that instead.
if [ "${#PARTS[@]}" -lt 5 ]; then
  echo "FAILED to read the config for '$CONFIG' / task '$TASK'."
  echo "Re-running it so you can see the actual error:"
  echo "---"
  python3 - "$CONFIG" "$TASK" <<'PY'
import sys, common
cfg = common.load_config(sys.argv[1])
ev = common.load_yaml("configs/eval.yaml")
task = sys.argv[2]
if task not in ev["tasks"]:
    raise SystemExit(f"unknown task {task!r}; known: {list(ev['tasks'])}")
print("config and task both loaded fine -- the failure is elsewhere")
PY
  echo "---"
  echo "Most likely: PYTHONPATH is not picking up scripts/, or configs/models/"
  echo "has no ${CONFIG}.yaml (run: python3 scripts/gen_configs.py)."
  exit 2
fi
MODEL_ARGS="${PARTS[0]}"
HARNESS_TASK="${PARTS[1]}"
FEWSHOT="${PARTS[2]}"
BATCH="${PARTS[3]}"
NEEDS_EXEC="${PARTS[4]}"

if [ "$NEEDS_EXEC" = "1" ]; then
  if [ "${I_AM_IN_A_CONTAINER:-0}" != "1" ]; then
    echo "REFUSING: $TASK executes model-generated code."
    echo "Run it inside a container or a throwaway VM, then set"
    echo "  export I_AM_IN_A_CONTAINER=1"
    echo "Never run this on a machine you care about."
    exit 2
  fi
  export HF_ALLOW_CODE_EVAL=1
fi

# The harness entrypoint has been spelled three ways across releases. Current
# docs document `lm-eval` (hyphen), which auto-inserts its `run` subcommand
# when you use the legacy flag-only form; `lm_eval` (underscore) is the older
# console script. Pick whichever exists rather than guessing.
if command -v lm-eval >/dev/null 2>&1; then
  LM_EVAL=(lm-eval)
elif command -v lm_eval >/dev/null 2>&1; then
  LM_EVAL=(lm_eval)
else
  LM_EVAL=(python3 -m lm_eval)
fi
echo "harness entrypoint: ${LM_EVAL[*]}"

# gen_kwargs is space-separated key=val, parsed with ast.literal_eval.
# ONLY temperature=0 here. Do NOT add do_sample=False: that is a transformers
# argument, and the vLLM backend builds SamplingParams from these kwargs, so an
# unknown key raises rather than being ignored.
CMD=("${LM_EVAL[@]}"
  --model vllm
  --model_args "$MODEL_ARGS"
  --tasks "$HARNESS_TASK"
  --num_fewshot "$FEWSHOT"
  --batch_size "$BATCH"
  --gen_kwargs temperature=0
  --log_samples
  --output_path "$OUT"
  --seed 1234
  "$@"
)

{
  echo "config   : $CONFIG"
  echo "task     : $TASK -> $HARNESS_TASK"
  echo "run_id   : $RUN_ID"
  echo "command  : ${CMD[*]}"
} | tee "$OUT/command.txt"

python3 -c "
import sys, json, common
common.write_provenance('$OUT/provenance.json',
    {'config': '$CONFIG', 'task': '$TASK', 'run_id': '$RUN_ID',
     'command': '''${CMD[*]}'''})
"

"${CMD[@]}" 2>&1 | tee "$OUT/stdout.log"

echo
echo "saved to $OUT"
echo "next: python scripts/collect_logs.py   # folds this into the tidy per-prompt table"
