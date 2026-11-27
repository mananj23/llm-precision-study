#!/usr/bin/env bash
# Everything left to do after the benchmark grid, in the right order, with the
# gate that the last two mishaps needed.
#
# Twice now a second job was started while the benchmark grid was mid-flight.
# Both times vLLM refused to allocate, the eval died, and one benchmark
# configuration was lost -- the second time the stray `pkill -f "vllm serve"`
# killed a server that was 31 cells into a 36-cell grid. So this script's first
# act is to refuse to run at all while anything else holds the GPU.
#
#   ./scripts/finish.sh
#
# Order is deliberate. The noise-floor run comes first because it is four
# minutes long and every disagreement number in the paper is uninterpretable
# without it; if the session dies after one step, that is the step to have.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
BASE="${FAMILY:-llama31-8b}__w-bf16__kv-bf16"
AWQ="${FAMILY:-llama31-8b}__w-int4awq__kv-bf16"
LOG="results/finish_$(date +%Y%m%dT%H%M%S).log"
mkdir -p results results/logs

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# ---------------------------------------------------------------- the gate --
say "gate: is anything else using the GPU?"

busy=0
if pgrep -f "run_all.sh" >/dev/null 2>&1; then
  echo "  run_all.sh is RUNNING (pid $(pgrep -f run_all.sh | tr '\n' ' '))"
  busy=1
fi
if pgrep -f "vllm serve" >/dev/null 2>&1; then
  echo "  a vllm server is RUNNING (pid $(pgrep -f 'vllm serve' | tr '\n' ' '))"
  busy=1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "  GPU memory in use: ${used} MiB"
  [ "${used:-0}" -gt 2000 ] && busy=1
fi

if [ "$busy" -ne 0 ]; then
  cat <<'EOF'

  REFUSING TO RUN. Something is still using the GPU.

  Starting a second job now is what cost a benchmark configuration twice
  before: vLLM cannot allocate, the eval dies, and in the worst case the
  running server is killed partway through its grid.

  Wait for the grid to finish, then check:
      pgrep -f run_all.sh >/dev/null && echo "still running" || echo "done"

  Read-only commands are safe meanwhile: tail, grep, ls, find, and the
  analysis scripts.
EOF
  exit 2
fi
echo "  clear."

exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

# ------------------------------------------------------- 1. the noise floor --
# THE CONTROL. Two identical runs of the baseline disagree on some prompts,
# because vLLM's continuous batching is not bitwise deterministic across runs.
# That rate is the floor for every disagreement and composition number in the
# study. Four minutes.
say "1/4  noise floor: a second full baseline run on GSM8K"

n_full=$(find "results/quality/$BASE/gsm8k" -name 'samples_*.jsonl' \
           -size +100k -printf '%h\n' 2>/dev/null | sort -u | wc -l)
echo "  full baseline GSM8K runs already present: ${n_full}"
if [ "${n_full:-0}" -ge 2 ]; then
  echo "  two or more already exist; skipping."
else
  ./scripts/run_quality.sh "$BASE" gsm8k \
    && echo "  ok" || echo "  FAILED -- the floor is still unmeasured; nothing"
fi

say "     computing the floor"
python3 - <<'PY' || echo "  (could not compute yet)"
import pandas as pd, pathlib, subprocess, sys
subprocess.run([sys.executable, "scripts/collect_logs.py", "--keep", "all"],
               capture_output=True)
p = pathlib.Path("results/logs/per_prompt.parquet")
if not p.exists():
    raise SystemExit("  no parquet yet")
df = pd.read_parquet(p)
import os
base = f"{os.environ.get('FAMILY','llama31-8b')}__w-bf16__kv-bf16"
d = df[(df.task == "gsm8k") & (df.config == base)]
runs = sorted(r for r, g in d.groupby("run_id") if len(g) > 1000)
print(f"  full runs found: {runs}")
if len(runs) < 2:
    raise SystemExit("  need two full baseline runs for a floor")
a = d[d.run_id == runs[0]].groupby("doc_id")["correct"].first()
b = d[d.run_id == runs[-1]].groupby("doc_id")["correct"].first()
j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
disagree = (j.a != j.b).mean()
damage = ((j.a == 1) & (j.b == 0)).mean()
print(f"  n = {len(j)}")
print(f"  baseline-vs-baseline DISAGREEMENT : {disagree:.4f}")
print(f"  baseline-vs-baseline DAMAGE       : {damage:.4f}   <-- pass to --floor")
pathlib.Path("results/logs/noise_floor.txt").write_text(
    f"disagreement={disagree:.6f}\ndamage={damage:.6f}\nn={len(j)}\n"
    f"runs={runs[0]},{runs[-1]}\n")
print("  wrote results/logs/noise_floor.txt")
PY

# ------------------------------------------------------------------ 2. AWQ --
# One attempt, as agreed. AWQ is the least valuable cell: GPTQ already covers
# W4A16 and no finding depends on having two 4-bit methods.
say "2/4  AWQ checkpoint (one attempt; failure is acceptable)"
if [ -d "${CHECKPOINT_DIR:-/home/jovyan/checkpoints}/${FAMILY:-llama31-8b}__w-int4awq" ]; then
  echo "  already built; skipping."
elif [ -x ".venv-compressor/bin/python" ]; then
  ( cd scripts && ../.venv-compressor/bin/python quantize_int4_awq.py "$AWQ" ) \
    && echo "  built" \
    || echo "  FAILED -- note it in LIMITATIONS.md and move on, as planned"
else
  echo "  no .venv-compressor; skipping"
fi

# ------------------------------------------- 3. the tasks that lost to deps --
say "3/4  IFEval and LongBench (dependencies now installed)"
if grep -q '^TASKS=(gsm8k mmlu)$' scripts/run_all.sh; then
  sed -i 's/^TASKS=(gsm8k mmlu)$/TASKS=(gsm8k mmlu ifeval longbench)/' scripts/run_all.sh
  echo "  restored the full task list"
fi
grep -n '^TASKS=' scripts/run_all.sh
./scripts/run_all.sh && echo "  run_all finished" || echo "  run_all exited non-zero"

# ------------------------------------------------------------- 4. the maths --
say "4/4  analysis"
python3 scripts/collect_logs.py --keep all || true
python3 scripts/collect_bench.py || true

FLOOR=""
if [ -f results/logs/noise_floor.txt ]; then
  FLOOR=$(sed -n 's/^damage=//p' results/logs/noise_floor.txt)
fi
for task in gsm8k mmlu; do
  if [ -n "$FLOOR" ]; then
    python3 analysis/interaction.py --task "$task" --floor "$FLOOR" \
      --json-out "results/logs/interaction_${task}.json" || true
  else
    python3 analysis/interaction.py --task "$task" \
      --json-out "results/logs/interaction_${task}.json" || true
  fi
done

say "done"
echo "  log: $LOG"
echo "  next: FAMILY=qwen25-7b ./scripts/run_all.sh   (the replication)"
