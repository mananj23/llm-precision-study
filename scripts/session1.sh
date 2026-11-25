#!/usr/bin/env bash
# SESSION 1 — prove the environment before spending real money on it.
#
#   ./scripts/session1.sh
#
# Runs eight checks in increasing order of cost, ending with a real 20-prompt
# evaluation that exercises the entire pipeline: vLLM loads the model, lm-eval
# runs it, samples are logged, and collect_logs.py parses them. Roughly 15-20
# minutes, almost all of it the model download.
#
# Why this exists: run_all.sh has never run on real hardware. Finding a broken
# flag here costs ~20 minutes. Finding it three hours into the full matrix
# costs three hours at the GPU's hourly rate. Everything this checks is
# something that has actually broken for someone.
#
# It prints GO or NO-GO at the end. Do not start run_all.sh on a NO-GO.
set -uo pipefail
cd "$(dirname "$0")/.."
# scripts/ holds common.py; the heredocs below import it after this cd,
# so the repo root alone is not enough. Without this every config lookup
# dies with ModuleNotFoundError: No module named 'common'.
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"

RATE="${RATE_PER_HOUR:-379}"       # for the running cost estimate
START=$(date +%s)
PASS=0; FAIL=0; WARN=0
FAILED_CHECKS=()

ok()   { echo "  ✓ $*"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $*"; FAIL=$((FAIL+1)); FAILED_CHECKS+=("$*"); }
warn() { echo "  ! $*"; WARN=$((WARN+1)); }
hdr()  { echo; echo "── $* ────────────────────────────────────────"; }

echo "SESSION 1 PREFLIGHT   started $(date -Is)"
echo "billing assumption: Rs ${RATE}/hr (override with RATE_PER_HOUR=...)"

# ----------------------------------------------------------------- 1. GPU --
hdr "1. GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "nvidia-smi not found — this is not a GPU machine"
else
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  echo
  ok "$NGPU GPU(s) visible"
  case "$CC" in
    8.9|9.0) ok "compute capability $CC — FP8, INT8 and INT4 all supported" ;;
    8.0|8.6) bad "compute capability $CC is Ampere — FP8 will SILENTLY run as W8A16. Wrong card." ;;
    10.*|11.*) warn "compute capability $CC is Blackwell — INT8 W8A8 unsupported; drop that row" ;;
    *) bad "compute capability $CC — too old for this project" ;;
  esac
  echo "  TP is set to: ${TP:-1}"
  if [ "${TP:-1}" -gt "$NGPU" ]; then
    bad "TP=${TP} exceeds $NGPU GPU(s) — export TP=$NGPU"
  else
    ok "TP=${TP:-1} matches the hardware"
  fi
fi

# ------------------------------------------------------------ 2. the libs --
hdr "2. libraries"
python3 - <<'PY'
import importlib, sys
need = {"vllm": "0.29", "lm_eval": "0.4.12", "torch": None,
        "transformers": None, "pandas": None, "numpy": None,
        "pyarrow": None, "sklearn": None, "matplotlib": None, "yaml": None}
missing = []
for pkg, minv in need.items():
    try:
        m = importlib.import_module(pkg)
        print(f"  ✓ {pkg} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  ✗ {pkg} MISSING ({type(e).__name__})")
        missing.append(pkg)
sys.exit(1 if missing else 0)
PY
[ $? -eq 0 ] && ok "all libraries import" || bad "libraries missing — pip install -r requirements-vllm.txt"

# vLLM version gate: old vLLM lacks `bench serve` and has different quant paths
python3 - <<'PY' || true
try:
    import vllm
    v = tuple(int(x) for x in vllm.__version__.split(".")[:2])
    print("  vllm version tuple:", v)
    if v < (0, 29):
        print("  ! vllm older than 0.29 — benchmark CLI and quant kernels differ")
except Exception:
    pass
PY

# ------------------------------------------------------- 3. the harness CLI --
hdr "3. lm-eval entrypoint"
LM=$(python3 -c "import sys; sys.path.insert(0,'scripts'); import common; print(common.lm_eval_cmd())")
echo "  resolved to: $LM"
if $LM --help >/dev/null 2>&1; then ok "harness responds to --help"; else bad "harness not callable"; fi

# ------------------------------------------------- 4. the benchmark CLI -----
hdr "4. vllm bench serve (needed in phase 4, not now)"
if vllm bench serve --help >/dev/null 2>&1; then
  ok "vllm bench serve exists"
else
  warn "vllm bench serve NOT available — benchmarks will fail later."
  warn "  Tell Claude before phase 4; run_bench.sh needs the older"
  warn "  benchmarks/benchmark_serving.py path instead."
fi

# ------------------------------------------------------ 5. the task names --
hdr "5. task names in your harness version"
AVAIL=$($LM --tasks list 2>/dev/null || true)
for t in gsm8k mmlu ifeval longbench; do
  if echo "$AVAIL" | grep -qiE "^[[:space:]|-]*${t}([[:space:]|]|$)"; then
    ok "task '$t' present"
  else
    warn "task '$t' not found by exact match — check: $LM --tasks list | grep $t"
  fi
done
echo "  (a warning here is often just formatting; confirm by hand before worrying)"

# -------------------------------------------------------- 6. HF + model ----
hdr "6. Hugging Face access"
python3 - <<'PY'
import sys
try:
    from huggingface_hub import whoami, model_info
    print("  ✓ logged in as:", whoami()["name"])
except Exception as e:
    print("  ✗ not logged in:", type(e).__name__, "— run: hf auth login")
    sys.exit(1)
try:
    mi = model_info("meta-llama/Llama-3.1-8B-Instruct")
    print("  ✓ gated model accessible:", mi.id)
except Exception as e:
    print("  ✗ cannot access Llama-3.1-8B-Instruct:", type(e).__name__)
    print("    Either access is not granted, or the token lacks permission.")
    print("    Fallback that needs no approval: Qwen/Qwen2.5-7B-Instruct")
    sys.exit(1)
PY
[ $? -eq 0 ] && ok "HF auth and model access fine" || bad "HF access problem"

# ---------------------------------------------------------- 7. disk/env ----
hdr "7. disk and environment"
df -h . | tail -1
AVAIL_GB=$(df -BG . | tail -1 | awk '{gsub("G","",$4); print $4}')
if [ "${AVAIL_GB:-0}" -ge 120 ]; then ok "${AVAIL_GB}G free"; else bad "${AVAIL_GB}G free — need 120G+"; fi
[ -n "${HF_HOME:-}" ] && ok "HF_HOME=$HF_HOME" || warn "HF_HOME unset — cache may land on a small disk"
[ -n "${CHECKPOINT_DIR:-}" ] && ok "CHECKPOINT_DIR=$CHECKPOINT_DIR" || warn "CHECKPOINT_DIR unset (defaults inside the repo)"
command -v tmux >/dev/null 2>&1 && ok "tmux present" || bad "tmux MISSING — install it before any long run"

# ------------------------------------------- 8. THE REAL TEST: 20 prompts --
hdr "8. end-to-end: 20 GSM8K prompts on BF16"
echo "  This downloads the model (~16GB, several minutes) then runs a tiny eval."
echo "  It exercises vLLM loading, the harness flags, sample logging and parsing."
echo
BASE="${FAMILY:-llama31-8b}__w-bf16__kv-bf16"
if ./scripts/run_quality.sh "$BASE" gsm8k --limit 20 >results/session1_eval.log 2>&1; then
  ok "eval completed"
  if grep -qiE "exact_match|acc" results/session1_eval.log; then
    echo
    grep -iE "\|.*(exact_match|acc).*\|" results/session1_eval.log | head -5
    echo
    ok "metrics produced"
  else
    warn "no metric line found — read results/session1_eval.log"
  fi
else
  bad "eval FAILED — read results/session1_eval.log (last 30 lines below)"
  tail -30 results/session1_eval.log
fi

hdr "9. parsing"
if python3 scripts/collect_logs.py >results/session1_collect.log 2>&1; then
  ok "collect_logs.py parsed the samples"
  python3 - <<'PY'
import pandas as pd, pathlib
p = pathlib.Path("results/logs/per_prompt.parquet")
if p.exists():
    d = pd.read_parquet(p)
    print(f"  rows: {len(d)}  configs: {d['config'].nunique()}  "
          f"mean correct: {d['correct'].mean():.3f}")
    need = {"doc_id","correct","input_len","prompt_text","config","task"}
    miss = need - set(d.columns)
    print("  ✗ missing columns:", miss) if miss else print("  ✓ schema complete")
PY
else
  bad "collect_logs.py FAILED — the analysis pipeline cannot read the output"
  tail -20 results/session1_collect.log
fi

# -------------------------------------------------------------- verdict ----
ELAPSED=$(( $(date +%s) - START ))
COST=$(python3 -c "print(f'{$ELAPSED/3600*$RATE:.0f}')")
hdr "VERDICT"
printf "passed %d   warnings %d   failed %d\n" "$PASS" "$WARN" "$FAIL"
printf "elapsed %dm %ds   approx cost so far: Rs %s\n" $((ELAPSED/60)) $((ELAPSED%60)) "$COST"
echo
if [ "$FAIL" -eq 0 ]; then
  echo "  ######   GO   ######"
  echo
  echo "  The pipeline works end to end. Start the full run:"
  echo
  echo "      tmux new -s all"
  echo "      ./scripts/run_all.sh 2>&1 | tee results/run_all.log"
  echo "      # Ctrl-B then D to detach"
  echo
  echo "  Check progress with:  grep FAILED results/run_all_status.tsv"
else
  echo "  ######  NO-GO  ######"
  echo
  echo "  Fix these before starting the full run:"
  for c in "${FAILED_CHECKS[@]}"; do echo "    - $c"; done
  echo
  echo "  STOP THE CONTAINER while you fix them. Do not debug on the clock."
fi
echo
