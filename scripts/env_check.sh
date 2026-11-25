#!/usr/bin/env bash
# Day-1 gate. Run this the moment you get a GPU, BEFORE anything else.
# Everything it prints goes into notes/<today>.md.
set -uo pipefail

echo "=== llm-precision-study environment check ==="
echo "date: $(date -Is)"
echo

echo "--- GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version \
             --format=csv
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')
  echo
  echo "compute capability: $CC"
  awk -v cc="$CC" 'BEGIN{
    if (cc+0 >= 10.0) {
      print "  -> Blackwell. FP8 OK. INT8 W8A8 is NOT supported: drop INT8 or change card."
    } else if (cc+0 >= 8.9) {
      print "  -> Ada/Hopper. FP8 W8A8 OK, INT8 OK, INT4 OK. This is the card you want."
    } else if (cc+0 >= 8.0) {
      print "  -> Ampere. FP8 W8A8 NOT supported (would silently fall back to W8A16 Marlin)."
      print "     Either move to Ada/Hopper, or cut FP8 and say so in the report."
    } else {
      print "  -> Too old for this project."
    }
  }'
else
  echo "nvidia-smi not found: no GPU on this machine."
  echo "Use this week for reading + writing the analysis code (that is already done"
  echo "in analysis/ and router/ -- test it with: make synthetic test)."
fi
echo

echo "--- disk ---"
df -h . | tail -n +1
echo "need >= 200 GB free: each 8B model is ~16 GB in BF16 and you keep several copies"
echo

echo "--- python / libs ---"
python3 -c "import sys; print('python', sys.version.split()[0])"
for pkg in vllm lm_eval llmcompressor torch transformers; do
  python3 - "$pkg" <<'PY' 2>/dev/null || echo "  $pkg: NOT INSTALLED in this env"
import importlib, sys
m = importlib.import_module(sys.argv[1])
print(f"  {sys.argv[1]}: {getattr(m, '__version__', 'unknown')}")
PY
done
echo
echo "reminder: vLLM and llm-compressor belong in SEPARATE venvs."
echo "  .venv-vllm       -> requirements-vllm.txt       (serve + lm_eval)"
echo "  .venv-compressor -> requirements-compressor.txt (build checkpoints)"
echo

echo "--- huggingface ---"
echo "HF_HOME=${HF_HOME:-<unset>}"
if [ -z "${HF_HOME:-}" ]; then
  echo "  WARNING: HF_HOME unset. Caches will land on the root disk and fill it."
  echo "  Fix: export HF_HOME=/path/on/big/disk/hf_cache"
fi
python3 -c "
from huggingface_hub import whoami
try:
    print('  logged in as:', whoami()['name'])
except Exception as e:
    print('  NOT logged in (run: hf auth login):', type(e).__name__)
" 2>/dev/null || echo "  huggingface_hub not installed"
echo

echo "--- tmux ---"
command -v tmux >/dev/null 2>&1 \
  && echo "  tmux present. Run every long job inside it." \
  || echo "  tmux MISSING. Install it before starting a 3-hour eval."
echo
echo "=== paste all of the above into notes/$(date +%F).md ==="
