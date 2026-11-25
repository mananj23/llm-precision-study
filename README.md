# llm-precision-study

**What does lower numerical precision actually cost you, and what does it save?**

A controlled study of weight and KV-cache quantization for 8B instruction-tuned
models served with vLLM: quality measured per prompt with paired confidence
intervals, serving performance measured on the same hardware, and an honest test
of whether choosing precision per request beats choosing one setting for
everything.

Status: **environment and analysis pipeline complete and tested; measurements
not yet run.** Every number in this repository today comes from a synthetic
fixture and is labelled as such.

---

## The question

Quantization results are usually reported as a single accuracy number on a
single benchmark, against a baseline that was measured with a different prompt
format, with no interval. That makes it impossible to tell a real 1-point
regression from noise, and impossible to know whether the setting that is best
for arithmetic is also best for long-context summarisation.

This project asks three things:

1. **Where does quality actually break?** Per task, per precision setting, with
   a 95% paired bootstrap interval on the change from BF16 — not a difference of
   two separately-estimated accuracies.
2. **What does each setting buy?** Throughput, time-to-first-token,
   time-per-output-token, KV-cache capacity and dollars per million output
   tokens, measured on the same card at a fixed grid of concurrencies and
   context lengths.
3. **Does per-request routing beat a fixed choice?** Measured against a random
   mix at matched cost, which is the only baseline that makes the question
   meaningful. **A negative answer is a valid result and will be reported as
   one.**

## Configurations

| Setting | Weights | KV cache | Built with |
|---|---|---|---|
| `w-bf16__kv-bf16` | BF16 | BF16 | baseline, no build |
| `w-fp8__kv-bf16` | FP8 W8A8 | BF16 | llm-compressor `FP8_DYNAMIC` |
| `w-bf16__kv-fp8` | BF16 | FP8 | vLLM `--kv-cache-dtype fp8` |
| `w-fp8__kv-fp8` | FP8 W8A8 | FP8 | both |
| `w-int8__kv-bf16` | INT8 W8A8 | BF16 | SmoothQuant(0.8) + GPTQ |
| `w-int4gptq__kv-bf16` | INT4 W4A16 | BF16 | GPTQ, group size 128 |
| `w-int4awq__kv-bf16` | INT4 W4A16 | BF16 | AWQ, group size 128 |

`w-bf16__kv-fp8` exists so that the effect of the FP8 KV cache can be separated
from the effect of FP8 weights. Without it, every FP8-KV row also has quantized
weights and the two causes are confounded.

GPTQ-INT4 and AWQ-INT4 are the one genuinely controlled algorithm comparison
here: same bit width, same group size, same calibration set, so a difference
between them is a difference between the two algorithms.

## How much can a benchmark this size actually resolve?

A result that costs nothing to produce and reframes the rest. For paired binary
outcomes only the *discordant* prompts carry information, so with discordance
`r` and effect `d` the minimum detectable effect is `1.96·sqrt(r/(n+1.96²))`.
At an 8% discordance rate:

| Benchmark | n | Smallest detectable effect | Power at 1 pt |
|---|---|---|---|
| HumanEval | 164 | 4.3 pts | 7% |
| GSM8K | 1,319 | 1.5 pts | **25%** |
| MMLU (full) | 14,042 | 0.5 pts | 99% |

Detecting a 1-point regression with 80% power takes about **6,300 prompts**.
The literature routinely reports 1–3 point effects on benchmarks of ~1,000.

`python analysis/power.py` — validates its own formulas against simulation
before printing anything.

**The method is not ours.** The resolution framework is Kotawala,
*Resolution Diagnostics for Paired LLM Evaluation* (arXiv:2605.30315,
ICML 2026); the case for paired, question-level inference is Miller,
*Adding Error Bars to Evals* (arXiv:2411.00640). What this repository adds is
narrow: applying it to the **quantization** literature specifically, with
discordance rates measured from real quantized checkpoints rather than
assumed. Below ~300 prompts the normal approximation understates uncertainty
(Bowyer et al., arXiv:2503.01747), so HumanEval-sized figures are lower
bounds.

## How to reproduce the table and the figures

```bash
make all          # collect raw output -> analysis/results.csv -> report/*.png
make power        # what effect size the benchmarks can resolve
```

Running the measurements on rented hardware: see **[RUNBOOK.md](RUNBOOK.md)**,
and run `./scripts/session1.sh` before anything long — it verifies the whole
pipeline with a 20-prompt evaluation and prints GO or NO-GO.

That is the whole of it. The table is never edited by hand; if a number is
wrong, the fix goes in `analysis/build_results.py` and the table is rebuilt.

To check the pipeline without a GPU or any real data:

```bash
make test         # synthetic fixture -> full pipeline -> guard tests
```

## Method notes that matter

**Intervals are paired.** Two configurations are scored on the same prompts, so
prompt difficulty cancels in a per-prompt difference. Resampling prompts and
recomputing that difference gives an interval roughly **7× tighter** than
treating the two accuracies as independent. On a simulated 1-point regression
the paired interval excludes zero and the unpaired one does not — the unpaired
method would report "no difference" for a real effect. Reproduce with
`python analysis/bootstrap.py`.

**"No detectable difference" is written as such.** Configurations whose
intervals overlap are not ranked against each other.

**Every checkpoint is verified before it is evaluated.**
`scripts/verify_checkpoint.py` checks the files, checks the memory actually
dropped, and runs 100 GSM8K prompts. A quantized model scoring near zero is a
broken checkpoint — wrong chat template, missing tokenizer files, bad
calibration — not evidence that quantization hurts.

**One calibration set, frozen.** `configs/calibration.yaml` is fixed at 512
UltraChat samples, seed 42, and is used by every calibrated method. A different
calibration set per method would make the comparison meaningless. It is
deliberately not drawn from any evaluation task.

**Benchmarks do not share the GPU and do not share a prefix cache.**
`run_bench.sh` refuses to start if another process is on the card. Benchmark
servers run with `--no-enable-prefix-caching` and each repeat uses a different
seed, because repeating a benchmark against a warm prefix cache reuses prompts
and inflates throughput.

**The router is evaluated against random mixing at matched cost**, split by
prompt and again with an entire task held out, and the cost of any confidence
signal it consults is charged on every request. Features that are only known
after the model has answered are rejected by a guard, not by discipline.

## Layout

```
configs/        one YAML per precision setting + frozen eval and calibration setup
scripts/        environment check, quantize, serve, evaluate, benchmark, collect
analysis/       paired bootstrap, results table, Pareto and long-context figures
router/         routing dataset, baselines, training and evaluation
tests/          synthetic fixture and guards against silent failure
notes/          daily lab log: what was run, which commit, what surprised me
results/        raw output (never edited) + tidy derived tables
report/         figures and the written report
```

## Hardware

FP8 W8A8 needs compute capability ≥ 8.9 (Ada or Hopper). On an Ampere card it
will still load, as weight-only W8A16 via FP8-Marlin — a *different* experiment,
and `scripts/common.py` refuses to build the checkpoint rather than let that
happen silently. INT8 W8A8 is not supported on Blackwell (cc ≥ 10.0). Run
`make check` on any new machine before anything else.

## Honest limitations

See [LIMITATIONS.md](LIMITATIONS.md). The short version: this measures
benchmark prompts, not production traffic; the routing result is an offline
replay of logged outputs, not a live serving experiment; and everything is
measured on one GPU model at one price on one date.

## Licence

MIT. See [LICENSE](LICENSE).
