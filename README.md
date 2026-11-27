# llm-precision-study

**What does lower numerical precision actually cost you, and what does it save?**

A controlled study of weight and KV-cache quantization for Llama-3.1-8B-Instruct
served with vLLM: quality measured **per prompt** with paired confidence
intervals, serving performance measured on the same card, and a verified
measurement floor so that small effects mean something.

Status: **measurements complete.** Six precision configurations, four
benchmarks, 216 throughput cells, all run on one NVIDIA L40S between 25 and 27
November 2026. Every number in this repository is generated from the measurement
data by `analysis/report_tables.py`; none is typed by hand.

📄 **[report/report.pdf](report/report.pdf)** — the full write-up, 13 pages.

---

## What came out of it

**The measurement floor is exactly zero.** The BF16 baseline was evaluated
twice, 17 hours and 22 minutes apart on the same GPU. All 1,319 extracted
answers were byte-identical; correctness disagreement **0.000%**. Every
difference below is attributable to precision rather than to run-to-run
variation — a claim most quantization results assert rather than demonstrate.

**A stable score hides a large change in behaviour.** FP8 weights with an FP8 KV
cache score **70.43%** on GSM8K against the baseline's **70.51%** — one question
in 1,319 — while **15.24%** of the answers change. The aggregate holds still
because ~101 prompts break and ~100 are repaired. On LongBench, **46%** of
documents score differently with every aggregate difference indistinguishable
from zero.

**The change lands on the model's near-ties, not on its confident answers.**
MMLU logs per-choice log-likelihoods, so a confidence margin is recoverable for
all 14,042 documents. The least-confident **25%** of prompts carry **98.6%** of
all FP8-weight damage. Above 1.75 nats — two-thirds of the set, 6,397 prompts —
FP8 weights damaged **1** prompt and the FP8 KV cache damaged **none**.

**Two quantizations compose additively — and the analysis that said otherwise
was ours.** Measured on a *damage set* (baseline-correct prompts, counting only
losses), the combination looks strongly sub-additive, resolved on three tasks.
Measured by the symmetric 2 × 2 interaction contrast, the interaction is
**unresolved on all four tasks**. Conditioning on baseline-correct selects
prompts the baseline answered luckily, and regression to the mean then
masquerades as structure. Both framings are reported, because the disagreement
between them is the more useful result.

**4-bit trades instruction-following for pattern-copying.** INT4 GPTQ almost
never fails to reproduce the answer format its five in-context exemplars
demonstrate — a format-miss rate of **0.91%** against the baseline's **7.35%**
(−6.44 pp, CI [−7.88, −5.00]) — while scoring **4.07 pp worse on IFEval**
(CI [−7.39, −0.74]), which tests compliance with instructions given in prose.
Both resolved; every 8-bit scheme is indistinguishable from BF16 on both.

**A prediction that failed.** A quantized KV cache stores every past token, so
its error should compound with context length. At up to 20,480 tokens on
LongBench the FP8 KV cache costs **−0.01 pp** — the smallest effect measured
anywhere in this study.

| configuration | GSM8K | MMLU | IFEval | LongBench | peak tok/s | KV tokens |
|---|---:|---:|---:|---:|---:|---:|
| BF16 / BF16 | 70.51% | 68.25% | 47.32% | 43.64% | 1,256 | 191,264 |
| FP8 W8A8 / BF16 KV | 70.28% | 68.23% | 45.84% | 43.97% | 1,714 | 248,832 |
| BF16 / FP8 KV | 69.60% | 67.96% | 46.21% | 43.63% | 1,353 | 389,824 |
| FP8 W8A8 / FP8 KV | 70.43% | 67.69%\* | 48.06% | 43.92% | **1,910** | 491,904 |
| INT8 W8A8 / BF16 KV | 70.20% | 67.70%\* | 46.40% | — | 1,753 | 242,336 |
| INT4 GPTQ / BF16 KV | 72.18% | 66.25%\* | 43.25%\* | — | 1,761 | 269,584 |

\* 95% paired-bootstrap interval excludes zero. Every other cell is unresolved
against BF16 and is **not** ranked. Full intervals: [report/tables.md](report/tables.md).

## What a benchmark this size can resolve

For paired binary outcomes only the *discordant* prompts carry information, so
with discordance `r` the minimum detectable effect is `1.96·sqrt(r/(n+1.96²))`.
Using the discordance rates **measured here** rather than assumed:

| Benchmark | n | measured discordance | smallest detectable effect |
|---|---:|---:|---:|
| IFEval | 541 | 13.3% | 3.06 pts |
| GSM8K | 1,319 | 15.2% | 2.10 pts |
| MMLU | 14,042 | 5.9% | 0.40 pts |

This is why GSM8K resolves **none** of the five comparisons here and MMLU
resolves three. GSM8K is the benchmark quantization papers reach for most often,
and at full size it cannot separate these configurations.

**The method is not ours.** The resolution framework is Kotawala, *Resolution
Diagnostics for Paired LLM Evaluation* ([arXiv:2605.30315](https://arxiv.org/abs/2605.30315),
ICML 2026); the case for paired, question-level inference is Miller, *Adding
Error Bars to Evals* ([arXiv:2411.00640](https://arxiv.org/abs/2411.00640)).
What this repository adds is narrow: applying it to quantization specifically,
with discordance measured from real quantized checkpoints rather than assumed.
Below ~300 prompts the normal approximation understates uncertainty (Bowyer et
al., [arXiv:2503.01747](https://arxiv.org/abs/2503.01747)).

## Configurations

| Setting | Weights | KV cache | Built with | Status |
|---|---|---|---|---|
| `w-bf16__kv-bf16` | BF16 | BF16 | baseline, no build | ✅ measured |
| `w-fp8__kv-bf16` | FP8 W8A8 | BF16 | llm-compressor `FP8_DYNAMIC` | ✅ measured |
| `w-bf16__kv-fp8` | BF16 | FP8 | vLLM `--kv-cache-dtype fp8` | ✅ measured |
| `w-fp8__kv-fp8` | FP8 W8A8 | FP8 | both | ✅ measured |
| `w-int8__kv-bf16` | INT8 W8A8 | BF16 | SmoothQuant(0.8) + GPTQ | ✅ measured |
| `w-int4gptq__kv-bf16` | INT4 W4A16 | BF16 | GPTQ, group size 128 | ✅ measured |
| `w-int4awq__kv-bf16` | INT4 W4A16 | BF16 | AWQ, group size 128 | ❌ **not built** |

`w-bf16__kv-fp8` exists so the effect of the FP8 KV cache can be separated from
the effect of FP8 weights. The first four form a **2 × 2 factorial** in weight
precision × KV precision, which is what makes the composition analysis possible.

**The AWQ arm was never built.** The installed llm-compressor exposes
`AWQModifier` as a factory function rather than a class, and the substitute path
did not produce a valid checkpoint inside the GPU budget. The GPTQ-vs-AWQ
algorithm comparison this repository was designed to make therefore **did not
happen**, and INT4 results here are GPTQ only.

**The per-request routing experiment was not run.** `router/` contains the
specification and code; no routing result is reported, and nothing in this
repository should be read as evidence for or against routing.

## Reproducing it

```bash
./scripts/env_check.sh                 # refuses to proceed on a mismatched stack
./scripts/run_all.sh                   # build, verify, evaluate, benchmark
python3 analysis/report_tables.py      # every number in the report
python3 analysis/plot_report.py        # every figure in the report
python3 analysis/margins.py            # the confidence analysis (needs raw samples)
python3 analysis/interaction_contrast.py
```

`TASKS="gsm8k mmlu" SKIP_BENCH=1 ./scripts/run_all.sh` runs quality only.
`FAMILY=qwen25-7b` selects the replication grid.

Without a GPU: `make test` runs the whole pipeline on a synthetic fixture.

## What is released

- **Per-prompt outcomes** for every configuration on every task
  (`results/logs/per_prompt.parquet`) — the churn, composition and confidence
  results all follow from these and none is recoverable from a score table.
- **All 216 throughput cells** (`results/logs/system.parquet`).
- **Confidence margins** for all 14,042 MMLU documents
  (`results/logs/mmlu_margins.parquet`).
- A provenance record per run, and the full console logs **including the
  failures**.

## Two analysis notes that changed numbers

**The filter column.** `scripts/collect_logs.py` wrote one row per (document,
lm-eval filter) without recording *which* filter, giving 2,638
indistinguishable rows per GSM8K configuration. A dedupe keeping one row per
document can silently retain `strict-match` for one configuration and
`flexible-extract` for another — a difference of ~7 points — which fabricates
effects. `analysis/load.py` recovers the filters from lm-eval's filter-major
write order and then **proves** the recovery against the accuracy table lm-eval
itself printed for that run. All seven runs confirm to within 1e-4. The loader
raises rather than returning unverified data.

**The noise floor is a gate, not a statistic.** `analysis/interaction.py`
refuses to interpret the novel-damage fraction without a measured floor to
compare against.

## Layout

```
configs/    one YAML per precision setting + frozen eval and calibration setup
scripts/    environment check, quantize, serve, evaluate, benchmark, collect
analysis/   paired bootstrap, filter recovery, tables, figures, margins
router/     specification only -- not run
tests/      synthetic fixture and guards against silent failure
notes/      daily lab log: what was run, which commit, what surprised me
results/    raw output (never edited) + tidy derived tables
report/     the written report and its figures
```

## Hardware

FP8 W8A8 needs compute capability ≥ 8.9 (Ada or Hopper); the L40S is 8.9. On an
Ampere card it loads as weight-only W8A16 via FP8-Marlin — a *different*
experiment — and `scripts/common.py` refuses to build rather than let that
happen silently. INT8 W8A8 is not supported on Blackwell (cc ≥ 10.0).
Run `make check` on any new machine first.

## Honest limitations

See [LIMITATIONS.md](LIMITATIONS.md) and §6 of the report. The short version:
one model family, one GPU, greedy decoding only, long context capped at 20,480
tokens, the confidence analysis is MMLU-only, no AWQ arm, no routing result, and
no correction for multiplicity across the comparison family.

## Licence

MIT. See [LICENSE](LICENSE).
