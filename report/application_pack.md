---
title: "LLM Precision Study — Application Reference Pack"
subtitle: "Source material for resume, SOP and interviews · Manan Jindal · November 2026"
---

# How to use this document

This is not the technical report. It is the **source material** for writing about
the project — resume bullets, SOP paragraphs, short-form answers, and the facts
behind each of them.

Every number here is taken from `report/tables.json` and `report/margins.json`,
which are generated from the measurement data rather than typed. If you edit a
sentence, do not edit a number. If a number here disagrees with the report, the
report is right and this document is stale.

**The single rule for everything below:** nothing claims the work is published,
peer-reviewed, novel, or state-of-the-art, because none of that is true. It does
not need to be. A committee that catches one overstatement discounts everything
else in the application.

Repository: `github.com/mananj23/llm-precision-study`
Technical report: `report/report.pdf` in that repository (12 pages)

---

# 1. The project in one sentence

Pick the register that fits the space:

**One line.**
: A controlled study of how reduced numerical precision changes what a language
  model actually answers — not just what it scores.

**Two lines.**
: I measured six weight and KV-cache precision configurations of
  Llama-3.1-8B-Instruct on three benchmarks, recording every individual prompt's
  outcome rather than only the aggregate. The headline: configurations that score
  identically answer a sixth of their prompts differently, and the differences
  land almost entirely where the model was least certain.

**For someone outside ML.**
: Running a large language model cheaply means storing its numbers at lower
  precision. Everyone reports what that does to a benchmark score. I measured
  what it does to the individual answers, and found that a model can keep its
  score while changing one answer in seven.

---

# 2. What was actually done

Facts, for when you need to be concrete.

| | |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Hardware | 1 × NVIDIA L40S (48 GB), rented, ~₹2,700 of compute consumed |
| Serving | vLLM 0.32.0, PyTorch 2.13.0+cu129 |
| Evaluation | lm-evaluation-harness 0.4.13, greedy decoding |
| Quantization | llm-compressor: FP8 dynamic, SmoothQuant+GPTQ W8A8, GPTQ W4A16 |
| Configurations | 6: BF16 baseline, FP8 W, FP8 KV, FP8 W + FP8 KV, INT8 W8A8, INT4 GPTQ |
| Quality data | GSM8K (n=1,319), MMLU (n=14,042), IFEval (n=541), LongBench (n=4,750, to 20,480 tokens), per-prompt |
| Throughput data | 216 cells = 6 configs × 3 shapes × 4 concurrencies × 3 seeds |
| Statistics | Paired bootstrap, 10,000 resamples, per-prompt pairing |
| Output | Public repository, per-prompt data released, 12-page technical report |

**Design choices worth mentioning**, because they are what distinguish this from
a benchmarking exercise:

- The configuration grid, tasks, sample sizes and statistical test were
  **written down and committed before the GPU was rented**. The effects turned
  out to be small enough that an analysis chosen after seeing the data could
  have manufactured a finding.
- The first four configurations form a **2 × 2 factorial design** in weight
  precision × KV precision. That is what makes the composition question in
  §3.4 below answerable at all.
- The analysis code **verifies its own inputs** against the evaluation harness
  before reporting anything, and refuses to return unverified data.

---

# 3. The findings

Each stated as precisely as the data supports, with what can and cannot be
claimed. Use the "say this" line; avoid the "don't say" line.

## 3.1 The measurement floor is exactly zero

The BF16 baseline was evaluated twice, **17 hours and 22 minutes apart**, on the
same GPU, with a day of other work in between. All 1,319 extracted answers were
byte-identical; all 1,319 correctness outcomes agreed. Disagreement: **0.000%**
on both measures.

**Say this:** "I measured the noise floor before trusting any comparison, and it
was zero, so every difference I report is attributable to precision rather than
to run-to-run variation."

**Don't say:** that evaluation is deterministic in general. It is deterministic
*in this configuration* — greedy decoding, fixed seed, prefix caching off,
one job on the GPU. Under sampling or with caching enabled it would not be.

**Why it matters:** this is the step most quantization papers skip. It costs one
repeated evaluation and it is what licenses every sub-point claim.

## 3.2 A stable score hides a large change in behaviour

FP8 weights with an FP8 KV cache: **70.43%** on GSM8K against the baseline's
**70.51%** — one question in 1,319 — while **15.24% of the answers changed**.
The aggregate holds still because ~101 prompts break and ~100 are repaired.

Churn across configurations:

| configuration | GSM8K changed | net Δ | MMLU changed | net Δ | IFEval changed | net Δ |
|---|---:|---:|---:|---:|---:|---:|
| FP8 W8A8 | 14.33% | −0.23 | 4.07% | −0.01 | 13.31% | −1.48 |
| BF16 + FP8 KV | 11.07% | −0.91 | 3.77% | −0.28 | 11.83% | −1.11 |
| FP8 + FP8 KV | 15.24% | −0.08 | 5.93% | −0.56 | 13.31% | +0.74 |
| INT8 W8A8 | 13.50% | −0.30 | 4.39% | −0.54 | 16.08% | −0.92 |
| INT4 GPTQ | 21.99% | +1.67 | 9.43% | −1.99 | 15.90% | −4.07 |

**Say this:** "'Accuracy preserved' and 'behaviour preserved' are different
claims, and only the first is normally tested."

**Don't say:** that the quantized model is worse. On GSM8K it isn't — the
interval contains zero. It is *different*.

## 3.3 The change lands almost entirely on the model's near-ties

MMLU is multiple-choice, so the per-choice log-likelihoods were logged and a
**confidence margin** is recoverable for all 14,042 documents. Binning the 9,583
prompts the baseline answers correctly:

| baseline margin (nats) | n | damaged by FP8 W | by FP8 KV | by both |
|---|---:|---:|---:|---:|
| < 0.63 | 1,543 | 17.3% | 17.4% | 24.6% |
| 0.63 – 1.75 | 1,643 | 1.2% | 1.0% | 4.1% |
| 1.75 – 2.88 | 1,602 | 0.1% | 0.0% | 0.4% |
| > 2.88 | 4,795 | 0.0% | 0.0% | ~0.0% |

The least-confident **25%** of prompts carry **98.6%** of all FP8-weight damage,
**98.9%** of FP8-KV damage and **95.8%** of the combination's. Above 1.75 nats —
6,397 prompts, two-thirds of the set — FP8 weights damaged **1** prompt, the FP8
KV cache damaged **0**, the combination damaged **7**.

**Say this:** "Quantization doesn't degrade confident predictions here; it
reshuffles the model's coin-flips. Whether that makes it free or expensive
depends on where your traffic sits on that distribution — and an aggregate
benchmark score averages over both, so it cannot tell you."

**Don't say:** that this proves quantization is safe. It is MMLU-only, and the
margin between top-two log-likelihoods is a proxy for confidence, not calibrated
probability. A model can be confidently wrong.

**This is the strongest thing to lead with in an interview**, because it is the
one finding that is both surprising and immediately actionable.

## 3.4 Two quantizations compose — and the analysis that said otherwise was mine

This is the result to lead with if someone asks whether you can be trusted with
a negative finding, because it is a correction to my own earlier conclusion.

**What I first measured.** Define *damage* as a prompt the baseline answers
correctly and a variant does not. FP8 weights alone damage 10.32% of
baseline-correct GSM8K prompts; an FP8 KV cache alone damages 8.49%;
independence predicts 17.94%; the combination damages **10.86%**. Strongly
sub-additive, and resolved on all three tasks it could be run on:

| | GSM8K | MMLU | IFEval |
|---|---:|---:|---:|
| excess over independence | −7.08 pp | −1.13 pp | −13.88 pp |
| 95% CI | [−9.54, −4.48] | [−1.65, −0.63] | [−19.25, −8.21] |

**Why that was wrong.** The damage set conditions on the baseline being correct
and counts only losses, discarding the recoveries — which are comparable in
size. The standard symmetric contrast for a 2 × 2 factorial,
*I* = score_C − score_W − score_K + score_base, makes no such choice:

| task | n | interaction | 95% CI | resolved |
|---|---:|---:|---|---|
| GSM8K | 1,319 | +1.06 | [−1.74, +3.87] | no |
| MMLU | 14,042 | −0.26 | [−0.74, +0.21] | no |
| IFEval | 541 | +3.33 | [−1.11, +7.76] | no |
| LongBench | 4,750 | −0.04 | [−0.62, +0.56] | no |

**Zero of four resolved.** The two effects compose additively as far as this
experiment can detect. Conditioning on baseline-correct selects prompts where
the baseline was lucky; regression to the mean then shows up as "damage" from
one component and "repair" from the second. §3.3 confirms the mechanism
independently — the damage sits at the decision boundary, which is precisely
where regression operates.

**Say this:** "I found what looked like a clean, replicated, statistically
resolved interaction between two quantization schemes. Then I ran the symmetric
contrast and it vanished on all four tasks. The original result was an artifact
of conditioning on baseline-correct prompts, and I report both framings because
the disagreement between them is the more useful finding — any metric that
conditions on a baseline outcome inherits regression to the mean."

**Don't say:** that quantization effects don't compose. On the aggregate they do.

**What survives:** the aggregate composes; the *set of prompts affected* does
not. Component measurements predict the score of a stacked configuration. They
do not predict which inputs change behaviour.

**Why this is worth leading with:** it is evidence you will run the test that
can kill your own result. That is rarer and more valuable than the result.

## 3.5 4-bit trades instruction-following for pattern-copying

Two resolved effects pointing opposite ways:

| | BF16 | INT4 GPTQ | Δ | 95% CI |
|---|---:|---:|---:|---|
| GSM8K format-miss (right answer, wrong format) | 7.35% | **0.91%** | −6.44 pp | [−7.88, −5.00] |
| IFEval (follows a *prose* instruction) | 47.32% | **43.25%** | −4.07 pp | [−7.39, −0.74] |

INT4 almost never fails to reproduce the answer format its five in-context
exemplars demonstrate, while getting measurably worse at obeying an instruction
stated in words. Every 8-bit scheme is indistinguishable from BF16 on both.

**Say this:** "The two measurements are consistent with 4-bit quantization
shifting the model toward copying the surface form in its context and away from
executing an instruction stated in prose."

**Don't say:** that this is demonstrated. It is an interpretation of two
resolved effects. The clean test — vary the number of exemplars and watch the
format-miss rate respond — was not run.

**Consequence worth stating:** this is *why* INT4's GSM8K delta is +1.67 under
`strict-match` and −4.78 under `flexible-extract` from identical outputs. Strict
extraction rewards exactly the format-copying 4-bit gains. A paper reporting one
filter without naming it tells a different story about the same run.

## 3.5b Long context: the biggest churn, and a failed prediction

LongBench, 4,750 documents across 21 subtasks, up to 20,480 tokens, on the four
configurations that isolate the KV contrast.

| configuration | mean score | Δ vs BF16 | 95% CI | resolved | docs changed | mean abs. change |
|---|---:|---:|---|---|---:|---:|
| BF16 / BF16 | 43.64% | — | — | — | — | — |
| FP8 W8A8 | 43.97% | +0.32 | [−0.12, +0.78] | no | 42.67% | 5.15 |
| BF16 + FP8 KV | 43.63% | −0.01 | [−0.40, +0.37] | no | 39.31% | 4.13 |
| FP8 + FP8 KV | 43.92% | +0.27 | [−0.22, +0.76] | no | 46.42% | 6.00 |

**The failed prediction is worth telling.** A quantized KV cache stores every
past token, so its error should compound with context length — long context is
where it should hurt most. At up to 20,480 tokens the FP8 KV cache costs
**−0.01 pp**, the smallest effect of any configuration on any task in the study.

**Say this:** "I expected the KV cache to be the component that degraded at long
context, because its error should compound. It didn't — it was the smallest
effect I measured anywhere. I report it because a prediction that fails is still
a measurement, and it's the kind that usually doesn't get written up."

**And the churn is the largest in the study:** 39–46% of documents score
differently, mean absolute change 4–6 points, with every aggregate difference
indistinguishable from zero.

**Caveat to state yourself:** LongBench's metric is continuous, so any change
registers — a one-token difference in a summary moves ROUGE. The mean absolute
change is the figure to quote, not "46% changed".

## 3.6 Throughput

| configuration | peak output tok/s | speedup | KV cache tokens |
|---|---:|---:|---:|
| FP8 W8A8 / FP8 KV | 1,909.7 | 1.52× | 491,904 |
| INT4 GPTQ | 1,761.0 | 1.40× | 269,584 |
| INT8 W8A8 | 1,752.9 | 1.40× | 242,336 |
| BF16 / BF16 | 1,256.5 | 1.00× | 191,264 |

A detail worth having ready: Llama-3.1-8B uses grouped-query attention with 8 KV
heads of dimension 128 over 32 layers, so one 16-bit KV token is
2 × 32 × 8 × 128 × 2 = **128 KiB**. Predicting the cache from reported free
memory gave ≈191,000 tokens against vLLM's 191,264 — **under 1% error**. That
confirms the serving stack was configured as intended.

**Note the inversion:** INT4 gives *less* throughput than FP8/FP8 on an L40S
while costing 1.99 pp on MMLU — dominated on both axes, because this card has
hardware FP8 and W4A16 must dequantize to compute. Its advantage is memory
footprint, on smaller cards, which this grid does not measure.

---

# 4. Resume entry

## Full version

> **LLM Inference Precision — Independent Research** · Oct–Nov 2026
> `github.com/mananj23/llm-precision-study`
>
> - Designed and ran a pre-registered controlled study of 6 weight/KV-cache precision configurations for Llama-3.1-8B-Instruct on one L40S, evaluating GSM8K (n=1,319), MMLU (n=14,042) and IFEval (n=541) with per-prompt outcomes and a 216-cell throughput grid.
> - Established a zero measurement floor — two baseline evaluations 17 hours apart produced byte-identical answers on all 1,319 prompts — making sub-point effects attributable to precision rather than run-to-run variance.
> - Showed FP8 weights + FP8 KV cache changes 15.2% of GSM8K answers while moving aggregate accuracy by one question in 1,319; on MMLU the least-confident 25% of prompts carry 98.6% of all damage, and only 1 of 6,397 high-confidence prompts is affected.
> - Showed that a natural "damage set" analysis reports a strongly sub-additive interaction between two quantizations, while the symmetric 2x2 contrast finds no resolved interaction on any of four tasks -- identifying the first as a regression-to-the-mean artifact of conditioning on baseline-correct prompts, and corrected my own earlier conclusion.
> - Released the full pipeline, per-prompt data and a 12-page technical report; analysis code verifies its own inputs against the evaluation harness before reporting.

## Three-bullet version, if space is tight

Keep bullets 1, 2 and 4. The zero-floor bullet is the one hardest to fake and
the one a technical reader will notice.

## One-line version, for a projects list

> **LLM Inference Precision** — controlled study of 6 quantization configurations
> (3 benchmarks, 216 throughput cells, per-prompt data released); showed
> score-preserving configurations change 15% of answers, concentrated at the
> model's decision boundary. `github.com/mananj23/llm-precision-study`

---

# 5. SOP material

## 5.1 The project paragraph

Replaces any sentence saying the project is planned or has no results yet.

> Over October and November I ran a controlled study of how reduced numerical
> precision changes a language model's behaviour, and the result surprised me. I
> served six weight and KV-cache precision configurations of
> Llama-3.1-8B-Instruct on a single rented L40S and evaluated each on GSM8K, MMLU
> and IFEval, recording the outcome of every individual prompt rather than only
> the aggregate score. Before trusting any comparison I evaluated the baseline
> twice, seventeen hours apart, and confirmed that all 1,319 answers were
> identical — a measurement floor of exactly zero, which most published
> quantization results assert rather than demonstrate. Against that floor, FP8
> weights with an FP8 KV cache scored within one question of the full-precision
> baseline while changing 15.2% of its answers. The aggregate was stable because
> roughly a hundred prompts broke and roughly a hundred were repaired. When I
> recovered the model's per-choice confidence from the evaluation logs, the churn
> turned out to be almost perfectly concentrated at the decision boundary: the
> least-confident quarter of prompts carried 98.6% of the damage, and across the
> two-thirds of prompts the model was confident about, quantization changed one
> answer in 6,397. What I took from this is that a benchmark score is a lossy
> summary of a system's behaviour, and that the loss is largest exactly where we
> are least inclined to look.

## 5.2 The gap paragraph

> That project taught me what I can do alone and where I stop. I could design the
> controls, measure the floor, and run the test that killed my own result: the
> cleanest finding I had — a replicated, statistically resolved interaction
> between two quantization schemes — turned out to be an artifact of how I had
> defined damage, and I found that by checking rather than by being told. What I
> could not do was situate the result: I do not yet have the grounding in systems
> and machine learning to know which of these observations is worth a year and
> which is a footnote, and I was working on one model family on one GPU because
> that is what I could afford. An MS is where I would learn to choose problems
> rather than only to execute them carefully.

## 5.3 The connecting sentence

Your analog IC work is about operating usefully with limited precision and noise.
This project is the same trade in digital inference. If the SOP currently reads
as three unrelated interests, this is the sentence that makes it one:

> What draws me to both analog circuit synthesis and low-precision inference is
> the same question in two materials: how much exactness can you give up before
> the behaviour, and not just the headline number, stops being the thing you
> designed.

Place it where you introduce the project, or where you pivot from the analog
work to the ML work.

## 5.4 A shorter variant, if you have ~120 words

> I spent October and November measuring what low-precision inference does to a
> language model's individual answers rather than to its benchmark score. Six
> precision configurations of Llama-3.1-8B, three benchmarks, every prompt's
> outcome recorded. I verified first that repeating an evaluation changed nothing
> at all — a zero noise floor — which let me trust sub-point differences. The
> result I did not expect: a configuration that scores within one question of the
> baseline changes 15% of its answers, and almost all of that change falls on
> prompts the model was nearly indifferent about. It taught me that an aggregate
> score is a lossy summary, and that the loss hides where we least look.

---

# 6. Short-form answers

Applications and forms often ask for these.

**"Describe your most significant technical project" (50 words)**

> I ran a controlled study of six low-precision configurations of a language
> model, measuring per-prompt outcomes rather than benchmark scores. I found that
> configurations scoring identically answer 15% of prompts differently, and that
> the differences concentrate almost entirely where the model was least
> confident. Code and data are public.

**"What research question interests you?" (50 words)**

> Aggregate benchmark scores are lossy summaries of model behaviour, and the loss
> is not random — in my own measurements it concentrates at the decision
> boundary. I want to work on evaluation that reports what a system does, not
> only what it scores, especially for systems modified after training.

**"What did you learn?" (40 words)**

> That the discipline is in the controls, not the result. Measuring the noise
> floor and pre-registering the analysis cost one extra evaluation each and are
> the only reasons any of my findings are interpretable.

---

# 7. Likely questions, and honest answers

**"Isn't this just benchmarking?"**
No — benchmarking reports scores. This records per-prompt outcomes, which is what
makes the churn, the composition structure and the confidence concentration
visible. None of them is recoverable from a score table.

**"Why should I believe a 0.5-point difference?"**
Because I measured what repeating an identical run does first, and it was zero —
not small, zero, on all 1,319 prompts. That is the only reason sub-point claims
are interpretable, and it is the step the literature usually skips.

**"What's novel here?"**
Not the quantization schemes — those are standard and I used off-the-shelf
implementations. What I'd defend is the measurement design: the zero floor, the
2 × 2 that makes composition testable, and conditioning the damage on the model's
own confidence. I'd call it careful rather than novel.

**"What would you do with more resources?"**
Replicate on a second model family — that's the biggest weakness and it's cheap.
Then measure long-context behaviour, where an FP8 KV cache should matter most and
where I have no data. Then the exemplar-count experiment, to test the
pattern-copying interpretation properly.

**"What went wrong?"**
Several things worth owning. A collection bug mixed two evaluation filters
together and would have flipped a result's sign by three points; I caught it and
rewrote the loader to verify itself against the harness's own output before
returning anything. I also ran the whole grid on a GPU twice while another job
held it, and lost a run 31 cells into 36. The second time I wrote a gate that
refuses to start if the GPU is occupied.

**"Is it published?"**
No. It's a public repository with a technical report. I'd want a second model
family before submitting it anywhere.

---

# 8. Guardrails

Things that are **not** true and must not appear anywhere:

- Not published, not peer-reviewed, not submitted, not under review.
- Not novel, not state-of-the-art, not a new method. The quantization schemes are
  standard; the contribution is measurement design.
- Not "proves" anything. The confidence analysis is MMLU-only. The
  pattern-copying account in §3.5 is an interpretation of two measurements, not a
  third measurement.
- Not generalizable across models — one model family, one GPU. Say so before
  anyone asks.
- Do not describe the GSM8K differences as degradation. Those intervals contain
  zero.
- **Do not say the quantization effects fail to compose.** On the aggregate they
  compose additively — zero of four tasks show a resolved interaction. The
  non-compositionality is in *which prompts change*, not in the score. An earlier
  draft of this project claimed otherwise; the corrected claim is §3.4.
- Do not round a confidence interval away. "−0.56 pp, 95% CI [−0.96, −0.15]" is
  the claim; "about half a point worse" is not.

**If the Qwen2.5-7B replication completes before you submit**, one phrase
upgrades: "on Llama-3.1-8B" becomes "replicated across two model families." Do
not write that sentence until the run has actually finished and the numbers are
in the repository.
