# Positioning: what this project is, and what it is not

Written 2026-11-25, after a deliberate search for prior work on every angle
this project could claim. Read this before writing the SOP or the 6 Dec note.

## 1. The novelty question, answered

**There is no open novelty gap in the directions we considered.** Each angle
has 2026 work on it:

| Angle we considered | Prior work that closes it |
|---|---|
| Paired bootstrap / MDE for quantization benchmarks | arXiv 2605.28873, *Pre-Registering the Detectable Effect: A Paired-MDE Budget for 4-bit Quantization Benchmarks* — derives δ* ≤ (z_{1-α/2}+z_{1-β})·√(ρ_d/m) from Miettinen (1968), exactly the formula in `analysis/power.py` |
| MDE is a property of a *contrast*, not of n | arXiv 2611.07329 §MDE; arXiv 2611.09807; arXiv 2611.13843 (PolyRoute) §7.2 |
| Sample complexity of LLM eval, MDE atlas across benchmarks | "The Sample Complexity of LLM Evaluation" (tmls.nyc) |
| Flips / discordance under quantization | arXiv 2607.08734 (*Illusion of Equivalency*), 2611.27155, 2607.04302 (PFR / Quant-Hurts / Quant-Helps), 2609.04748 |
| *Predicting* flips from decision margins | arXiv 2608.06564 — margin shrinkage predicts flip rate to a median of 1.0 pp across 16 models |
| Quantization effect vs. the engine/restart/batch noise floor | arXiv 2611.15671 — paired flip-rate differences against a measured noise floor, bootstrap CIs, fixed record set |
| Per-prompt routing between precisions | GitHub `quant-precision-router` (7-rung GGUF ladder, learned required-precision predictor, found routing does not beat the best single rung); arXiv 2609.12550; 2609.24322 (routing 25% of inputs recovers 85–93% of lost accuracy); DP-LLM; MoBiQuant; AutoEMatic |
| Disagreement between two precisions as a routing signal | arXiv 2611.16478, *One Model, Two Precisions* |
| Oracle routing headroom ≫ what a learned router recovers | arXiv 2609.32996; arXiv 2605.09618 (oracle +14 pp, scouting recovers 11–19%); arXiv 2606.07587, *The Routing Plateau* |
| "You can only exploit what you cannot measure" duality | Covered from both sides: 2611.16478 states the trade-off is "visible only from one side: the signal is best where the quantization damage is small"; *The Routing Plateau* decomposes disagreement into symmetric exchanged wins, which is the (r, d) decomposition |

I claimed novelty twice in this project and was wrong twice. The third check
says the same thing. **Stop looking for a gap here.** Any claim of the form
"we are the first to..." in this area will be found false by a reviewer in
one search, and that is a worse outcome than claiming nothing.

## 2. Why that does not make this a bad project

Admissions committees do not score novelty. They score *demonstrated research
capability*, and they read it through two channels:

- **The SOP.** Harvard SEAS: describe the research projects you undertook, the
  challenges you faced, and what you learned — not your early interest in
  computers. NYU: the most important factor is the applicant's likely ability
  to succeed in original research, and the committee pays significant
  attention to the SOP and to letters that speak to research contributions.
- **The letter.** Cornell: the level of detail a recommender can provide
  matters more than their reputation. Brown: research experience is not
  required but boosts chances considerably — tell us about the project and ask
  a recommender to comment on it.

Also worth knowing (UCSD CS PhD info session): a reviewer will look at
anything you *highlight* in the application, but will not go and find your
GitHub. The repo only counts if the SOP links it and says what is in it.

So the asset is not a novel finding. The asset is:

1. A question stated before the data existed, with a pre-registered analysis.
2. A controlled matrix, including the cell that isolates the confound.
3. Honest negatives, reported with a power statement.
4. A professor who can write "he designed and ran this himself."

Every one of those survives the fact that the topic is crowded.

## 3. The actual positioning

Not: *"a novel method for X."*

Instead: **a controlled measurement study of the quality/throughput/cost
surface of 8B serving precisions, on one machine, with one calibration set,
pre-registered, with the statistical resolution of the comparison reported
alongside every result.**

What is defensibly *ours* in that sentence:

- **The matrix, not any single cell.** Most published work takes one slice:
  flips, or routing, or the noise floor, or MDE. The seven-config matrix here
  measures quality *and* throughput *and* cost *and* discordance *and* routing
  headroom on the same checkpoints, on the same hardware, in one window.
  Nobody publishes that combination because it is expensive and unglamorous —
  which is exactly why doing it is evidence of capability.
- **The isolating cell.** `w-bf16__kv-fp8` separates the FP8 KV-cache effect
  from the FP8-weights effect. Most comparisons confound the two, because
  every other FP8-KV configuration also has quantized weights.
- **Reporting the resolution.** Our GSM8K n=1319 has ~25% power to detect a
  1-point effect; HumanEval n=164 has ~7%; 80% power at 1 point needs ~6,272
  prompts. Papers that report "FP8 loses 0.3 points" without this are making
  a claim their design cannot support. We will say so about our own numbers.
- **Reproducibility.** Frozen calibration set (512 UltraChat, seed 42), raw
  per-prompt JSONL released, provenance on every artifact, a tarball that
  reproduces every figure from the raw logs.

## 4. Publication target: be realistic

- **Not** a main-conference paper with a novelty claim. It would be rejected,
  correctly, on related work.
- **Realistic and worth doing:** an efficiency/efficient-NLP workshop, an
  MLSys-style measurement or experience track, or a reproducibility track.
  Workshop papers are reviewed, citable, and above the bar for an MS
  application. They are a reasonable signal for PhD too.
- **Also real:** a well-written technical report on arXiv plus the repo. An
  arXiv preprint is not peer review and should never be described as if it
  were, but it is a legitimate way to make work readable and citable, and the
  SOP can link it.
- **The honest framing for a reviewer:** "we replicate and extend the flip and
  MDE literature on a single controlled platform, and report where our
  measurement cannot resolve the effects that platform-level papers claim."
  That is a contribution of a genre that exists and gets accepted.

## 5. What this means for the 29-hour window

Do not spend GPU hours chasing a novel result. Spend them on **completeness of
the matrix**, because the matrix is the asset:

Priority order (already encoded in `scripts/run_all.sh`):

1. All seven quality evals on GSM8K — this is the paired-bootstrap core.
2. The benchmark grid on the four most important configs (bf16/bf16,
   fp8/bf16, bf16/fp8, int4gptq/bf16) — this is the Pareto figure.
3. HumanEval and the second task, with the power caveat stated.
4. The long-context shape, and the KV-cache caps where it fails. A cell that
   does not fit is a result, not a gap.
5. Routing headroom from the logs already collected. Offline only, labelled
   offline.

Anything not finished is a limitation in `LIMITATIONS.md`, which is a normal
part of a paper, not a failure.

## 6. One cheap thing that raises the ceiling

**Pre-register before the GPU window opens.** Push the design — the matrix,
the frozen calibration set, the metrics, the power analysis, and the decision
rules — to the public repo, with a timestamped commit, before any result
exists. Then no reviewer and no admissions reader can wonder whether the
analysis was chosen after seeing the data, and the SOP can say "I
pre-registered the analysis." It costs one commit and it is the single
clearest signal of research training in the whole project.
