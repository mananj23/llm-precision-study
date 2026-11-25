# The cost of precision: quality, throughput and per-request routing for 8B LLM serving

Manan Jindal · draft skeleton, 2026-11-16 · target 6–8 pages

> **How to use this file.** Sections marked ✅ are written and need only light
> revision. Sections marked 🔲 are placeholders, and each one names the exact
> pipeline output that fills it. Do not write a 🔲 section from expectation —
> if the file it names does not exist yet, the section stays empty.
>
> Rule for the whole document: **every numeric claim must be traceable to a
> cell in `analysis/results.csv` or a figure in `report/`.** If a sentence
> cannot be checked against one of those, delete it.

---

## 1. Introduction 🔲 (outline written, prose after Phase 1)

**Paragraph 1 — the practical question.** Serving an 8B instruct model, you
choose a numerical precision once and live with it. Lower precision shrinks
weights and the KV cache, which raises throughput and lowers cost per token.
What it costs in answer quality is reported inconsistently, usually as a single
average on one benchmark, against a baseline measured under a different prompt
format, with no statement of uncertainty.

**Paragraph 2 — what is already known.** State plainly that Kurtic et al. (ACL
2025) have already measured much of this across the Llama-3.1 family in vLLM at
very large scale, and summarise their headline findings. *Do not bury this.*
See `related_work.md` §5.

**Paragraph 3 — what this study adds.** Four things, in this order:
per-prompt paired intervals so that "no detectable difference" can be said and
meant; the KV cache treated as an axis independent of the weights; an honest
test of per-request precision routing against random mixing at matched cost;
and an independent single-GPU replication with every version, price and date
recorded.

**Paragraph 4 — summary of findings.** Write last. One sentence per
contribution, each with a number and an interval.

**Paragraph 5 — the negative results.** Say here, not in a footnote, which
hypotheses failed. Fill from `report/router_summary.json` and the
"no detectable difference" count printed by `analysis/build_results.py`.

## 2. Related work ✅

See `report/related_work.md` — written in full, ~1.5 pages. Four threads:
weight/activation quantization; KV-cache compression; reporting practice and
the per-prompt argument; routing and cascades. Then the positioning section,
which must be revisited once results exist.

## 3. Method ✅ (written; update only if a config changes)

### 3.1 Models and configurations

Llama-3.1-8B-Instruct, served with vLLM, in seven configurations spanning the
weight precision and the KV-cache dtype:

| Setting | Weights | KV cache | Built with |
|---|---|---|---|
| `w-bf16__kv-bf16` | BF16 | BF16 | baseline |
| `w-fp8__kv-bf16` | FP8 W8A8 | BF16 | llm-compressor `FP8_DYNAMIC` |
| `w-bf16__kv-fp8` | BF16 | FP8 | vLLM `--kv-cache-dtype fp8` |
| `w-fp8__kv-fp8` | FP8 W8A8 | FP8 | both |
| `w-int8__kv-bf16` | INT8 W8A8 | BF16 | SmoothQuant(0.8) + GPTQ |
| `w-int4gptq__kv-bf16` | INT4 W4A16 | BF16 | GPTQ, group 128, symmetric |
| `w-int4awq__kv-bf16` | INT4 W4A16 | BF16 | AWQ, group 128, symmetric |

Mistral-7B-Instruct-v0.3 is a second phase at BF16 and FP8 only, and is cut
first if the schedule slips.

Two design points deserve explicit mention in the paper:

*The third row exists to break a confound.* Every published FP8-KV result also
quantizes the weights, so the cache's contribution cannot be separated from the
weights'. Running BF16 weights with an FP8 cache isolates it.

*The two INT4 rows are matched deliberately.* The llm-compressor reference
examples default GPTQ to a symmetric 4-bit scheme and AWQ to an asymmetric one.
Taking both defaults would have compared algorithm *and* zero-point at once.
Both are run symmetric, at group size 128, on the same calibration set, so the
only difference is the algorithm. The asymmetric AWQ variant — which is what
most published AWQ checkpoints ship — is reported separately if time allows,
and labelled as such. (`scripts/w4a16_config.py`.)

### 3.2 Quantization procedure

Checkpoints are built with llm-compressor, following its reference recipes with
only the model name changed; no quantizer was written for this project.
FP8 uses `FP8_DYNAMIC` (static per-channel weights, dynamic per-token
activations) and needs no calibration. INT8 uses SmoothQuant at smoothing
strength 0.8 followed by GPTQ W8A8. Both INT4 variants use an explicit 4-bit
group-128 configuration.

Every calibrated method uses **one frozen calibration set**: 512 UltraChat
samples, shuffle seed 42, maximum sequence length 2048, the model's own chat
template applied, fixed on 2026-11-16 (`configs/calibration.yaml`). A different
calibration set per method would make the comparison meaningless. The set is
deliberately not drawn from any evaluation task, to avoid leaking the test
distribution into the weights.

Every checkpoint is verified before it is evaluated: files and quantization
config present, memory actually reduced by roughly the expected factor, and a
100-prompt GSM8K sanity check. A quantized model scoring near zero indicates a
broken checkpoint — wrong chat template, missing tokenizer files, failed
calibration — and is rebuilt, not reported.

### 3.3 Evaluation

Tasks: GSM8K (5-shot), MMLU on a fixed 10-subject subset (5-shot), HumanEval
(0-shot, executed in a container), IFEval, and a LongBench subset chosen to
overlap with Agrawal and Mayer (2026). Greedy decoding throughout
(temperature 0), a fixed seed, and settings frozen in `configs/eval.yaml`
before the baseline was run.

`add_bos_token=True` is set for **every** configuration including BF16.
Quantized models are sensitive to the presence of the BOS token and the
evaluation harness does not add one by default; setting it only for the
quantized runs would have confounded precision with tokenization, and setting
it for none would have understated quantized accuracy.

Per-prompt outputs and correctness flags are logged for every run
(`--log_samples`) and folded into a single tidy table, one row per
(model, configuration, task, prompt). This table is the input to both the
intervals and the router.

### 3.4 Intervals

Differences from BF16 are computed by a **paired bootstrap over per-prompt
correctness**: prompt indices are resampled with replacement 10,000 times and
the per-prompt difference is averaged within each resample, giving a 95%
interval on the change. Prompt difficulty is the dominant source of variance in
an accuracy estimate and it is shared between the two runs, so it cancels in
the per-prompt difference; estimating two accuracies independently and
subtracting discards that cancellation.

The size of the effect is worth stating, because it determines what this study
can conclude. On a simulated one-point regression over 2,000 prompts, the
paired interval is roughly seven times tighter than the unpaired alternative,
and the two procedures disagree about whether the effect exists at all: the
paired interval excludes zero, the unpaired one contains it. Reproducible with
`python analysis/bootstrap.py`.

Where an interval contains zero, the result is reported as **no detectable
difference** and the configurations involved are not ranked against each other.

### 3.5 System measurements

Each configuration is served alone on the GPU — the benchmark script refuses to
start if another process holds the card — and driven with vLLM's serving
benchmark over a grid of four concurrencies (1, 8, 32, 64) and three
input/output shapes (512/128, 4096/256, 16384/256). Each cell is run three
times with different seeds after a discarded warm-up, and the median is
reported with the spread retained as the system noise floor.

Benchmark servers run with prefix caching disabled. Repeating a benchmark
against the same server reuses prompts left in the prefix cache and inflates
throughput; this is a measurement artefact, not a property of the precision
setting.

Two memory figures are recorded, because they answer different questions: the
weights' size, and the KV-cache capacity in tokens from vLLM's startup log. The
second is the one that explains throughput differences, since a smaller cache
per request lets more requests run concurrently.

Cost is `price_per_hour / (output_tokens_per_second × 3600) × 10⁶` dollars per
million output tokens, quoted at a stated reference concurrency, with the GPU
price and the date it was taken recorded. Quoting a cost without the
concurrency is meaningless, since throughput per dollar varies by an order of
magnitude across the grid.

### 3.6 Routing

For a (cheap, expensive) pair, the cheap configuration is labelled
**sufficient** on a prompt when it is correct, *or* when the expensive one is
also wrong — in the second case nothing was lost by being cheap. Defining the
label as "the cheap one is correct" would penalise the router for prompts no
configuration can answer.

Features are restricted to what is known before generation: prompt length and
prompt text. Output length is not a feature; a guard rejects any feature drawn
from a post-hoc column. Task identity is included in one variant and excluded
in another, and both are reported, because real traffic does not arrive
labelled with its benchmark.

Logistic regression first, then gradient-boosted trees. Evaluation uses two
splits: grouped by prompt, and with an entire task held out. The second is the
one that shows whether the router generalises.

The comparison that matters is against **a random mix at matched cost**: if the
router sends a fraction *f* of prompts to the cheap configuration, the
benchmark is sending a random *f* of them. Beating "always expensive" on cost
and "always cheap" on quality demonstrates nothing, since any mixture does
that. An oracle with perfect foresight gives the upper bound and therefore the
headroom. If the router consults the cheap model for a confidence signal, that
run is charged on every prompt, including those escalated to the expensive
configuration.

Two operating points are reported: a reference threshold chosen in advance to
route about half the prompts cheap, and the best threshold on the sweep —
the latter selected on the data it is scored on, and labelled as optimistic.

## 4. Results: quality 🔲

**Fills from:** `analysis/results.csv`, `report/fig_pareto.png`.

- [ ] Table: change from BF16 in percentage points with 95% interval, rows =
      configurations, columns = tasks. Generated, not typed.
- [ ] Count of cells showing no detectable difference, out of the total.
- [ ] Where quality breaks first — which task, which configuration, and the
      interval that supports the claim.
- [ ] GPTQ vs AWQ at matched symmetry and group size: the controlled comparison.
- [ ] Comparison with Kurtic et al.: do FP8-lossless and INT8-at-1–3% replicate?
- [ ] Baseline rerun: does the repeat match, and what is the noise floor?

**Discipline:** for every sentence, find the interval that supports it. Delete
any sentence an interval does not support.

## 5. Results: system performance 🔲

**Fills from:** `analysis/system_summary.csv`.

- [ ] Throughput, TTFT, TPOT per configuration across the grid.
- [ ] KV-cache capacity in tokens per configuration, and the argument that it
      explains the throughput ordering at high concurrency.
- [ ] Which setting is fastest at concurrency 1 and which at 64 — one sentence
      each, as required by the plan.
- [ ] Shapes that failed to run, and at what concurrency. **A configuration
      that cannot hold 64 × 16k is a result; report it, do not omit the row.**
- [ ] Spread across the three repeats, as the system noise floor.

## 6. Results: the quality–cost frontier 🔲

**Fills from:** `report/fig_pareto.png` and the frontier comparison printed by
`analysis/plot_pareto.py`.

- [ ] The Pareto frontier for the average over tasks, and per task.
- [ ] **Is the frontier the same in every panel?** If yes, one fixed setting
      wins everywhere and the router has little to exploit — say so, and say it
      before Section 8 rather than after. If no, name the tasks that disagree;
      that disagreement is the router's motivation.
- [ ] Cost per correct answer as a headline metric.
- [ ] Price sensitivity: re-run `build_results.py --price-per-hour` at a
      different GPU price and state whether the frontier changes. If it does,
      every cost conclusion is conditional on one rental price on one date.

## 7. Results: long context 🔲

**Fills from:** `report/fig_longcontext.png`.

- [ ] Accuracy against input length, per KV setting, with intervals.
- [ ] Do the FP8-KV and BF16-KV bands separate anywhere? Only claim a
      long-context cost if they do.
- [ ] Does any gap widen with length? That is the KIVI-motivated prediction.
- [ ] Confirm no prompts were silently truncated at `max_model_len` —
      truncation looks exactly like a quality drop.

## 8. Results: routing 🔲

**Fills from:** `report/fig_router.png`, `report/fig_router_no_task_flag.png`,
`report/router_summary.json`.

- [ ] AUC for predicting sufficiency, by prompt split and held-out task.
- [ ] Router vs random mix at matched cost, at the reference operating point.
- [ ] Gap to the oracle: how much headroom existed in principle.
- [ ] With and without the task-identity feature. **Lead with the number
      without it.**
- [ ] Consider reporting CPT and APGR (RouteLLM's metrics) for legibility.
- [ ] The wording constraint, stated in the paper and not only here: this is an
      offline simulation replayed over logged benchmark outputs. It does not
      show the router would work on live traffic or in a serving system.

## 9. How much can a benchmark of this size resolve? ✅ (method done, numbers pending)

**Fills from:** `analysis/power.csv`, `report/fig_power.png` (`make power`).

**The method here is not new, and this section must open by saying so.**
Kotawala (arXiv:2605.30315, ICML 2026) derives this exact resolution framework
-- MDE, required paired N*, resolution ratio -- and applies it to public
leaderboards. Miller (arXiv:2411.00640) had already argued for paired,
question-level inference in evals. Claiming novelty here would be found out in
minutes.

What this project adds is narrow and worth stating as such: applying
resolution diagnostics to the **quantization** literature specifically, using
discordance rates measured from real quantized checkpoints rather than assumed
values. Nobody has asked whether the 1-3 point FP8/INT8/INT4 effects are
resolvable on the benchmarks they are reported on.

For paired binary outcomes, only the prompts where the two configurations
*disagree* carry information. Writing `r` for that discordance rate and `d`
for the effect, the standard error of the measured difference is
`sqrt((r − d²)/n)`, so the smallest detectable effect is

    MDE = 1.96 · sqrt(r / (n + 1.96²))

The consequence is counter-intuitive and worth stating plainly: **the sample
size you need depends on how often the two configurations disagree, not on how
accurate either one is.** Two settings that agree on 95% of prompts separate
with far fewer prompts than two that disagree constantly.

Applying it to standard benchmarks at an 8% discordance rate:

| Benchmark | n | MDE | Power at 1 pt | at 3 pts |
|---|---|---|---|---|
| HumanEval | 164 | 4.3 pts | 7% | 28% |
| IFEval | 541 | 2.4 pts | 13% | 70% |
| GSM8K | 1,319 | 1.5 pts | **25%** | 97% |
| MMLU (full) | 14,042 | 0.5 pts | 99% | 100% |

Detecting a 1-point regression with 80% power needs about **6,300 prompts**.

- [ ] Replace the assumed 8% with your measured discordance (`make power`)
- [ ] State, for each cell in your results table, whether the effect was
      resolvable at that benchmark's size
- [ ] The claim to make, carefully: effects of the size this literature
      reports sit near the resolution limit of the benchmarks it reports them
      on, so a point estimate without an interval cannot distinguish a real
      1-point regression from none. **Say this about method, not about any
      individual paper's conclusions being wrong.**
- [ ] Verify the formula still passes `python analysis/power.py` (it validates
      itself against simulation before printing anything)

## 10. Limitations 🔲

Import `LIMITATIONS.md` and add whatever went wrong: tasks that could not be
run, recipes that fell back to a published checkpoint, configurations cut
because the card did not support them.

## 11. Conclusion 🔲

Three sentences. What was safe to quantize, where quality broke, what it saved.
Then one sentence on whether per-request routing was worth it.

---

## Pre-submission checklist

- [ ] Every number traceable to `analysis/results.csv` or a figure
- [ ] Every claim has an interval; sentences without one deleted
- [ ] "No detectable difference" used wherever an interval contains zero
- [ ] No configuration ranked against another with overlapping intervals
- [ ] Kurtic et al. acknowledged in the **first** paragraph of the introduction
- [ ] GPU model, price and date stated
- [ ] Library versions pinned and committed
- [ ] Negative results in the introduction, not hidden in Section 8
- [ ] Table and figures regenerate with `make all` from the raw files
- [ ] `LIMITATIONS.md` has nothing removed because it became inconvenient
- [ ] Sent to Prof. Sayak Ray Chowdhury or Prof. Preeti Malakar in the week of
      11 Jan with the question: *which claim here is least supported?*
      Then fix that one before finalising.
