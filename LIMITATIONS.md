# Limitations: what this study does not show

Written before the measurements, so that the limitations are not chosen after
seeing which results were inconvenient. Add to it as you go; do not remove
items because the numbers came out well.

## Scope

- **One model family at a time.** Phase 1 is Llama-3.1-8B-Instruct only.
  Mistral-7B-Instruct-v0.3 is a second phase and may be cut. Nothing here
  generalises to 70B models, MoE models, or base (non-instruct) models, and the
  cost of quantization is widely believed to differ with scale.
- **Benchmark prompts, not traffic.** GSM8K, MMLU, HumanEval, IFEval and
  LongBench are short, clean, and English. Real requests are messier, longer in
  the system prompt, and often multi-turn. A precision setting that is safe here
  may not be safe on production traffic.
- **One GPU model, one price, one date.** Every dollar figure is
  `price_per_hour / throughput`. A different card, a different provider, or a
  spot price changes the Pareto frontier without any model changing at all. The
  price used and the date it was taken are recorded in
  `configs/bench_grid.yaml`.

## Measurement

- **Accuracy is measured with one harness version and one prompt format.** The
  absolute numbers will not match published model-card numbers, because answer
  extraction and few-shot formatting differ. Only the *differences from the
  BF16 baseline*, measured with the identical setup, are meaningful here.
- **`add_bos_token=True` is set for every configuration, including BF16.**
  Quantized models are sensitive to the BOS token and lm-eval does not add one
  by default. Setting it only for quantized runs would have confounded the
  comparison; setting it for none would have understated quantized accuracy.
- **Greedy decoding only.** No temperature sweep, so nothing here says how
  precision interacts with sampling.
- **Token counts in the per-prompt logs are approximations**, not tokenizer
  counts. They are used for routing features and length bucketing, not for any
  figure in the results table.
- **Three repeats, median reported.** The spread across repeats is the system
  noise floor and is recorded; if it is large relative to a difference between
  configurations, that difference is not reported as real.

## Statistics

- Intervals are 95% paired bootstrap intervals on the change from BF16, over
  per-prompt correctness. They capture **prompt sampling variability only**.
  They do not capture: harness version differences, prompt-format sensitivity,
  non-deterministic kernels, or calibration-set variability.
- **Calibration variability is not measured.** Each calibrated checkpoint was
  built once, from one 512-sample draw. Rebuilding with a different seed would
  give a slightly different checkpoint, and that variance is not in any interval
  reported here.
- Many comparisons are made. No multiple-comparison correction is applied, so
  among many "no detectable difference" results some individually-significant
  ones will be chance. Treat a single significant task in isolation with
  suspicion.

## The router

- **It is an offline simulation replayed over logged benchmark outputs.** It
  does not show that the router would work on real traffic, and it does not
  show that it would work in a live serving system, where both configurations
  would have to be resident or swapped, which costs memory or latency that is
  not modelled here.
- Routing-time features are prompt length, prompt text, and optionally the task
  identity. **The task identity does not exist in real traffic**; results are
  reported both with and without it, and the number without it is the honest
  one.
- The cost of any confidence signal the router consults is charged on every
  request, including requests that are then sent to the expensive model.
- The "best operating point" is selected on the same data it is scored on and
  is therefore optimistic. A reference operating point chosen in advance is
  reported alongside it.
- Serving-level effects of mixing precisions — batching two model copies,
  cache contention, model-swap latency — are entirely unmodelled.

## Known gaps to state in the report

- *(fill in as they arise: tasks that could not be run, shapes that OOMed, a
  recipe that had to fall back to a published checkpoint, a card that forced a
  configuration to be cut)*
