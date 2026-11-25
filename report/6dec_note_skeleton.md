# Early results: FP8 vs BF16 serving of Llama-3.1-8B-Instruct (vLLM)

> One page, due Sunday 6 Dec. Fill the brackets **only from your own results**.
> If a bracket cannot be filled from a number you measured, delete the sentence
> rather than estimating it. Draft by Fri 4 Dec so there is a day to fix it.

**Question.** How much quality does FP8 cost, and what does it save?

**Setup.** [GPU], vLLM [version], greedy decoding, [k]-shot, `add_bos_token=True`
for every configuration including the baseline, library versions pinned in the
repo. Calibration set frozen at 512 UltraChat samples, seed 42 (unused for FP8
dynamic, which needs none).

**Quality.** Change from BF16, 95% paired bootstrap over per-prompt correctness:

- [task 1]: BF16 [x]%, FP8 [y]%, difference [d] points (95% CI [a, b])
- [task 2]: BF16 [x]%, FP8 [y]%, difference [d] points (95% CI [a, b])

> If an interval contains zero, write "no detectable difference" and do not
> rank the two settings.

**System.** Weights [GB] vs [GB]; KV-cache capacity [tokens] vs [tokens];
throughput at concurrency [c]: [t1] vs [t2] tokens/s; cost per million output
tokens $[c1] vs $[c2] at $[price]/hr on [date].

**What this suggests.** [One sentence. No more than the intervals support.]

**What I have not tested.** [Other tasks, long context, other models, real
traffic, calibration variability.]

**Method note worth one line.** Intervals are paired over per-prompt
correctness. On a simulated 1-point regression the unpaired alternative reports
no detectable difference; the paired interval excludes zero. That is why every
number here is a paired difference rather than a difference of two accuracies.

**Next.** INT8, INT4 (GPTQ and AWQ), FP8 KV cache, then per-request routing
evaluated against random mixing at matched cost.

**Repo.** [link]

---

## Checklist before sending

- [ ] Every bracket replaced with a measured number, or the sentence deleted
- [ ] Every interval stated, none dropped because it was inconvenient
- [ ] "No detectable difference" used wherever an interval contains zero
- [ ] The GPU price and the date it was taken are stated
- [ ] The baseline was rerun and matched (so the pipeline is trustworthy)
- [ ] Repo link works for someone who is not logged in
- [ ] SOP paragraph updated to match this wording
