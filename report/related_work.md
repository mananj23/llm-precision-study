# Related work

Drafted 2026-11-16, before any measurement. Sections 1–4 are final prose.
Section 5 is the positioning argument and should be re-read once results exist,
because it is the part most likely to need honest revision.

---

## 1. Weight and activation quantization

The modern line starts with the observation that transformer activations
contain systematic outliers. Dettmers et al. (2022) showed that these outliers
sit in a small number of emergent feature dimensions and that routing just
those dimensions through a 16-bit matrix multiplication, while more than 99.9%
of values stay in 8-bit, preserves full-precision quality at half the inference
memory. The result is a memory result rather than a speed one, and the
mixed-precision decomposition it depends on is awkward for hardware.

Two responses followed. SmoothQuant (Xiao et al., ICML 2023) notes that the
outliers occupy *fixed* activation channels and applies a mathematically
equivalent offline rescaling that moves the difficulty out of the activations
and into the weights, where uniform quantization copes; this makes a
hardware-friendly INT8 W8A8 possible without training. GPTQ (Frantar et al.,
ICLR 2023) attacks the weights instead, using approximate second-order
information to quantize one-shot to 3–4 bits with little loss, at a cost of a
few GPU hours even for very large models. AWQ (Lin et al., MLSys 2024) argues
that weight importance should be read off the activation distribution rather
than the weight values, and protects roughly the top 1% of salient channels by
per-channel scaling, avoiding mixed precision altogether.

FP8 enters from the numerics side. Micikevicius et al. (2022) specify the E4M3
and E5M2 encodings and show that FP8 matches 16-bit training quality across
architectures. The finding most relevant here is an aside in that paper: FP8
post-training quantization succeeded on models that had resisted fixed-point
INT8 quantization. A floating-point format spends bits on dynamic range, which
is exactly what an outlier-heavy activation distribution needs. This gives a
mechanistic reason to expect FP8 to degrade less than INT8 at the same
bit width — a prediction this study is in a position to test rather than assume.

## 2. KV-cache compression

Serving throughput is bounded by how many requests fit concurrently, and since
PagedAttention (Kwon et al., SOSP 2023) made KV-cache allocation efficient
rather than fragmented, the remaining lever is the cache's size in bytes.
Quantizing it is the direct approach.

KIVI (Liu et al., ICML 2024) is the most careful treatment. Its central finding
is an asymmetry: the key cache should be quantized along the channel dimension
and the value cache along the token dimension, because per-token quantization
confines error within a token and matches the streaming structure of decoding.
With that asymmetry respected, 2-bit is viable, giving roughly 2.6× lower peak
memory, up to 4× larger batches, and 2.35–3.47× throughput.

This matters for the present study because the FP8 KV cache that vLLM exposes
as a runtime flag does not, by default, do any of this: scales default to 1.0
with no calibration, and calibrated per-head scales are an optional extra.
KIVI therefore supplies the hypothesis to test — if an FP8 KV cache costs
quality anywhere, it should cost it at long context, where the cache is largest
and errors accumulate across the most tokens.

Agrawal and Mayer (2026) make the methodological argument that this project
follows: KV-cache techniques are hard to compare because each was evaluated in
a different setup, so a benchmark should measure task quality and system
performance jointly under the same long-context workload. Their metric set —
task accuracy, mean output throughput, mean time-to-first-token and KV-cache
memory footprint — is adopted here. KV Pareto (2025) performs a related
Pareto analysis over KV-cache quantization and AWQ weight quantization on
LongBench, reporting large memory savings at a few points of long-context
accuracy.

## 3. Reporting practice, and why this study logs per prompt

A recurring criticism of this literature is that an averaged accuracy figure
hides the structure of the damage. "Rethinking Key-Value Cache Compression
Techniques for LLM Serving" (2025) states it directly: benchmark studies tend
to conclude that compression costs little on average, but the aggregate can
obscure bias toward particular tasks and particular kinds of example, and
existing work rarely analyses what happens to individual responses.
KVDiagnosis (2026) operationalises the complaint with a correct-to-wrong rate:
of the prompts the uncompressed model answered correctly, what fraction does
the compressed one get wrong?

That quantity is per-prompt, and it is also what a paired confidence interval
needs. The present study logs every prompt's correctness for every
configuration and computes differences from the BF16 baseline by resampling
prompts, not by subtracting two independently-estimated accuracies. The
justification is empirical and is reproducible from this repository: on a
simulated one-point regression over 2,000 prompts, the paired interval excludes
zero while the unpaired interval spans it. The unpaired procedure does not
merely give a wider interval — it reaches the opposite conclusion about whether
an effect exists.

## 4. Routing and cascades

FrugalGPT (Chen, Zaharia and Zou, TMLR 2024) established the cost-cascade idea
for commercial APIs, where posted prices differ by two orders of magnitude:
query a cheap model, judge the answer, escalate if necessary. RouteLLM (Ong et
al., ICLR 2025) replaces the cascade with a *router* trained on human
preference data from Chatbot Arena, deciding before generation which of a
strong and a weak model should answer, and reports cost reductions above 2×
with transfer to unseen model pairs. Its evaluation metrics — call-performance
threshold and average performance gain recovered — are the right vocabulary for
this kind of result.

Both differ from the question asked here in ways that change the accounting.
They route between *different models*, so the two options differ in weights,
training data and capability; here the two options are the *same model at
different numerical precision*, which is a much narrower difference and
therefore a harder routing problem. A cascade pays for the cheap model and
then sometimes the expensive one, so its cost is additive; a router pays for
one, unless it consults the cheap model for a confidence signal, in which case
it pays for that on every request including the escalated ones. And preference
labels are not correctness labels: they cannot express the case that matters
most here, where the cheap configuration is *sufficient* because the expensive
one was also wrong, so nothing was lost by being cheap.

## 5. Positioning this study — and the honest problem

The closest prior work is Kurtic et al. (ACL 2025), which evaluates FP8, INT8
and INT4 across the whole Llama-3.1 family through vLLM, with over half a
million individual evaluations, and reports that FP8 W8A8 is effectively
lossless at all scales, well-tuned INT8 W8A8 costs roughly 1–3% per task, and
INT4 weight-only is more competitive than its reputation, with W4A16 the best
cost-efficiency choice for synchronous deployment.

**That overlaps substantially with Phase 1 of this study: same model family,
same formats, same serving framework.** The headline question — what does
lower precision cost on standard benchmarks — is largely answered there, and
this report should say so in its first paragraph rather than imply otherwise.

What remains, and what this study should claim:

1. **Uncertainty, per prompt.** Kurtic et al. report averages across a very
   large number of evaluations. They do not report, for any configuration and
   task, an interval on the change from BF16, so they cannot say which of their
   differences are distinguishable from noise. A study at this much smaller
   scale cannot beat them on coverage, but it can state for every cell whether
   the difference is detectable at all — and, given the 1–3% effects they
   report, how many prompts are needed before such an effect is detectable is
   itself a useful finding for anyone planning a similar evaluation.
2. **The KV cache as an independent axis.** Weight-quantization studies hold
   the cache fixed; KV-cache studies hold the weights fixed. The
   `w-bf16__kv-fp8` configuration — unquantized weights, FP8 cache — is the
   cell that separates the two causes, and it is rarely run.
3. **Per-request precision routing.** Routing work swaps models; quantization
   work picks one setting for everything. Whether the quality-cost frontier
   differs enough *between prompts of the same task* to be worth exploiting is
   open, and a negative answer is a result.
4. **Replication value.** An independent reproduction on a single rented GPU,
   with the exact recipes, versions, prices and dates recorded, either confirms
   a 500,000-evaluation industrial study or contradicts it. Both outcomes are
   worth writing down, and neither requires the study to be novel.

The framing to use is therefore *replication plus extension*, not discovery.
That is a stronger position than it sounds: it is honest, it is checkable, and
it is what the per-prompt machinery in this repository is actually good for.

---

## References

Agrawal, N., & Mayer, R. (2026). Benchmarking KV-Cache Optimizations across
Task Quality and System Performance for Long-Context Serving. arXiv:2607.05399.

Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to Use Large Language
Models While Reducing Cost and Improving Performance. arXiv:2305.05176. TMLR 2024.

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8():
8-bit Matrix Multiplication for Transformers at Scale. arXiv:2208.07339.

Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers.
arXiv:2210.17323. ICLR 2023.

Kurtic, E., Marques, A., Pandit, S., Kurtz, M., & Alistarh, D. (2024). "Give Me
BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization.
arXiv:2411.02355. ACL 2025.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E.,
Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language
Model Serving with PagedAttention. arXiv:2309.06180. SOSP 2023.

Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G.,
Dang, X., Gan, C., & Han, S. (2023). AWQ: Activation-aware Weight Quantization
for LLM Compression and Acceleration. arXiv:2306.00978. MLSys 2024.

Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V., Chen, B., & Hu, X.
(2024). KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
arXiv:2402.02750. ICML 2024.

Micikevicius, P., Stosic, D., Burgess, N., Cornea, M., Dubey, P.,
Grisenthwaite, R., Ha, S., Heinecke, A., Judd, P., Kamalu, J., Mellempudi, N.,
Oberman, S., Shoeybi, M., Siu, M., & Wu, H. (2022). FP8 Formats for Deep
Learning. arXiv:2209.05433.

Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E.,
Kadous, M. W., & Stoica, I. (2024). RouteLLM: Learning to Route LLMs with
Preference Data. arXiv:2406.18665. ICLR 2025.

Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2022).
SmoothQuant: Accurate and Efficient Post-Training Quantization for Large
Language Models. arXiv:2211.10438. ICML 2023.

*Also cited, verify details before final submission:* KV Pareto
(arXiv:2512.01953); Rethinking Key-Value Cache Compression Techniques for Large
Language Model Serving (arXiv:2503.24000); KVDiagnosis (arXiv:2608.09412).
