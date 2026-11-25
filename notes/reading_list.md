# Week-1 reading list

Three lines each: **what they claim**, **what they measured it on**, **what they
did not test**. The third line is where this project's questions come from, so
it is written to be specific rather than dismissive — "did not test X" means X
is genuinely absent, not merely out of scope.

All bibliographic details checked against arXiv on 2026-11-16.

---

## 0. The one to read first — the closest prior work

### "Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization
Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, Dan Alistarh
(Red Hat AI / ISTA). arXiv 2411.02355, Nov 2024. ACL 2025.

- **Claim.** FP8 W8A8 is effectively lossless at every model scale; a
  well-tuned INT8 W8A8 loses only about 1–3% accuracy per task, far less than
  the 10%+ reported in earlier work; and INT4 weight-only is more competitive
  than expected, rivalling 8-bit. W4A16 gives the best cost-efficiency for
  synchronous deployments and for asynchronous deployment on mid-tier GPUs.
- **Measured on.** The entire Llama-3.1 family, academic benchmarks plus
  real-world tasks, over 500,000 individual evaluations, with inference
  performance analysed through vLLM. Their FP8 recipe is dynamic per-token
  activation quantization with symmetric round-to-nearest weights; their INT8
  is SmoothQuant or dynamic activations paired with GPTQ symmetric weights.
- **Not tested.** Per-prompt confidence intervals on the change from BF16 —
  results are reported as averages, so there is no statement of which
  differences are distinguishable from noise and which are not. The KV cache is
  not treated as an independent axis. And nothing about choosing precision
  per request.

> **Read this before writing the introduction.** It answers a large part of
> Phase 1's headline question already, and pretending otherwise would be found
> out in five minutes by any reviewer. See `report/related_work.md` for how this
> project should be positioned against it — the short version is that it becomes
> a replication-plus-extension, which is a stronger and more honest framing than
> a false claim of novelty.

---

## 1. Efficient Memory Management for LLM Serving with PagedAttention (vLLM)
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu,
Joseph E. Gonzalez, Hao Zhang, Ion Stoica. arXiv 2309.06180, Sep 2023. SOSP 2023.

- **Claim.** KV-cache memory is the binding constraint on serving throughput
  because it grows and shrinks per request and is badly fragmented by
  contiguous allocation. PagedAttention borrows virtual memory and paging from
  operating systems, storing attention keys and values in non-contiguous paged
  blocks, and copy-on-write handles shared prefixes.
- **Measured on.** Throughput of a serving system against prior systems; gains
  are larger for longer sequences, larger models and more complex decoding.
- **Not tested.** Numerical precision is not a variable anywhere. The paper is
  about *how much* KV cache you can fit, not what happens to answer quality
  when you store it in fewer bits — which is precisely the gap this project
  works in.

**Why it matters here.** This is why `kv_cache_tokens` from the vLLM startup
log is recorded as a first-class result: in this framework, KV-cache capacity
is the mechanism that converts a memory saving into a throughput gain.

## 2. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang,
Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han. arXiv 2306.00978, Jun 2023.
MLSys 2024 — **Best Paper**.

- **Claim.** Weights are not equally important: protecting roughly 1% of
  salient weight channels sharply reduces quantization error, and the salient
  channels are identified from the *activation* distribution, not from the
  weights. Rather than mixed precision (hardware-unfriendly), AWQ derives a
  per-channel scaling that protects those channels. It uses no backpropagation
  or reconstruction, so it is claimed not to overfit the calibration set.
- **Measured on.** Language modelling, common-sense QA and domain-specific
  benchmarks; instruction-tuned and multi-modal models; a reported 1.45×
  speedup over GPTQ and 1.85× over cuBLAS FP16, with TinyChat kernels on
  desktop, mobile and edge GPUs.
- **Not tested.** No FP8, no KV-cache quantization, and no measurement under a
  continuous-batching server at varying concurrency — the speedups are kernel
  and edge-device numbers, not throughput of a loaded serving system.

## 3. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh.
arXiv 2210.17323, Oct 2022. ICLR 2023.

- **Claim.** One-shot weight quantization using approximate second-order
  (Hessian) information can take a 175B model to 3–4 bits per weight with
  negligible accuracy loss, in about four GPU hours, and still gives reasonable
  accuracy at 2-bit or ternary.
- **Measured on.** Mostly perplexity and generative-task fidelity on the
  OPT/BLOOM generation of models; reported end-to-end inference speedups over
  FP16 of roughly 3.25× on A100 and 4.5× on cheaper GPUs.
- **Not tested.** Instruction-tuned chat models (they did not exist in this
  form yet), task-level reasoning accuracy such as GSM8K, and anything about
  the KV cache. Perplexity is a weak proxy for the multi-step arithmetic this
  project measures.

**Why it matters here.** GPTQ and AWQ are the head-to-head pair in this study.
Note that their reference recipes default to *different* symmetry settings —
see `scripts/w4a16_config.py` for the trap that creates.

## 4. SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs
Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han.
arXiv 2211.10438, Nov 2022. ICML 2023.

- **Claim.** Weights are easy to quantize and activations are not, because
  systematic outliers sit in *fixed* activation channels and stretch the range.
  A mathematically equivalent offline transformation migrates that difficulty
  from activations into weights, making W8A8 feasible without training.
- **Measured on.** OPT-175B, BLOOM-176B, GLM-130B and MT-NLG 530B (later
  versions add Llama-1/2, Falcon, Mistral, Mixtral); up to 1.56× speedup and 2×
  memory reduction integrated into FasterTransformer, serving 530B on one node.
- **Not tested.** FP8 as an alternative to INT8, the KV cache, and per-task
  accuracy breakdowns on modern instruct models with any statement of
  uncertainty.

**Why it matters here.** This is the "SmoothQuant" half of the INT8 recipe:
`SmoothQuantModifier(0.8)` then GPTQ. Running GPTQ W8A8 *without* it is a
different and worse recipe, which is why the script does not make it optional.

## 5. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer.
arXiv 2208.07339, Aug 2022.

- **Claim.** Int8 matmul for feed-forward and attention projection layers
  halves inference memory with no performance loss, because emergent outlier
  features can be isolated: a mixed-precision decomposition routes the outlier
  dimensions through a 16-bit matmul while over 99.9% of values stay in 8-bit,
  with vector-wise quantization and separate normalization constants elsewhere.
- **Measured on.** Models up to 175B parameters, loaded from an existing 16/32-bit
  checkpoint and converted without retraining.
- **Not tested.** Speed — the mixed-precision decomposition is a memory result,
  not a throughput one, and in practice this method is often *slower* than
  FP16. Also no KV cache, and no modern instruct-model task suite.

**Why it matters here.** It is the origin of the "outlier features" framing that
SmoothQuant and AWQ both respond to, and a useful warning that *memory saved*
and *time saved* are different claims — which is why this project measures both.

## 6. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu,
Vladimir Braverman, Beidi Chen, Xia Hu. arXiv 2402.02750, Feb 2024. ICML 2024.

- **Claim.** Keys and values need *different* quantization axes: the key cache
  should be quantized per channel, the value cache per token. Per-token
  quantization confines the error within each token and suits the streaming
  nature of autoregressive decoding. The resulting tuning-free 2-bit scheme
  keeps quality nearly unchanged at 2.6× less peak memory, enabling up to 4×
  larger batches and 2.35–3.47× throughput.
- **Measured on.** Llama (Llama-2), Falcon and Mistral, on real inference
  workloads.
- **Not tested.** FP8 KV cache specifically, and integration with a production
  paged-attention server — the comparison is against their own baselines, not
  against what vLLM ships.

**Why it matters here.** This is the sharpest reason to be suspicious of
vLLM's *default* FP8 KV cache, where all scales are 1.0 with no calibration and
no key/value asymmetry. If `w-bf16__kv-fp8` shows a long-context regression,
KIVI is the explanation to reach for first.

## 7. FrugalGPT: How to Use LLMs While Reducing Cost and Improving Performance
Lingjiao Chen, Matei Zaharia, James Zou (Stanford).
arXiv 2305.05176, May 2023. TMLR 2024.

- **Claim.** LLM API pricing is heterogeneous — fees differ by two orders of
  magnitude — so cost can be cut by prompt adaptation, LLM approximation and an
  LLM *cascade* that learns which model combination to use per query. Reported
  savings of up to 98% of the best single API's cost while matching its
  performance, or +4% accuracy at equal cost.
- **Measured on.** News classification, reading comprehension and scientific
  question answering, against commercial APIs.
- **Not tested.** Anything self-hosted. The "cheap" and "expensive" options are
  different *vendors' models* at posted prices, not precision settings of one
  model on one GPU, and a cascade *runs the cheap model first and then maybe
  the expensive one*, so its cost model is additive in a way a router's is not.

**Why it matters here.** It is the origin of the cost-cascade idea, but the
cost accounting does not transfer. This is why `router/dataset.py` charges the
cheap model's confidence signal on *every* prompt if it is used.

## 8. RouteLLM: Learning to Route LLMs with Preference Data
Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu,
Joseph E. Gonzalez, M Waleed Kadous, Ion Stoica.
arXiv 2406.18665, Jun 2024. ICLR 2025.

- **Claim.** A router trained on human preference data (Chatbot Arena) with
  data augmentation can choose between a strong and a weak model per query,
  cutting cost by more than 2× with minimal quality loss, and transfers to
  model pairs it was not trained on.
- **Measured on.** Public benchmarks including MT-Bench, with two headline
  metrics worth borrowing: **call-performance threshold (CPT)** and **average
  performance gain recovered (APGR)**.
- **Not tested.** Routing between *precision settings of the same model* —
  every pair is two different models. Preference labels, not correctness, so
  there is no notion of "the cheap one was sufficient because both were wrong."
  And no serving-level cost of keeping two configurations resident.

**Why it matters here.** The closest methodological relative of Step 7, and the
source of the right baseline discipline. Consider reporting CPT and APGR
alongside the random-mix comparison so the numbers are legible to anyone who
knows this paper.

## 9. Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving
Nikita Agrawal, Ruben Mayer. arXiv 2607.05399, May 2026.

- **Claim.** KV-cache compression techniques are hard to compare because each
  was evaluated on a different setup; a workload-aware benchmark should
  evaluate quantization and pruning methods jointly on task quality *and*
  system performance.
- **Measured on.** Models including Mistral-7B-Instruct-v0.3, with six LongBench
  datasets across four categories (multi-document QA, single-document QA,
  few-shot learning, summarization), reporting task quality, mean output
  throughput, mean time-to-first-token and KV-cache memory footprint; system
  performance assessed on NarrativeQA, GovReport and Qasper.
- **Not tested.** Weight precision as a co-varying axis, per-prompt intervals,
  and per-request routing.

**Why it matters here.** This is the methodological template for Step 5, and it
independently validates the design: the LongBench subsets in
`configs/eval.yaml` were chosen to overlap with theirs, and the metric set is
the same one `collect_bench.py` extracts. **Check whether their numbers for
Mistral agree with yours** — an agreement is a cheap credibility win and a
disagreement is the most interesting thing in the report.

## 10. FP8 Formats for Deep Learning
Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey,
Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd,
John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu,
Hao Wu (NVIDIA / Intel / Arm). arXiv 2209.05433, Sep 2022.

- **Claim.** Two FP8 encodings: E4M3 (4-bit exponent, 3-bit mantissa) and E5M2
  (5 exponent, 2 mantissa). E5M2 follows IEEE 754 for special values; E4M3
  extends dynamic range by dropping infinities and keeping a single NaN
  bit-pattern. FP8 matches 16-bit training quality across CNNs, RNNs and
  Transformers with hyperparameters unchanged.
- **Measured on.** Image and language tasks, training runs up to 175B
  parameters; also FP8 post-training quantization of models that had *resisted*
  fixed-point INT8 quantization.
- **Not tested.** Serving throughput, the KV cache, and instruction-following
  or reasoning benchmarks of the kind used here. It is a numerics paper.

**Why it matters here.** The last point is the key one: FP8 PTQ worked where
INT8 PTQ failed, because a floating-point format handles outlier activations
that a fixed-point one cannot. That is the mechanistic reason to expect FP8 ≥
INT8 in this study, and a hypothesis the results can confirm or contradict.

---

## Also worth a skim (found while checking the list above)

- **KV Pareto: Systems-Level Optimization of KV Cache and Model Compression for
  Long Context Inference** (arXiv 2512.01953). Jointly evaluates chunked
  prefill, KV-cache quantization and AWQ weight-only quantization across
  granularities and precisions, finding Pareto-optimal configurations on
  LongBench, with 68–78% total memory savings at 1–3% long-context accuracy
  loss. *This is a Pareto analysis over the same two axes as this project* —
  read it before claiming the frontier framing is new, and cite it in Related
  Work.
- **Rethinking Key-Value Cache Compression Techniques for LLM Serving**
  (arXiv 2503.24000). Makes exactly this project's per-prompt argument:
  benchmark studies suggest only minor average degradation, but an aggregated
  number can obscure bias toward specific tasks and sample types, and existing
  studies lack analysis of how compression affects *individual* examples.
  **This is the strongest citation available for why Step 4 logs per prompt.**
- **KVDiagnosis** (arXiv 2608.09412). Reports a "correct → wrong" rate per
  method and setting: the fraction of prompts the full-cache model got right
  that the compressed one gets wrong. That is a per-prompt flip count, the same
  quantity this project's paired bootstrap is built on, and it is a good extra
  column for the results table — cheap to compute from
  `results/logs/per_prompt.parquet` and more interpretable than a delta.

---

## What the gaps add up to

Across all of it, three things are genuinely under-reported:

1. **Uncertainty.** Almost everything reports point estimates. Where this
   project can say "no detectable difference" with a paired interval, most of
   this literature says a number and leaves the reader to guess.
2. **The KV cache as an independent axis.** Weight-quantization papers hold it
   fixed; KV-cache papers hold weights fixed. The `w-bf16__kv-fp8` cell exists
   because almost nobody runs it.
3. **Per-request precision choice.** Routing work swaps *models*; quantization
   work picks one setting for everything. Whether the frontier differs enough
   *by prompt* to be worth exploiting is open — and the honest answer may be no.
