# Measured tables

All filter recoveries confirmed against lm-eval output: **True**

## Noise floor (BF16 evaluated twice)

- runs `20261125T211450` and `20261126T143602`, n = 1319
- accuracy 70.508% vs 70.508%
- per-prompt correctness disagreement **0.0%**
- per-prompt extracted-answer disagreement **0.0%**

## GSM8K exact_match (strict-match)

| config | n | acc % | Δ vs BF16 (pp) | 95% CI | resolved | answers changed % |
|---|---:|---:|---:|---|---|---:|
| BF16 / BF16 | 1319 | 70.51 | +0.0 | [+0.0, +0.0] | no | 0.0 |
| FP8 / BF16 | 1319 | 70.28 | -0.23 | [-2.27, +1.82] | no | 14.33 |
| BF16 / FP8 | 1319 | 69.6 | -0.91 | [-2.73, +0.91] | no | 11.07 |
| FP8 / FP8 | 1319 | 70.43 | -0.08 | [-2.2, +2.05] | no | 15.24 |
| INT8 W8A8 / BF16 | 1319 | 70.2 | -0.3 | [-2.35, +1.67] | no | 13.5 |
| INT4 GPTQ / BF16 | 1319 | 72.18 | +1.67 | [-0.83, +4.25] | no | 21.99 |

## GSM8K exact_match (flexible-extract)

| config | n | acc % | Δ vs BF16 (pp) | 95% CI | resolved | answers changed % |
|---|---:|---:|---:|---|---|---:|
| BF16 / BF16 | 1319 | 77.79 | +0.0 | [+0.0, +0.0] | no | 0.0 |
| FP8 / BF16 | 1319 | 77.56 | -0.23 | [-1.97, +1.44] | no | 10.08 |
| BF16 / FP8 | 1319 | 77.18 | -0.61 | [-2.12, +0.99] | no | 8.04 |
| FP8 / FP8 | 1319 | 76.95 | -0.83 | [-2.58, +0.91] | no | 10.39 |
| INT8 W8A8 / BF16 | 1319 | 77.71 | -0.08 | [-1.67, +1.59] | no | 9.02 |
| INT4 GPTQ / BF16 | 1319 | 73.01 | -4.78 | [-6.97, -2.58] | yes | 17.21 |

## MMLU acc

| config | n | acc % | Δ vs BF16 (pp) | 95% CI | resolved | answers changed % |
|---|---:|---:|---:|---|---|---:|
| BF16 / BF16 | 14042 | 68.25 | +0.0 | [+0.0, +0.0] | no | 0.0 |
| FP8 / BF16 | 14042 | 68.23 | -0.01 | [-0.35, +0.33] | no | 4.07 |
| BF16 / FP8 | 14042 | 67.96 | -0.28 | [-0.61, +0.04] | no | 3.77 |
| FP8 / FP8 | 14042 | 67.69 | -0.56 | [-0.96, -0.15] | yes | 5.93 |
| INT8 W8A8 / BF16 | 14042 | 67.7 | -0.54 | [-0.88, -0.2] | yes | 4.39 |
| INT4 GPTQ / BF16 | 14042 | 66.25 | -1.99 | [-2.49, -1.49] | yes | 9.43 |

## IFEval prompt_level_strict_acc

| config | n | acc % | Δ vs BF16 (pp) | 95% CI | resolved | answers changed % |
|---|---:|---:|---:|---|---|---:|
| BF16 / BF16 | 541 | 47.32 | +0.0 | [+0.0, +0.0] | no | 0.0 |
| FP8 / BF16 | 541 | 45.84 | -1.48 | [-4.62, +1.66] | no | 13.31 |
| BF16 / FP8 | 541 | 46.21 | -1.11 | [-3.88, +1.85] | no | 11.83 |
| FP8 / FP8 | 541 | 48.06 | +0.74 | [-2.4, +3.7] | no | 13.31 |
| INT8 W8A8 / BF16 | 541 | 46.4 | -0.92 | [-4.25, +2.4] | no | 16.08 |
| INT4 GPTQ / BF16 | 541 | 43.25 | -4.07 | [-7.39, -0.74] | yes | 15.9 |

## GSM8K format-miss (flexible correct, strict wrong)

| config | n | acc % | Δ vs BF16 (pp) | 95% CI | resolved | answers changed % |
|---|---:|---:|---:|---|---|---:|
| BF16 / BF16 | 1319 | 7.35 | +0.0 | [+0.0, +0.0] | no | 0.0 |
| FP8 / BF16 | 1319 | 7.35 | +0.0 | [-1.36, +1.36] | no | 6.52 |
| BF16 / FP8 | 1319 | 7.66 | +0.3 | [-0.91, +1.52] | no | 5.0 |
| FP8 / FP8 | 1319 | 6.6 | -0.76 | [-2.2, +0.68] | no | 7.28 |
| INT8 W8A8 / BF16 | 1319 | 7.58 | +0.23 | [-1.14, +1.59] | no | 6.6 |
| INT4 GPTQ / BF16 | 1319 | 0.91 | -6.44 | [-7.88, -5.0] | yes | 7.51 |

## Filter sensitivity (same runs, two GSM8K extraction filters)

| config | strict Δ (pp) | 95% CI | flexible Δ (pp) | 95% CI | sign flips |
|---|---:|---|---:|---|---|
| BF16 / BF16 | +0.0 | [0.0, 0.0] | +0.0 | [0.0, 0.0] | no |
| FP8 / BF16 | -0.23 | [-2.27, 1.82] | -0.23 | [-1.9, 1.52] | no |
| BF16 / FP8 | -0.91 | [-2.65, 0.83] | -0.61 | [-2.12, 0.91] | no |
| FP8 / FP8 | -0.08 | [-2.12, 2.05] | -0.83 | [-2.5, 0.91] | no |
| INT8 W8A8 / BF16 | -0.3 | [-2.35, 1.67] | -0.08 | [-1.74, 1.52] | no |
| INT4 GPTQ / BF16 | +1.67 | [-0.91, 4.25] | -4.78 | [-7.05, -2.58] | no |

## Composition -- GSM8K (strict-match)

- prompts the baseline answered correctly: 930
- damaged by weights only (FP8 W): 10.32% (96)
- damaged by KV only (FP8 KV): 8.49% (79)
- damaged by both together: 10.86% (101)
- independence prediction: 17.94%
- excess over independence: -7.08 pp, CI [-9.54, -4.48]
- Jaccard overlap of the two damage sets: 0.224, CI [0.158, 0.292]
- combined damage in neither single-component set: 40.59%, CI [31.11, 50.0]

## Composition -- MMLU

- prompts the baseline answered correctly: 9583
- damaged by weights only (FP8 W): 2.99% (287)
- damaged by KV only (FP8 KV): 2.97% (285)
- damaged by both together: 4.75% (455)
- independence prediction: 5.88%
- excess over independence: -1.13 pp, CI [-1.65, -0.63]
- Jaccard overlap of the two damage sets: 0.177, CI [0.143, 0.212]
- combined damage in neither single-component set: 48.35%, CI [43.79, 52.86]

## Composition -- IFEval

- prompts the baseline answered correctly: 256
- damaged by weights only (FP8 W): 15.62% (40)
- damaged by KV only (FP8 KV): 13.67% (35)
- damaged by both together: 13.28% (34)
- independence prediction: 27.16%
- excess over independence: -13.88 pp, CI [-19.25, -8.21]
- Jaccard overlap of the two damage sets: 0.339, CI [0.217, 0.468]
- combined damage in neither single-component set: 29.41%, CI [14.29, 45.45]

## Throughput

- 216 measured cells: 6 configs x 3 shapes x 4 concurrencies x 3 seeds
- shapes ['long', 'medium', 'short'], concurrencies [1, 8, 32, 64]

| config | peak output tok/s | shape | conc | speedup vs BF16 | KV cache tokens |
|---|---:|---|---:|---:|---:|
| FP8 / FP8 | 1909.65 | short | 64 | 1.52x | 491904 |
| INT4 GPTQ / BF16 | 1761.03 | short | 64 | 1.4x | 269584 |
| INT8 W8A8 / BF16 | 1752.88 | short | 64 | 1.4x | 242336 |
| FP8 / BF16 | 1713.87 | short | 64 | 1.36x | 248832 |
| BF16 / FP8 | 1352.68 | short | 64 | 1.08x | 389824 |
| BF16 / BF16 | 1256.48 | short | 64 | 1.0x | 191264 |
