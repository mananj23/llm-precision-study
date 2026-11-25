# Run matrix

Tick a cell only when the raw JSON is saved AND `scripts/collect_logs.py`
parsed it without warnings. "Running" is not "done".

## Llama-3.1-8B-Instruct, quality

| config | gsm8k | mmlu | humaneval | ifeval | longbench |
|---|---|---|---|---|---|
| w-bf16__kv-bf16 (baseline) | | | | | |
| w-fp8__kv-bf16 | | | | | |
| w-bf16__kv-fp8 | | | | | |
| w-fp8__kv-fp8 | | | | | |
| w-int8__kv-bf16 | | | | | |
| w-int4gptq__kv-bf16 | | | | | |
| w-int4awq__kv-bf16 | | | | | |

## Llama-3.1-8B-Instruct, system benchmark

| config | short | medium | long | notes (KV capacity, OOMs) |
|---|---|---|---|---|
| w-bf16__kv-bf16 | | | | |
| w-fp8__kv-bf16 | | | | |
| w-bf16__kv-fp8 | | | | |
| w-fp8__kv-fp8 | | | | |
| w-int8__kv-bf16 | | | | |
| w-int4gptq__kv-bf16 | | | | |
| w-int4awq__kv-bf16 | | | | |

## Checkpoints built

| config | built | size GB | verified (100-prompt check) | build minutes |
|---|---|---|---|---|
| w-fp8 | | | | |
| w-int8 | | | | |
| w-int4gptq | | | | |
| w-int4awq | | | | |

## Mistral-7B-Instruct-v0.3 (phase 2, cut first if behind)

| config | gsm8k | mmlu |
|---|---|---|
| w-bf16__kv-bf16 | | |
| w-fp8__kv-bf16 | | |
