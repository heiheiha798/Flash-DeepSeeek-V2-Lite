# Triton Kernels

This directory contains the active DeepSeek-V2-Lite decode kernels used by
`src/sota.py`.

Target assumptions:

- model: `DeepSeek-V2-Lite-Chat`
- GPU: RTX A6000 / SM86 for current batch sweeps; kernels are Triton bf16 decode kernels.
- dtype: bf16
- decode: q_len=1 with shared batched paths for batch sizes 1 through 256
- graph-safe execution where possible

## Active Kernels

- `rmsnorm.py`: Triton RMSNorm for hidden size 2048 and kv-lora rank 512 paths.
- `moe_router.py`: router softmax + fixed topk6.
- `moe_grouped_gemv.py`: shared batched routed-MoE grouped gate/up/down/reduce path with shape-keyed tile autotuning.
- `mlp_elementwise.py`: MLP `silu(gate) * up` elementwise fusion.
- `attention_prepost.py`: decode input preparation, cache write, residual add.
- `attention_decode.py`: DeepSeek-V2-Lite shared batched attention decode path and packed linear helpers.

## Tooling

Micro profile and validation scripts live in `tools/`, not in this package.
This directory should contain kernel implementations only.

## Optimization Policy

Prefer changes that can be proven with `ncu` before running full end-to-end
benchmarks. Do not keep a kernel rewrite just because it looks cleaner; keep it
only if single-kernel metrics or end-to-end graph metrics justify it.

## Current Batch Tradeoff

The current `src/sota.py` path intentionally routes `bsz=1` and `bsz>1` through
the same decode-attention and grouped-MoE implementation. This removes the old
API-level batch-size split and improves scaling at larger batch sizes, but it
reduces batch=1 throughput versus the previous single-token-specialized path.
Current measured `bsz=1` is about 111 tok/s, versus about 136 tok/s before the
unified batched path.
