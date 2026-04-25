# Triton Kernels

This directory contains the active DeepSeek-V2-Lite decode kernels used by
`src/run.py --kernel-family small|batch`.

Target assumptions:

- model: `DeepSeek-V2-Lite-Chat`
- GPU: NVIDIA A100 80GB / SM80 for current batch sweeps; kernels are Triton bf16 decode kernels.
- dtype: bf16
- decode: q_len=1 with shared batched paths for batch sizes 1 through 256
- graph-safe execution where possible

## Active Kernels

- `rmsnorm.py`: Triton RMSNorm for hidden size 2048 and kv-lora rank 512 paths.
- `moe_router.py`: router softmax + fixed topk6.
- `moe_batch.py`: grouped batched routed-MoE gate/up/down/reduce path with shape-keyed tile autotuning.
- `moe_small_gemv.py`: batched API implemented with the bsz=1-style route-local GEMV template.
- `mlp_elementwise.py`: MLP `silu(gate) * up` elementwise fusion.
- `attention_prepost.py`: decode input preparation, cache write, residual add.
- `attention_decode_batch.py`: DeepSeek-V2-Lite grouped batching attention decode path and packed linear helpers.
- `attention_decode_small.py`: DeepSeek-V2-Lite small-GEMV attention decode path.

## Tooling

Micro profile and validation scripts live in `tools/`, not in this package.
This directory should contain kernel implementations only.

## Optimization Policy

Prefer changes that can be proven with `ncu` before running full end-to-end
benchmarks. Do not keep a kernel rewrite just because it looks cleaner; keep it
only if single-kernel metrics or end-to-end graph metrics justify it.

## Current Batch Tradeoff

The current runner intentionally exposes two kernel families:

- `--kernel-family small`: batched API with a small-GEMV implementation template.
- `--kernel-family batch`: grouped batching kernels that reuse weights across
  batch and expert routes.

These are selected by an explicit runtime argument so full-batch sweeps compare
implementation methods directly rather than hiding the tradeoff behind an
auto-selected path.
