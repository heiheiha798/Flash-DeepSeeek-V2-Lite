# Triton Kernels

This directory contains the active DeepSeek-V2-Lite decode kernels used by
`src/sota.py`.

Target assumptions:

- model: `DeepSeek-V2-Lite-Chat`
- GPU: A100 / SM80
- dtype: bf16
- decode: batch=1, q_len=1
- graph-safe execution where possible

## Active Kernels

- `rmsnorm.py`: Triton RMSNorm for hidden size 2048 and kv-lora rank 512 paths.
- `moe_router.py`: router softmax + fixed topk6.
- `moe_grouped_gemv.py`: routed MoE topk-only gate/up/down/reduce path.
- `mlp_elementwise.py`: MLP `silu(gate) * up` elementwise fusion.
- `attention_prepost.py`: decode input preparation, cache write, residual add.
- `attention_decode.py`: DeepSeek-V2-Lite attention decode path and GEMV helpers.

## Tooling

Micro profile and validation scripts live in `tools/`, not in this package.
This directory should contain kernel implementations only.

## Optimization Policy

Prefer changes that can be proven with `ncu` before running full end-to-end
benchmarks. Do not keep a kernel rewrite just because it looks cleaner; keep it
only if single-kernel metrics or end-to-end graph metrics justify it.
