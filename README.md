# DeepSeek-V2-Lite Triton Inference Optimization

This repository is now used as a focused performance playground for
`DeepSeek-V2-Lite-Chat` decode inference on `A100 / SM80` with `bf16`.

The original codebase started from DeepSeek FlashMLA, but the legacy FlashMLA
CUDA extension has been removed from the active tree. The active development
path is now:

- Hugging Face model structure
- prefill in eager mode
- single-token decode with CUDA graph replay
- DeepSeek-V2-Lite-specific Triton kernels for hot decode operators
- `nsys` / `ncu` driven optimization

## Current Scope

Target configuration:

- Model: `/data/models/DeepSeek-V2-Lite-Chat`
- GPU: NVIDIA A100 80GB, SM80
- dtype: `bf16`
- batch: `1`
- decode shape: `q_len=1`
- default output length: `100` new tokens

The goal is not general inference serving. The active runner compares two
q_len=1 Triton decode kernel families:

- `--kernel-family small`: batched API with a small-GEMV implementation template.
- `--kernel-family batch`: grouped batching kernels for larger batch efficiency.

Serving baselines under `baselines/` are capped at batch size 256 for comparable
plots.

The kernel optimization goal is to iterate quickly on fixed-shape decode kernels
and measure whether they improve real decode latency.

## Main Entrypoints

Run the small-GEMV kernel family:

```bash
CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family small --device cuda:0 --max-new-tokens 100
```

Run the grouped batching kernel family:

```bash
CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100
```

Run the two reporting sweeps over common batch sizes:

```bash
CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family small --device cuda:0 --max-new-tokens 100 \
  --batch-sizes "1 2 4 8 16 32 64 128 256"

CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100 \
  --batch-sizes "1 2 4 8 16 32 64 128 256"
```

Interpretation: the two `--kernel-family` values intentionally use different
kernel implementations so their curves show method tradeoffs, not a hidden
auto-selected best path.

For performance runs, check `nvidia-smi` first and do not run two GPU workloads
on the same card. GPU3 has been the preferred card for final numbers, but GPU0
is acceptable for relative micro comparisons when GPU3 is busy.

## Profiling

Node-level `nsys` profile:

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --output=nsys-reps/sota_gpu0_node \
  /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100
```

MoE kernel `ncu` micro profile:

```bash
CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --kernel-name regex:_fused_gate_up_swiglu_kernel \
  --launch-skip 1 \
  --launch-count 1 \
  --set full \
  --export ncu-reps/moe_gate_profile \
  --force-overwrite \
  /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  tools/profile_moe_ncu.py
```

## Current Performance Snapshot

The latest stable GPU3 batch-kernel snapshot before repo cleanup:

- command: `CUDA_VISIBLE_DEVICES=3 ... src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100`
- decode throughput: about `195+ TPS`
- hardware: `NVIDIA A100 80GB PCIe`, `sm80`, `80 GB`, `INTEL(R) XEON(R) PLATINUM 8558P`, `96C/192T`, `503.53 GiB RAM`
- software: `Python 3.10.20`, `torch 2.10.0+cu130`, `torch CUDA 13.0`, `triton 3.6.0`, `transformers 4.57.6`
- node-level graph: `651 kernels/step`
- graph GPU time: about `5.145 ms/step`
- report: `nsys-reps/sota_gpu3_gate_combined_node.nsys-rep`

GPU0 is slower on this machine, but useful for relative micro profiling.

## Active Directory Layout

- `src/`: runnable DeepSeek-V2-Lite decode scripts.
- `src/run.py`: CUDA graph runner with `--kernel-family small|batch`.
- `triton_kernels/`: custom Triton kernel implementations only.
- `tools/`: micro profile and validation scripts.
- `baselines/`: third-party baseline wrappers, including llama.cpp, SGLang, and vLLM.
- `docs/`: profiling summaries, Triton optimization notes, and performance reporting rules.
- `nsys-reps/`: local Nsight Systems reports, gitignored.
- `ncu-reps/`: local Nsight Compute reports, gitignored.

## Removed Legacy Areas

The earlier FlashMLA extension path has been physically removed from the active
repo to keep this project focused on DeepSeek-V2-Lite inference optimization.
Removed areas include the old `flash_mla/` package, CUDA `csrc/`, FlashMLA
benchmarks, earlier `hf_inference/` prototypes, and extension build scripts.

New work should go through `src/`, `triton_kernels/`, and `docs/`. Performance reports must follow `docs/performance_reporting.md`.

## Dependencies

The root `requirements.txt` tracks the active optimization path. The existing
conda environment used during development is:

```bash
/data/home/tianjianyang/.conda/envs/flashmla
```

Install Python dependencies as needed:

```bash
pip install -r requirements.txt
```

## Documentation

- `docs/hf_profile_summary.md`: historical profile conclusions for the earlier baseline/SOTA split.
- `docs/triton_kernel_optimizations.md`: effective Triton optimizations and failed attempts.
- `docs/performance_reporting.md`: TPS report format with hardware and software context.

## Acknowledgement

This repository started from the DeepSeek FlashMLA codebase:

- https://github.com/deepseek-ai/FlashMLA/

The current active work is a downstream DeepSeek-V2-Lite Triton inference
optimization effort rather than an upstream-compatible FlashMLA package.
