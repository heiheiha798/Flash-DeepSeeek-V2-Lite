# vLLM Baseline

This baseline records the official vLLM path for `DeepSeek-V2-Lite-Chat` on
A100. The goal is to keep a third-party serving baseline, not to tune vLLM.

## Environment

Conda env:

```bash
/data/home/tianjianyang/.conda/envs/dsv2lite-vllm
```

Important installed packages:

```text
vllm 0.16.0
torch 2.9.1+cu130
triton 3.5.1
flashinfer-python 0.6.3
```

Check the environment:

```bash
baselines/vllm/check_env.sh
```

The wrapper runs from `/tmp` by default so this repository's local
`triton_kernels/` package does not shadow optional vLLM imports.

## Bench

Run the latency benchmark on GPU0:

```bash
GPU=0 baselines/vllm/bench_dsv2_lite_latency.sh
```

Run the fixed batch sweep:

```bash
GPU=0 baselines/vllm/bench_dsv2_lite_batch_sweep.sh
```

Default sweep batch sizes:

```text
1 2 4 8 16 32 64 128 256
```

Override with `BATCH_SIZES`, for example `BATCH_SIZES="1 8 64"`. Sweep logs
are written under `RESULT_DIR`, defaulting to `/tmp/vllm_dsv2lite_batch_sweep`.
The 512-request run is excluded in this setup because vLLM scheduled it as two 256-request waves rather than one true `bsz=512` batch.

Default benchmark shape:

- batch size: `1`
- prompt/input tokens: `24`
- generated tokens: `100`
- dtype: `bfloat16`
- CUDA graph capture sizes: `BATCH_SIZE` in the sweep, `1` in the single-batch default

Observed backend logs:

```text
Using TRITON_MLA attention backend
Using FlashAttention prefill for MLA
Using TRITON backend for Unquantized MoE
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with TritonMLABackend
setting cudagraph_mode=PIECEWISE
```

Latest observed GPU0 decode result:

```text
output_len=100 avg latency: 1.017024 s
output_len=1 avg latency:   0.044115 s
decode TPS estimate:        101.76 tok/s
hardware: NVIDIA A100 80GB PCIe, sm80, 80 GB; INTEL(R) XEON(R) PLATINUM 8558P, 96C/192T, 503.53 GiB RAM
software: vllm 0.16.0, torch 2.9.1+cu130, triton 3.5.1, flashinfer-python 0.6.3
```

Decode TPS is estimated by subtracting the `output_len=1` latency from the
`output_len=100` latency:

```text
99 / (1.017024 - 0.044115) = 101.76 tok/s
```

The script prints `nvidia-smi` first and refuses to run if the selected GPU has
an existing compute process. Set `ALLOW_BUSY_GPU=1` only for intentional manual
overrides.
