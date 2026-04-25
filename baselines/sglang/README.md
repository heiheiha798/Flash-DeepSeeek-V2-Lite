# SGLang Baseline

This baseline records the official SGLang path for `DeepSeek-V2-Lite-Chat`. On
A100/SM80, SGLang's default MLA attention backend is `triton`; CUDA graph is
enabled by default. The single-batch script defaults to batch size 1; the sweep script captures each tested batch size separately.

## Environment

Conda env:

```bash
/data/home/tianjianyang/.conda/envs/dsv2lite-sglang-fa311
```

Important installed packages:

```text
torch 2.9.1+cu130
triton 3.5.1
flash_attn 2.8.3
torchvision 0.24.1+cu130
torchaudio 2.9.1+cu130
sglang 0.5.9
torchao 0.9.0
flashinfer-python 0.6.3
sgl-kernel 0.3.21
```

`flash_attn` FA2 is installed from the local wheel, but SGLang 0.5.9 does not
use an explicit FA2 backend for DeepSeek-V2-Lite MLA on A100/SM80. The official
default backend is `triton`, which is the baseline used here.

Check the environment:

```bash
baselines/sglang/check_env.sh
```

## Bench

```bash
GPU=0 baselines/sglang/bench_dsv2_lite.sh
```

Run the fixed batch sweep:

```bash
GPU=0 baselines/sglang/bench_dsv2_lite_batch_sweep.sh
```

Default sweep batch sizes:

```text
1 2 4 8 16 32 64 128 256
```

Override with `BATCH_SIZES`, for example `BATCH_SIZES="1 8 64"`. Sweep logs
and JSONL outputs are written under `RESULT_DIR`, defaulting to
`/tmp/sglang_dsv2lite_batch_sweep`.

Default benchmark shape:

- batch size: `1`
- prompt/input tokens: `24`
- generated tokens: `100`
- dtype: `bfloat16`
- CUDA graph batch sizes: `BATCH_SIZE` in the sweep, `1` in the single-batch default

Latest observed A100 GPU3 decode result:

```text
median decode latency: 0.00605 s
decode TPS: 165.16 tok/s
log: /tmp/dsv2lite_a100_rerun_20260425_162309/sglang/bs1.jsonl
hardware: NVIDIA A100 80GB PCIe, sm80, 80 GB; INTEL(R) XEON(R) PLATINUM 8558P, 96C/192T, 503.53 GiB RAM
software: torch 2.9.1+cu130, triton 3.5.1, sglang 0.5.9, flash_attn 2.8.3
```

The script prints `nvidia-smi` first and refuses to run if the selected GPU has
an existing compute process. Set `ALLOW_BUSY_GPU=1` only for intentional manual
overrides.

## Current Caveat

On A100, SGLang reports missing fused MoE tuning configs for `E=64,N=1408,device_name=NVIDIA_A100_80GB_PCIe*.json`. The result is valid for the official default path, but tuned MoE configs may improve it.
