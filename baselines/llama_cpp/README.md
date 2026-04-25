# llama.cpp Baseline

This baseline keeps llama.cpp outside this repository and records the exact
commands used to compare against the DeepSeek-V2-Lite Triton path.

## Paths

- llama.cpp source: `/data/home/tianjianyang/code/llama.cpp`
- conda env: `/data/home/tianjianyang/.conda/envs/dsv2lite-llamacpp`
- CUDA build: `/data/home/tianjianyang/code/llama.cpp/build-cuda-a100`
- default model: `/data/models/gguf-models/DeepSeek-V2-Lite-Chat-F16.gguf`

The similarly named `/data/models/gguf_models/DeepSeek-V2-Lite-Chat-F16.gguf`
was not used because it is a zero-byte file without read permission for this
user.

## Setup

```bash
baselines/llama_cpp/setup_llama_cpp.sh
```

The setup script clones llama.cpp over SSH if needed and builds with:

```bash
-DGGML_CUDA=ON
-DGGML_CUDA_GRAPHS=ON
-DGGML_CUDA_FA=ON
-DCMAKE_CUDA_ARCHITECTURES=80
```

## Bench

Run the fixed decode benchmark on GPU0:

```bash
GPU=0 baselines/llama_cpp/bench_dsv2_lite.sh
```

Run the fixed real-parallel `llama-batched-bench -npl` sweep:

```bash
GPU=0 baselines/llama_cpp/bench_dsv2_lite_batch_sweep.sh
```

Default sweep values:

```text
1 2 4 8 16 32 64 128 256
```

This sweep uses `llama-batched-bench -npl`, so `BATCH_SIZES` maps to real
parallel sequences. The 512-point is excluded in this setup because the current
llama.cpp build rejects `npl=512` with `n_seq_max must be <= 256`. Do not fall
back to `llama-bench -b/-ub`, which is not concurrent-request batch size.
Override with `BATCH_SIZES`, for example `BATCH_SIZES="1 8 64"`. Sweep logs are
written under `RESULT_DIR`, defaulting to `/tmp/llama_cpp_dsv2lite_batch_sweep`.

Default benchmark shape:

- prompt tokens: `24`
- generated tokens: `100`
- repetitions: `3`
- GPU layers: `99`
- llama.cpp FlashAttention: `on`
- eval batch / micro-batch: `BATCH_SIZE=2048`, `UBATCH_SIZE=512` in the single-run script

Latest observed A100 GPU3 `llama-batched-bench -npl` batch-size sweep result:

```text
bsz=1 tg100:   137.01 tok/s
bsz=256 tg100: 2051.89 tok/s
log: /tmp/dsv2lite_a100_rerun_20260425_162309/llama_cpp/llama_batched_bench_npl_1_2_4_8_16_32_64_128_256.jsonl
hardware: NVIDIA A100 80GB PCIe, sm80, 80 GB; INTEL(R) XEON(R) PLATINUM 8558P, 96C/192T, 503.53 GiB RAM
software: llama.cpp CUDA build with CUDA graphs and GGML CUDA FlashAttention
```

## Generate

Run one non-interactive generation:

```bash
GPU=0 MAX_NEW_TOKENS=100 baselines/llama_cpp/generate_dsv2_lite.sh
```

Default prompt:

```text
Write me a 500 word novel
```

The scripts print `nvidia-smi` first and refuse to run if the selected GPU has
an existing compute process. Set `ALLOW_BUSY_GPU=1` only for intentional manual
overrides.
