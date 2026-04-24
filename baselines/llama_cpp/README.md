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

Run the fixed `llama-bench` batch-parameter sweep:

```bash
GPU=0 baselines/llama_cpp/bench_dsv2_lite_batch_sweep.sh
```

Default sweep values:

```text
1 2 4 8 16 32 64 128 256 512
```

For llama.cpp, this sweep maps to `llama-bench -b` and defaults `-ub` to the
same value. This is not the same concept as SGLang/vLLM concurrent request
batch size; it is llama.cpp's internal eval batch / micro-batch parameter.
Override with `BATCH_SIZES`, for example `BATCH_SIZES="1 8 64"`. Sweep logs are
written under `RESULT_DIR`, defaulting to `/tmp/llama_cpp_dsv2lite_batch_sweep`.

Default benchmark shape:

- prompt tokens: `24`
- generated tokens: `100`
- repetitions: `3`
- GPU layers: `99`
- llama.cpp FlashAttention: `on`
- eval batch / micro-batch: `BATCH_SIZE=2048`, `UBATCH_SIZE=512` in the single-run script

Latest observed GPU0 result from `bench_dsv2_lite.sh`:

```text
pp24:  1045.36 +/- 27.80 tok/s
tg100: 115.90 +/- 0.05 tok/s
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
