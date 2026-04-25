# Third-Party Baselines

This note records the reproducible third-party baseline setup for
`DeepSeek-V2-Lite-Chat` downloaded from ModelScope.

## Model Artifacts

- HF/ModelScope weights: `/data/models/DeepSeek-V2-Lite-Chat`
- llama.cpp F16 GGUF: `/data/models/gguf-models/DeepSeek-V2-Lite-Chat-F16.gguf`

The baseline wrappers default to these local paths and can be overridden with
`MODEL_PATH=...` or `DSV2_GGUF=...`.

## Environments

| Baseline | Environment | Entry |
| --- | --- | --- |
| SGLang | `/data/home/tianjianyang/.conda/envs/dsv2lite-sglang-fa311` | `baselines/sglang/bench_dsv2_lite.sh` |
| vLLM | `/data/home/tianjianyang/.conda/envs/dsv2lite-vllm` | `baselines/vllm/bench_dsv2_lite_latency.sh` |
| llama.cpp | `/data/home/tianjianyang/.conda/envs/dsv2lite-llamacpp` plus `/data/home/tianjianyang/code/llama.cpp/build-cuda-a100` | `baselines/llama_cpp/bench_dsv2_lite.sh` |

## Single-Batch Decode Results

All results below use NVIDIA A100 80GB PCIe, batch size 1, input length 24,
output length 100, BF16/F16 weights as supported by each backend, and physical
GPU 3 when rerun.

| Backend | Command shape | Decode TPS | Log |
| --- | --- | ---: | --- |
| SGLang 0.5.9 | `bench_one_batch --batch-size 1 --input-len 24 --output-len 100 --cuda-graph-bs 1` | 165.16 tok/s | `/tmp/dsv2lite_a100_rerun_20260425_162309/sglang/bs1.jsonl` |
| vLLM 0.16.0 | `vllm bench latency --batch-size 1 --input-len 24 --output-len 100 --cudagraph-capture-sizes 1` | 78.44 tok/s | `/tmp/dsv2lite_a100_rerun_20260425_162309/vllm/bs1.log` |
| llama.cpp | `llama-batched-bench -npp 24 -ntg 100 -npl 1 -ngl 99 -fa on` | 137.01 tok/s | `/tmp/dsv2lite_a100_rerun_20260425_162309/llama_cpp/llama_batched_bench_npl_1_2_4_8_16_32_64_128_256.jsonl` |

## SGLang Notes

SGLang is faster than vLLM and llama.cpp in this reproduction for the comparable
single-batch decode path. The April 25, 2026 A100 rerun reports 165.16 tok/s.

The SGLang log still warns that the A100 MoE tuning config is missing:

```text
E=64,N=1408,device_name=NVIDIA_A100_80GB_PCIe.json
E=64,N=1408,device_name=NVIDIA_A100_80GB_PCIe_down.json
```

So the current SGLang result is a functional official default, but not a fully
auto-tuned A100 MoE result. On hardware with tuned configs, or after generating
those configs with SGLang's fused MoE benchmark tools, SGLang may improve.

## Batch-Size Sweep Results

The third-party batch-size sweep below was rerun serially on physical GPU 3 on
April 25, 2026. Shape is input length 24 and output length 100. Logs are under
`/tmp/dsv2lite_a100_rerun_20260425_162309`.

<div align="center">
  <img src="figures/batch_scaling.svg" alt="Batch-size throughput scaling" width="760">
</div>

The plot includes the third-party baselines plus the custom Triton
`src/run.py --kernel-family batch` and `--kernel-family small` paths.

| Batch | SGLang decode tok/s | vLLM output tok/s | llama.cpp batched tg tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 165.16 | 78.44 | 137.01 |
| 2 | 256.48 | 185.52 | 226.56 |
| 4 | 379.13 | 390.01 | 367.57 |
| 8 | 599.85 | 585.65 | 593.93 |
| 16 | 953.28 | 930.27 | 849.41 |
| 32 | 1585.99 | 1549.51 | 1285.17 |
| 64 | 2833.83 | 2638.93 | 1838.71 |
| 128 | 3945.98 | 3655.33 | 2389.42 |
| 256 | 7094.12 | 6113.90 | 2051.89 |

Notes:

- SGLang values are `median_decode_throughput` from `bench_one_batch` JSONL with `--cuda-graph-bs` matching batch size.
- vLLM values are `batch_size * output_len / Avg latency` from `vllm bench latency`; this includes prefill plus decode for the request batch, so it is not exactly equivalent to SGLang decode-only median. The attempted `batch_size=512` run is excluded because vLLM scheduled it as two 256-request waves (`Running: 256 reqs, Waiting: 256 reqs`), so it was not a valid bsz=512 measurement.
- llama.cpp values are from `llama-batched-bench -npl`, i.e. real parallel sequences. The tool refused `npl=512` with `n_seq_max must be <= 256`, so the valid llama.cpp curve stops at 256.


## src Batch Sweep

The current custom Triton path was rerun on physical GPU 3 on April 25, 2026
after the batch attention linear tile update, and the small-GEMV curve was
refreshed on April 26, 2026 after restoring combined gate/up dot in the small
MoE kernels. Shape is input length 24 and output length 100. Batch-kernel logs
are under `/tmp/dsv2lite_src_rerun_20260425_232359`; refreshed small-GEMV logs
are under `/tmp/dsv2lite_small_combined_all_20260426_000346`. The table compares
two forced dispatch modes: batching kernels for every batch size, and
small-GEMV kernels for every measured batch size.

| Batch | Forced batching tok/s | Forced small-GEMV tok/s |
| ---: | ---: | ---: |
| 1 | 154.03 | 197.98 |
| 2 | 270.65 | 272.59 |
| 4 | 518.80 | 460.22 |
| 8 | 958.35 | 614.82 |
| 16 | 1914.46 | 802.99 |
| 32 | 3608.94 | 918.80 |
| 64 | 6237.99 | 1006.51 |
| 128 | 9109.06 | 1057.46 |
| 256 | 11428.26 | 1089.71 |

Note: all plotted backends are capped at batch size 256. The earlier 512 point
is intentionally excluded from the plot and table.

The plotted `src/small GEMV` series is diagnostic. It runs the `bsz=1`-template
GEMV family through `B=256` to compare a distinct kernel implementation method
against full batching kernels; it is not presented as the final dispatch policy.

## Reproduction Commands

```bash
RUN_DIR=/tmp/dsv2lite_a100_rerun_20260425_162309
mkdir -p "${RUN_DIR}"

RESULT_DIR="${RUN_DIR}/sglang" GPU=3 MODEL_PATH=/data/models/DeepSeek-V2-Lite-Chat BATCH_SIZES="1 2 4 8 16 32 64 128 256" \
  baselines/sglang/bench_dsv2_lite_batch_sweep.sh 2>&1 | tee "${RUN_DIR}/sglang.log"

RESULT_DIR="${RUN_DIR}/vllm" GPU=3 MODEL_PATH=/data/models/DeepSeek-V2-Lite-Chat BATCH_SIZES="1 2 4 8 16 32 64 128 256" \
  baselines/vllm/bench_dsv2_lite_batch_sweep.sh 2>&1 | tee "${RUN_DIR}/vllm.log"

RESULT_DIR="${RUN_DIR}/llama_cpp" GPU=3 DSV2_GGUF=/data/models/gguf-models/DeepSeek-V2-Lite-Chat-F16.gguf BATCH_SIZES="1 2 4 8 16 32 64 128 256" \
  baselines/llama_cpp/bench_dsv2_lite_batch_sweep.sh 2>&1 | tee "${RUN_DIR}/llama_cpp.log"

CUDA_VISIBLE_DEVICES=3 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family batch \
  --model-path /data/models/DeepSeek-V2-Lite-Chat --device cuda:0 \
  --max-new-tokens 100 --batch-sizes "1 2 4 8 16 32 64 128 256" \
  2>&1 | tee "${RUN_DIR}/src_batch_sweep.log"

CUDA_VISIBLE_DEVICES=3 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family small \
  --model-path /data/models/DeepSeek-V2-Lite-Chat --device cuda:0 \
  --max-new-tokens 100 --batch-sizes "1 2 4 8 16 32 64 128 256" \
  2>&1 | tee "${RUN_DIR}/src_small_gemv_sweep.log"
```

The refreshed `src/run.py` logs for the current table used:

```bash
RUN_DIR=/tmp/dsv2lite_src_rerun_20260425_232359
SMALL_RUN_DIR=/tmp/dsv2lite_small_combined_all_20260426_000346
```
