# Third-Party Baselines

This note records the reproducible third-party baseline setup for
`DeepSeek-V2-Lite-Chat` downloaded from ModelScope.

## Model Artifacts

- HF/ModelScope weights: `/data/home/tianjianyang/models/DeepSeek-V2-Lite-Chat`
- llama.cpp F16 GGUF: `/data/home/tianjianyang/models/gguf-models/DeepSeek-V2-Lite-Chat-F16.gguf`

The baseline wrappers default to these local paths and can be overridden with
`MODEL_PATH=...` or `DSV2_GGUF=...`.

## Environments

| Baseline | Environment | Entry |
| --- | --- | --- |
| SGLang | `/data/home/tianjianyang/.conda/envs/dsv2lite-sglang-fa311` | `baselines/sglang/bench_dsv2_lite.sh` |
| vLLM | `/data/home/tianjianyang/.conda/envs/dsv2lite-vllm` | `baselines/vllm/bench_dsv2_lite_latency.sh` |
| llama.cpp | `/data/home/tianjianyang/.conda/envs/dsv2lite-llamacpp` plus `/data/home/tianjianyang/code/llama.cpp/build-cuda-a100` | `baselines/llama_cpp/bench_dsv2_lite.sh` |

## Single-Batch Decode Results

All results below use RTX A6000, batch size 1, input length 24, output length
100, BF16/F16 weights as supported by each backend, and GPU 2 when rerun.

| Backend | Command shape | Decode TPS | Log |
| --- | --- | ---: | --- |
| SGLang 0.5.9 | `bench_one_batch --batch-size 1 --input-len 24 --output-len 100 --cuda-graph-bs 1` | 110.21 tok/s | `/tmp/sglang_dsv2_lite_gpu2_rerun.log` |
| vLLM 0.16.0 | `vllm bench latency --batch-size 1 --input-len 24 --output-len 100 --cudagraph-capture-sizes 1` | 62.89 tok/s | `/tmp/vllm_dsv2_lite_gpu2_out1.log` |
| llama.cpp | `llama-bench -p 24 -n 100 -ngl 99 -fa 1 -b 2048 -ub 512` | 101.56 tok/s | `/tmp/llama_cpp_dsv2_lite_gpu2.log` |

## SGLang Notes

SGLang is faster than vLLM and llama.cpp in this reproduction for the comparable
single-batch decode path. The earlier 109.06 tok/s value was from the same path;
the rerun reports 110.21 tok/s.

The SGLang log still warns that the RTX A6000 MoE tuning config is missing:

```text
E=64,N=1408,device_name=NVIDIA_RTX_A6000.json
E=64,N=1408,device_name=NVIDIA_RTX_A6000_down.json
```

So the current SGLang result is a functional official default, but not a fully
auto-tuned A6000 MoE result. On hardware with tuned configs, or after generating
those configs with SGLang's fused MoE benchmark tools, SGLang may improve.

## Reproduction Commands

```bash
GPU=2 RESULT_FILENAME=/tmp/sglang_dsv2lite_bench_rerun.jsonl \
  baselines/sglang/bench_dsv2_lite.sh 2>&1 | tee /tmp/sglang_dsv2_lite_gpu2_rerun.log

GPU=2 baselines/vllm/bench_dsv2_lite_latency.sh 2>&1 | tee /tmp/vllm_dsv2_lite_gpu2_rerun.log

GPU=2 baselines/llama_cpp/bench_dsv2_lite.sh 2>&1 | tee /tmp/llama_cpp_dsv2_lite_gpu2_rerun.log
```
