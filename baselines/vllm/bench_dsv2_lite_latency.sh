#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

BATCH_SIZE=${BATCH_SIZE:-1}
INPUT_LEN=${INPUT_LEN:-24}
OUTPUT_LEN=${OUTPUT_LEN:-100}
NUM_ITERS_WARMUP=${NUM_ITERS_WARMUP:-1}
NUM_ITERS=${NUM_ITERS:-3}
DTYPE=${DTYPE:-bfloat16}
CUDAGRAPH_CAPTURE_SIZE=${CUDAGRAPH_CAPTURE_SIZE:-1}

require_file "${MODEL_PATH}/config.json"
require_executable "${VLLM_BIN}"
check_gpu_available

# Run outside this repository so repo-local triton_kernels/ does not shadow
# optional packages imported by vLLM.
cd "${RUN_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${VLLM_BIN}" bench latency \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --input-len "${INPUT_LEN}" \
  --output-len "${OUTPUT_LEN}" \
  --num-iters-warmup "${NUM_ITERS_WARMUP}" \
  --num-iters "${NUM_ITERS}" \
  --cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZE}" \
  --disable-detokenize
