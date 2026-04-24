#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

BATCH_SIZE=${BATCH_SIZE:-1}
INPUT_LEN=${INPUT_LEN:-24}
OUTPUT_LEN=${OUTPUT_LEN:-100}
WARMUPS=${WARMUPS:-1}
DTYPE=${DTYPE:-bfloat16}
CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-1}
RESULT_FILENAME=${RESULT_FILENAME:-/tmp/sglang_dsv2lite_bench.jsonl}

require_file "${MODEL_PATH}/config.json"
require_executable "${SGLANG_PYTHON}"
check_gpu_available

CUDA_VISIBLE_DEVICES="${GPU}" "${SGLANG_PYTHON}" -m sglang.bench_one_batch \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --input-len "${INPUT_LEN}" \
  --output-len "${OUTPUT_LEN}" \
  --warmups "${WARMUPS}" \
  --cuda-graph-bs "${CUDA_GRAPH_BS}" \
  --result-filename "${RESULT_FILENAME}"
