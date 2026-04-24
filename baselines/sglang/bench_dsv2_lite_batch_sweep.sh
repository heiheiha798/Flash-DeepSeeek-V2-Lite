#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

BATCH_SIZES=${BATCH_SIZES:-"1 2 4 8 16 32 64 128 256 512"}
RESULT_DIR=${RESULT_DIR:-/tmp/sglang_dsv2lite_batch_sweep}

normalize_batch_sizes() {
  local sizes=${BATCH_SIZES//,/ }
  sizes=${sizes//，/ }
  echo "${sizes}"
}

require_file "${MODEL_PATH}/config.json"
require_executable "${SGLANG_PYTHON}"
check_gpu_available
mkdir -p "${RESULT_DIR}"

read -r -a batch_size_array <<< "$(normalize_batch_sizes)"
for batch_size in "${batch_size_array[@]}"; do
  if [[ -z "${batch_size}" ]]; then
    continue
  fi

  cuda_graph_bs=${CUDA_GRAPH_BS:-${batch_size}}
  result_file="${RESULT_DIR}/bs${batch_size}.jsonl"
  log_file="${RESULT_DIR}/bs${batch_size}.log"

  echo "=== SGLang DeepSeek-V2-Lite batch_size=${batch_size} cuda_graph_bs=${cuda_graph_bs} ==="
  BATCH_SIZE="${batch_size}" \
  CUDA_GRAPH_BS="${cuda_graph_bs}" \
  RESULT_FILENAME="${result_file}" \
  SKIP_GPU_CHECK=1 \
    "${SCRIPT_DIR}/bench_dsv2_lite.sh" 2>&1 | tee "${log_file}"
done
