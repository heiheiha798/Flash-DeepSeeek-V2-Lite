#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

BATCH_SIZES=${BATCH_SIZES:-"1 2 4 8 16 32 64 128 256"}
RESULT_DIR=${RESULT_DIR:-/tmp/vllm_dsv2lite_batch_sweep}

normalize_batch_sizes() {
  local sizes=${BATCH_SIZES//,/ }
  sizes=${sizes//，/ }
  echo "${sizes}"
}

require_file "${MODEL_PATH}/config.json"
require_executable "${VLLM_BIN}"
check_gpu_available
mkdir -p "${RESULT_DIR}"

read -r -a batch_size_array <<< "$(normalize_batch_sizes)"
for batch_size in "${batch_size_array[@]}"; do
  if [[ -z "${batch_size}" ]]; then
    continue
  fi

  capture_size=${CUDAGRAPH_CAPTURE_SIZE:-${batch_size}}
  log_file="${RESULT_DIR}/bs${batch_size}.log"

  echo "=== vLLM DeepSeek-V2-Lite batch_size=${batch_size} cudagraph_capture_size=${capture_size} ==="
  BATCH_SIZE="${batch_size}" \
  CUDAGRAPH_CAPTURE_SIZE="${capture_size}" \
  SKIP_GPU_CHECK=1 \
    "${SCRIPT_DIR}/bench_dsv2_lite_latency.sh" 2>&1 | tee "${log_file}"
done
