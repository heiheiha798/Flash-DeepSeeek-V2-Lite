#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

BATCH_SIZES=${BATCH_SIZES:-"1 2 4 8 16 32 64 128 256 512"}
RESULT_DIR=${RESULT_DIR:-/tmp/llama_cpp_dsv2lite_batch_sweep}

normalize_batch_sizes() {
  local sizes=${BATCH_SIZES//,/ }
  sizes=${sizes//，/ }
  echo "${sizes}"
}

require_file "${DSV2_GGUF}"
require_executable "$(llama_cpp_bin llama-bench)"
check_gpu_available
mkdir -p "${RESULT_DIR}"

read -r -a batch_size_array <<< "$(normalize_batch_sizes)"
for batch_size in "${batch_size_array[@]}"; do
  [[ -z "${batch_size}" ]] && continue
  ubatch_size=${UBATCH_SIZE:-${batch_size}}
  result_file="${RESULT_DIR}/b${batch_size}_ub${ubatch_size}.md"
  log_file="${RESULT_DIR}/b${batch_size}_ub${ubatch_size}.log"

  echo "=== llama.cpp DeepSeek-V2-Lite -b ${batch_size} -ub ${ubatch_size} ==="
  BATCH_SIZE="${batch_size}" UBATCH_SIZE="${ubatch_size}" OUTPUT_FORMAT=md SKIP_GPU_CHECK=1 \
    "${SCRIPT_DIR}/bench_dsv2_lite.sh" 2>&1 | tee "${log_file}" | tee "${result_file}" >/dev/null
done
