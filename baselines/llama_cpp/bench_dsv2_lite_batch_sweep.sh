#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

BATCH_SIZES=${BATCH_SIZES:-"1 2 4 8 16 32 64 128 256"}
RESULT_DIR=${RESULT_DIR:-/tmp/llama_cpp_dsv2lite_batch_sweep}
LLAMA_BATCHED_BENCH=$(llama_cpp_bin llama-batched-bench)
PROMPT_TOKENS=${PROMPT_TOKENS:-24}
GEN_TOKENS=${GEN_TOKENS:-100}
CTX_SIZE=${CTX_SIZE:-65536}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
N_GPU_LAYERS=${N_GPU_LAYERS:-99}
FLASH_ATTN=${FLASH_ATTN:-on}

normalize_batch_sizes() {
  local sizes=${BATCH_SIZES//,/ }
  sizes=${sizes//，/ }
  echo "${sizes}"
}

require_file "${DSV2_GGUF}"
require_executable "${LLAMA_BATCHED_BENCH}"
check_gpu_available
mkdir -p "${RESULT_DIR}"

parallel_sizes=$(normalize_batch_sizes | tr ' ' ',')
log_file="${RESULT_DIR}/llama_batched_bench_npl_${parallel_sizes//,/_}.jsonl"

echo "=== llama.cpp DeepSeek-V2-Lite llama-batched-bench -npl ${parallel_sizes} ==="
CUDA_VISIBLE_DEVICES="${GPU}" "${LLAMA_BATCHED_BENCH}" \
  -m "${DSV2_GGUF}" \
  -c "${CTX_SIZE}" \
  -b "${BATCH_SIZE}" \
  -ub "${UBATCH_SIZE}" \
  -ngl "${N_GPU_LAYERS}" \
  -fa "${FLASH_ATTN}" \
  --output-format jsonl \
  -npp "${PROMPT_TOKENS}" \
  -ntg "${GEN_TOKENS}" \
  -npl "${parallel_sizes}" \
  2>&1 | tee "${log_file}"
