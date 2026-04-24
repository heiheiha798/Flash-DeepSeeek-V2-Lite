#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

LLAMA_BENCH=$(llama_cpp_bin llama-bench)
PROMPT_TOKENS=${PROMPT_TOKENS:-24}
GEN_TOKENS=${GEN_TOKENS:-100}
REPETITIONS=${REPETITIONS:-3}
N_GPU_LAYERS=${N_GPU_LAYERS:-99}
FLASH_ATTN=${FLASH_ATTN:-1}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
OUTPUT_FORMAT=${OUTPUT_FORMAT:-md}

require_file "${DSV2_GGUF}"
require_executable "${LLAMA_BENCH}"
check_gpu_available

CUDA_VISIBLE_DEVICES="${GPU}" "${LLAMA_BENCH}" \
  -m "${DSV2_GGUF}" \
  -p "${PROMPT_TOKENS}" \
  -n "${GEN_TOKENS}" \
  -ngl "${N_GPU_LAYERS}" \
  -fa "${FLASH_ATTN}" \
  -b "${BATCH_SIZE}" \
  -ub "${UBATCH_SIZE}" \
  -r "${REPETITIONS}" \
  -o "${OUTPUT_FORMAT}"
