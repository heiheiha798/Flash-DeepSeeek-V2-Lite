#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

LLAMA_CLI=$(llama_cpp_bin llama-cli)
PROMPT=${PROMPT:-Write me a 500 word novel}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-100}
CTX_SIZE=${CTX_SIZE:-2048}
N_GPU_LAYERS=${N_GPU_LAYERS:-99}
FLASH_ATTN=${FLASH_ATTN:-on}
TEMP=${TEMP:-0}

require_file "${DSV2_GGUF}"
require_executable "${LLAMA_CLI}"
check_gpu_available

CUDA_VISIBLE_DEVICES="${GPU}" "${LLAMA_CLI}" \
  -m "${DSV2_GGUF}" \
  -p "${PROMPT}" \
  -n "${MAX_NEW_TOKENS}" \
  -ngl "${N_GPU_LAYERS}" \
  -fa "${FLASH_ATTN}" \
  -c "${CTX_SIZE}" \
  --temp "${TEMP}" \
  --no-display-prompt \
  --single-turn \
  --perf
