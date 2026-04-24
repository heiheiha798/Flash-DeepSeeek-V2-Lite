#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-/data/home/tianjianyang/code/llama.cpp}
LLAMA_CPP_BUILD_DIR=${LLAMA_CPP_BUILD_DIR:-${LLAMA_CPP_DIR}/build-cuda-a100}
LLAMA_CPP_REPO=${LLAMA_CPP_REPO:-git@github.com:ggml-org/llama.cpp.git}
LLAMA_CPP_ENV=${LLAMA_CPP_ENV:-/data/home/tianjianyang/.conda/envs/dsv2lite-llamacpp}
DSV2_GGUF=${DSV2_GGUF:-/data/home/tianjianyang/models/gguf-models/DeepSeek-V2-Lite-Chat-F16.gguf}
GPU=${GPU:-0}

llama_cpp_bin() {
  local name=$1
  printf '%s/bin/%s\n' "${LLAMA_CPP_BUILD_DIR}" "${name}"
}

require_file() {
  local path=$1
  if [[ ! -f "${path}" ]]; then
    echo "missing file: ${path}" >&2
    exit 1
  fi
}

require_executable() {
  local path=$1
  if [[ ! -x "${path}" ]]; then
    echo "missing executable: ${path}" >&2
    echo "run: baselines/llama_cpp/setup_llama_cpp.sh" >&2
    exit 1
  fi
}

check_gpu_available() {
  if [[ "${SKIP_GPU_CHECK:-0}" == "1" ]]; then
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found" >&2
    exit 1
  fi

  nvidia-smi

  local busy
  busy=$(nvidia-smi --id="${GPU}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)
  if [[ -n "${busy}" && "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
    echo "GPU ${GPU} already has compute processes:" >&2
    echo "${busy}" >&2
    echo "set ALLOW_BUSY_GPU=1 only if you intentionally want to override this guard" >&2
    exit 1
  fi
}
