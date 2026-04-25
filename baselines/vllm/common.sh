#!/usr/bin/env bash
set -euo pipefail

VLLM_ENV=${VLLM_ENV:-/data/home/tianjianyang/.conda/envs/dsv2lite-vllm}
VLLM_BIN=${VLLM_BIN:-${VLLM_ENV}/bin/vllm}
VLLM_PYTHON=${VLLM_PYTHON:-${VLLM_ENV}/bin/python}
MODEL_PATH=${MODEL_PATH:-/data/models/DeepSeek-V2-Lite-Chat}
GPU=${GPU:-0}
RUN_DIR=${RUN_DIR:-/tmp}

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
