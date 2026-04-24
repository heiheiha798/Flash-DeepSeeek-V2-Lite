#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

CUDA_ARCH=${CUDA_ARCH:-80}
BUILD_JOBS=${BUILD_JOBS:-16}

if [[ ! -d "${LLAMA_CPP_ENV}" ]]; then
  conda create -y -p "${LLAMA_CPP_ENV}" python=3.11 cmake ninja git
fi

if [[ ! -d "${LLAMA_CPP_DIR}/.git" ]]; then
  git clone "${LLAMA_CPP_REPO}" "${LLAMA_CPP_DIR}"
else
  git -C "${LLAMA_CPP_DIR}" remote -v
fi

conda run -p "${LLAMA_CPP_ENV}" cmake \
  -S "${LLAMA_CPP_DIR}" \
  -B "${LLAMA_CPP_BUILD_DIR}" \
  -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DCMAKE_BUILD_TYPE=Release

conda run -p "${LLAMA_CPP_ENV}" cmake \
  --build "${LLAMA_CPP_BUILD_DIR}" \
  --config Release \
  -j "${BUILD_JOBS}"
