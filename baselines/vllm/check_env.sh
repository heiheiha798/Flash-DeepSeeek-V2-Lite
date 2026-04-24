#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

require_executable "${VLLM_PYTHON}"

cd "${RUN_DIR}"
"${VLLM_PYTHON}" - <<'PY'
import importlib.metadata as metadata

import torch
import triton
import vllm

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__)
print("vllm", metadata.version("vllm"))
print("flashinfer-python", metadata.version("flashinfer-python"))

try:
    import triton_kernels
except ModuleNotFoundError:
    print("triton_kernels import", "not found from RUN_DIR, no repo shadowing")
else:
    print("triton_kernels", triton_kernels.__file__)
PY
