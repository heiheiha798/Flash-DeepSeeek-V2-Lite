#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

require_executable "${SGLANG_PYTHON}"

"${SGLANG_PYTHON}" - <<'PY'
import importlib.metadata as metadata

import flash_attn
import torch
import torchaudio
import torchvision
import triton
from flash_attn import flash_attn_func

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__)
print("flash_attn", flash_attn.__version__)
print("flash_attn_func ok", flash_attn_func is not None)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
for name in ["sglang", "torchao", "flashinfer-python", "flashinfer-cubin", "sgl-kernel"]:
    print(name, metadata.version(name))
PY
