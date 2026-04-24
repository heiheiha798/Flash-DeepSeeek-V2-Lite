import os
import platform
import subprocess
from importlib import metadata
from typing import Any

import psutil
import torch


def _round_gib(num_bytes: int) -> float:
    return round(num_bytes / 1024**3, 2)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    first_line = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
    return first_line or None


def _cuda_device_index(device: str | torch.device) -> int | None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None
    if torch_device.index is not None:
        return torch_device.index
    return torch.cuda.current_device()


def collect_hardware_info(device: str | torch.device) -> dict[str, Any]:
    cuda_index = _cuda_device_index(device)
    gpu: dict[str, Any] | None = None
    if cuda_index is not None:
        props = torch.cuda.get_device_properties(cuda_index)
        gpu = {
            "visible_index": cuda_index,
            "name": props.name,
            "capability": f"sm{props.major}{props.minor}",
            "sm_count": props.multi_processor_count,
            "memory_gb": _round_gib(props.total_memory),
        }

    return {
        "cpu": {
            "model": _cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "memory_gb": _round_gib(psutil.virtual_memory().total),
        },
        "gpu": gpu,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvidia_driver": _nvidia_driver_version(),
            "psutil": psutil.__version__,
            "triton": _package_version("triton"),
            "transformers": _package_version("transformers"),
        },
        "env": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
