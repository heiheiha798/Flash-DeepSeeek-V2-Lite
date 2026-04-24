from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_kernels.moe_grouped_gemv import PackedRoutedExperts, grouped_routed_moe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile MoE Triton v2 kernels with NCU.")
    return parser.parse_args()


def main() -> None:
    _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    hidden_size = 2048
    intermediate_size = 1408
    num_experts = 64
    topk = 6

    x = torch.randn((1, hidden_size), device=device, dtype=dtype)
    topk_ids = torch.tensor([[1, 7, 9, 13, 21, 42]], device=device, dtype=torch.long)
    topk_weight = torch.randn((1, topk), device=device, dtype=torch.float32).softmax(dim=-1)
    gate_up_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
    )
    down_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=dtype,
    )

    packed = PackedRoutedExperts(
        gate_up_weights=gate_up_weights,
        down_weights=down_weights,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
    )
    # warmup launch (compile + first run)
    grouped_routed_moe(x, topk_ids, topk_weight, packed)
    torch.cuda.synchronize()
    # profiled launch
    grouped_routed_moe(x, topk_ids, topk_weight, packed)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
