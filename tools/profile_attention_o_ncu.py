from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_kernels.attention_decode_small import _gemv_2048x2048_o_kernel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile attention o_proj GEMV kernel with NCU.")
    parser.add_argument("--block-row", type=int, default=64)
    parser.add_argument("--block-col", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn((2048,), device=device, dtype=dtype)
    w = torch.randn((2048, 2048), device=device, dtype=dtype)
    out = torch.empty((2048,), device=device, dtype=dtype)

    grid = (2048 // args.block_row,)

    for _ in range(args.warmup):
        _gemv_2048x2048_o_kernel[grid](
            x,
            w,
            out,
            BLOCK_ROW=args.block_row,
            BLOCK_COL=args.block_col,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
    torch.cuda.synchronize()

    for _ in range(args.iters):
        _gemv_2048x2048_o_kernel[grid](
            x,
            w,
            out,
            BLOCK_ROW=args.block_row,
            BLOCK_COL=args.block_col,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
