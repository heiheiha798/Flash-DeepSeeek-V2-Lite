from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import triton

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_kernels.attention_decode import _gemv_contig_kernel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile attention q_kv_a GEMV kernel with NCU.")
    parser.add_argument("--out-rows", type=int, default=3648)
    parser.add_argument("--in-cols", type=int, default=2048)
    parser.add_argument("--block-row", type=int, default=64)
    parser.add_argument("--block-col", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=3)
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

    x = torch.randn((args.in_cols,), device=device, dtype=dtype)
    w = torch.randn((args.out_rows, args.in_cols), device=device, dtype=dtype)
    out = torch.empty((args.out_rows,), device=device, dtype=dtype)

    grid = (triton.cdiv(args.out_rows, args.block_row),)
    exact_rows = args.out_rows % args.block_row == 0

    for _ in range(args.warmup):
        _gemv_contig_kernel[grid](
            x,
            w,
            out,
            args.out_rows,
            BLOCK_ROW=args.block_row,
            BLOCK_COL=args.block_col,
            IN_COLS=args.in_cols,
            EXACT_ROWS=exact_rows,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
    torch.cuda.synchronize()

    for _ in range(args.iters):
        _gemv_contig_kernel[grid](
            x,
            w,
            out,
            args.out_rows,
            BLOCK_ROW=args.block_row,
            BLOCK_COL=args.block_col,
            IN_COLS=args.in_cols,
            EXACT_ROWS=exact_rows,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
