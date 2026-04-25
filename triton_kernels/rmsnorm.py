from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    n_cols,
    eps,
    x_stride_row,
    x_stride_col,
    y_stride_row,
    y_stride_col,
    BLOCK_COL: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_COL)

    # pass 1: accumulate sum(x^2) in fp32
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    col_start = 0
    while col_start < n_cols:
        cols = col_start + col_offsets
        mask = cols < n_cols
        x = tl.load(
            x_ptr + row_idx * x_stride_row + cols * x_stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
        col_start += BLOCK_COL

    inv_rms = tl.rsqrt(sum_sq / n_cols + eps)

    # pass 2: normalize + scale
    col_start = 0
    while col_start < n_cols:
        cols = col_start + col_offsets
        mask = cols < n_cols
        x = tl.load(
            x_ptr + row_idx * x_stride_row + cols * x_stride_col,
            mask=mask,
            other=0.0,
        )
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x.to(tl.float32) * inv_rms * w
        tl.store(
            y_ptr + row_idx * y_stride_row + cols * y_stride_col,
            y.to(x.dtype),
            mask=mask,
        )
        col_start += BLOCK_COL


def rmsnorm_triton(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # decode path expected to be bf16/fp16 on CUDA; fallback should be handled by caller.
    if hidden_states.device.type != "cuda":
        raise NotImplementedError("rmsnorm_triton requires CUDA tensor")
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError("rmsnorm_triton only supports bf16/fp16 input")
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError("rmsnorm_triton only supports bf16/fp16/fp32 weight")
    if not weight.is_contiguous():
        raise NotImplementedError("rmsnorm_triton requires contiguous weight")

    x = hidden_states
    shape = x.shape
    n_cols = int(shape[-1])
    x_2d = x.view(-1, n_cols)
    if x_2d.stride(1) != 1:
        raise NotImplementedError("rmsnorm_triton requires last dimension to be contiguous")
    rows = int(x_2d.shape[0])

    y = torch.empty((rows, n_cols), device=x.device, dtype=x.dtype)
    block_col = 512

    _rmsnorm_kernel[(rows,)](
        x_2d,
        weight,
        y,
        n_cols,
        float(eps),
        x_2d.stride(0),
        x_2d.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_COL=block_col,
        num_warps=4,
    )
    return y.reshape(shape)
