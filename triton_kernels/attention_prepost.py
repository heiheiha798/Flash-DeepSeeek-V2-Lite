from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_decode_inputs_kernel(
    pos_ptr,
    attention_mask_ptr,
    position_ids_ptr,
    attention_mask_index_ptr,
    max_cache_len,
    batch_size,
    attn_stride_b,
    attn_stride_s,
    pos_ids_stride_b,
    pos_ids_stride_s,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    pid = tl.program_id(1)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < max_cache_len

    if batch_idx >= batch_size:
        return

    pos = tl.load(pos_ptr).to(tl.int64)
    attn_index = tl.load(attention_mask_index_ptr + offsets, mask=mask, other=0).to(tl.int64)
    valid = attn_index <= pos
    tl.store(attention_mask_ptr + batch_idx * attn_stride_b + offsets * attn_stride_s, valid.to(tl.int64), mask=mask)

    if pid == 0:
        tl.store(position_ids_ptr + batch_idx * pos_ids_stride_b + 0 * pos_ids_stride_s, pos)


@triton.jit
def _cache_write_q1_kernel(
    src_ptr,
    dst_ptr,
    pos_ptr,
    num_heads,
    head_dim,
    src_stride_b,
    src_stride_h,
    src_stride_s,
    src_stride_d,
    dst_stride_b,
    dst_stride_h,
    dst_stride_s,
    dst_stride_d,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    d_block = tl.program_id(2)
    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    if head_idx >= num_heads:
        return

    # cache_position is a device tensor in graph path; load in-kernel to stay graph-safe.
    pos = tl.load(pos_ptr).to(tl.int32)
    src_ptrs = (
        src_ptr
        + batch_idx * src_stride_b
        + head_idx * src_stride_h
        + 0 * src_stride_s
        + d_offsets * src_stride_d
    )
    dst_ptrs = (
        dst_ptr
        + batch_idx * dst_stride_b
        + head_idx * dst_stride_h
        + pos * dst_stride_s
        + d_offsets * dst_stride_d
    )
    x = tl.load(src_ptrs, mask=d_mask, other=0.0)
    tl.store(dst_ptrs, x, mask=d_mask)


@triton.jit
def _residual_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_cols,
    a_stride_row,
    a_stride_col,
    b_stride_row,
    b_stride_col,
    out_stride_row,
    out_stride_col,
    BLOCK_COL: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)
    col_offsets = col_block * BLOCK_COL + tl.arange(0, BLOCK_COL)
    col_mask = col_offsets < n_cols

    a_ptrs = a_ptr + row_idx * a_stride_row + col_offsets * a_stride_col
    b_ptrs = b_ptr + row_idx * b_stride_row + col_offsets * b_stride_col
    out_ptrs = out_ptr + row_idx * out_stride_row + col_offsets * out_stride_col

    a = tl.load(a_ptrs, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptrs, mask=col_mask, other=0.0).to(tl.float32)
    tl.store(out_ptrs, (a + b).to(tl.bfloat16), mask=col_mask)


def copy_single_token_to_cache(
    src: torch.Tensor,
    dst: torch.Tensor,
    cache_position: torch.Tensor,
) -> None:
    # expected decode shapes: src=[batch, heads, 1, dim], dst=[batch, heads, max_seq, dim]
    if src.device.type != "cuda" or dst.device.type != "cuda":
        raise NotImplementedError("copy_single_token_to_cache requires CUDA tensors")
    if src.ndim != 4 or dst.ndim != 4 or src.shape[2] != 1 or src.shape[0] != dst.shape[0]:
        raise NotImplementedError("copy_single_token_to_cache only supports [B, H, 1, D] -> [B, H, S, D]")
    if src.dtype != dst.dtype:
        raise ValueError("src and dst dtype must match")
    if cache_position.device.type != "cuda" or cache_position.numel() != 1:
        raise ValueError("cache_position must be a CUDA tensor with one element")

    batch_size = int(src.shape[0])
    num_heads = int(src.shape[1])
    head_dim = int(src.shape[3])
    block_d = 128
    grid = (batch_size, num_heads, triton.cdiv(head_dim, block_d))
    _cache_write_q1_kernel[grid](
        src,
        dst,
        cache_position,
        num_heads,
        head_dim,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        src.stride(3),
        dst.stride(0),
        dst.stride(1),
        dst.stride(2),
        dst.stride(3),
        BLOCK_D=block_d,
        num_warps=2,
    )


def prepare_decode_inputs_triton(
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask_index: torch.Tensor,
) -> None:
    if cache_position.device.type != "cuda":
        raise NotImplementedError("prepare_decode_inputs_triton requires CUDA tensors")
    if cache_position.numel() != 1:
        raise ValueError("cache_position must have one element")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, max_cache_len]")
    if position_ids.ndim != 2 or position_ids.shape[0] != attention_mask.shape[0] or position_ids.shape[1] != 1:
        raise ValueError("position_ids must have shape [batch, 1]")
    if attention_mask_index.ndim != 1 or attention_mask_index.shape[0] != attention_mask.shape[1]:
        raise ValueError("attention_mask_index must be 1D with length max_cache_len")
    if attention_mask.dtype != torch.long or position_ids.dtype != torch.long or attention_mask_index.dtype != torch.long:
        raise ValueError("prepare_decode_inputs_triton requires int64 tensors")

    batch_size = int(attention_mask.shape[0])
    max_cache_len = int(attention_mask.shape[1])
    block = 256
    grid = (batch_size, triton.cdiv(max_cache_len, block))
    _prepare_decode_inputs_kernel[grid](
        cache_position,
        attention_mask,
        position_ids,
        attention_mask_index,
        max_cache_len,
        batch_size,
        attention_mask.stride(0),
        attention_mask.stride(1),
        position_ids.stride(0),
        position_ids.stride(1),
        BLOCK=block,
        num_warps=2,
    )


def residual_add_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise NotImplementedError("residual_add_triton requires CUDA tensors")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype:
        raise NotImplementedError("residual_add_triton supports bf16/fp16 with same dtype")
    if not a.is_contiguous() or not b.is_contiguous():
        raise NotImplementedError("residual_add_triton requires contiguous inputs")

    n_cols = int(a.shape[-1])
    rows = int(a.numel() // n_cols)
    a_2d = a.reshape(rows, n_cols)
    b_2d = b.reshape(rows, n_cols)
    out = torch.empty_like(a_2d)

    block_col = 256
    grid = (rows, triton.cdiv(n_cols, block_col))
    _residual_add_kernel[grid](
        a_2d,
        b_2d,
        out,
        n_cols,
        a_2d.stride(0),
        a_2d.stride(1),
        b_2d.stride(0),
        b_2d.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_COL=block_col,
        num_warps=4,
    )
    return out.reshape(a.shape)
