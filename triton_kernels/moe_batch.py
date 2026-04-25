from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import triton
import triton.language as tl


@dataclass
class PackedRoutedExperts:
    # gate_up_weights: [num_experts, 2 * intermediate_size, hidden_size]
    # down_weights:    [num_experts, hidden_size, intermediate_size]
    gate_up_weights: torch.Tensor
    down_weights: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int

    autotune_cache: ClassVar[dict[tuple[int, int, int, torch.device], tuple[int, int, int]]] = {}

    def to(self, *args, **kwargs) -> "PackedRoutedExperts":
        self.gate_up_weights = self.gate_up_weights.to(*args, **kwargs)
        self.down_weights = self.down_weights.to(*args, **kwargs)
        return self


MOE_GROUPED_TILE_CANDIDATES: tuple[tuple[int, int, int], ...] = (
    (1, 32, 128),
    (2, 32, 128),
    (2, 32, 256),
    (4, 16, 64),
    (4, 32, 128),
    (4, 32, 256),
    (8, 16, 64),
    (8, 32, 64),
    (8, 32, 128),
    (8, 64, 128),
    (16, 16, 64),
    (16, 32, 64),
    (16, 64, 128),
    (32, 32, 64),
    (32, 64, 128),
)

# Kept disabled until the route-major path beats grouped kernels end to end.
MOE_ROUTE_GEMV_MAX_ROUTES = 0


@triton.jit
def _batched_down_reduce_topk6_kernel(
    partial_ptr,
    topk_weight_ptr,
    out_ptr,
    hidden_size,
    topk,
    partial_stride_r,
    partial_stride_m,
    topk_weight_stride_b,
    topk_weight_stride_s,
    out_stride_b,
    out_stride_m,
    BLOCK_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    row_block_idx = tl.program_id(1)
    row_offsets = row_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < hidden_size
    route_base = token_idx * topk

    w0 = tl.load(topk_weight_ptr + token_idx * topk_weight_stride_b + 0 * topk_weight_stride_s).to(tl.float32)
    w1 = tl.load(topk_weight_ptr + token_idx * topk_weight_stride_b + 1 * topk_weight_stride_s).to(tl.float32)
    w2 = tl.load(topk_weight_ptr + token_idx * topk_weight_stride_b + 2 * topk_weight_stride_s).to(tl.float32)
    w3 = tl.load(topk_weight_ptr + token_idx * topk_weight_stride_b + 3 * topk_weight_stride_s).to(tl.float32)
    w4 = tl.load(topk_weight_ptr + token_idx * topk_weight_stride_b + 4 * topk_weight_stride_s).to(tl.float32)
    w5 = tl.load(topk_weight_ptr + token_idx * topk_weight_stride_b + 5 * topk_weight_stride_s).to(tl.float32)

    p0 = tl.load(partial_ptr + (route_base + 0) * partial_stride_r + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)
    p1 = tl.load(partial_ptr + (route_base + 1) * partial_stride_r + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)
    p2 = tl.load(partial_ptr + (route_base + 2) * partial_stride_r + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)
    p3 = tl.load(partial_ptr + (route_base + 3) * partial_stride_r + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)
    p4 = tl.load(partial_ptr + (route_base + 4) * partial_stride_r + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)
    p5 = tl.load(partial_ptr + (route_base + 5) * partial_stride_r + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)

    out = p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4 + p5 * w5
    tl.store(out_ptr + token_idx * out_stride_b + row_offsets * out_stride_m, out.to(OUT_DTYPE), mask=row_mask)


@triton.jit
def _zero_i32_kernel(ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offsets, tl.zeros((BLOCK,), dtype=tl.int32), mask=offsets < n_elements)


@triton.jit
def _zero_fp32_kernel(ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=offsets < n_elements)


@triton.jit
def _build_route_indices_kernel(
    topk_ids_ptr,
    counts_ptr,
    route_indices_ptr,
    total_routes,
    topk,
    topk_ids_stride_b,
    topk_ids_stride_s,
):
    route_idx = tl.program_id(0)
    token_idx = route_idx // topk
    slot_idx = route_idx - token_idx * topk
    expert_id = tl.load(topk_ids_ptr + token_idx * topk_ids_stride_b + slot_idx * topk_ids_stride_s).to(tl.int32)
    offset = tl.atomic_add(counts_ptr + expert_id, 1, sem="relaxed")
    tl.store(route_indices_ptr + expert_id * total_routes + offset, route_idx)


@triton.jit
def _build_route_blocks_kernel(
    counts_ptr,
    block_count_ptr,
    block_experts_ptr,
    block_offsets_ptr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    count = tl.load(counts_ptr + expert_id)
    start = 0
    while start < count:
        block_idx = tl.atomic_add(block_count_ptr, 1, sem="relaxed")
        tl.store(block_experts_ptr + block_idx, expert_id)
        tl.store(block_offsets_ptr + block_idx, start)
        start += BLOCK_N


@triton.jit
def _grouped_gate_up_swiglu_kernel(
    x_ptr,
    gate_up_ptr,
    route_indices_ptr,
    counts_ptr,
    block_count_ptr,
    block_experts_ptr,
    block_offsets_ptr,
    hidden_ptr,
    total_routes,
    topk,
    hidden_size,
    intermediate_size,
    x_stride_b,
    x_stride_k,
    gate_up_stride_e,
    gate_up_stride_m,
    gate_up_stride_k,
    hidden_stride_r,
    hidden_stride_m,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_idx = tl.program_id(0)
    row_block_idx = tl.program_id(1)
    if block_idx >= tl.load(block_count_ptr):
        return

    expert_id = tl.load(block_experts_ptr + block_idx).to(tl.int32)
    route_start = tl.load(block_offsets_ptr + block_idx).to(tl.int32)
    expert_count = tl.load(counts_ptr + expert_id).to(tl.int32)
    route_offsets = route_start + tl.arange(0, BLOCK_N)
    route_mask = route_offsets < expert_count
    route_idx = tl.load(
        route_indices_ptr + expert_id * total_routes + route_offsets,
        mask=route_mask,
        other=0,
    ).to(tl.int32)
    token_idx = route_idx // topk

    row_start = row_block_idx * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < intermediate_size
    k_offsets_base = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    k_block = 0
    while k_block * BLOCK_K < hidden_size:
        k_start = k_block * BLOCK_K
        k_offsets = k_start + k_offsets_base
        k_mask = k_offsets < hidden_size
        x_tile = tl.load(
            x_ptr + token_idx[:, None] * x_stride_b + k_offsets[None, :] * x_stride_k,
            mask=route_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        gate_w = tl.load(
            gate_up_ptr
            + expert_id * gate_up_stride_e
            + k_offsets[:, None] * gate_up_stride_k
            + row_offsets[None, :] * gate_up_stride_m,
            mask=k_mask[:, None] & row_mask[None, :],
            other=0.0,
        )
        up_w = tl.load(
            gate_up_ptr
            + expert_id * gate_up_stride_e
            + k_offsets[:, None] * gate_up_stride_k
            + (intermediate_size + row_offsets)[None, :] * gate_up_stride_m,
            mask=k_mask[:, None] & row_mask[None, :],
            other=0.0,
        )
        gate_acc += tl.dot(x_tile, gate_w)
        up_acc += tl.dot(x_tile, up_w)
        k_block += 1

    routed_hidden = gate_acc * tl.sigmoid(gate_acc) * up_acc
    tl.store(
        hidden_ptr + route_idx[:, None] * hidden_stride_r + row_offsets[None, :] * hidden_stride_m,
        routed_hidden.to(tl.bfloat16),
        mask=route_mask[:, None] & row_mask[None, :],
    )


@triton.jit
def _grouped_down_partial_kernel(
    hidden_ptr,
    down_ptr,
    route_indices_ptr,
    counts_ptr,
    block_count_ptr,
    block_experts_ptr,
    block_offsets_ptr,
    partial_ptr,
    total_routes,
    topk,
    hidden_size,
    intermediate_size,
    hidden_stride_r,
    hidden_stride_k,
    down_stride_e,
    down_stride_m,
    down_stride_k,
    partial_stride_r,
    partial_stride_m,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_idx = tl.program_id(0)
    row_block_idx = tl.program_id(1)
    if block_idx >= tl.load(block_count_ptr):
        return

    expert_id = tl.load(block_experts_ptr + block_idx).to(tl.int32)
    route_start = tl.load(block_offsets_ptr + block_idx).to(tl.int32)
    expert_count = tl.load(counts_ptr + expert_id).to(tl.int32)
    route_offsets = route_start + tl.arange(0, BLOCK_N)
    route_mask = route_offsets < expert_count
    route_idx = tl.load(
        route_indices_ptr + expert_id * total_routes + route_offsets,
        mask=route_mask,
        other=0,
    ).to(tl.int32)

    row_offsets = row_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < hidden_size
    k_offsets_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    k_block = 0
    while k_block * BLOCK_K < intermediate_size:
        k_offsets = k_block * BLOCK_K + k_offsets_base
        k_mask = k_offsets < intermediate_size
        hidden = tl.load(
            hidden_ptr + route_idx[:, None] * hidden_stride_r + k_offsets[None, :] * hidden_stride_k,
            mask=route_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        down_w = tl.load(
            down_ptr
            + expert_id * down_stride_e
            + k_offsets[:, None] * down_stride_k
            + row_offsets[None, :] * down_stride_m,
            mask=k_mask[:, None] & row_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(hidden, down_w)
        k_block += 1

    tl.store(
        partial_ptr + route_idx[:, None] * partial_stride_r + row_offsets[None, :] * partial_stride_m,
        acc,
        mask=route_mask[:, None] & row_mask[None, :],
    )


@triton.jit
def _cast_output_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    OUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, x.to(OUT_DTYPE), mask=mask)


def pack_routed_experts(moe: torch.nn.Module, device: torch.device | str | None = None) -> PackedRoutedExperts:
    if getattr(moe, "ep_size", 1) != 1:
        raise NotImplementedError("pack_routed_experts currently only supports ep_size == 1")

    experts = [expert for expert in moe.experts if expert is not None]
    if not experts:
        raise ValueError("No local routed experts found to pack")

    first_expert = experts[0]
    if device is None:
        device = first_expert.gate_proj.weight.device
    dtype = first_expert.gate_proj.weight.dtype
    hidden_size = int(first_expert.gate_proj.weight.shape[1])
    intermediate_size = int(first_expert.gate_proj.weight.shape[0])
    num_experts = len(experts)
    topk = int(moe.num_experts_per_tok)

    gate_up_weights = torch.empty(
        (num_experts, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
    )
    down_weights = torch.empty(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=dtype,
    )
    for expert_idx, expert in enumerate(experts):
        gate_up_weights[expert_idx, :intermediate_size].copy_(expert.gate_proj.weight)
        gate_up_weights[expert_idx, intermediate_size:].copy_(expert.up_proj.weight)
        down_weights[expert_idx].copy_(expert.down_proj.weight)

    return PackedRoutedExperts(
        gate_up_weights=gate_up_weights,
        down_weights=down_weights,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
    )


def packed_routed_moe_eager(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    packed: PackedRoutedExperts,
) -> torch.Tensor:
    if packed.gate_up_weights.device != x.device or packed.down_weights.device != x.device:
        packed.to(device=x.device)
    output = torch.zeros((x.shape[0], packed.hidden_size), device=x.device, dtype=torch.float32)
    for expert_id in range(packed.num_experts):
        mask = topk_ids == expert_id
        if not bool(mask.any()):
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        expert_input = x[token_idx]
        gate_up = torch.nn.functional.linear(expert_input, packed.gate_up_weights[expert_id])
        gate, up = gate_up.split(packed.intermediate_size, dim=-1)
        hidden = torch.nn.functional.silu(gate) * up
        expert_output = torch.nn.functional.linear(hidden, packed.down_weights[expert_id])
        weighted = expert_output.float() * topk_weight[token_idx, slot_idx].float().unsqueeze(-1)
        output.index_add_(0, token_idx, weighted)
    return output.to(dtype=x.dtype)


def packed_routed_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    packed: PackedRoutedExperts,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if x.device.type == "cuda" and x.ndim == 2 and topk_ids.ndim == 2 and topk_ids.shape[0] == x.shape[0]:
        return batched_grouped_routed_moe(x, topk_ids, topk_weight, packed, output_dtype=output_dtype)
    return packed_routed_moe_eager(x, topk_ids, topk_weight, packed)


def batched_grouped_routed_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    packed: PackedRoutedExperts,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if x.ndim != 2 or topk_ids.ndim != 2 or topk_weight.ndim != 2:
        raise ValueError("batched_grouped_routed_moe expects x=[B,H], topk tensors=[B,topk]")
    if topk_ids.shape != topk_weight.shape or topk_ids.shape[0] != x.shape[0]:
        raise ValueError("topk_ids/topk_weight must have shape [B, topk]")
    if topk_ids.shape[1] != packed.topk:
        raise ValueError(f"Expected topk={packed.topk}, got {topk_ids.shape[1]}")
    if x.device.type != "cuda":
        raise NotImplementedError("batched_grouped_routed_moe requires CUDA")
    if packed.topk != 6:
        raise NotImplementedError(f"batched_grouped_routed_moe currently supports topk=6, got {packed.topk}")
    if packed.gate_up_weights.device != x.device or packed.down_weights.device != x.device:
        packed.to(device=x.device)

    return batched_grouped_routed_moe_grouped_triton(x, topk_ids, topk_weight, packed, output_dtype=output_dtype)


def _run_batched_grouped_routed_moe_grouped_triton(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    packed: PackedRoutedExperts,
    output_dtype: torch.dtype,
    block_n: int,
    block_m: int,
    block_k: int,
) -> torch.Tensor:
    batch_size = int(x.shape[0])
    topk = packed.topk
    total_routes = batch_size * topk
    hidden_size = packed.hidden_size
    intermediate_size = packed.intermediate_size
    max_blocks = total_routes

    counts = torch.empty((packed.num_experts,), device=x.device, dtype=torch.int32)
    route_indices = torch.empty((packed.num_experts, total_routes), device=x.device, dtype=torch.int32)
    block_count = torch.empty((1,), device=x.device, dtype=torch.int32)
    block_experts = torch.empty((max_blocks,), device=x.device, dtype=torch.int32)
    block_offsets = torch.empty((max_blocks,), device=x.device, dtype=torch.int32)
    routed_hidden = torch.empty((total_routes, intermediate_size), device=x.device, dtype=torch.bfloat16)
    partial = torch.empty((total_routes, hidden_size), device=x.device, dtype=torch.float32)
    out = torch.empty((batch_size, hidden_size), device=x.device, dtype=output_dtype)

    _zero_i32_kernel[(triton.cdiv(packed.num_experts + 1, 256),)](counts, packed.num_experts, BLOCK=256, num_warps=4)
    _zero_i32_kernel[(1,)](block_count, 1, BLOCK=1, num_warps=1)
    _build_route_indices_kernel[(total_routes,)](
        topk_ids,
        counts,
        route_indices,
        total_routes,
        topk,
        topk_ids.stride(0),
        topk_ids.stride(1),
        num_warps=1,
    )
    _build_route_blocks_kernel[(packed.num_experts,)](
        counts,
        block_count,
        block_experts,
        block_offsets,
        BLOCK_N=block_n,
        num_warps=1,
    )
    _grouped_gate_up_swiglu_kernel[(max_blocks, triton.cdiv(intermediate_size, block_m))](
        x,
        packed.gate_up_weights,
        route_indices,
        counts,
        block_count,
        block_experts,
        block_offsets,
        routed_hidden,
        total_routes,
        topk,
        hidden_size,
        intermediate_size,
        x.stride(0),
        x.stride(1),
        packed.gate_up_weights.stride(0),
        packed.gate_up_weights.stride(1),
        packed.gate_up_weights.stride(2),
        routed_hidden.stride(0),
        routed_hidden.stride(1),
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
    _grouped_down_partial_kernel[(max_blocks, triton.cdiv(hidden_size, block_m))](
        routed_hidden,
        packed.down_weights,
        route_indices,
        counts,
        block_count,
        block_experts,
        block_offsets,
        partial,
        total_routes,
        topk,
        hidden_size,
        intermediate_size,
        routed_hidden.stride(0),
        routed_hidden.stride(1),
        packed.down_weights.stride(0),
        packed.down_weights.stride(1),
        packed.down_weights.stride(2),
        partial.stride(0),
        partial.stride(1),
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
    _batched_down_reduce_topk6_kernel[(batch_size, triton.cdiv(hidden_size, block_m))](
        partial,
        topk_weight,
        out,
        hidden_size,
        topk,
        partial.stride(0),
        partial.stride(1),
        topk_weight.stride(0),
        topk_weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        OUT_DTYPE=tl.bfloat16 if output_dtype is torch.bfloat16 else tl.float32,
        num_warps=4,
    )
    return out


def _default_moe_grouped_tile(batch_size: int) -> tuple[int, int, int]:
    if batch_size >= 32:
        return 32, 64, 128
    if batch_size >= 16:
        return 4, 32, 256
    if batch_size >= 8:
        return 32, 64, 128
    if batch_size >= 4:
        return 2, 32, 128
    if batch_size >= 2:
        return 16, 64, 128
    return 8, 32, 64


def _select_moe_grouped_tile(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    packed: PackedRoutedExperts,
    output_dtype: torch.dtype,
) -> tuple[int, int, int]:
    batch_size = int(x.shape[0])
    key = (batch_size, packed.hidden_size, packed.intermediate_size, x.device)
    cached = PackedRoutedExperts.autotune_cache.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        tile = _default_moe_grouped_tile(batch_size)
        PackedRoutedExperts.autotune_cache[key] = tile
        return tile

    best_tile = _default_moe_grouped_tile(batch_size)
    best_ms = float("inf")
    for block_n, block_m, block_k in MOE_GROUPED_TILE_CANDIDATES:
        _run_batched_grouped_routed_moe_grouped_triton(
            x,
            topk_ids,
            topk_weight,
            packed,
            output_dtype,
            block_n,
            block_m,
            block_k,
        )
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(3):
            _run_batched_grouped_routed_moe_grouped_triton(
                x,
                topk_ids,
                topk_weight,
                packed,
                output_dtype,
                block_n,
                block_m,
                block_k,
            )
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start_event.elapsed_time(end_event)) / 3.0
        if elapsed_ms < best_ms:
            best_ms = elapsed_ms
            best_tile = (block_n, block_m, block_k)
    PackedRoutedExperts.autotune_cache[key] = best_tile
    return best_tile


def batched_grouped_routed_moe_grouped_triton(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    packed: PackedRoutedExperts,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    x = x.contiguous()
    topk_ids = topk_ids.contiguous()
    topk_weight = topk_weight.contiguous()
    if output_dtype is None:
        output_dtype = x.dtype
    block_n, block_m, block_k = _select_moe_grouped_tile(x, topk_ids, topk_weight, packed, output_dtype)
    return _run_batched_grouped_routed_moe_grouped_triton(
        x,
        topk_ids,
        topk_weight,
        packed,
        output_dtype,
        block_n,
        block_m,
        block_k,
    )


