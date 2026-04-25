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
def _fused_gate_up_swiglu_kernel(
    x_ptr,
    topk_ids_ptr,
    gate_up_ptr,
    hidden_ptr,
    num_rows,
    hidden_size,
    intermediate_size,
    gate_up_stride_e,
    gate_up_stride_m,
    gate_up_stride_k,
    hidden_stride_e,
    hidden_stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_idx = tl.program_id(0)
    row_block_idx = tl.program_id(1)
    row_start = row_block_idx * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < num_rows

    expert_id = tl.load(topk_ids_ptr + expert_idx).to(tl.int32)
    gate_up_base_ptr = gate_up_ptr + expert_id * gate_up_stride_e
    k_offsets_base = tl.arange(0, BLOCK_K)
    gate_up_acc = tl.zeros((2, BLOCK_M), dtype=tl.float32)
    k_block = 0
    while k_block * BLOCK_K < hidden_size:
        k_start = k_block * BLOCK_K
        k_offsets = k_start + k_offsets_base
        k_mask = k_offsets < hidden_size
        x = tl.load(x_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        x_col = x[:, None]

        gate_up_block_ptr = tl.make_block_ptr(
            base=gate_up_base_ptr,
            shape=(2 * intermediate_size, hidden_size),
            strides=(gate_up_stride_m, gate_up_stride_k),
            offsets=(row_start, k_start),
            block_shape=(2 * BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        gate_up_w = tl.load(gate_up_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        gate_up_acc += tl.reshape(tl.dot(gate_up_w, x_col), (2, BLOCK_M))
        k_block += 1

    gate_up_rows = tl.arange(0, 2)[:, None]
    gate_acc = tl.sum(tl.where(gate_up_rows == 0, gate_up_acc, 0.0), axis=0)
    up_acc = tl.sum(tl.where(gate_up_rows == 1, gate_up_acc, 0.0), axis=0)
    routed_hidden = gate_acc * tl.sigmoid(gate_acc) * up_acc
    hidden_ptrs = hidden_ptr + expert_idx * hidden_stride_e + row_offsets * hidden_stride_m
    tl.store(hidden_ptrs, routed_hidden, mask=row_mask)


@triton.jit
def _down_partial_kernel(
    hidden_ptr,
    topk_ids_ptr,
    down_ptr,
    partial_ptr,
    topk,
    hidden_size,
    intermediate_size,
    down_stride_e,
    down_stride_m,
    down_stride_k,
    hidden_stride_e,
    hidden_stride_k,
    partial_stride_e,
    partial_stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    slot_idx = tl.program_id(0)
    row_block_idx = tl.program_id(1)

    if slot_idx >= topk:
        return

    expert_id = tl.load(topk_ids_ptr + slot_idx).to(tl.int32)
    row_offsets = row_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < hidden_size
    k_offsets_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    k_block = 0
    while k_block * BLOCK_K < intermediate_size:
        k_offsets = k_block * BLOCK_K + k_offsets_base
        k_mask = k_offsets < intermediate_size
        hidden = tl.load(
            hidden_ptr + slot_idx * hidden_stride_e + k_offsets * hidden_stride_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        down_ptrs = (
            down_ptr
            + expert_id * down_stride_e
            + row_offsets[:, None] * down_stride_m
            + k_offsets[None, :] * down_stride_k
        )
        down_w = tl.load(down_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(down_w * hidden[None, :], axis=1)
        k_block += 1

    tl.store(partial_ptr + slot_idx * partial_stride_e + row_offsets * partial_stride_m, acc, mask=row_mask)


@triton.jit
def _down_reduce_topk6_kernel(
    partial_ptr,
    topk_weight_ptr,
    out_ptr,
    hidden_size,
    partial_stride_e,
    partial_stride_m,
    BLOCK_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    row_block_idx = tl.program_id(0)
    row_offsets = row_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < hidden_size

    w0 = tl.load(topk_weight_ptr + 0).to(tl.float32)
    w1 = tl.load(topk_weight_ptr + 1).to(tl.float32)
    w2 = tl.load(topk_weight_ptr + 2).to(tl.float32)
    w3 = tl.load(topk_weight_ptr + 3).to(tl.float32)
    w4 = tl.load(topk_weight_ptr + 4).to(tl.float32)
    w5 = tl.load(topk_weight_ptr + 5).to(tl.float32)

    p0 = tl.load(partial_ptr + 0 * partial_stride_e + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)
    p1 = tl.load(partial_ptr + 1 * partial_stride_e + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)
    p2 = tl.load(partial_ptr + 2 * partial_stride_e + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)
    p3 = tl.load(partial_ptr + 3 * partial_stride_e + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)
    p4 = tl.load(partial_ptr + 4 * partial_stride_e + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)
    p5 = tl.load(partial_ptr + 5 * partial_stride_e + row_offsets * partial_stride_m, mask=row_mask, other=0.0).to(tl.float32)

    out = p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4 + p5 * w5
    tl.store(out_ptr + row_offsets, out.to(OUT_DTYPE), mask=row_mask)


@triton.jit
def _batched_fused_gate_up_swiglu_kernel(
    x_ptr,
    topk_ids_ptr,
    gate_up_ptr,
    hidden_ptr,
    hidden_size,
    intermediate_size,
    topk,
    x_stride_b,
    x_stride_k,
    topk_ids_stride_b,
    topk_ids_stride_s,
    gate_up_stride_e,
    gate_up_stride_m,
    gate_up_stride_k,
    hidden_stride_r,
    hidden_stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route_idx = tl.program_id(0)
    row_block_idx = tl.program_id(1)
    token_idx = route_idx // topk
    slot_idx = route_idx - token_idx * topk
    row_start = row_block_idx * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < intermediate_size

    expert_id = tl.load(topk_ids_ptr + token_idx * topk_ids_stride_b + slot_idx * topk_ids_stride_s).to(tl.int32)
    gate_up_base_ptr = gate_up_ptr + expert_id * gate_up_stride_e
    k_offsets_base = tl.arange(0, BLOCK_K)
    gate_up_acc = tl.zeros((2, BLOCK_M), dtype=tl.float32)
    k_block = 0
    while k_block * BLOCK_K < hidden_size:
        k_start = k_block * BLOCK_K
        k_offsets = k_start + k_offsets_base
        k_mask = k_offsets < hidden_size
        x = tl.load(x_ptr + token_idx * x_stride_b + k_offsets * x_stride_k, mask=k_mask, other=0.0).to(tl.float32)
        x_col = x[:, None]

        gate_up_block_ptr = tl.make_block_ptr(
            base=gate_up_base_ptr,
            shape=(2 * intermediate_size, hidden_size),
            strides=(gate_up_stride_m, gate_up_stride_k),
            offsets=(row_start, k_start),
            block_shape=(2 * BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        gate_up_w = tl.load(gate_up_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        gate_up_acc += tl.reshape(tl.dot(gate_up_w, x_col), (2, BLOCK_M))
        k_block += 1

    gate_up_rows = tl.arange(0, 2)[:, None]
    gate_acc = tl.sum(tl.where(gate_up_rows == 0, gate_up_acc, 0.0), axis=0)
    up_acc = tl.sum(tl.where(gate_up_rows == 1, gate_up_acc, 0.0), axis=0)
    routed_hidden = gate_acc * tl.sigmoid(gate_acc) * up_acc
    hidden_ptrs = hidden_ptr + route_idx * hidden_stride_r + row_offsets * hidden_stride_m
    tl.store(hidden_ptrs, routed_hidden, mask=row_mask)


@triton.jit
def _batched_down_partial_kernel(
    hidden_ptr,
    topk_ids_ptr,
    down_ptr,
    partial_ptr,
    topk,
    hidden_size,
    intermediate_size,
    topk_ids_stride_b,
    topk_ids_stride_s,
    down_stride_e,
    down_stride_m,
    down_stride_k,
    hidden_stride_r,
    hidden_stride_k,
    partial_stride_r,
    partial_stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    route_idx = tl.program_id(0)
    row_block_idx = tl.program_id(1)
    token_idx = route_idx // topk
    slot_idx = route_idx - token_idx * topk

    expert_id = tl.load(topk_ids_ptr + token_idx * topk_ids_stride_b + slot_idx * topk_ids_stride_s).to(tl.int32)
    row_offsets = row_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < hidden_size
    k_offsets_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    k_block = 0
    while k_block * BLOCK_K < intermediate_size:
        k_offsets = k_block * BLOCK_K + k_offsets_base
        k_mask = k_offsets < intermediate_size
        hidden = tl.load(
            hidden_ptr + route_idx * hidden_stride_r + k_offsets * hidden_stride_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        down_ptrs = (
            down_ptr
            + expert_id * down_stride_e
            + row_offsets[:, None] * down_stride_m
            + k_offsets[None, :] * down_stride_k
        )
        down_w = tl.load(down_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(down_w * hidden[None, :], axis=1)
        k_block += 1

    tl.store(partial_ptr + route_idx * partial_stride_r + row_offsets * partial_stride_m, acc, mask=row_mask)


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


def batched_grouped_routed_moe_route_gemv(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    packed: PackedRoutedExperts,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    x = x.contiguous()
    topk_ids = topk_ids.contiguous()
    topk_weight = topk_weight.contiguous()
    batch_size = int(x.shape[0])
    topk = packed.topk
    total_routes = batch_size * topk
    hidden_size = packed.hidden_size
    intermediate_size = packed.intermediate_size
    if output_dtype is None:
        output_dtype = x.dtype

    block_m = 32
    block_k = 128
    routed_hidden = torch.empty((total_routes, intermediate_size), device=x.device, dtype=torch.bfloat16)
    partial = torch.empty((total_routes, hidden_size), device=x.device, dtype=torch.float32)
    out = torch.empty((batch_size, hidden_size), device=x.device, dtype=output_dtype)

    _batched_fused_gate_up_swiglu_kernel[(total_routes, triton.cdiv(intermediate_size, block_m))](
        x,
        topk_ids,
        packed.gate_up_weights,
        routed_hidden,
        hidden_size,
        intermediate_size,
        topk,
        x.stride(0),
        x.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        packed.gate_up_weights.stride(0),
        packed.gate_up_weights.stride(1),
        packed.gate_up_weights.stride(2),
        routed_hidden.stride(0),
        routed_hidden.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=2,
    )

    _batched_down_partial_kernel[(total_routes, triton.cdiv(hidden_size, block_m))](
        routed_hidden,
        topk_ids,
        packed.down_weights,
        partial,
        topk,
        hidden_size,
        intermediate_size,
        topk_ids.stride(0),
        topk_ids.stride(1),
        packed.down_weights.stride(0),
        packed.down_weights.stride(1),
        packed.down_weights.stride(2),
        routed_hidden.stride(0),
        routed_hidden.stride(1),
        partial.stride(0),
        partial.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=2,
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
        num_warps=2,
    )
    return out


def grouped_routed_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    packed: PackedRoutedExperts,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # decode-only specialization: x=[1, hidden], topk ids=[1, topk]
    if x.ndim != 2 or x.shape[0] != 1:
        raise NotImplementedError("grouped_routed_moe only supports single-token decode")
    if topk_ids.ndim != 2 or topk_ids.shape[0] != 1:
        raise NotImplementedError("topk_ids must have shape [1, topk]")
    if topk_ids.shape[1] != packed.topk:
        raise ValueError(f"Expected topk={packed.topk}, got {topk_ids.shape[1]}")
    if x.device.type != "cuda":
        raise NotImplementedError("grouped_routed_moe requires CUDA")
    if packed.topk > 8:
        raise NotImplementedError(f"MAX_TOPK=8 kernel only, got topk={packed.topk}")

    if packed.gate_up_weights.device != x.device or packed.down_weights.device != x.device:
        packed.to(device=x.device)

    token = x[0].contiguous()
    routed_ids = topk_ids[0].to(torch.int32).contiguous()
    routed_weight = topk_weight[0].to(dtype=torch.float32).contiguous()

    topk = int(routed_ids.shape[0])
    hidden_size = packed.hidden_size
    intermediate_size = packed.intermediate_size

    block_m = 32
    block_k = 128

    routed_hidden = torch.empty(
        (topk, intermediate_size),
        device=x.device,
        dtype=torch.float32,
    )
    if output_dtype is None:
        output_dtype = x.dtype
    partial = torch.empty((topk, hidden_size), device=x.device, dtype=torch.float32)
    out = torch.empty((hidden_size,), device=x.device, dtype=output_dtype)

    # Kernel1: gate/up GEMV + SiLU*mul
    grid = (topk, triton.cdiv(intermediate_size, block_m))
    _fused_gate_up_swiglu_kernel[grid](
        token,
        routed_ids,
        packed.gate_up_weights,
        routed_hidden,
        intermediate_size,
        hidden_size,
        intermediate_size,
        packed.gate_up_weights.stride(0),
        packed.gate_up_weights.stride(1),
        packed.gate_up_weights.stride(2),
        routed_hidden.stride(0),
        routed_hidden.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=2,
    )

    # Kernel2: down GEMV per route, still parallel across topk slots.
    grid = (topk, triton.cdiv(hidden_size, block_m))
    _down_partial_kernel[grid](
        routed_hidden,
        routed_ids,
        packed.down_weights,
        partial,
        topk,
        hidden_size,
        intermediate_size,
        packed.down_weights.stride(0),
        packed.down_weights.stride(1),
        packed.down_weights.stride(2),
        routed_hidden.stride(0),
        routed_hidden.stride(1),
        partial.stride(0),
        partial.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=2,
    )

    # Kernel3: deterministic route weight + topk reduction + final dtype store.
    _down_reduce_topk6_kernel[(triton.cdiv(hidden_size, block_m),)](
        partial,
        routed_weight,
        out,
        hidden_size,
        partial.stride(0),
        partial.stride(1),
        BLOCK_M=block_m,
        OUT_DTYPE=tl.bfloat16 if output_dtype is torch.bfloat16 else tl.float32,
        num_warps=2,
    )
    return out.unsqueeze(0)
