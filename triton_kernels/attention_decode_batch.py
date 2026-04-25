from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl

from triton_kernels.rmsnorm import rmsnorm_triton


DSV2L_HIDDEN = 2048
DSV2L_NUM_HEADS = 16
DSV2L_Q_DIM = 192
DSV2L_QK_NOPE_DIM = 128
DSV2L_QK_ROPE_DIM = 64
DSV2L_V_DIM = 128
DSV2L_KV_LORA_RANK = 512
DSV2L_Q_ROWS = DSV2L_NUM_HEADS * DSV2L_Q_DIM
DSV2L_KV_A_ROWS = DSV2L_KV_LORA_RANK + DSV2L_QK_ROPE_DIM
DSV2L_KVB_ROWS = DSV2L_NUM_HEADS * (DSV2L_QK_NOPE_DIM + DSV2L_V_DIM)


@dataclass
class AttentionPackedWeights:
    # Q and KV-A packed together: [3648, 2048]
    q_kv_a_weight: torch.Tensor
    # KV-B packed as [4096, 512]
    kv_b_weight: torch.Tensor
    # O: [2048, 2048]
    o_weight: torch.Tensor
    kv_a_ln_weight: torch.Tensor
    kv_a_ln_eps: float
    workspace_by_device: dict[int, "AttentionDecodeWorkspace"] = field(default_factory=dict)


@dataclass
class AttentionDecodeWorkspace:
    q_kv_a_out: torch.Tensor
    q_proj: torch.Tensor
    kv_a_out: torch.Tensor
    kv_lora: torch.Tensor
    k_pe: torch.Tensor
    kvb_proj: torch.Tensor
    kvb_proj_flat: torch.Tensor
    q_out: torch.Tensor
    k_new: torch.Tensor
    v_new: torch.Tensor
    attn_ctx: torch.Tensor
    attn_ctx_flat: torch.Tensor
    o_out: torch.Tensor


@triton.jit
def _cache_write_q1_dual_kernel(
    k_new_ptr,
    v_new_ptr,
    k_cache_ptr,
    v_cache_ptr,
    pos_ptr,
    q_dim,
    v_dim,
    k_new_stride_b,
    k_new_stride_h,
    k_new_stride_s,
    k_new_stride_d,
    v_new_stride_b,
    v_new_stride_h,
    v_new_stride_s,
    v_new_stride_d,
    k_cache_stride_b,
    k_cache_stride_h,
    k_cache_stride_s,
    k_cache_stride_d,
    v_cache_stride_b,
    v_cache_stride_h,
    v_cache_stride_s,
    v_cache_stride_d,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    d_block = tl.program_id(2)
    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)

    pos = tl.load(pos_ptr).to(tl.int32)

    k_mask = d_offsets < q_dim
    k = tl.load(
        k_new_ptr
        + batch_idx * k_new_stride_b
        + head_idx * k_new_stride_h
        + 0 * k_new_stride_s
        + d_offsets * k_new_stride_d,
        mask=k_mask,
        other=0.0,
    )
    tl.store(
        k_cache_ptr
        + batch_idx * k_cache_stride_b
        + head_idx * k_cache_stride_h
        + pos * k_cache_stride_s
        + d_offsets * k_cache_stride_d,
        k,
        mask=k_mask,
    )

    v_mask = d_offsets < v_dim
    v = tl.load(
        v_new_ptr
        + batch_idx * v_new_stride_b
        + head_idx * v_new_stride_h
        + 0 * v_new_stride_s
        + d_offsets * v_new_stride_d,
        mask=v_mask,
        other=0.0,
    )
    tl.store(
        v_cache_ptr
        + batch_idx * v_cache_stride_b
        + head_idx * v_cache_stride_h
        + pos * v_cache_stride_s
        + d_offsets * v_cache_stride_d,
        v,
        mask=v_mask,
    )


@triton.jit
def _decode_attention_q1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pos_ptr,
    out_ptr,
    q_dim,
    v_dim,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_s,
    k_stride_d,
    v_stride_h,
    v_stride_s,
    v_stride_d,
    out_stride_h,
    out_stride_d,
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    head_idx = tl.program_id(0)
    v_block = tl.program_id(1)

    v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offsets < v_dim

    m_i = tl.full((1,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

    pos = tl.load(pos_ptr).to(tl.int32)

    n_start = 0
    while n_start <= pos:
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_valid = n_offsets <= pos

        logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

        q_start = 0
        while q_start < q_dim:
            q_offsets = q_start + tl.arange(0, BLOCK_Q)
            q_mask = q_offsets < q_dim

            q_vec = tl.load(
                q_ptr + head_idx * q_stride_h + q_offsets * q_stride_d,
                mask=q_mask,
                other=0.0,
            ).to(tl.float32)
            k_block = tl.load(
                k_ptr + head_idx * k_stride_h + n_offsets[:, None] * k_stride_s + q_offsets[None, :] * k_stride_d,
                mask=n_valid[:, None] & q_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            logits += tl.reshape(tl.dot(k_block, q_vec[:, None]), (BLOCK_N,))
            q_start += BLOCK_Q

        logits = logits * scale
        logits = tl.where(n_valid, logits, -float("inf"))

        m_ij = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new)

        v_block_data = tl.load(
            v_ptr + head_idx * v_stride_h + n_offsets[:, None] * v_stride_s + v_offsets[None, :] * v_stride_d,
            mask=n_valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v_block_data, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        n_start += BLOCK_N

    out = acc / l_i
    tl.store(out_ptr + head_idx * out_stride_h + v_offsets * out_stride_d, out.to(tl.bfloat16), mask=v_mask)


def pack_attention_weights(attn: torch.nn.Module) -> AttentionPackedWeights:
    if (
        int(attn.hidden_size) != DSV2L_HIDDEN
        or int(attn.num_heads) != DSV2L_NUM_HEADS
        or int(attn.q_head_dim) != DSV2L_Q_DIM
        or int(attn.qk_nope_head_dim) != DSV2L_QK_NOPE_DIM
        or int(attn.qk_rope_head_dim) != DSV2L_QK_ROPE_DIM
        or int(attn.v_head_dim) != DSV2L_V_DIM
        or int(attn.kv_lora_rank) != DSV2L_KV_LORA_RANK
    ):
        raise NotImplementedError("attention_decode_only_triton is specialized for DeepSeek-V2-Lite dimensions")

    q_weight_2d = attn.q_proj.weight.view(DSV2L_Q_ROWS, DSV2L_HIDDEN).contiguous()
    kv_a_weight = attn.kv_a_proj_with_mqa.weight.contiguous()
    q_kv_a_weight = torch.cat((q_weight_2d, kv_a_weight), dim=0).contiguous()
    kv_b_weight = attn.kv_b_proj.weight.view(DSV2L_KVB_ROWS, DSV2L_KV_LORA_RANK).contiguous()
    o_weight = attn.o_proj.weight.contiguous()
    return AttentionPackedWeights(
        q_kv_a_weight=q_kv_a_weight,
        kv_b_weight=kv_b_weight,
        o_weight=o_weight,
        kv_a_ln_weight=attn.kv_a_layernorm.weight.contiguous(),
        kv_a_ln_eps=float(attn.kv_a_layernorm.variance_epsilon),
    )


@triton.jit
def _batched_decode_attention_q1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pos_ptr,
    out_ptr,
    q_dim,
    v_dim,
    q_stride_b,
    q_stride_h,
    q_stride_s,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_s,
    k_stride_d,
    v_stride_b,
    v_stride_h,
    v_stride_s,
    v_stride_d,
    out_stride_b,
    out_stride_h,
    out_stride_s,
    out_stride_d,
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    v_block = tl.program_id(2)

    v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offsets < v_dim

    m_i = tl.full((1,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    pos = tl.load(pos_ptr).to(tl.int32)

    n_start = 0
    while n_start <= pos:
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_valid = n_offsets <= pos
        logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

        q_start = 0
        while q_start < q_dim:
            q_offsets = q_start + tl.arange(0, BLOCK_Q)
            q_mask = q_offsets < q_dim
            q_vec = tl.load(
                q_ptr
                + batch_idx * q_stride_b
                + head_idx * q_stride_h
                + 0 * q_stride_s
                + q_offsets * q_stride_d,
                mask=q_mask,
                other=0.0,
            ).to(tl.float32)
            k_block = tl.load(
                k_ptr
                + batch_idx * k_stride_b
                + head_idx * k_stride_h
                + n_offsets[:, None] * k_stride_s
                + q_offsets[None, :] * k_stride_d,
                mask=n_valid[:, None] & q_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            logits += tl.reshape(tl.dot(k_block, q_vec[:, None]), (BLOCK_N,))
            q_start += BLOCK_Q

        logits = tl.where(n_valid, logits * scale, -float("inf"))
        m_ij = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new)

        v_block_data = tl.load(
            v_ptr
            + batch_idx * v_stride_b
            + head_idx * v_stride_h
            + n_offsets[:, None] * v_stride_s
            + v_offsets[None, :] * v_stride_d,
            mask=n_valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v_block_data, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        n_start += BLOCK_N

    out = acc / l_i
    tl.store(
        out_ptr
        + batch_idx * out_stride_b
        + head_idx * out_stride_h
        + 0 * out_stride_s
        + v_offsets * out_stride_d,
        out.to(tl.bfloat16),
        mask=v_mask,
    )


def batched_attention_decode_q1_triton(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    if query_states.device.type != "cuda":
        raise NotImplementedError("batched_attention_decode_q1_triton requires CUDA tensors")
    if query_states.dtype != torch.bfloat16 or key_states.dtype != torch.bfloat16 or value_states.dtype != torch.bfloat16:
        raise NotImplementedError("batched_attention_decode_q1_triton supports bf16 only")
    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("query/key/value states must be rank-4 tensors")
    if query_states.shape[2] != 1:
        raise ValueError("batched_attention_decode_q1_triton only supports q_len=1")
    if cache_position.numel() != 1 or cache_position.dtype != torch.long:
        raise ValueError("cache_position must be CUDA int64 tensor with one element")

    batch_size = int(query_states.shape[0])
    num_heads = int(query_states.shape[1])
    q_dim = int(query_states.shape[3])
    v_dim = int(value_states.shape[3])
    out = torch.empty(
        (batch_size, num_heads, 1, v_dim),
        device=query_states.device,
        dtype=query_states.dtype,
    )
    _batched_decode_attention_q1_kernel[(batch_size, num_heads, triton.cdiv(v_dim, 128))](
        query_states,
        key_states,
        value_states,
        cache_position.view(-1),
        out,
        q_dim,
        v_dim,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        key_states.stride(3),
        value_states.stride(0),
        value_states.stride(1),
        value_states.stride(2),
        value_states.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        float(softmax_scale),
        BLOCK_N=32,
        BLOCK_Q=64,
        BLOCK_V=128,
        num_warps=4,
    )
    return out


@dataclass
class BatchedAttentionDecodeWorkspace:
    batch_size: int
    q_kv_a_out: torch.Tensor
    kv_lora: torch.Tensor
    k_pe: torch.Tensor
    kv_lora_norm: torch.Tensor
    kvb_proj: torch.Tensor
    q_out: torch.Tensor
    k_new: torch.Tensor
    v_new: torch.Tensor
    attn_ctx: torch.Tensor
    o_out: torch.Tensor


@triton.jit
def _batched_linear_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    out_rows,
    in_cols,
    x_stride_b,
    x_stride_k,
    w_stride_o,
    w_stride_k,
    out_stride_b,
    out_stride_o,
    BLOCK_B: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_block = tl.program_id(0)
    out_block = tl.program_id(1)
    batch_offsets = batch_block * BLOCK_B + tl.arange(0, BLOCK_B)
    out_offsets = out_block * BLOCK_O + tl.arange(0, BLOCK_O)
    batch_mask = batch_offsets < batch_size
    out_mask = out_offsets < out_rows

    acc = tl.zeros((BLOCK_B, BLOCK_O), dtype=tl.float32)
    k_offsets_base = tl.arange(0, BLOCK_K)
    k_start = 0
    while k_start < in_cols:
        k_offsets = k_start + k_offsets_base
        k_mask = k_offsets < in_cols
        x = tl.load(
            x_ptr + batch_offsets[:, None] * x_stride_b + k_offsets[None, :] * x_stride_k,
            mask=batch_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + out_offsets[:, None] * w_stride_o + k_offsets[None, :] * w_stride_k,
            mask=out_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(x, tl.trans(w))
        k_start += BLOCK_K

    tl.store(
        out_ptr + batch_offsets[:, None] * out_stride_b + out_offsets[None, :] * out_stride_o,
        acc.to(tl.bfloat16),
        mask=batch_mask[:, None] & out_mask[None, :],
    )


def _select_batched_linear_tile(batch_size: int) -> tuple[int, int, int]:
    forced_tile = _parse_batched_linear_tile_env()
    if forced_tile is not None:
        return forced_tile
    if batch_size == 1:
        return 1, 64, 256
    if batch_size >= 128:
        return 32, 64, 128
    return 8, 64, 128


def _parse_batched_linear_tile_env() -> tuple[int, int, int] | None:
    value = os.environ.get("DSV2_BATCH_LINEAR_TILE")
    if not value:
        return None
    parts = value.replace(",", " ").replace("x", " ").split()
    if len(parts) != 3:
        raise ValueError(f"DSV2_BATCH_LINEAR_TILE must contain three integers, got {value!r}")
    tile = tuple(int(part) for part in parts)
    if any(part <= 0 for part in tile):
        raise ValueError(f"DSV2_BATCH_LINEAR_TILE must contain positive integers, got {value!r}")
    return tile  # type: ignore[return-value]


@triton.jit
def _batched_build_qkv_rope_kernel(
    q_kv_a_ptr,
    kvb_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_out_ptr,
    k_new_ptr,
    v_new_ptr,
    batch_size,
    qkv_stride_b,
    qkv_stride_d,
    kvb_stride_b,
    kvb_stride_d,
    cos_stride_s,
    cos_stride_d,
    sin_stride_s,
    sin_stride_d,
    q_out_stride_b,
    q_out_stride_h,
    q_out_stride_s,
    q_out_stride_d,
    k_new_stride_b,
    k_new_stride_h,
    k_new_stride_s,
    k_new_stride_d,
    v_new_stride_b,
    v_new_stride_h,
    v_new_stride_s,
    v_new_stride_d,
    Q_ROWS: tl.constexpr,
    Q_DIM: tl.constexpr,
    QK_NOPE_DIM: tl.constexpr,
    QK_ROPE_DIM: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    d_block = tl.program_id(2)
    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)

    if batch_idx >= batch_size:
        return

    pos = tl.load(pos_ptr).to(tl.int32)
    q_base = head_idx * Q_DIM
    kvb_base = head_idx * (QK_NOPE_DIM + V_DIM)

    q_mask = d_offsets < Q_DIM
    q_raw = tl.load(
        q_kv_a_ptr + batch_idx * qkv_stride_b + (q_base + d_offsets) * qkv_stride_d,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)

    q_nope_mask = d_offsets < QK_NOPE_DIM
    rope_idx = d_offsets - QK_NOPE_DIM
    rope_mask = (d_offsets >= QK_NOPE_DIM) & (rope_idx < QK_ROPE_DIM)
    half = QK_ROPE_DIM // 2
    first_half = rope_idx < half
    pair_base = tl.where(first_half, rope_idx * 2, (rope_idx - half) * 2)

    q_even = tl.load(
        q_kv_a_ptr + batch_idx * qkv_stride_b + (q_base + QK_NOPE_DIM + pair_base) * qkv_stride_d,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    q_odd = tl.load(
        q_kv_a_ptr + batch_idx * qkv_stride_b + (q_base + QK_NOPE_DIM + pair_base + 1) * qkv_stride_d,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    k_pe_base = Q_ROWS + KV_LORA_RANK
    k_even = tl.load(
        q_kv_a_ptr + batch_idx * qkv_stride_b + (k_pe_base + pair_base) * qkv_stride_d,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    k_odd = tl.load(
        q_kv_a_ptr + batch_idx * qkv_stride_b + (k_pe_base + pair_base + 1) * qkv_stride_d,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)

    cos_idx = tl.where(first_half, rope_idx, rope_idx - half)
    cos_v = tl.load(cos_ptr + pos * cos_stride_s + cos_idx * cos_stride_d, mask=rope_mask, other=1.0).to(tl.float32)
    sin_v = tl.load(sin_ptr + pos * sin_stride_s + cos_idx * sin_stride_d, mask=rope_mask, other=0.0).to(tl.float32)

    first_q = q_even * cos_v - q_odd * sin_v
    second_q = q_odd * cos_v + q_even * sin_v
    first_k = k_even * cos_v - k_odd * sin_v
    second_k = k_odd * cos_v + k_even * sin_v
    q_rope = tl.where(first_half, first_q, second_q)
    k_rope = tl.where(first_half, first_k, second_k)

    q_out = tl.where(q_nope_mask, q_raw, 0.0)
    q_out = tl.where(rope_mask, q_rope, q_out)

    k_nope = tl.load(
        kvb_ptr + batch_idx * kvb_stride_b + (kvb_base + d_offsets) * kvb_stride_d,
        mask=q_nope_mask,
        other=0.0,
    ).to(tl.float32)
    k_out = tl.where(q_nope_mask, k_nope, 0.0)
    k_out = tl.where(rope_mask, k_rope, k_out)

    v_offsets = d_offsets
    v_mask = v_offsets < V_DIM
    v_out = tl.load(
        kvb_ptr + batch_idx * kvb_stride_b + (kvb_base + QK_NOPE_DIM + v_offsets) * kvb_stride_d,
        mask=v_mask,
        other=0.0,
    ).to(tl.float32)

    tl.store(
        q_out_ptr
        + batch_idx * q_out_stride_b
        + head_idx * q_out_stride_h
        + 0 * q_out_stride_s
        + d_offsets * q_out_stride_d,
        q_out.to(tl.bfloat16),
        mask=q_mask,
    )
    tl.store(
        k_new_ptr
        + batch_idx * k_new_stride_b
        + head_idx * k_new_stride_h
        + 0 * k_new_stride_s
        + d_offsets * k_new_stride_d,
        k_out.to(tl.bfloat16),
        mask=q_mask,
    )
    tl.store(
        v_new_ptr
        + batch_idx * v_new_stride_b
        + head_idx * v_new_stride_h
        + 0 * v_new_stride_s
        + v_offsets * v_new_stride_d,
        v_out.to(tl.bfloat16),
        mask=v_mask,
    )


def _get_batched_workspace(
    packed: AttentionPackedWeights,
    device: torch.device,
    batch_size: int,
) -> BatchedAttentionDecodeWorkspace:
    device_key = device.index if device.index is not None else 0
    key = (device_key, batch_size)
    workspace = getattr(packed, "batched_workspace_by_device", {}).get(key)
    if workspace is not None:
        return workspace
    if not hasattr(packed, "batched_workspace_by_device"):
        packed.batched_workspace_by_device = {}

    q_kv_a_out = torch.empty((batch_size, DSV2L_Q_ROWS + DSV2L_KV_A_ROWS), device=device, dtype=torch.bfloat16)
    workspace = BatchedAttentionDecodeWorkspace(
        batch_size=batch_size,
        q_kv_a_out=q_kv_a_out,
        kv_lora=q_kv_a_out[:, DSV2L_Q_ROWS : DSV2L_Q_ROWS + DSV2L_KV_LORA_RANK],
        k_pe=q_kv_a_out[:, DSV2L_Q_ROWS + DSV2L_KV_LORA_RANK :],
        kv_lora_norm=torch.empty((batch_size, DSV2L_KV_LORA_RANK), device=device, dtype=torch.bfloat16),
        kvb_proj=torch.empty((batch_size, DSV2L_KVB_ROWS), device=device, dtype=torch.bfloat16),
        q_out=torch.empty((batch_size, DSV2L_NUM_HEADS, 1, DSV2L_Q_DIM), device=device, dtype=torch.bfloat16),
        k_new=torch.empty((batch_size, DSV2L_NUM_HEADS, 1, DSV2L_Q_DIM), device=device, dtype=torch.bfloat16),
        v_new=torch.empty((batch_size, DSV2L_NUM_HEADS, 1, DSV2L_V_DIM), device=device, dtype=torch.bfloat16),
        attn_ctx=torch.empty((batch_size, DSV2L_NUM_HEADS, 1, DSV2L_V_DIM), device=device, dtype=torch.bfloat16),
        o_out=torch.empty((batch_size, 1, DSV2L_HIDDEN), device=device, dtype=torch.bfloat16),
    )
    packed.batched_workspace_by_device[key] = workspace
    return workspace


def attention_decode_triton(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    packed: AttentionPackedWeights,
    softmax_scale: float,
) -> torch.Tensor:
    if hidden_states.device.type != "cuda":
        raise NotImplementedError("attention_decode_triton requires CUDA tensors")
    if hidden_states.dtype != torch.bfloat16:
        raise NotImplementedError("attention_decode_triton supports bf16 only")
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1 or hidden_states.shape[2] != DSV2L_HIDDEN:
        raise ValueError("hidden_states shape must be [B,1,2048]")
    if position_ids.ndim != 2 or position_ids.shape[1] != 1 or position_ids.dtype != torch.long:
        raise ValueError("position_ids must be CUDA int64 tensor with shape [B,1]")
    if cache_position.numel() != 1 or cache_position.dtype != torch.long:
        raise ValueError("cache_position must be CUDA int64 tensor with one element")

    batch_size = int(hidden_states.shape[0])
    x = hidden_states.view(batch_size, DSV2L_HIDDEN)
    workspace = _get_batched_workspace(packed, x.device, batch_size)

    block_b, block_o, block_k = _select_batched_linear_tile(batch_size)
    _batched_linear_kernel[(triton.cdiv(batch_size, block_b), triton.cdiv(DSV2L_Q_ROWS + DSV2L_KV_A_ROWS, block_o))](
        x,
        packed.q_kv_a_weight,
        workspace.q_kv_a_out,
        batch_size,
        DSV2L_Q_ROWS + DSV2L_KV_A_ROWS,
        DSV2L_HIDDEN,
        x.stride(0),
        x.stride(1),
        packed.q_kv_a_weight.stride(0),
        packed.q_kv_a_weight.stride(1),
        workspace.q_kv_a_out.stride(0),
        workspace.q_kv_a_out.stride(1),
        BLOCK_B=block_b,
        BLOCK_O=block_o,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=4,
    )

    kv_lora_norm = rmsnorm_triton(
        workspace.kv_lora.view(batch_size, 1, DSV2L_KV_LORA_RANK),
        packed.kv_a_ln_weight,
        packed.kv_a_ln_eps,
    ).view(batch_size, DSV2L_KV_LORA_RANK)
    workspace.kv_lora_norm.copy_(kv_lora_norm)

    block_b, block_o, block_k = _select_batched_linear_tile(batch_size)
    _batched_linear_kernel[(triton.cdiv(batch_size, block_b), triton.cdiv(DSV2L_KVB_ROWS, block_o))](
        workspace.kv_lora_norm,
        packed.kv_b_weight,
        workspace.kvb_proj,
        batch_size,
        DSV2L_KVB_ROWS,
        DSV2L_KV_LORA_RANK,
        workspace.kv_lora_norm.stride(0),
        workspace.kv_lora_norm.stride(1),
        packed.kv_b_weight.stride(0),
        packed.kv_b_weight.stride(1),
        workspace.kvb_proj.stride(0),
        workspace.kvb_proj.stride(1),
        BLOCK_B=block_b,
        BLOCK_O=block_o,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=4,
    )

    _batched_build_qkv_rope_kernel[(batch_size, DSV2L_NUM_HEADS, triton.cdiv(DSV2L_Q_DIM, 64))](
        workspace.q_kv_a_out,
        workspace.kvb_proj,
        cos,
        sin,
        cache_position.view(-1),
        workspace.q_out,
        workspace.k_new,
        workspace.v_new,
        batch_size,
        workspace.q_kv_a_out.stride(0),
        workspace.q_kv_a_out.stride(1),
        workspace.kvb_proj.stride(0),
        workspace.kvb_proj.stride(1),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        workspace.q_out.stride(0),
        workspace.q_out.stride(1),
        workspace.q_out.stride(2),
        workspace.q_out.stride(3),
        workspace.k_new.stride(0),
        workspace.k_new.stride(1),
        workspace.k_new.stride(2),
        workspace.k_new.stride(3),
        workspace.v_new.stride(0),
        workspace.v_new.stride(1),
        workspace.v_new.stride(2),
        workspace.v_new.stride(3),
        Q_ROWS=DSV2L_Q_ROWS,
        Q_DIM=DSV2L_Q_DIM,
        QK_NOPE_DIM=DSV2L_QK_NOPE_DIM,
        QK_ROPE_DIM=DSV2L_QK_ROPE_DIM,
        KV_LORA_RANK=DSV2L_KV_LORA_RANK,
        V_DIM=DSV2L_V_DIM,
        BLOCK_D=64,
        num_warps=4,
    )

    _cache_write_q1_dual_kernel[(batch_size, DSV2L_NUM_HEADS, triton.cdiv(max(DSV2L_Q_DIM, DSV2L_V_DIM), 64))](
        workspace.k_new,
        workspace.v_new,
        key_cache,
        value_cache,
        cache_position.view(-1),
        DSV2L_Q_DIM,
        DSV2L_V_DIM,
        workspace.k_new.stride(0),
        workspace.k_new.stride(1),
        workspace.k_new.stride(2),
        workspace.k_new.stride(3),
        workspace.v_new.stride(0),
        workspace.v_new.stride(1),
        workspace.v_new.stride(2),
        workspace.v_new.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        BLOCK_D=64,
        num_warps=4,
    )

    workspace.attn_ctx.copy_(
        batched_attention_decode_q1_triton(
            workspace.q_out,
            key_cache,
            value_cache,
            cache_position,
            softmax_scale,
        )
    )
    attn_ctx_flat = workspace.attn_ctx.view(batch_size, DSV2L_NUM_HEADS * DSV2L_V_DIM)
    o_out_flat = workspace.o_out.view(batch_size, DSV2L_HIDDEN)
    block_b, block_o, block_k = _select_batched_linear_tile(batch_size)
    _batched_linear_kernel[(triton.cdiv(batch_size, block_b), triton.cdiv(DSV2L_HIDDEN, block_o))](
        attn_ctx_flat,
        packed.o_weight,
        o_out_flat,
        batch_size,
        DSV2L_HIDDEN,
        DSV2L_HIDDEN,
        attn_ctx_flat.stride(0),
        attn_ctx_flat.stride(1),
        packed.o_weight.stride(0),
        packed.o_weight.stride(1),
        o_out_flat.stride(0),
        o_out_flat.stride(1),
        BLOCK_B=block_b,
        BLOCK_O=block_o,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=4,
    )
    return workspace.o_out
