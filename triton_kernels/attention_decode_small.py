from __future__ import annotations

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
def _gemv_contig_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    out_rows,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    IN_COLS: tl.constexpr,
    EXACT_ROWS: tl.constexpr,
):
    row_block = tl.program_id(0)
    row_offsets = row_block * BLOCK_ROW + tl.arange(0, BLOCK_ROW)

    acc = tl.zeros((BLOCK_ROW,), dtype=tl.float32)
    col_offsets_base = tl.arange(0, BLOCK_COL)
    col_start = 0
    while col_start < IN_COLS:
        cols = col_start + col_offsets_base
        x = tl.load(x_ptr + cols)
        x_col = x[:, None]

        w_block_ptr = tl.make_block_ptr(
            base=w_ptr,
            shape=(out_rows, IN_COLS),
            strides=(IN_COLS, 1),
            offsets=(row_block * BLOCK_ROW, col_start),
            block_shape=(BLOCK_ROW, BLOCK_COL),
            order=(1, 0),
        )
        if EXACT_ROWS:
            w = tl.load(w_block_ptr, boundary_check=(1,), padding_option="zero")
        else:
            w = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += tl.reshape(tl.dot(w, x_col), (BLOCK_ROW,))
        col_start += BLOCK_COL

    out_ptrs = out_ptr + row_offsets
    if EXACT_ROWS:
        tl.store(out_ptrs, acc.to(tl.bfloat16))
    else:
        row_mask = row_offsets < out_rows
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _gemv_2048x2048_o_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    row_block = tl.program_id(0)
    row_offsets = row_block * BLOCK_ROW + tl.arange(0, BLOCK_ROW)

    acc = tl.zeros((BLOCK_ROW,), dtype=tl.float32)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(2048, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_COL, 1),
        order=(1, 0),
    )
    w_block_ptr = tl.make_block_ptr(
        base=w_ptr,
        shape=(2048, 2048),
        strides=(2048, 1),
        offsets=(row_block * BLOCK_ROW, 0),
        block_shape=(BLOCK_ROW, BLOCK_COL),
        order=(1, 0),
    )

    for _ in range(0, 2048, BLOCK_COL):
        x_col = tl.load(x_block_ptr)
        w = tl.load(w_block_ptr)
        acc += tl.reshape(tl.dot(w, x_col), (BLOCK_ROW,))

        x_block_ptr = tl.advance(x_block_ptr, (BLOCK_COL, 0))
        w_block_ptr = tl.advance(w_block_ptr, (0, BLOCK_COL))

    out_ptrs = out_ptr + row_offsets
    tl.store(out_ptrs, acc.to(tl.bfloat16))


@triton.jit
def _build_qkv_rope_kernel(
    q_proj_ptr,
    kvb_proj_ptr,
    k_pe_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    q_out_ptr,
    k_new_ptr,
    v_new_ptr,
    num_heads,
    q_dim,
    qk_nope_dim,
    qk_rope_dim,
    v_dim,
    q_proj_stride_h,
    q_proj_stride_d,
    kvb_stride_h,
    kvb_stride_d,
    cos_stride_s,
    cos_stride_d,
    sin_stride_s,
    sin_stride_d,
    q_out_stride_h,
    q_out_stride_d,
    k_new_stride_h,
    k_new_stride_d,
    v_new_stride_h,
    v_new_stride_d,
    BLOCK_D: tl.constexpr,
):
    head_idx = tl.program_id(0)
    d_block = tl.program_id(1)
    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask_q = d_offsets < q_dim

    if head_idx >= num_heads:
        return

    pos = tl.load(pos_ptr).to(tl.int32)

    # q_nope direct copy
    q_nope_mask = d_offsets < qk_nope_dim
    q_nope = tl.load(
        q_proj_ptr + head_idx * q_proj_stride_h + d_offsets * q_proj_stride_d,
        mask=q_nope_mask,
        other=0.0,
    ).to(tl.float32)

    # rope part in DeepSeek layout:
    # output rope dim is [first_half(rot-even), second_half(rot-odd)].
    rope_idx = d_offsets - qk_nope_dim
    rope_mask = (d_offsets >= qk_nope_dim) & (rope_idx < qk_rope_dim)
    half = qk_rope_dim // 2
    first_half = rope_idx < half
    pair_base = tl.where(first_half, rope_idx * 2, (rope_idx - half) * 2)

    q_even = tl.load(
        q_proj_ptr + head_idx * q_proj_stride_h + (qk_nope_dim + pair_base) * q_proj_stride_d,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    q_odd = tl.load(
        q_proj_ptr + head_idx * q_proj_stride_h + (qk_nope_dim + pair_base + 1) * q_proj_stride_d,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    k_even = tl.load(k_pe_ptr + pair_base, mask=rope_mask, other=0.0).to(tl.float32)
    k_odd = tl.load(k_pe_ptr + pair_base + 1, mask=rope_mask, other=0.0).to(tl.float32)

    cos_idx = tl.where(first_half, rope_idx, rope_idx - half)
    cos_v = tl.load(
        cos_ptr + pos * cos_stride_s + cos_idx * cos_stride_d,
        mask=rope_mask,
        other=1.0,
    ).to(tl.float32)
    sin_v = tl.load(
        sin_ptr + pos * sin_stride_s + cos_idx * sin_stride_d,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)

    first_q = q_even * cos_v - q_odd * sin_v
    second_q = q_odd * cos_v + q_even * sin_v
    first_k = k_even * cos_v - k_odd * sin_v
    second_k = k_odd * cos_v + k_even * sin_v

    q_rope_out = tl.where(first_half, first_q, second_q)
    k_rope_out = tl.where(first_half, first_k, second_k)

    q_out = tl.where(q_nope_mask, q_nope, 0.0)
    q_out = tl.where(rope_mask, q_rope_out, q_out)

    k_nope = tl.load(
        kvb_proj_ptr + head_idx * kvb_stride_h + d_offsets * kvb_stride_d,
        mask=q_nope_mask,
        other=0.0,
    ).to(tl.float32)
    k_new = tl.where(q_nope_mask, k_nope, 0.0)
    k_new = tl.where(rope_mask, k_rope_out, k_new)

    v_offsets = d_offsets
    v_mask = v_offsets < v_dim
    v = tl.load(
        kvb_proj_ptr + head_idx * kvb_stride_h + (qk_nope_dim + v_offsets) * kvb_stride_d,
        mask=v_mask,
        other=0.0,
    ).to(tl.float32)

    tl.store(
        q_out_ptr + head_idx * q_out_stride_h + d_offsets * q_out_stride_d,
        q_out.to(tl.bfloat16),
        mask=d_mask_q,
    )
    tl.store(
        k_new_ptr + head_idx * k_new_stride_h + d_offsets * k_new_stride_d,
        k_new.to(tl.bfloat16),
        mask=d_mask_q,
    )
    tl.store(
        v_new_ptr + head_idx * v_new_stride_h + v_offsets * v_new_stride_d,
        v.to(tl.bfloat16),
        mask=v_mask,
    )


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


def _run_gemv_2048(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
) -> None:
    out_rows = int(out.numel())
    if out_rows == 2048:
        _gemv_contig_kernel[(triton.cdiv(out_rows, 64),)](
            x,
            w,
            out,
            out_rows,
            BLOCK_ROW=64,
            BLOCK_COL=256,
            IN_COLS=2048,
            EXACT_ROWS=True,
            num_warps=4,
            num_stages=4,
        )
    else:
        _gemv_contig_kernel[(triton.cdiv(out_rows, 32),)](
            x,
            w,
            out,
            out_rows,
            BLOCK_ROW=32,
            BLOCK_COL=256,
            IN_COLS=2048,
            EXACT_ROWS=True,
            num_warps=4,
            num_stages=3,
        )


def _run_gemv_o_proj_2048(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _gemv_2048x2048_o_kernel[(64,)](
        x,
        w,
        out,
        BLOCK_ROW=32,
        BLOCK_COL=256,
        num_warps=4,
        num_stages=3,
    )


def _run_gemv_512(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
) -> None:
    out_rows = int(out.numel())
    _gemv_contig_kernel[(triton.cdiv(out_rows, 32),)](
        x,
        w,
        out,
        out_rows,
        BLOCK_ROW=32,
        BLOCK_COL=128,
        IN_COLS=512,
        EXACT_ROWS=True,
        num_warps=8,
        num_stages=3,
    )


def _get_workspace(packed: AttentionPackedWeights, device: torch.device) -> AttentionDecodeWorkspace:
    device_key = device.index if device.index is not None else 0
    workspace = packed.workspace_by_device.get(device_key)
    if workspace is not None:
        return workspace

    workspace = AttentionDecodeWorkspace(
        q_kv_a_out=torch.empty((DSV2L_Q_ROWS + DSV2L_KV_A_ROWS,), device=device, dtype=torch.bfloat16),
        q_proj=torch.empty((0,), device=device, dtype=torch.bfloat16),
        kv_a_out=torch.empty((0,), device=device, dtype=torch.bfloat16),
        kv_lora=torch.empty((0,), device=device, dtype=torch.bfloat16),
        k_pe=torch.empty((0,), device=device, dtype=torch.bfloat16),
        kvb_proj=torch.empty((DSV2L_NUM_HEADS, DSV2L_QK_NOPE_DIM + DSV2L_V_DIM), device=device, dtype=torch.bfloat16),
        kvb_proj_flat=torch.empty((0,), device=device, dtype=torch.bfloat16),
        q_out=torch.empty((DSV2L_NUM_HEADS, DSV2L_Q_DIM), device=device, dtype=torch.bfloat16),
        k_new=torch.empty((DSV2L_NUM_HEADS, DSV2L_Q_DIM), device=device, dtype=torch.bfloat16),
        v_new=torch.empty((DSV2L_NUM_HEADS, DSV2L_V_DIM), device=device, dtype=torch.bfloat16),
        attn_ctx=torch.empty((DSV2L_NUM_HEADS, DSV2L_V_DIM), device=device, dtype=torch.bfloat16),
        attn_ctx_flat=torch.empty((0,), device=device, dtype=torch.bfloat16),
        o_out=torch.empty((DSV2L_HIDDEN,), device=device, dtype=torch.bfloat16),
    )
    workspace.q_proj = workspace.q_kv_a_out[:DSV2L_Q_ROWS].view(DSV2L_NUM_HEADS, DSV2L_Q_DIM)
    workspace.kv_a_out = workspace.q_kv_a_out[DSV2L_Q_ROWS:]
    workspace.kv_lora = workspace.kv_a_out[:DSV2L_KV_LORA_RANK]
    workspace.k_pe = workspace.kv_a_out[DSV2L_KV_LORA_RANK:]
    workspace.kvb_proj_flat = workspace.kvb_proj.view(-1)
    workspace.attn_ctx_flat = workspace.attn_ctx.view(-1)
    packed.workspace_by_device[device_key] = workspace
    return workspace


def attention_decode_only_triton(
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
        raise NotImplementedError("attention_decode_only_triton requires CUDA tensors")
    if hidden_states.dtype != torch.bfloat16:
        raise NotImplementedError("attention_decode_only_triton supports bf16 only")
    if hidden_states.shape != (1, 1, DSV2L_HIDDEN):
        raise ValueError("hidden_states shape must be [1,1,2048]")
    if position_ids.shape != (1, 1) or position_ids.dtype != torch.long:
        raise ValueError("position_ids must be CUDA int64 tensor with shape [1,1]")
    if cache_position.numel() != 1 or cache_position.dtype != torch.long:
        raise ValueError("cache_position must be CUDA int64 tensor with one element")

    x = hidden_states.view(DSV2L_HIDDEN)
    workspace = _get_workspace(packed, x.device)
    position_ids_1d = position_ids.view(-1)
    cache_position_1d = cache_position.view(-1)

    _run_gemv_2048(x=x, w=packed.q_kv_a_weight, out=workspace.q_kv_a_out)
    kv_lora_norm = rmsnorm_triton(workspace.kv_lora.view(1, 1, DSV2L_KV_LORA_RANK), packed.kv_a_ln_weight, packed.kv_a_ln_eps).view(-1)

    _run_gemv_512(x=kv_lora_norm, w=packed.kv_b_weight, out=workspace.kvb_proj_flat)

    _build_qkv_rope_kernel[(DSV2L_NUM_HEADS, triton.cdiv(DSV2L_Q_DIM, 64))](
        workspace.q_proj,
        workspace.kvb_proj,
        workspace.k_pe,
        cos,
        sin,
        position_ids_1d,
        workspace.q_out,
        workspace.k_new,
        workspace.v_new,
        DSV2L_NUM_HEADS,
        DSV2L_Q_DIM,
        DSV2L_QK_NOPE_DIM,
        DSV2L_QK_ROPE_DIM,
        DSV2L_V_DIM,
        workspace.q_proj.stride(0),
        workspace.q_proj.stride(1),
        workspace.kvb_proj.stride(0),
        workspace.kvb_proj.stride(1),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        workspace.q_out.stride(0),
        workspace.q_out.stride(1),
        workspace.k_new.stride(0),
        workspace.k_new.stride(1),
        workspace.v_new.stride(0),
        workspace.v_new.stride(1),
        BLOCK_D=64,
        num_warps=4,
    )

    _cache_write_q1_dual_kernel[(1, DSV2L_NUM_HEADS, triton.cdiv(max(DSV2L_Q_DIM, DSV2L_V_DIM), 64))](
        workspace.k_new,
        workspace.v_new,
        key_cache,
        value_cache,
        cache_position_1d,
        DSV2L_Q_DIM,
        DSV2L_V_DIM,
        0,
        workspace.k_new.stride(0),
        0,
        workspace.k_new.stride(1),
        0,
        workspace.v_new.stride(0),
        0,
        workspace.v_new.stride(1),
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

    _decode_attention_q1_kernel[(DSV2L_NUM_HEADS, triton.cdiv(DSV2L_V_DIM, 64))](
        workspace.q_out,
        key_cache[0],
        value_cache[0],
        cache_position_1d,
        workspace.attn_ctx,
        DSV2L_Q_DIM,
        DSV2L_V_DIM,
        workspace.q_out.stride(0),
        workspace.q_out.stride(1),
        key_cache[0].stride(0),
        key_cache[0].stride(1),
        key_cache[0].stride(2),
        value_cache[0].stride(0),
        value_cache[0].stride(1),
        value_cache[0].stride(2),
        workspace.attn_ctx.stride(0),
        workspace.attn_ctx.stride(1),
        float(softmax_scale),
        BLOCK_N=64,
        BLOCK_Q=64,
        BLOCK_V=64,
        num_warps=4,
    )

    # o_proj: [hidden, heads*v_dim] x [heads*v_dim]
    o_out = workspace.o_out
    _run_gemv_o_proj_2048(x=workspace.attn_ctx_flat, w=packed.o_weight, out=o_out)
    return o_out.view(1, 1, DSV2L_HIDDEN)


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
def _small_batch_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    out_rows,
    in_cols,
    x_stride_b,
    x_stride_k,
    w_stride_o,
    w_stride_k,
    out_stride_b,
    out_stride_o,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    EXACT_ROWS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    row_block = tl.program_id(1)
    row_offsets = row_block * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    row_mask = row_offsets < out_rows

    acc = tl.zeros((BLOCK_ROW,), dtype=tl.float32)
    col_offsets_base = tl.arange(0, BLOCK_COL)
    col_start = 0
    while col_start < in_cols:
        col_offsets = col_start + col_offsets_base
        col_mask = col_offsets < in_cols
        x = tl.load(
            x_ptr + batch_idx * x_stride_b + col_offsets * x_stride_k,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        w_ptrs = w_ptr + row_offsets[:, None] * w_stride_o + col_offsets[None, :] * w_stride_k
        if EXACT_ROWS:
            w = tl.load(w_ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
        else:
            w = tl.load(w_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.reshape(tl.dot(w, x[:, None]), (BLOCK_ROW,))
        col_start += BLOCK_COL

    if EXACT_ROWS:
        tl.store(out_ptr + batch_idx * out_stride_b + row_offsets * out_stride_o, acc.to(tl.bfloat16))
    else:
        tl.store(out_ptr + batch_idx * out_stride_b + row_offsets * out_stride_o, acc.to(tl.bfloat16), mask=row_mask)


def _run_small_batch_gemv(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    out_rows: int,
    in_cols: int,
    exact_rows: bool,
) -> None:
    _small_batch_gemv_kernel[(int(x.shape[0]), triton.cdiv(out_rows, 64))](
        x,
        w,
        out,
        out_rows,
        in_cols,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_ROW=64,
        BLOCK_COL=256,
        EXACT_ROWS=exact_rows,
        num_warps=4,
        num_stages=4,
    )


def attention_decode_small_batch_triton(
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
        raise NotImplementedError("attention_decode_small_batch_triton requires CUDA tensors")
    if hidden_states.dtype != torch.bfloat16:
        raise NotImplementedError("attention_decode_small_batch_triton supports bf16 only")
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1 or hidden_states.shape[2] != DSV2L_HIDDEN:
        raise ValueError("hidden_states shape must be [B,1,2048]")
    if position_ids.ndim != 2 or position_ids.shape[1] != 1 or position_ids.dtype != torch.long:
        raise ValueError("position_ids must be CUDA int64 tensor with shape [B,1]")
    if cache_position.numel() != 1 or cache_position.dtype != torch.long:
        raise ValueError("cache_position must be CUDA int64 tensor with one element")

    batch_size = int(hidden_states.shape[0])
    x = hidden_states.view(batch_size, DSV2L_HIDDEN)
    workspace = _get_batched_workspace(packed, x.device, batch_size)

    _run_small_batch_gemv(
        x=x,
        w=packed.q_kv_a_weight,
        out=workspace.q_kv_a_out,
        out_rows=DSV2L_Q_ROWS + DSV2L_KV_A_ROWS,
        in_cols=DSV2L_HIDDEN,
        exact_rows=True,
    )

    kv_lora_norm = rmsnorm_triton(
        workspace.kv_lora.view(batch_size, 1, DSV2L_KV_LORA_RANK),
        packed.kv_a_ln_weight,
        packed.kv_a_ln_eps,
    ).view(batch_size, DSV2L_KV_LORA_RANK)
    workspace.kv_lora_norm.copy_(kv_lora_norm)

    _run_small_batch_gemv(
        x=workspace.kv_lora_norm,
        w=packed.kv_b_weight,
        out=workspace.kvb_proj,
        out_rows=DSV2L_KVB_ROWS,
        in_cols=DSV2L_KV_LORA_RANK,
        exact_rows=True,
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
    _run_small_batch_gemv(
        x=attn_ctx_flat,
        w=packed.o_weight,
        out=o_out_flat,
        out_rows=DSV2L_HIDDEN,
        in_cols=DSV2L_HIDDEN,
        exact_rows=True,
    )
    return workspace.o_out


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


