from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


def silu_mul_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.device.type != "cuda" or up.device.type != "cuda":
        raise NotImplementedError("silu_mul_triton requires CUDA tensors")
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
        raise NotImplementedError("silu_mul_triton supports bf16 only")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise NotImplementedError("silu_mul_triton requires contiguous inputs")

    out = torch.empty_like(gate)
    n_elements = gate.numel()
    block = 1024
    _silu_mul_kernel[(triton.cdiv(n_elements, block),)](
        gate,
        up,
        out,
        n_elements,
        BLOCK=block,
        num_warps=4,
    )
    return out
