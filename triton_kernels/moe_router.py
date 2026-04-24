from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _router_softmax_topk6_kernel(
    logits_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    scaling: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_E)
    logits = tl.load(logits_ptr + offsets).to(tl.float32)
    logits = logits - tl.max(logits, axis=0)
    scores = tl.exp(logits)
    scores = scores / tl.sum(scores, axis=0)

    neg_inf = tl.full((BLOCK_E,), -float("inf"), dtype=tl.float32)
    work = scores

    val0 = tl.max(work, axis=0)
    idx0 = tl.argmax(work, axis=0)
    work = tl.where(offsets == idx0, neg_inf, work)

    val1 = tl.max(work, axis=0)
    idx1 = tl.argmax(work, axis=0)
    work = tl.where(offsets == idx1, neg_inf, work)

    val2 = tl.max(work, axis=0)
    idx2 = tl.argmax(work, axis=0)
    work = tl.where(offsets == idx2, neg_inf, work)

    val3 = tl.max(work, axis=0)
    idx3 = tl.argmax(work, axis=0)
    work = tl.where(offsets == idx3, neg_inf, work)

    val4 = tl.max(work, axis=0)
    idx4 = tl.argmax(work, axis=0)
    work = tl.where(offsets == idx4, neg_inf, work)

    val5 = tl.max(work, axis=0)
    idx5 = tl.argmax(work, axis=0)

    tl.store(topk_ids_ptr + 0, idx0.to(tl.int64))
    tl.store(topk_ids_ptr + 1, idx1.to(tl.int64))
    tl.store(topk_ids_ptr + 2, idx2.to(tl.int64))
    tl.store(topk_ids_ptr + 3, idx3.to(tl.int64))
    tl.store(topk_ids_ptr + 4, idx4.to(tl.int64))
    tl.store(topk_ids_ptr + 5, idx5.to(tl.int64))

    tl.store(topk_weights_ptr + 0, val0 * scaling)
    tl.store(topk_weights_ptr + 1, val1 * scaling)
    tl.store(topk_weights_ptr + 2, val2 * scaling)
    tl.store(topk_weights_ptr + 3, val3 * scaling)
    tl.store(topk_weights_ptr + 4, val4 * scaling)
    tl.store(topk_weights_ptr + 5, val5 * scaling)


def router_softmax_topk6_triton(
    logits: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.device.type != "cuda":
        raise NotImplementedError("router_softmax_topk6_triton requires CUDA tensors")
    if logits.dtype != torch.float32:
        raise NotImplementedError("router_softmax_topk6_triton expects fp32 logits")
    if logits.shape != (1, 64):
        raise NotImplementedError("router_softmax_topk6_triton is specialized for [1, 64] logits")
    if not logits.is_contiguous():
        raise NotImplementedError("router_softmax_topk6_triton requires contiguous logits")

    topk_ids = torch.empty((1, 6), device=logits.device, dtype=torch.long)
    topk_weights = torch.empty((1, 6), device=logits.device, dtype=torch.float32)
    _router_softmax_topk6_kernel[(1,)](
        logits,
        topk_ids,
        topk_weights,
        scaling=float(scaling),
        BLOCK_E=64,
        num_warps=2,
    )
    return topk_ids, topk_weights
