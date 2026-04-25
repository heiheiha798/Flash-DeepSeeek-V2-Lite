from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_kernels.moe_grouped_gemv import batched_grouped_routed_moe, grouped_routed_moe, pack_routed_experts


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/data/models/DeepSeek-V2-Lite-Chat"))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    moe = None
    for layer in model.model.layers:
        if layer.mlp.__class__.__name__ == "DeepseekV2MoE":
            moe = layer.mlp
            break
    if moe is None:
        raise RuntimeError("Failed to find DeepseekV2MoE layer")
    config = model.config

    x = torch.randn((1, config.hidden_size), device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.tensor([[1, 7, 9, 13, 21, 42]], device="cuda", dtype=torch.long)
    topk_weight = torch.randn((1, config.num_experts_per_tok), device="cuda", dtype=torch.float32).softmax(dim=-1)

    packed = pack_routed_experts(moe)

    with torch.inference_mode():
        ref = torch.zeros_like(x)
        for slot in range(config.num_experts_per_tok):
            expert_idx = int(topk_ids[0, slot].item())
            expert_out = moe.experts[expert_idx](x)
            ref += expert_out * topk_weight[:, slot : slot + 1].to(dtype=expert_out.dtype)

        out = grouped_routed_moe(x, topk_ids, topk_weight, packed)

    max_abs_err = torch.max(torch.abs(ref.float() - out.float())).item()

    batch_size = 16
    xb = torch.randn((batch_size, config.hidden_size), device="cuda", dtype=torch.bfloat16)
    topk_ids_b = torch.randint(
        0,
        config.n_routed_experts,
        (batch_size, config.num_experts_per_tok),
        device="cuda",
        dtype=torch.long,
    )
    topk_weight_b = torch.randn((batch_size, config.num_experts_per_tok), device="cuda", dtype=torch.float32).softmax(dim=-1)
    ref_b = torch.zeros_like(xb)
    for slot in range(config.num_experts_per_tok):
        for row in range(batch_size):
            expert_idx = int(topk_ids_b[row, slot].item())
            expert_out = moe.experts[expert_idx](xb[row : row + 1])
            ref_b[row : row + 1] += expert_out * topk_weight_b[row : row + 1, slot : slot + 1].to(dtype=expert_out.dtype)
    out_b = batched_grouped_routed_moe(xb, topk_ids_b, topk_weight_b, packed)

    batch_max_abs_err = torch.max(torch.abs(ref_b.float() - out_b.float())).item()
    result = {
        "max_abs_err": max_abs_err,
        "ref_norm": torch.linalg.vector_norm(ref.float()).item(),
        "out_norm": torch.linalg.vector_norm(out.float()).item(),
        "batch_max_abs_err": batch_max_abs_err,
        "batch_ref_norm": torch.linalg.vector_norm(ref_b.float()).item(),
        "batch_out_norm": torch.linalg.vector_norm(out_b.float()).item(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
