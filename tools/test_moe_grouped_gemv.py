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

    packed = pack_routed_experts(moe)

    def reference(x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        ref = torch.zeros_like(x)
        for slot in range(config.num_experts_per_tok):
            for row in range(x.shape[0]):
                expert_idx = int(topk_ids[row, slot].item())
                expert_out = moe.experts[expert_idx](x[row : row + 1])
                ref[row : row + 1] += expert_out * topk_weight[row : row + 1, slot : slot + 1].to(dtype=expert_out.dtype)
        return ref

    results = {}
    with torch.inference_mode():
        for batch_size in (1, 2, 4, 8, 16):
            x = torch.randn((batch_size, config.hidden_size), device="cuda", dtype=torch.bfloat16)
            topk_ids = torch.randint(
                0,
                config.n_routed_experts,
                (batch_size, config.num_experts_per_tok),
                device="cuda",
                dtype=torch.long,
            )
            topk_weight = torch.randn(
                (batch_size, config.num_experts_per_tok),
                device="cuda",
                dtype=torch.float32,
            ).softmax(dim=-1)
            ref = reference(x, topk_ids, topk_weight)
            out = batched_grouped_routed_moe(x, topk_ids, topk_weight, packed)
            results[f"batch_{batch_size}"] = {
                "max_abs_err": torch.max(torch.abs(ref.float() - out.float())).item(),
                "ref_norm": torch.linalg.vector_norm(ref.float()).item(),
                "out_norm": torch.linalg.vector_norm(out.float()).item(),
            }

    result = {
        "results": results,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
