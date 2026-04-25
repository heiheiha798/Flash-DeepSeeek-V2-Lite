import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, CacheLayerMixin


MODEL_PATH = Path("/data/models/DeepSeek-V2-Lite-Chat")
DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_PROMPT = "Write me a 500 word novel."
GRAPH_WARMUP_STEPS = 3

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hardware_info import collect_hardware_info
from triton_kernels.moe_grouped_gemv import PackedRoutedExperts, packed_routed_moe, pack_routed_experts
from triton_kernels.moe_router import router_softmax_topk6_triton
from triton_kernels.attention_prepost import (
    copy_single_token_to_cache,
    prepare_decode_inputs_triton,
    residual_add_triton,
)
from triton_kernels.attention_decode import attention_decode_triton, pack_attention_weights
from triton_kernels.mlp_elementwise import silu_mul_triton
from triton_kernels.rmsnorm import rmsnorm_triton


class GraphCacheLayer(CacheLayerMixin):
    is_compileable = False

    def __init__(self, max_cache_len: int) -> None:
        super().__init__()
        self.max_cache_len = max_cache_len
        self.max_batch_size = 1
        self.seq_len = 0
        self.static_mode = False
        self.cache_position: Optional[torch.Tensor] = None
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        self.max_batch_size, self.num_heads, _, self.key_head_dim = key_states.shape
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.key_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.is_initialized = True

    def _lazy_initialize_values(self, value_states: torch.Tensor) -> None:
        self.value_head_dim = value_states.shape[-1]
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.value_head_dim),
            dtype=value_states.dtype,
            device=value_states.device,
        )
        self.cache_position = torch.zeros((1,), dtype=torch.long, device=value_states.device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, object]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        if self.values is None:
            self._lazy_initialize_values(value_states)

        q_len = key_states.shape[-2]
        if self.static_mode:
            if q_len != 1:
                raise NotImplementedError("Graph decode path only supports q_len=1 in static mode")
            position = self.cache_position
            if cache_kwargs is not None and cache_kwargs.get("cache_position") is not None:
                position = cache_kwargs["cache_position"]
            if position is None:
                raise RuntimeError("cache_position must be initialized before static decode")
            if getattr(self, "use_triton_cache_write", False):
                copy_single_token_to_cache(key_states, self.keys, position)
                copy_single_token_to_cache(value_states, self.values, position)
            else:
                self.keys.index_copy_(2, position, key_states)
                self.values.index_copy_(2, position, value_states)
            return self.keys, self.values

        start = self.seq_len
        next_seq_len = start + q_len
        if next_seq_len > self.max_cache_len:
            raise ValueError(
                f"Graph cache overflow: next_seq_len={next_seq_len}, max_cache_len={self.max_cache_len}"
            )
        positions = torch.arange(start, next_seq_len, dtype=torch.long, device=key_states.device)
        self.keys.index_copy_(2, positions, key_states)
        self.values.index_copy_(2, positions, value_states)
        self.seq_len = next_seq_len
        return self.keys[:, :, :next_seq_len, :], self.values[:, :, :next_seq_len, :]

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        if self.static_mode:
            return self.max_cache_len, 0
        return self.seq_len + cache_position.shape[0], 0

    def get_seq_length(self) -> int:
        return self.seq_len

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def set_static_mode(self, enabled: bool) -> None:
        self.static_mode = enabled

    def set_cache_position_tensor(self, cache_position: torch.Tensor) -> None:
        self.cache_position = cache_position

    def set_seq_len(self, seq_len: int) -> None:
        self.seq_len = seq_len

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        if not self.is_initialized or self.keys is None or self.values is None:
            raise RuntimeError("Cannot snapshot an uninitialized cache layer")
        return self.keys[:, :, : self.seq_len, :].clone(), self.values[:, :, : self.seq_len, :].clone(), self.seq_len

    def restore(self, snapshot: tuple[torch.Tensor, torch.Tensor, int]) -> None:
        keys, values, seq_len = snapshot
        self.keys[:, :, :seq_len, :].copy_(keys)
        self.values[:, :, :seq_len, :].copy_(values)
        self.seq_len = seq_len


class GraphCache(Cache):
    def __init__(self, num_layers: int, max_cache_len: int) -> None:
        super().__init__(layers=[GraphCacheLayer(max_cache_len=max_cache_len) for _ in range(num_layers)])
        self._max_cache_len = max_cache_len
        self._static_mode = False

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        if self._static_mode:
            return self._max_cache_len - new_seq_length
        return self.get_seq_length(layer_idx)

    def get_max_length(self) -> int:
        return self._max_cache_len

    @property
    def seen_tokens(self) -> int:
        return self.get_seq_length(0)

    def set_static_mode(self, enabled: bool) -> None:
        self._static_mode = enabled
        for layer in self.layers:
            layer.set_static_mode(enabled)

    def share_cache_position(self) -> torch.Tensor:
        cache_position = next((layer.cache_position for layer in self.layers if layer.cache_position is not None), None)
        if cache_position is None:
            raise RuntimeError("cache_position is not initialized yet")
        for layer in self.layers:
            layer.set_cache_position_tensor(cache_position)
        return cache_position

    def set_seq_len(self, seq_len: int) -> None:
        for layer in self.layers:
            layer.set_seq_len(seq_len)

    def set_triton_cache_write(self, enabled: bool) -> None:
        for layer in self.layers:
            layer.use_triton_cache_write = enabled

    def snapshot(self) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        return [layer.snapshot() for layer in self.layers]

    def restore(self, snapshot: list[tuple[torch.Tensor, torch.Tensor, int]]) -> None:
        for layer, layer_snapshot in zip(self.layers, snapshot):
            layer.restore(layer_snapshot)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF decode-only CUDA graph SOTA path with Triton kernels.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=str, default=None, help='Comma/space separated sweep, e.g. "1 2 4".')
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dump-token-ids", action="store_true")
    parser.add_argument("--dump-step-trace", action="store_true")
    return parser.parse_args()


def _parse_batch_sizes(args: argparse.Namespace) -> list[int]:
    if args.batch_sizes is None:
        batch_sizes = [args.batch_size]
    else:
        raw_sizes = args.batch_sizes.replace(",", " ").replace("，", " ").split()
        batch_sizes = [int(size) for size in raw_sizes]
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    return batch_sizes


def _greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def _initialize_packed_moe(model: torch.nn.Module) -> None:
    def _graph_gate_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            hidden_states.device.type == "cuda"
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape == (1, 1, 2048)
            and self.top_k == 6
            and self.n_routed_experts == 64
            and self.topk_method == "greedy"
            and self.scoring_func == "softmax"
            and not self.norm_topk_prob
        ):
            logits = torch.nn.functional.linear(
                hidden_states.view(1, 2048).float(),
                self._graph_router_weight_fp32,
                None,
            )
            topk_ids, topk_weights = router_softmax_topk6_triton(logits, float(self.routed_scaling_factor))
            return topk_ids, topk_weights, None
        return self._graph_original_gate_forward(hidden_states)

    def _packed_moe_infer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        packed = PackedRoutedExperts(
            gate_up_weights=self._graph_packed_gate_up_weights,
            down_weights=self._graph_packed_down_weights,
            hidden_size=self._graph_packed_hidden_size,
            intermediate_size=self._graph_packed_intermediate_size,
            num_experts=self._graph_packed_num_experts,
            topk=self._graph_packed_topk,
        )
        return packed_routed_moe(x, topk_ids, topk_weight, packed, output_dtype=x.dtype)

    for layer in model.model.layers:
        mlp = layer.mlp
        if mlp.__class__.__name__ != "DeepseekV2MoE":
            continue
        if getattr(mlp, "_graph_moe_patched", False):
            continue

        packed = pack_routed_experts(mlp, device="cpu")
        num_experts = len(mlp.experts)
        mlp.experts = torch.nn.ModuleList([None for _ in range(num_experts)])
        mlp.register_buffer("_graph_packed_gate_up_weights", packed.gate_up_weights, persistent=False)
        mlp.register_buffer("_graph_packed_down_weights", packed.down_weights, persistent=False)
        mlp._graph_packed_hidden_size = packed.hidden_size
        mlp._graph_packed_intermediate_size = packed.intermediate_size
        mlp._graph_packed_num_experts = packed.num_experts
        mlp._graph_packed_topk = packed.topk
        mlp.moe_infer = _packed_moe_infer.__get__(mlp, type(mlp))
        mlp.gate.register_buffer(
            "_graph_router_weight_fp32",
            mlp.gate.weight.detach().float().contiguous(),
            persistent=False,
        )
        mlp.gate._graph_original_gate_forward = mlp.gate.forward
        mlp.gate.forward = _graph_gate_forward.__get__(mlp.gate, type(mlp.gate))
        mlp._graph_moe_patched = True

def _patch_rmsnorm_for_graph(model: torch.nn.Module) -> None:
    def _graph_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type != "cuda":
            return self._graph_original_rmsnorm_forward(hidden_states)
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            return self._graph_original_rmsnorm_forward(hidden_states)
        if hidden_states.shape[-1] != int(self.weight.shape[0]):
            return self._graph_original_rmsnorm_forward(hidden_states)
        if not hidden_states.is_contiguous() or not self.weight.is_contiguous():
            return self._graph_original_rmsnorm_forward(hidden_states)
        return rmsnorm_triton(hidden_states, self.weight, float(self.variance_epsilon))

    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV2RMSNorm":
            continue
        if getattr(module, "_graph_rmsnorm_patched", False):
            continue
        module._graph_original_rmsnorm_forward = module.forward
        module.forward = _graph_rmsnorm_forward.__get__(module, type(module))
        module._graph_rmsnorm_patched = True


def _patch_mlp_elementwise_for_graph(model: torch.nn.Module) -> None:
    def _graph_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if (
            gate.device.type == "cuda"
            and gate.dtype == torch.bfloat16
            and up.dtype == torch.bfloat16
            and gate.is_contiguous()
            and up.is_contiguous()
        ):
            return self.down_proj(silu_mul_triton(gate, up))
        return self._graph_original_mlp_forward(x)

    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV2MLP":
            continue
        if getattr(module, "_graph_mlp_elementwise_patched", False):
            continue
        module._graph_original_mlp_forward = module.forward
        module.forward = _graph_mlp_forward.__get__(module, type(module))
        module._graph_mlp_elementwise_patched = True


def _patch_attention_for_graph(model: torch.nn.Module) -> None:
    def _triton_decode_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Cache,
    ) -> tuple[torch.Tensor, None, Cache]:
        layer_cache = past_key_value.layers[self.layer_idx]
        kv_seq_len = layer_cache.max_cache_len if layer_cache.static_mode else layer_cache.seq_len + int(hidden_states.shape[1])
        cos, sin = self.rotary_emb(hidden_states, seq_len=kv_seq_len)
        attn_output = attention_decode_triton(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            cache_position=layer_cache.cache_position,
            key_cache=layer_cache.keys,
            value_cache=layer_cache.values,
            packed=self._graph_packed_attention_weights,
            softmax_scale=float(self.softmax_scale),
        )
        return attn_output, None, past_key_value

    def _graph_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if (
            hidden_states.device.type == "cuda"
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[1] == 1
            and past_key_value is not None
            and use_cache
            and position_ids is not None
            and attention_mask is not None
            and not output_attentions
            and getattr(self, "_graph_triton_full_attention", False)
        ):
            return _triton_decode_attention_forward(self, hidden_states, position_ids, past_key_value)

        return self._graph_original_attention_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

    for layer in model.model.layers:
        attn = layer.self_attn
        if attn.__class__.__name__ != "DeepseekV2Attention":
            continue
        if getattr(attn, "_graph_attention_patched", False):
            continue
        attn._graph_original_attention_forward = attn.forward
        attn._graph_packed_attention_weights = pack_attention_weights(attn)
        attn._graph_triton_full_attention = attn.q_lora_rank is None
        attn.forward = _graph_attention_forward.__get__(attn, type(attn))
        attn._graph_attention_patched = True


def _patch_residual_add_for_graph(model: torch.nn.Module) -> None:
    def _graph_decoder_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual_add_triton(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual_add_triton(residual, hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    for layer in model.model.layers:
        if layer.__class__.__name__ != "DeepseekV2DecoderLayer":
            continue
        if getattr(layer, "_graph_residual_patched", False):
            continue
        layer._graph_original_forward = layer.forward
        layer.forward = _graph_decoder_layer_forward.__get__(layer, type(layer))
        layer._graph_residual_patched = True


def _sync_cuda() -> None:
    torch.cuda.synchronize()


def _append_step_trace(
    step_trace: list[dict[str, object]],
    enabled: bool,
    step: int,
    token_ids: list[int],
    cache_position: int,
) -> None:
    if not enabled:
        return
    entry: dict[str, object] = {"step": step, "cache_position": cache_position}
    if len(token_ids) == 1:
        entry["token_id"] = token_ids[0]
    else:
        entry["token_ids"] = token_ids
    step_trace.append(entry)


def _prefill_once(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: Cache,
) -> tuple[torch.Tensor, float]:
    _sync_cuda()
    prefill_start = time.perf_counter()
    prefill_outputs = model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        past_key_values=cache,
        use_cache=True,
        return_dict=False,
    )
    last_hidden = prefill_outputs[0][:, -1:, :]
    logits = model.lm_head(last_hidden).float()
    first_token = _greedy_decode(logits)
    _sync_cuda()
    return first_token, time.perf_counter() - prefill_start


def _reset_graph_prompt_state(
    graph_cache: GraphCache,
    prompt_snapshot: list[tuple[torch.Tensor, torch.Tensor, int]],
    prompt_len: int,
    cache_position: torch.Tensor,
) -> None:
    graph_cache.restore(prompt_snapshot)
    graph_cache.set_seq_len(prompt_len)
    cache_position.fill_(prompt_len)


def _graph_decode_step(
    model: torch.nn.Module,
    cache: Cache,
    buffers: dict[str, torch.Tensor],
    use_triton_prepare_inputs: bool,
) -> None:
    if use_triton_prepare_inputs:
        prepare_decode_inputs_triton(
            buffers["cache_position"],
            buffers["attention_mask"],
            buffers["position_ids"],
            buffers["attention_mask_index"],
        )
    else:
        buffers["position_ids"].copy_(buffers["cache_position"].view(1, 1).expand_as(buffers["position_ids"]))
        attention_mask = (buffers["attention_mask_index"].view(1, -1) <= buffers["cache_position"]).to(
            dtype=buffers["attention_mask"].dtype
        )
        buffers["attention_mask"].copy_(attention_mask.expand_as(buffers["attention_mask"]))
    outputs = model(
        input_ids=buffers["input_ids"],
        attention_mask=buffers["attention_mask"],
        position_ids=buffers["position_ids"],
        past_key_values=cache,
        use_cache=True,
        return_dict=False,
    )
    buffers["logits"].copy_(outputs[0])
    buffers["next_token"].copy_(_greedy_decode(buffers["logits"]))
    buffers["input_ids"].copy_(buffers["next_token"])
    buffers["cache_position"].add_(1)


def _run_graph_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    dump_step_trace: bool = False,
    dump_token_ids: bool = False,
) -> dict:
    model_device = next(model.parameters()).device
    batch_size = int(input_ids.shape[0])
    prompt_len = int(input_ids.shape[1])
    max_cache_len = prompt_len + max_new_tokens + 8
    use_cuda_graph_decode = True
    graph_cache = GraphCache(num_layers=model.config.num_hidden_layers, max_cache_len=max_cache_len)
    graph_cache.set_triton_cache_write(True)
    _patch_rmsnorm_for_graph(model)
    _patch_mlp_elementwise_for_graph(model)
    _patch_attention_for_graph(model)
    _patch_residual_add_for_graph(model)

    with torch.inference_mode():
        first_token, prefill_seconds = _prefill_once(model=model, input_ids=input_ids, cache=graph_cache)
        prompt_snapshot = graph_cache.snapshot()
        graph_cache.set_static_mode(True)

        collect_tokens = dump_token_ids or dump_step_trace
        generated_ids: list[list[int]] | None = None
        if collect_tokens:
            generated_ids = [[int(token)] for token in first_token.detach().cpu().view(-1).tolist()]
        step_trace: list[dict[str, object]] = []
        if generated_ids is not None:
            _append_step_trace(
                step_trace,
                dump_step_trace,
                0,
                [tokens[0] for tokens in generated_ids],
                prompt_len,
            )

        graph_cache_position = graph_cache.share_cache_position()
        graph_buffers = {
            "input_ids": torch.empty((batch_size, 1), dtype=torch.long, device=model_device),
            "position_ids": torch.empty((batch_size, 1), dtype=torch.long, device=model_device),
            "attention_mask": torch.empty((batch_size, max_cache_len), dtype=torch.long, device=model_device),
            "attention_mask_index": torch.arange(max_cache_len, dtype=torch.long, device=model_device),
            "cache_position": graph_cache_position,
            "logits": torch.empty((batch_size, 1, model.config.vocab_size), dtype=torch.float32, device=model_device),
            "next_token": torch.empty((batch_size, 1), dtype=torch.long, device=model_device),
        }

        if use_cuda_graph_decode:
            for _ in range(GRAPH_WARMUP_STEPS):
                graph_buffers["input_ids"].copy_(first_token)
                graph_buffers["cache_position"].fill_(prompt_len)
                _graph_decode_step(model, graph_cache, graph_buffers, use_triton_prepare_inputs=True)
                _sync_cuda()
                _reset_graph_prompt_state(graph_cache, prompt_snapshot, prompt_len, graph_buffers["cache_position"])

        graph_buffers["input_ids"].copy_(first_token)
        graph_buffers["cache_position"].fill_(prompt_len)
        graph = None
        if use_cuda_graph_decode:
            graph = torch.cuda.CUDAGraph()
            _sync_cuda()
            with torch.cuda.graph(graph):
                _graph_decode_step(model, graph_cache, graph_buffers, use_triton_prepare_inputs=True)
            _sync_cuda()
            _reset_graph_prompt_state(graph_cache, prompt_snapshot, prompt_len, graph_buffers["cache_position"])
            graph_buffers["input_ids"].copy_(first_token)

        _sync_cuda()
        decode_start = time.perf_counter()
        for step in range(1, max_new_tokens):
            if graph is None:
                _graph_decode_step(model, graph_cache, graph_buffers, use_triton_prepare_inputs=False)
            else:
                graph.replay()
            if generated_ids is not None:
                step_token_ids = [int(token) for token in graph_buffers["next_token"].detach().cpu().view(-1).tolist()]
                for row_idx, token_id in enumerate(step_token_ids):
                    generated_ids[row_idx].append(token_id)
                _append_step_trace(step_trace, dump_step_trace, step, step_token_ids, prompt_len + step)
        _sync_cuda()
        decode_end = time.perf_counter()

    if generated_ids is not None and batch_size == 1:
        output_generated_ids: list[int] | list[list[int]] | None = generated_ids[0]
    else:
        output_generated_ids = generated_ids

    return {
        "batch_size": batch_size,
        "path": "triton_decode_graph",
        "generated_ids": output_generated_ids,
        "decode_tokens": batch_size * max(max_new_tokens - 1, 0),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_end - decode_start,
        "step_trace": step_trace,
    }


def _build_result(run_result: dict, include_token_ids: bool, include_step_trace: bool) -> dict:
    decode_seconds = run_result["decode_seconds"]
    decode_tokens = run_result["decode_tokens"]
    result = {
        "batch_size": run_result["batch_size"],
        "path": run_result["path"],
        "ttft_ms": run_result["prefill_seconds"] * 1000.0,
        "tps": decode_tokens / decode_seconds if decode_seconds > 0 else 0.0,
        "decode_tokens": decode_tokens,
    }
    if include_token_ids:
        result["generated_token_ids"] = run_result["generated_ids"]
    if include_step_trace:
        result["step_trace"] = run_result["step_trace"]
    return result


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    batch_sizes = _parse_batch_sizes(args)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    _initialize_packed_moe(model)
    model.to(args.device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    results = []
    for batch_size in batch_sizes:
        batch_input_ids = input_ids.expand(batch_size, -1).contiguous()
        run_result = _run_graph_decode(
            model=model,
            input_ids=batch_input_ids,
            max_new_tokens=args.max_new_tokens,
            dump_step_trace=args.dump_step_trace,
            dump_token_ids=args.dump_token_ids,
        )
        results.append(_build_result(run_result, args.dump_token_ids, args.dump_step_trace))

    hardware = collect_hardware_info(next(model.parameters()).device)
    if len(results) == 1:
        result = results[0]
        result["hardware"] = hardware
    else:
        result = {"results": results, "hardware": hardware}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
