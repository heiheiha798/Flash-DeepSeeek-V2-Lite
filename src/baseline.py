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
        return self.keys.clone(), self.values.clone(), self.seq_len

    def restore(self, snapshot: tuple[torch.Tensor, torch.Tensor, int]) -> None:
        keys, values, seq_len = snapshot
        self.keys.copy_(keys)
        self.values.copy_(values)
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

    def snapshot(self) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        return [layer.snapshot() for layer in self.layers]

    def restore(self, snapshot: list[tuple[torch.Tensor, torch.Tensor, int]]) -> None:
        for layer, layer_snapshot in zip(self.layers, snapshot):
            layer.restore(layer_snapshot)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF baseline decode-only CUDA graph benchmark.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dump-token-ids", action="store_true")
    parser.add_argument("--dump-step-trace", action="store_true")
    return parser.parse_args()


def _greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def _initialize_packed_moe(model: torch.nn.Module) -> None:
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
        mlp._graph_moe_patched = True

def _sync_cuda() -> None:
    torch.cuda.synchronize()


def _append_step_trace(
    step_trace: list[dict[str, int]],
    enabled: bool,
    step: int,
    token_id: int,
    cache_seen_tokens: int,
) -> None:
    if not enabled:
        return
    step_trace.append(
        {
            "step": step,
            "token_id": token_id,
            "cache_seen_tokens": cache_seen_tokens,
        }
    )


def _prefill_once(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: Cache,
) -> tuple[torch.Tensor, float]:
    _sync_cuda()
    prefill_start = time.perf_counter()
    prefill_outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        past_key_values=cache,
        use_cache=True,
        return_dict=False,
    )
    first_token = _greedy_decode(prefill_outputs[0])
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
) -> None:
    buffers["position_ids"].copy_(buffers["cache_position"].view(1, 1))
    buffers["attention_mask"].copy_(
        (buffers["attention_mask_index"] <= buffers["cache_position"])
        .to(dtype=buffers["attention_mask"].dtype)
        .view(1, -1)
    )
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
) -> dict:
    model_device = next(model.parameters()).device
    prompt_len = int(input_ids.shape[1])
    max_cache_len = prompt_len + max_new_tokens + 8
    graph_cache = GraphCache(num_layers=model.config.num_hidden_layers, max_cache_len=max_cache_len)

    with torch.inference_mode():
        first_token, prefill_seconds = _prefill_once(model=model, input_ids=input_ids, cache=graph_cache)
        prompt_snapshot = graph_cache.snapshot()
        graph_cache.set_static_mode(True)

        generated_ids = [int(first_token.item())]
        step_trace: list[dict[str, int]] = []
        _append_step_trace(step_trace, dump_step_trace, 0, generated_ids[-1], int(graph_cache.seen_tokens))

        graph_cache_position = graph_cache.share_cache_position()
        graph_buffers = {
            "input_ids": torch.empty((1, 1), dtype=torch.long, device=model_device),
            "position_ids": torch.empty((1, 1), dtype=torch.long, device=model_device),
            "attention_mask": torch.empty((1, max_cache_len), dtype=torch.long, device=model_device),
            "attention_mask_index": torch.arange(max_cache_len, dtype=torch.long, device=model_device),
            "cache_position": graph_cache_position,
            "logits": torch.empty((1, 1, model.config.vocab_size), dtype=torch.float32, device=model_device),
            "next_token": torch.empty((1, 1), dtype=torch.long, device=model_device),
        }

        for _ in range(GRAPH_WARMUP_STEPS):
            graph_buffers["input_ids"].copy_(first_token)
            graph_buffers["cache_position"].fill_(prompt_len)
            _graph_decode_step(model, graph_cache, graph_buffers)
            _sync_cuda()
            _reset_graph_prompt_state(graph_cache, prompt_snapshot, prompt_len, graph_buffers["cache_position"])

        graph_buffers["input_ids"].copy_(first_token)
        graph_buffers["cache_position"].fill_(prompt_len)
        graph = torch.cuda.CUDAGraph()
        _sync_cuda()
        with torch.cuda.graph(graph):
            _graph_decode_step(model, graph_cache, graph_buffers)
        _sync_cuda()
        _reset_graph_prompt_state(graph_cache, prompt_snapshot, prompt_len, graph_buffers["cache_position"])
        graph_buffers["input_ids"].copy_(first_token)

        _sync_cuda()
        decode_start = time.perf_counter()
        while len(generated_ids) < max_new_tokens:
            graph.replay()
            next_token = graph_buffers["next_token"]
            generated_ids.append(int(next_token.item()))
            _append_step_trace(
                step_trace,
                dump_step_trace,
                len(generated_ids) - 1,
                generated_ids[-1],
                int(graph_cache.seen_tokens),
            )
        _sync_cuda()
        decode_end = time.perf_counter()

    return {
        "generated_ids": generated_ids,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_end - decode_start,
        "step_trace": step_trace,
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

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

    run_result = _run_graph_decode(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        dump_step_trace=args.dump_step_trace,
    )

    decode_tokens = max(len(run_result["generated_ids"]) - 1, 0)
    decode_seconds = run_result["decode_seconds"]
    result = {
        "ttft_ms": run_result["prefill_seconds"] * 1000.0,
        "tps": decode_tokens / decode_seconds if decode_seconds > 0 else 0.0,
        "hardware": collect_hardware_info(next(model.parameters()).device),
    }
    if args.dump_token_ids:
        result["generated_token_ids"] = run_result["generated_ids"]
    if args.dump_step_trace:
        result["step_trace"] = run_result["step_trace"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
