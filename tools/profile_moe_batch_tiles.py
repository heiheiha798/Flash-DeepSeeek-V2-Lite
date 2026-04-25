from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_kernels.moe_batch import PackedRoutedExperts, _run_batched_grouped_routed_moe_grouped_triton


TILES: tuple[tuple[int, int, int], ...] = (
    (8, 32, 64),
    (8, 32, 128),
    (16, 16, 64),
    (16, 32, 64),
    (16, 32, 128),
    (16, 64, 128),
    (32, 16, 64),
    (32, 32, 64),
    (32, 32, 128),
    (32, 64, 128),
)


def _parse_tile(value: str) -> tuple[int, int, int]:
    parts = value.replace(",", " ").replace("x", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("tile must be N,M,K")
    tile = tuple(int(part) for part in parts)
    if any(part <= 0 for part in tile):
        raise argparse.ArgumentTypeError("tile values must be positive")
    return tile  # type: ignore[return-value]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro-benchmark batch grouped MoE tile choices.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=1408)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--gate-tile", type=_parse_tile, default=None)
    parser.add_argument("--down-tile", type=_parse_tile, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-counts", action="store_true")
    return parser.parse_args()


def _make_topk_ids(batch_size: int, topk: int, num_experts: int, device: str) -> torch.Tensor:
    route_idx = torch.arange(batch_size * topk, device=device, dtype=torch.long)
    # Deterministic balanced-ish routing with unique experts per token.
    token_idx = route_idx // topk
    slot_idx = route_idx - token_idx * topk
    return ((token_idx * 13 + slot_idx * 7) % num_experts).view(batch_size, topk).contiguous()


def _time_case(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    packed: PackedRoutedExperts,
    gate_tile: tuple[int, int, int],
    down_tile: tuple[int, int, int],
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        _run_batched_grouped_routed_moe_grouped_triton(
            x,
            topk_ids,
            topk_weight,
            packed,
            x.dtype,
            gate_tile,
            down_tile,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _run_batched_grouped_routed_moe_grouped_triton(
            x,
            topk_ids,
            topk_weight,
            packed,
            x.dtype,
            gate_tile,
            down_tile,
        )
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16
    x = torch.randn((args.batch_size, args.hidden_size), device=device, dtype=dtype)
    topk_ids = _make_topk_ids(args.batch_size, args.topk, args.num_experts, device)
    topk_weight = torch.randn((args.batch_size, args.topk), device=device, dtype=torch.float32).softmax(dim=-1)
    gate_up_weights = torch.randn(
        (args.num_experts, 2 * args.intermediate_size, args.hidden_size),
        device=device,
        dtype=dtype,
    )
    down_weights = torch.randn(
        (args.num_experts, args.hidden_size, args.intermediate_size),
        device=device,
        dtype=dtype,
    )
    packed = PackedRoutedExperts(
        gate_up_weights=gate_up_weights,
        down_weights=down_weights,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        topk=args.topk,
    )

    counts = torch.bincount(topk_ids.view(-1).cpu(), minlength=args.num_experts)
    if args.dump_counts:
        print(json.dumps({"counts": counts.tolist()}, indent=2))

    if args.sweep:
        results = []
        for gate_tile in TILES:
            for down_tile in TILES:
                ms = _time_case(
                    x,
                    topk_ids,
                    topk_weight,
                    packed,
                    gate_tile,
                    down_tile,
                    args.warmup,
                    args.iters,
                )
                results.append({"gate_tile": gate_tile, "down_tile": down_tile, "ms": ms})
                print(json.dumps(results[-1]))
        best = min(results, key=lambda item: item["ms"])
        print(json.dumps({"best": best}, indent=2))
        return

    gate_tile = args.gate_tile or (32, 64, 128)
    down_tile = args.down_tile or (32, 64, 128)
    ms = _time_case(
        x,
        topk_ids,
        topk_weight,
        packed,
        gate_tile,
        down_tile,
        args.warmup,
        args.iters,
    )
    print(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "gate_tile": gate_tile,
                "down_tile": down_tile,
                "ms": ms,
                "count_min": int(counts.min()),
                "count_max": int(counts.max()),
                "count_mean": float(counts.float().mean()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
