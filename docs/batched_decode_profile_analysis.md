# Batched Decode Profiling Notes

This note records the A100 profiling state for the current unified
`triton_decode_graph` path and the current explanation for two observations:

- Batch scaling from `bsz=128` to `bsz=256` is lower than the small-batch growth.
- `bsz=1` is slower than the earlier single-token-only implementation, even
  though the current path is functionally more general.

The intended direction is not to restore a separate `bsz=1` fast path. The
target is still one batched implementation that serves all batch sizes. The
important distinction is that one implementation does not require one fixed
tile shape or one fixed route-grouping policy for every batch size.

## Profile Setup

- GPU: NVIDIA A100 80GB PCIe GPU3
- Current path: `src/sota.py` `triton_decode_graph`
- Current profiles:
  - `nsys-reps/sota_bsz64_a100_node.nsys-rep`
  - `nsys-reps/sota_bsz128_a100_node.nsys-rep`
  - `nsys-reps/sota_bsz256_a100_node.nsys-rep`
  - `nsys-reps/sota_bsz1_unified_a100_node.nsys-rep`
- Historical comparison profile:
  - `nsys-reps/sota_bsz1_fc39f1f_a100_node.nsys-rep`
- NCU reports:
  - `ncu-reps/grouped_gate_up_bsz256_metrics.ncu-rep`
  - `ncu-reps/grouped_down_bsz256_metrics.ncu-rep`
  - `ncu-reps/attention_decode_bsz256_metrics.ncu-rep`
  - `ncu-reps/batched_linear_bsz256_metrics.ncu-rep`

## Batch Scaling At 128 And 256

The reduced relative TPS growth at `bsz=128` and `bsz=256` is expected for the
current kernels. The main kernels no longer look launch-bound at these sizes;
they scale close to batch size in GPU time and are already dominated by memory
traffic.

`nsys` kernel-time summary:

| Kernel | bsz=64 | bsz=128 | bsz=256 | 64->128 | 128->256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `_grouped_gate_up_swiglu_kernel` | 351.397 ms | 587.349 ms | 1109.756 ms | 1.67x | 1.89x |
| `_grouped_down_partial_kernel` | 199.941 ms | 364.313 ms | 684.170 ms | 1.82x | 1.88x |
| `_batched_linear_kernel` | 260.076 ms | 411.094 ms | 756.798 ms | 1.58x | 1.84x |
| `_batched_decode_attention_q1_kernel` | 144.122 ms | 240.867 ms | 439.926 ms | 1.67x | 1.83x |
| Total GPU kernel time | 1356.719 ms | 2085.852 ms | 3661.934 ms | 1.54x | 1.76x |

At `bsz=256`, NCU shows the major kernels are memory-throughput limited:

| Kernel | DRAM throughput | Compute-memory throughput | SM throughput |
| --- | ---: | ---: | ---: |
| `_grouped_gate_up_swiglu_kernel` | 100% | 100% | 34.78% |
| `_grouped_down_partial_kernel` | 100% | 100% | 33.51% |
| `_batched_decode_attention_q1_kernel` | 100% | 100% | 40.65% |
| `_batched_linear_kernel` | 100% | 100% | about 44% |

Conclusion: for the current implementation, the 128/256 growth-rate reduction
is not primarily a scheduling or launch issue. The dominant kernels are moving
large amounts of activation and weight data, and the A100 profile is consistent
with a memory-bandwidth ceiling.

## bsz=1 Regression

The `bsz=1` slowdown is not explained by Python overhead or graph launch
overhead. It is visible inside CUDA node/kernel time.

`nsys` comparison:

| Path | Total GPU kernel time | Decode TPS in profile |
| --- | ---: | ---: |
| Current unified batched path | 748.914 ms | 141.35 tok/s |
| Historical `fc39f1f` single-token path | 612.449 ms | 192.33 tok/s |

Largest differences:

| Area | Current unified batched path | Historical path |
| --- | ---: | ---: |
| Attention linear projections | `_batched_linear_kernel`: 172.376 ms | `_gemv_contig_kernel` + `_gemv_2048x2048_o_kernel`: 81.348 ms |
| MoE gate/up + down | `_grouped_gate_up_swiglu_kernel` + `_grouped_down_partial_kernel`: 253.107 ms | `_fused_gate_up_swiglu_kernel` + `_down_partial_kernel`: 159.043 ms |
| Decode attention | `_batched_decode_attention_q1_kernel`: 35.353 ms | `_decode_attention_q1_kernel`: 22.880 ms |
| RMSNorm and router | effectively unchanged | effectively unchanged |

The current regression is therefore mostly from attention projection kernels and
MoE expert kernels, not from the unchanged small kernels.

## Why A Fixed Batched Tile Hurts bsz=1

The current implementation is a unified code path, but several low-level choices
are tuned for larger batches and are expensive when the active batch dimension
is one.

Attention linears currently use `_batched_linear_kernel` for q/kv-a, kv-b, and
o-proj with `BLOCK_B=8`, `BLOCK_O=64`, and `BLOCK_K=128`. At `bsz=1`, that
kernel still uses an 8-row batch tile and masks out seven inactive rows. The
operation is functionally correct, but the tile shape, accumulator layout, and
memory access pattern are not the shape that a one-row GEMV-like workload wants.
The old single-token kernels used narrower row semantics and larger K tiles for
the vector case. This is a tile/shape problem inside the batched kernel, not a
need to return to a separate public `bsz=1` path.

MoE changed more substantially. The current batched MoE first constructs grouped
route metadata (`counts`, `route_indices`, `block_count`, `block_experts`,
`block_offsets`) and then runs grouped expert kernels. That is useful when many
routes can be grouped by expert, but at `bsz=1` there are only six routes. The
metadata and grouped-block machinery have almost no reuse to amortize. The
default tile for small batches is also `BLOCK_N=4`, `BLOCK_M=32`,
`BLOCK_K=64`, while the older single-token kernels effectively used route-local
GEMV work with a larger K tile. This is both an algorithmic decomposition issue
and a tile-parameter issue.

Decode attention is a smaller contributor. The current batched decode attention
and QKV/cache kernels carry batch strides and masks, and use shapes intended to
handle multiple rows. At `bsz=1` the overhead is measurable, but it is not the
dominant regression compared with attention linears and MoE.

## Direction Without Restoring bsz=1 Specialization

The goal should remain one batched implementation. However, one implementation
should be shape-adaptive:

- Keep the same public `attention_decode_triton(...)` and
  `packed_routed_moe(...)` APIs for every batch size.
- Keep kernels written as batched kernels over `[B, ...]`, not as a resurrected
  single-token-only path.
- Make batch tile shape a compile-time/autotuned parameter. For example,
  attention linears should be able to compile the same `_batched_linear_kernel`
  source with `BLOCK_B=1` when `B` is small, rather than always using
  `BLOCK_B=8`.
- Retune K tiles for small batch. The current `BLOCK_K=128` for attention
  linears and `BLOCK_K=64` for small-batch MoE are likely too small for the
  one-row case; larger K tiles reduce loop count and better match GEMV-like
  reuse.
- Rework MoE so the batched implementation has a low-route-count mode that does
  not pay full route-group metadata cost when `B * topk` is tiny. This should be
  implemented as a general batched route-major or adaptive grouped kernel, not
  as `if bsz == 1: call the old single-token kernels`.
- Remove avoidable intermediate copies in the unified attention path where a
  producer can write directly into the workspace consumed by the next batched
  kernel.

In short, the problem is not that batching support itself is incompatible with
good `bsz=1` performance. The problem is that the current batched kernels use
large-batch-oriented tiling and grouped-route machinery even when the batch
dimension degenerates to one. Matching the old `bsz=1` speed while keeping one
batched implementation requires shape-adaptive batched kernels, not a separate
single-token specialization path.

## Small-Batch Optimization Plan

The next optimization target is small-batch decode, not just `bsz=1`. The same
large-batch assumptions affect `bsz=2`, `bsz=4`, `bsz=8`, and in some kernels
possibly `bsz=16`. The implementation should therefore use batch and route
regimes instead of a `batch_size == 1` special case.

### 1. Shape-Adaptive Attention Linear Tiles

Keep `_batched_linear_kernel` as the shared batched linear kernel, but stop
launching every projection with `BLOCK_B=8`, `BLOCK_O=64`, `BLOCK_K=128`.
Introduce a tile selector used by q/kv-a, kv-b, and o-proj:

| Batch regime | Initial tile candidates |
| ---: | --- |
| `B <= 2` | `BLOCK_B=1`, `BLOCK_O=64`, `BLOCK_K=256` |
| `B <= 4` | `BLOCK_B=2`, `BLOCK_O=64`, `BLOCK_K=256` |
| `B <= 8` | `BLOCK_B=4`, `BLOCK_O=64`, `BLOCK_K=128 or 256` |
| `B <= 32` | `BLOCK_B=8 or 16`, `BLOCK_O=64`, `BLOCK_K=128` |
| `B >= 64` | current large-batch tile unless profiling says otherwise |

This keeps one batched source kernel and lets Triton compile shape-appropriate
variants through constexpr meta-parameters.

### 2. Low-Route Batched MoE Regime

MoE should choose its execution regime by route count:

```text
total_routes = batch_size * topk
```

For small `total_routes`, the current expert-grouped path pays for route
metadata construction before there is enough expert reuse to amortize it. Add a
low-route batched implementation that is still shaped as `[B, H]` plus
`[B, topk]`, but computes directly over routes instead of first constructing
full grouped expert blocks.

Initial policy:

| Route regime | Initial policy |
| ---: | --- |
| `total_routes <= 32` | route-major batched MoE, no full grouped-block metadata |
| `32 < total_routes <= 64` | benchmark route-major versus grouped |
| `total_routes > 64` | current grouped expert path |

The low-route path must remain a batched kernel and must support any `B` in the
regime. It should not call the old single-token-only kernels.

### 3. Retune MoE Tiles Per Regime

The current default small-batch tile is `(BLOCK_N=4, BLOCK_M=32, BLOCK_K=64)`.
For low-route and mid-route batches, test at least:

| Parameter | Candidates |
| --- | --- |
| `BLOCK_N` | `1`, `2`, `4`, `8` |
| `BLOCK_M` | `32`, `64` |
| `BLOCK_K` | `128`, `256` |

The first priority is increasing `BLOCK_K` for the small-route case, because the
current `BLOCK_K=64` increases loop count on GEMV-like work.

### 4. Secondary Attention Cleanup

After attention linear and MoE changes, revisit smaller attention overheads:

- benchmark `BLOCK_V=64` versus `128` for small-batch decode attention;
- remove avoidable workspace copies when a producer can write directly to the
  next consumer's workspace;
- keep batch strides and masks in the batched kernels, but avoid tile shapes
  that create mostly inactive lanes.

### Validation Matrix

Use the existing A100 sweep setup and compare both TPS and node-level kernel
time:

| Batch | Purpose |
| ---: | --- |
| `1`, `2`, `4`, `8` | primary small-batch regression target |
| `16`, `32` | transition point between low-route and grouped regimes |
| `64`, `128`, `256` | guard against large-batch regressions |

Required checks:

- numerical correctness versus baseline for touched kernels;
- no public API split between single-token and batched decode;
- `git diff --check`;
- A100 TPS sweep before and after;
- `nsys` node-level profiles for representative small and large batches.

## 2026-04-25 Small-Batch Implementation Result

The first small-batch pass kept the unified batched decode API and did not
restore the old single-token decode path.

Enabled changes:

- `_batched_linear_kernel` now uses a shape-adaptive launch tile for `B=1`:
  `BLOCK_B=1`, `BLOCK_O=64`, `BLOCK_K=256`. For `B>=2`, it keeps the previous
  `BLOCK_B=8`, `BLOCK_O=64`, `BLOCK_K=128` tile because microbenchmarks showed
  smaller `BLOCK_B` values were not consistently faster for `B=2/4/8`.
- The grouped MoE autotune candidate set now includes small-batch-friendly
  tiles such as `(2, 32, 128)`, `(4, 32, 256)`, `(8, 64, 128)`, and
  `(32, 64, 128)`. The graph warmup path can select these before CUDA graph
  capture.
- The default grouped MoE tile fallback was updated by batch regime so graph
  capture has a better no-autotune fallback.

Evaluated but not enabled:

- The existing route-major batched MoE path was repaired to load packed gate and
  up weights correctly and to use bf16 `routed_hidden`, but end-to-end decode
  was slower for `B=2/4/8`. It remains disabled with
  `MOE_ROUTE_GEMV_MAX_ROUTES = 0`.
- Direct MoE microbenchmarks showed route-major can beat grouped kernels for a
  single isolated MoE call at `B<=8`, but that did not transfer to full decode
  under the current graph/warmup/workspace behavior. The active path therefore
  keeps the grouped batched MoE kernels.

A100 GPU3, `max_new_tokens=100`, input length 24:

| Batch | Previous TPS | New TPS | Change |
| ---: | ---: | ---: | ---: |
| 1 | 144.78 | 159.59 | +10.2% |
| 2 | 266.97 | 276.67 | +3.6% |
| 4 | 514.33 | 537.08 | +4.4% |
| 8 | 951.82 | 974.91 | +2.4% |
| 16 | 1920.58 | 1957.76 | +1.9% |
| 32 | 3624.22 | 3622.55 | -0.0% |
| 64 | 5746.40 | 5742.33 | -0.1% |
| 128 | 7902.84 | 7902.50 | -0.0% |
| 256 | 9395.99 | 9340.77 | -0.6% |

The current result improves the targeted small-batch range without materially
changing large-batch throughput. The remaining main opportunity is still MoE:
the route-major idea needs either a fused/reduced workspace form or a different
graph-captured allocation strategy before it should replace grouped kernels for
small batches.

## Next Direction: Dedicated Small-Batch Kernel Family

The latest `bsz=1` `nsys` comparison still shows the current batching kernels do
not fully match the earlier GEMV-oriented implementation:

| Area | Historical `fc39f1f` | Current batching path | Delta |
| --- | ---: | ---: | ---: |
| Attention projections | `_gemv_contig` + `_gemv_2048x2048_o`: 81.35 ms | `_batched_linear_kernel`: 139.77 ms | +58.43 ms |
| MoE main work and routing | route-local gate/up, down, reduce: 164.26 ms | grouped gate/up, grouped down, reduce, route metadata: 297.62 ms | +133.36 ms |
| Decode attention | `_decode_attention_q1_kernel`: 22.88 ms | `_batched_decode_attention_q1_kernel`: 35.38 ms | +12.50 ms |
| Total GPU kernel time | 612.45 ms | 737.17 ms | +124.72 ms |

This supports a clearer design split: keep a common front-end and common batched
semantics, but add a small-batch kernel family for `B <= 8`. The front-end can
select the kernel family from the known static batch size before CUDA graph
capture.

Planned split:

- `B <= 8`: small-batch kernels that are still batched over `[B, ...]`, but use
  GEMV-like projection tiles and route-major MoE kernels that avoid full grouped
  route metadata.
- `B > 8`: current grouped batched kernels with shape-keyed autotuning.

The small-batch family should not be a `bsz=1`-only path. It should support
`B=1/2/4/8` through the same API so the scheduler only chooses between
small-batch and grouped-batch kernel families.
