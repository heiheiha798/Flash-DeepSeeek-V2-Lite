# Triton Kernel Optimizations

本文档记录当前 repo 中 DeepSeek-V2-Lite Triton 推理优化的有效改动、实测结果和剩余优化空间。

当前项目定位：

- 主线是 `DeepSeek-V2-Lite-Chat` 的 Triton decode 推理优化
- 优化目标是 `A100 / SM80 / bf16 / q_len=1`
- 当前入口是 `src/run.py --kernel-family small|batch`

## 当前文件结构

Triton kernels：

- `triton_kernels/rmsnorm.py`：RMSNorm
- `triton_kernels/moe_router.py`：router softmax + topk6
- `triton_kernels/moe_small_gemv.py`：small-GEMV routed MoE gate/up/down/reduce
- `triton_kernels/moe_batch.py`：grouped batching routed MoE gate/up/down/reduce
- `triton_kernels/mlp_elementwise.py`：MLP `silu * up`
- `triton_kernels/attention_prepost.py`：attention 前后的小操作、cache write、residual add
- `triton_kernels/attention_decode_small.py`：small-GEMV attention decode 路径
- `triton_kernels/attention_decode_batch.py`：grouped batching attention decode 路径

Profile / test scripts：

- `tools/profile_moe_ncu.py`
- `tools/profile_attention_o_ncu.py`
- `tools/profile_attention_qkv_ncu.py`
- `tools/profile_attention_kvb_ncu.py`
- `tools/test_moe_grouped_gemv.py`

当前保留两套 routed MoE 实验实现：small-GEMV family 和 grouped batching
family。它们由 `src/run.py --kernel-family` 显式选择，不再通过环境变量
或 batch-size 阈值自动切换。

## 实验配置

- 模型：`/data/models/DeepSeek-V2-Lite-Chat`
- 精度：`bf16`
- 设备：`NVIDIA A100 80GB PCIe / SM80`
- CPU：`INTEL(R) XEON(R) PLATINUM 8558P`, `96C/192T`, `503.53 GiB RAM`
- 软件：`Python 3.10.20`, `torch 2.10.0+cu130`, `torch CUDA 13.0`, `triton 3.6.0`, `transformers 4.57.6`
- GPU：优先 `GPU3`
- 场景：`decode-only`, `batch=1`, `q_len=1`, `max_new_tokens=100`
- 环境：`/data/home/tianjianyang/.conda/envs/flashmla`

端到端命令：

```bash
CUDA_VISIBLE_DEVICES=3 /data/home/tianjianyang/.conda/envs/flashmla/bin/python src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100
```

## 当前快照

最新非 profile 结果：

- `tps = 195.02 ~ 197.61`
- 当前稳定档位约 `195+ TPS`
- 报告 TPS 时必须同时记录硬件和软件上下文，格式见 `docs/performance_reporting.md`

最新 `nsys` decode graph 结果：

- report：`nsys-reps/sota_gpu3_gate_combined_node.nsys-rep`
- sqlite：`nsys-reps/sota_gpu3_gate_combined_node.sqlite`
- 每步 decode graph kernel 数：`651`
- 每步 decode graph GPU 时间：`5.145 ms/step`

最新 `ncu` 关键结果：

- baseline：`ncu-reps/moe_down_twostage_baseline.ncu-rep`
- combined gate/up dot：`ncu-reps/moe_gate_combined_dot.ncu-rep`

`_fused_gate_up_swiglu_kernel`：

- `gpu__time_duration.sum`: `60.736 -> 46.464`，下降 `23.5%`
- `launch__registers_per_thread`: `142 -> 144`
- `launch__occupancy_limit_registers`: `6 -> 6`
- `sm__throughput`: `13.26% -> 17.24%`
- `smsp__issue_active`: `17.72% -> 23.93%`
- `sass__inst_executed_register_spilling`: `0 -> 0`

## 当前 MoE 路径

`triton_kernels/moe_small_gemv.py` 中 small-GEMV routed MoE decode 路径由三个 kernel 组成。

### 1. `_fused_gate_up_swiglu_kernel`

输入：

- 单 token hidden
- 当前 token 的 routed expert ids
- packed `gate/up` 权重

计算：

- `gate_proj(x)`
- `up_proj(x)`
- `silu(gate) * up`

当前关键优化：

- 将 gate/up 权重按 `[expert, 2 * intermediate, hidden]` 打包
- 每个 program 处理一个 expert slot 的一个 row tile
- 一次 `tl.load` 读取 `[2 * BLOCK_M, BLOCK_K]` 的 contiguous gate/up block
- 一次 `tl.dot` 得到 `[2, BLOCK_M]`
- 再拆成 gate/up accumulator

这个优化比旧的“两次 load + 两次 dot”明显更好，且没有造成寄存器爆炸。

### 2. `_down_partial_kernel`

输入：

- routed hidden
- routed expert ids
- packed `down` 权重

计算：

- 每个 topk slot 独立执行 `down_proj`

输出：

- `partial[topk, hidden]`

当前保留 two-stage 形式，是因为单 kernel 串行 topk reduce 会破坏并行度，端到端明显变慢。

### 3. `_down_reduce_topk6_kernel`

输入：

- `partial[topk, hidden]`
- route weights

计算：

- 固定 `topk=6` 的 weighted reduce

输出：

- 最终 bf16 routed MoE output

当前优势：

- 避免 atomic add
- 避免 `out.zero_()`
- 避免单独 fp32 -> bf16 cast kernel

## 已生效优化

### 1. Router Softmax/TopK Fusion

文件：

- `triton_kernels/moe_router.py`

内容：

- 将 router `softmax + topk6 + routed_scaling` 合并到 `_router_softmax_topk6_kernel`
- 在 runner 中缓存 router fp32 weight，避免 graph 内重复 `self.weight.float()`

历史收益：

- router fusion 后：`755 -> 703 kernels/step`
- router fp32 weight cache 后：`703 -> 677 kernels/step`
- graph 时间：`5.867 -> 5.725 -> 5.559 ms/step`
- 端到端：约 `173.5 -> 177.7 -> 182.5 TPS`

### 2. MLP SiLU*Mul Fusion

文件：

- `triton_kernels/mlp_elementwise.py`

内容：

- 将 `silu(gate_proj(x)) * up_proj(x)` 的 elementwise 部分替换为 `_silu_mul_kernel`
- 不替换主 GEMM/GEMV

历史收益：

- `782 -> 755 kernels/step`
- `5.934 -> 5.867 ms/step`
- 端到端：约 `171.5 -> 173.5 TPS`

### 3. RMSNorm Triton 化

文件：

- `triton_kernels/rmsnorm.py`

内容：

- 单 kernel 两 pass：先算 `sum(x^2)`，再 normalize + scale
- 替换 HF 中拆开的 `pow / mean / rsqrt / mul`

历史收益：

- elementwise/reduce kernel 显著下降
- decode graph 内 torch reduce 基本被吸收

当前状态：

- pass2 已改成代码层面单次 input load。
- `ncu` 显示 global load 指令和 sectors 未变，说明编译器原本基本能消除重复 load。
- 该改动主要保留为代码清理，不作为显著性能优化。

### 4. Attention Pre/Post Triton 化

文件：

- `triton_kernels/attention_prepost.py`

内容：

- decode attention mask / position ids 准备
- q_len=1 cache write
- residual add

收益：

- 将部分 graph 外或 graph 内 torch 小 kernel 收进 Triton
- 降低 graph shell 侧噪声

### 5. Attention Decode Triton Path

文件：

- `triton_kernels/attention_decode_small.py`
- `triton_kernels/attention_decode_batch.py`

内容：

- q 与 kv_a projection packing
- kv_a RMSNorm
- kv_b projection
- RoPE + q/k/v materialization
- KV cache write
- q_len=1 attention core
- o_proj GEMV

已做过的有效 micro 优化：

- `_gemv_2048x2048_o_kernel` 用 `block_ptr + tl.advance` 降低地址计算和寄存器压力
- q_kv_a GEMV 调整为更适合当前 shape 的 block 配置

历史 `ncu` 结果：

- `o_proj` 旧实现：`regs/thread = 136`, `time = 18.5013`
- `block_ptr + advance` 后：`regs/thread = 70`, `time = 12.3893`
- 当前候选：`regs/thread = 70`, `time = 11.4560`

### 6. Two-stage Down Reduce

旧路径：

- `out.zero_()`
- `_fused_down_weight_atomic_kernel`
- fp32 output 再 cast 到 bf16

当前路径：

- `_down_partial_kernel`
- `_down_reduce_topk6_kernel`

收益：

- 去掉 `out.zero_()`
- 去掉 `atomic_add`
- 去掉单独 cast kernel
- 每步 decode graph kernel 数从 `677` 降到 `651`
- 每步 graph 时间从 `5.559 ms` 降到 `5.521 ms`
- 端到端从约 `182.5 TPS` 到约 `183.8 TPS`

### 7. Combined Gate/Up Dot

旧 `_fused_gate_up_swiglu_kernel`：

- 分别 load gate weight block 和 up weight block
- 分别执行两次 `tl.dot`

当前 `_fused_gate_up_swiglu_kernel`：

- 一次 load `[2 * BLOCK_M, BLOCK_K]` 的 contiguous gate/up weight block
- 一次 `tl.dot` 得到 `[2, BLOCK_M]`
- 再拆成 gate/up 做 `silu(gate) * up`

结果：

- kernel 时间下降 `23.5%`
- registers/thread 只从 `142` 到 `144`
- 没有 spilling
- 端到端从约 `183.8 ~ 183.9 TPS` 到 `195.46 ~ 197.61 TPS`

结论：

- 这是当前最有效的 MoE kernel 内部优化之一
- 后续类似优化应优先找“减少 load/dot 次数但不显著增加寄存器”的点

## 当前热点

基于 `nsys-reps/sota_gpu3_gate_combined_node.sqlite`：

- 每步 decode graph kernel 数：`651`
- 每步 graph GPU 时间：`5.145 ms/step`

主要热点：

- `cuBLAS bf16 GEMM 64x64`：`53/step`, `0.916 ms/step`
- `MoE gate/up/SwiGLU Triton`：`26/step`, `0.910 ms/step`
- `MoE down partial Triton`：`26/step`, `0.652 ms/step`
- `Triton GEMV generic`：`54/step`, `0.564 ms/step`
- `cuBLAS GEMV`：`26/step`, `0.379 ms/step`
- `Triton RMSNorm`：`82/step`, `0.343 ms/step`
- `Triton o_proj GEMV`：`27/step`, `0.238 ms/step`
- `Triton attention core`：`27/step`, `0.226 ms/step`
- `Torch elementwise/copy`：`84/step`, `0.208 ms/step`

## 最新收口 profile

GPU0 node-level profile：

- report：`nsys-reps/sota_gpu0_final_node.nsys-rep`
- sqlite：`nsys-reps/sota_gpu0_final_node.sqlite`
- 每步 decode graph kernel 数：`651`
- GPU0 profile 下平均 graph wall time：`6.25 ms/step`
- kernel duration sum：`6.21 ms/step`
- graph 内 gap：约 `0.037 ms/step`

GPU0 比 GPU3 慢，但 kernel 结构一致。这个 profile 用于确认当前优化空间已经主要集中在 MoE、GEMV/GEMM 和少量 torch elementwise。

## 无效或撤回的尝试

### Gate/Up accumulator split

尝试内容：

- 将 `_fused_gate_up_swiglu_kernel` 末尾的 `tl.where + tl.sum` 拆分改为 `tl.split(tl.trans(gate_up_acc))`。

结果：

- report：`ncu-reps/moe_gate_split_candidate.ncu-rep`
- baseline：`ncu-reps/moe_gate_split_baseline.ncu-rep`
- duration：`45.152 us -> 45.856 us`，变慢 `1.56%`
- registers/thread：`144 -> 144`
- global load/store 不变
- issue active 和 SM throughput 略降

结论：

- 指令数略降，但实际 kernel 时间变差。
- 不保留。

### Down partial block_ptr + tl.dot

尝试内容：

- 将 `_down_partial_kernel` 的 `tl.sum(down_w * hidden)` 改为 `block_ptr + tl.dot(down_w, hidden[:, None])`。

结果：

- report：`ncu-reps/moe_down_partial_dot_candidate.ncu-rep`
- baseline：`ncu-reps/moe_down_partial_baseline.ncu-rep`
- duration：`31.008 us -> 33.760 us`，变慢 `8.88%`
- registers/thread：`96 -> 88`
- inst_executed 降低 `18.1%`
- issue active：`23.69% -> 17.79%`
- SM throughput：`17.70% -> 13.31%`

结论：

- 虽然寄存器和指令数下降，但调度/吞吐变差，实际更慢。
- 当前显式 `tl.sum(down_w * hidden)` 更好。

### RMSNorm duplicate load cleanup

尝试内容：

- 将 RMSNorm pass2 代码层面的两次 input `tl.load` 改为一次 load。

结果：

- reports：`ncu-reps/rmsnorm_baseline_2048.ncu-rep`、`ncu-reps/rmsnorm_single_load_2048.ncu-rep`、`ncu-reps/rmsnorm_baseline_512.ncu-rep`、`ncu-reps/rmsnorm_single_load_512.ncu-rep`
- global load 指令、L1 load requests/sectors、registers 都不变。
- 2048 shape duration 小幅下降，512 shape 小幅上升，整体无显著性能结论。

结论：

- 编译器大概率已经消除了重复 load。
- 改动作为合理代码清理保留，但不计为主要性能优化。

### x tile 复用

尝试：

- 一个 program 同时处理两个 row tile
- 目标是复用同一个 input `x` tile load

结果：

- `_fused_gate_up_swiglu_kernel` 时间：`60.736 -> 89.120`
- registers/thread：`142 -> 222`
- 端到端掉到约 `166.55 TPS`

结论：

- 复用 `x` tile 的收益小于寄存器和 occupancy 损失
- 不保留

### 单 kernel deterministic down reduce

尝试：

- 一个 program 内串行处理 topk=6 并直接输出 bf16

结果：

- 能运行但端到端掉到约 `150.9 TPS`

结论：

- 串行 topk 破坏并行度
- 不保留

### 删除 `out.zero_()` 的 atomic 变体

尝试：

- slot0 用 store
- slot1-5 用 atomic_add

结果：

- 存在跨 CTA 顺序 race
- 输出 token 明显漂移

结论：

- 不安全
- 不保留

### 替换 cuBLAS/CUTLASS 主路径

当前判断：

- cuBLAS/CUTLASS 路径已经较优化
- 直接替换工程复杂度高，收益不确定
- 目前优先级低于 MoE Triton kernel 内部优化和剩余 torch 小 kernel 清理

## 剩余优化空间

### 1. MoE gate/up kernel

当前最大自研热点之一。

观察点：

- 继续检查是否还有重复 `tl.load` 或重复地址计算
- 保持当前 combined gate/up dot 形态
- 不再尝试两个 row tile 共享 `x` 的高寄存器版本
- 如果要继续优化，优先找低寄存器成本的 load/layout 改写

### 2. MoE down partial kernel

当前另一个大热点，但 `block_ptr + tl.dot` 版本已经被 `ncu` 否掉。

当前判断：

- hidden vector 每个 row block 重新 load 是主要成本之一。
- 直接改成 `tl.dot` 会降低指令和寄存器，但实际吞吐变差。
- 如果继续优化，需要新的 memory/layout 思路，而不是简单替换成 `tl.dot`。
- 不回到 atomic reduce。

### 3. RMSNorm kernel

当前判断：

- pass2 单次 load cleanup 已完成。
- `ncu` 显示底层 load 指令不变，因此不是有效性能大头。
- 进一步优化只有专门针对 2048/512 shape 改写 reduction 结构才可能有意义，但优先级不高。

### 4. Attention decode

观察点：

- `_gemv_contig_kernel` 仍是通用实现，可继续对固定 shape 特化
- `_build_qkv_rope_kernel` 同时处理 q/k/v 和 RoPE，需关注寄存器压力
- `_decode_attention_q1_kernel` 当前不是最大热点，不优先大改

### 5. 剩余 torch elementwise/copy

当前仍有：

- `84/step`
- `0.208 ms/step`

这部分可能是最容易继续减少 kernel 数的空间，但必须确认来源，不能盲目替换。

## 当前结论

当前仓库的有效优化状态：

- 目标是 `DeepSeek-V2-Lite-Chat` decode。
- `src/run.py --kernel-family small|batch` 是当前统一运行入口。
- `small` 和 `batch` 是两套实验 kernel family，不代表自动组合出的最终 SOTA。
- FlashMLA 接入不在当前活跃代码路径中。
- 剩余主要空间在 MoE Triton kernel 和少量 torch 小 kernel。

## Batch Decode Extension Status

Current `src/run.py` uses a CUDA graph decode path for q_len=1 decode across
batch sizes 1 through 256. `--kernel-family batch` calls the grouped batching
attention and MoE kernels. `--kernel-family small` calls the small-GEMV
attention and route-local MoE kernels through the same batched runner API.

The target is a unified batched implementation that remains efficient at
`bsz=1`, not a return to a separate single-token-only decode path. The current
implementation improves batch scaling substantially, but profiling shows the
`bsz=1` regression is inside the generalized Triton kernels themselves: large
batch-oriented tiles and MoE route grouping are applied even when the active
batch dimension is one. See `docs/batched_decode_profile_analysis.md` for the
A100 `nsys`/`ncu` breakdown and the shape-adaptive kernel direction.

Validated on NVIDIA A100 80GB PCIe GPU3 with input length 24 and
`max_new_tokens=100`:

The table below records the forced batching run. A separate forced small-GEMV
full sweep is tracked in `docs/batched_decode_profile_analysis.md` and in the
SVG plot.

| Batch | Path | Decode TPS | Log |
| ---: | --- | ---: | --- |
| 1 | `triton_batching_graph` | 158.07 | `src/run.py --kernel-family batch` A100 GPU3 sweep |
| 2 | `triton_batching_graph` | 277.16 | `src/run.py --kernel-family batch` A100 GPU3 sweep |
| 4 | `triton_batching_graph` | 537.06 | `src/run.py --kernel-family batch` A100 GPU3 sweep |
| 8 | `triton_batching_graph` | 959.59 | `src/run.py --kernel-family batch` A100 GPU3 sweep |
| 16 | `triton_batching_graph` | 1958.82 | `src/run.py --kernel-family batch` A100 GPU3 sweep |
| 32 | `triton_batching_graph` | 3631.14 | `src/run.py --kernel-family batch` A100 GPU3 sweep |
| 64 | `triton_batching_graph` | 5745.78 | `src/run.py --kernel-family batch` A100 GPU3 sweep |
| 128 | `triton_batching_graph` | 7902.87 | `src/run.py --kernel-family batch` A100 GPU3 sweep |
| 256 | `triton_batching_graph` | 9402.64 | `src/run.py --kernel-family batch` A100 GPU3 sweep |

Node-level `nsys` reports for the current path:

- `nsys-reps/sota_bsz64_node_nsys202506.nsys-rep`
- `nsys-reps/sota_bsz256_node_nsys202506.nsys-rep`

The node-level profiles show the remaining large-batch bottleneck is still MoE,
not attention: at `bsz=64`, grouped gate/up is 35.7% and grouped down is 26.5%;
at `bsz=256`, grouped gate/up is 38.8% and grouped down is 27.6%. `ncu` roofline
collection is currently blocked by `ERR_NVGPUCTRPERM` for this user, so the
memory-bound versus compute-bound split cannot be stated from hardware counters
yet.

## Batch MoE Tight Grid

After the combined gate/up dot optimization, the next useful MoE cleanup was the
grouped-kernel launch bound. The old launch used `total_routes` as the grid-x
upper bound for both gate/up and down. At `bsz=256`, `topk=6`,
`total_routes=1536`; with `BLOCK_N=32`, the actual route-block count is bounded
by `ceil(total_routes / BLOCK_N) + nonempty_experts <= 112`. The remaining CTAs
only load `block_count` and return.

The implementation now uses this tighter upper bound by default while keeping
`DSV2_BATCH_MOE_TIGHT_GRID=0` as an A/B fallback.

Validated on A100 GPU3, `bsz=256`, fixed tile
`DSV2_BATCH_MOE_GATE_TILE=32,64,128`,
`DSV2_BATCH_MOE_DOWN_TILE=32,64,128`:

| Kernel | Old grid | Tight grid | Old duration | Tight duration |
| --- | ---: | ---: | ---: | ---: |
| `_grouped_gate_up_swiglu_kernel` | 33,792 | 2,464 | 273.66 us | 254.91 us |
| `_grouped_down_partial_kernel` | 49,152 | 3,584 | 177.31 us | 158.72 us |

Reports:

- `ncu-reps/batch_bsz256_grouped_gate_up_tight_grid_m64_decode.ncu-rep`
- `ncu-reps/batch_bsz256_grouped_down_tight_grid_m64_decode.ncu-rep`
- baseline combined gate/up:
  `ncu-reps/batch_bsz256_grouped_gate_up_combined_m64_decode_source_0882f05.ncu-rep`
- baseline down:
  `ncu-reps/batch_bsz256_grouped_down_m64_decode_source_0882f05.ncu-rep`

End-to-end fixed-tile comparison on A100 GPU3:

| Mode | Decode TPS |
| --- | ---: |
| `DSV2_BATCH_MOE_TIGHT_GRID=0` | 9478.89 |
| `DSV2_BATCH_MOE_TIGHT_GRID=1` | 9747.18 |
| default env | 9731.97 |

Conclusion: this is a low-risk structural optimization. It does not change the
math or tile shape; it only avoids launching CTAs that are guaranteed to return
immediately.

## Batch MoE Metadata Kernel Cleanup

A follow-up free-lunch review found two additional metadata cleanups in the same
grouped MoE path:

- When gate/up and down use the same `BLOCK_N`, reuse the route block list
  built before gate/up instead of clearing `block_count` and rebuilding it before
  down.
- Fuse the initial `counts` zero and `block_count` zero into one metadata
  kernel.

These changes do not alter the mathematical kernels or tile choices. The
fallback path still rebuilds route blocks if gate/up and down use different
`BLOCK_N` values.

Short `nsys --cuda-graph-trace=node` profile at `bsz=256`,
`max_new_tokens=20`:

| Change point | Before | After |
| --- | ---: | ---: |
| total kernel instances | 22,149 | 21,431 |
| `_build_route_blocks_kernel` instances | 718 | 718 |
| zero metadata kernel instances | 1,436 `_zero_i32_kernel` | 718 `_zero_counts_and_block_count_kernel` |

The route-block reuse was already included in the "after" count above: one
`_build_route_blocks_kernel` remains per MoE layer invocation instead of two.
The zero fusion removes one additional tiny metadata launch per invocation.

End-to-end `bsz=256`, `max_new_tokens=100` after this cleanup measured
`9789.83` decode TPS on A100 GPU3. The small TPS delta is close to run-to-run
noise, but the kernel-count reduction is deterministic and low risk.

Also tested and reverted during this review:

- Direct-writing batch attention output into `workspace.attn_ctx`.
- Adding an optional output tensor to RMSNorm for `kv_lora_norm`.

Those changes did not reduce visible kernel counts in `nsys`, so they were not
kept.

## Batch Attention Linear Tile

After excluding the heavily optimized MoE gate/up and down kernels, the next
largest custom Triton hotspot at `bsz=256` was `_batched_linear_kernel`, used by
the batched attention projections:

- q + kv-a projection: `[B, 2048] x [3648, 2048]`
- kv-b projection: `[B, 512] x [4096, 512]`
- o projection: `[B, 2048] x [2048, 2048]`

This kernel is a straightforward row-batched linear projection. One CTA computes
one `(BLOCK_B, BLOCK_O)` output tile and iterates over the input dimension in
`BLOCK_K` chunks:

```text
acc[BLOCK_B, BLOCK_O] += x[BLOCK_B, BLOCK_K] @ w[BLOCK_O, BLOCK_K]^T
```

The previous default tile for `batch_size > 1` was `(BLOCK_B, BLOCK_O, BLOCK_K)
= (8, 64, 128)`. For small and medium batch sizes this is a reasonable
decomposition: each CTA is light, register pressure is moderate, and the grid is
large enough to expose parallelism. At high batch sizes the same tile becomes
over-decomposed. It splits a dense `B x out_rows` projection into many small
CTAs even though each CTA still pays the fixed costs of scheduling, masks,
pointer arithmetic, loop setup, and stores.

For `batch_size >= 128`, the default is now `(32, 64, 128)`. The math is
unchanged. The new tile only groups four times as many batch rows into each CTA:

| Tile | Output elements per CTA | K tile | Main difference |
| --- | ---: | ---: | --- |
| old `(8,64,128)` | 512 | 128 | more CTAs, lower per-CTA register use |
| new `(32,64,128)` | 2048 | 128 | fewer CTAs, higher per-CTA register use |

At `B=256`, this cuts the grid by exactly 4x for the three real projection
shapes:

| Projection | Old grid | New grid | Reduction |
| --- | ---: | ---: | ---: |
| q + kv-a | 1824 | 456 | 4.0x |
| kv-b | 2048 | 512 | 4.0x |
| o | 1024 | 256 | 4.0x |

This is why the source diff looks surprisingly small. The old kernel was not
doing wrong arithmetic; it was launching too many independently scheduled CTAs
for these high-batch projection shapes. The new tile leaves the inner K loop,
output dimension tile, dtype path, and memory layout unchanged, but amortizes the
fixed per-CTA work across four times as many batch rows.

NCU comparison at `bsz=256`, first three steady-state `_batched_linear_kernel`
launches:

| Projection | Tile | Grid | Duration | SM Busy | Mem Busy | Issue Active | Eligible Warps | Active Warps | Reg/thread |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q + kv-a | `(8,64,128)` | 1824 | 188.51 us | 42.02% | 67.59% | 44.15% | 0.734 | 4.558 | 94 |
| q + kv-a | `(32,64,128)` | 456 | 105.34 us | 30.89% | 48.41% | 31.29% | 0.386 | 2.242 | 162 |
| kv-b | `(8,64,128)` | 2048 | 59.78 us | 40.81% | 62.46% | 44.91% | 0.774 | 4.661 | 94 |
| kv-b | `(32,64,128)` | 512 | 36.03 us | 29.07% | 47.00% | 32.38% | 0.422 | 2.516 | 162 |
| o | `(8,64,128)` | 1024 | 111.49 us | 40.69% | 65.05% | 44.41% | 0.739 | 4.600 | 94 |
| o | `(32,64,128)` | 256 | 64.58 us | 29.67% | 46.79% | 31.29% | 0.387 | 2.308 | 162 |

The counter-intuitive part is that several local utilization counters become
lower with the faster tile. This is expected here:

- `Reg/thread` increases from 94 to 162 because the accumulator grows from
  `8 x 64` to `32 x 64`.
- `SM Busy`, memory busy, issue active, active warps, and eligible warps all
  drop in the sampled kernels.
- Despite that, total duration drops from about `359.8 us` to `206.0 us` across
  these three launches.

The interpretation is that the old tile was keeping the GPU busier at the CTA
scheduler level, but a meaningful part of that activity was overhead from
over-decomposition. The new tile has worse-looking per-SM instantaneous activity,
yet completes the projection sooner because the amount of CTA-level bookkeeping
is much smaller and each CTA performs more useful dot work before retiring.
Occupancy-like counters alone would point in the wrong direction for this case;
the decisive metric is end-to-end kernel duration on the actual projection
shapes.

This should not be generalized into "larger `BLOCK_B` is always better". The
tradeoff is shape-specific:

- At small batch sizes, `BLOCK_B=32` would waste lanes on masked batch rows and
  carry the higher register footprint without enough useful rows to amortize it.
- At high batch sizes, especially `B=128/256`, the projections are large enough
  that reducing CTA count dominates the extra register pressure.
- `BLOCK_O=64` and `BLOCK_K=128` were intentionally left unchanged, so this
  optimization isolates the batch-axis tiling effect instead of mixing in a new
  output/K pipeline.

The practical rule kept in code is therefore conservative: keep the existing
`(8,64,128)` tile for `1 < batch_size < 128`, keep the existing `bsz=1` tile,
and only switch to `(32,64,128)` for `batch_size >= 128`.

Reports:

- `ncu-reps/batch_bsz256_batched_linear_default_decode_8c21ced.ncu-rep`
- `ncu-reps/batch_bsz256_batched_linear_b32_decode.ncu-rep`

Correctness:

- Direct A/B of `(8,64,128)` versus `(32,64,128)` on the three real shapes was
  bitwise equal.
- The override `DSV2_BATCH_LINEAR_TILE=N,M,K` remains available for experiments.

End-to-end `bsz=256`, `max_new_tokens=100` measured `11390.90` decode TPS on
A100 GPU3 after this change. `bsz=128` also improved in the quick A/B check:
`8608.96` TPS with old `(8,64,128)` versus `9482.83` TPS with the new default.

Open follow-ups:

- Sweep `B=64/96/128/192/256` to confirm whether the threshold should remain
  exactly `128`.
- Consider per-projection tile choices only if profiling shows a clear split
  between q/kv-a, kv-b, and o. The current single rule is intentionally simpler
  and already gives the main gain.
