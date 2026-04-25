# HF Profile Summary

本文档记录早期 `DeepSeek-V2-Lite-Chat` baseline 与 Triton SOTA 路径的
profile 结论。当前活跃入口已收敛为 `src/run.py --kernel-family
small|batch`；本文保留作为历史 profile 参考。

当前主线：

- HF decode-only CUDA graph baseline
- DeepSeek-V2-Lite 专用 Triton kernel 优化
- `nsys` / `ncu` 数据驱动优化

## 实验对象

模型：

- `/data/models/DeepSeek-V2-Lite-Chat`

环境：

- conda：`/data/home/tianjianyang/.conda/envs/flashmla`
- GPU：`NVIDIA A100 80GB PCIe / SM80`
- CPU：`INTEL(R) XEON(R) PLATINUM 8558P`, `96C/192T`, `503.53 GiB RAM`
- 软件：`Python 3.10.20`, `torch 2.10.0+cu130`, `torch CUDA 13.0`, `triton 3.6.0`, `transformers 4.57.6`
- 性能测试优先 `GPU3`
- dtype：`bf16`

默认推理口径：

- prompt：`Write me a 500 word novel.`
- max new tokens：`100`
- batch：`1`
- decode：`q_len=1`

主要脚本：

- `src/baseline.py`：当前 HF decode-only CUDA graph baseline
- `src/sota.py`：当前 Triton SOTA 路径

## 当前 SOTA Profile 快照

命令：

```bash
CUDA_VISIBLE_DEVICES=3 /data/home/tianjianyang/.conda/envs/flashmla/bin/python src/sota.py --device cuda:0 --max-new-tokens 100
```

非 profile 结果：

- `tps = 195.02 ~ 197.61`
- 当前稳定档位约 `195+ TPS`
- 硬件：`NVIDIA A100 80GB PCIe`, `sm80`, `80 GB`; `INTEL(R) XEON(R) PLATINUM 8558P`, `96C/192T`, `503.53 GiB RAM`
- 软件：`Python 3.10.20`, `torch 2.10.0+cu130`, `torch CUDA 13.0`, `triton 3.6.0`, `transformers 4.57.6`

最新 node-level nsys：

- `nsys-reps/sota_gpu3_gate_combined_node.nsys-rep`
- `nsys-reps/sota_gpu3_gate_combined_node.sqlite`

decode graph 统计：

- `651 kernels/step`
- `5.145 ms/step`

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

当前判断：

- graph 外 GPU kernel 已基本被收进 graph
- 当前瓶颈主要在 graph 内模型主体
- MoE 和 GEMV/GEMM 仍是主要耗时
- 剩余 torch elementwise/copy 还有继续 Triton 化空间

## 为什么旧 HF Graph 和 Eager 接近

旧版 `hf_graph.py` 曾经出现 graph 和 eager TPS 接近的问题。关键原因不是 CUDA graph 本身无效，而是旧 graph patch 改变了 MoE 计算路径。

DeepSeek-V2-Lite MoE 配置：

- routed experts：`64`
- topk experts：`6`
- shared experts：`2`
- MoE 层数：`26`

旧 graph patch 为了 graph-safe，把 routed experts 从 eager 的 `topk=6` 实际变成了“全 64 experts 都算一遍”。每个 expert 有三个主要线性层：

- `gate_proj`
- `up_proj`
- `down_proj`

因此额外 GEMV 数量理论上是：

- `(64 - 6) * 3 * 26 = 4524`

这与旧 profile 中 graph 相比 eager 多出的 `internal::gemvx::kernel` 数量对齐：

- 旧 graph：`5072`
- 旧 eager：`548`
- 差值：`4524`

结论：

- 旧 graph 版本性能不理想，主因是 MoE patch 错误地全算 experts
- 不能据此否定 CUDA graph
- 当前 SOTA 已用 topk-only Triton MoE 路径替换该旧逻辑

## Graph 外 Kernel 收敛

早期 grouped MoE 版本中，decode replay 之间仍有少量图外 GPU kernel：

- `vectorized_elementwise_kernel`
- `reduce_kernel`

来源主要是：

- 图外 `torch.argmax`
- decode 外壳里的 cache position / buffer 更新

后续已经将这些操作并入 graph capture。当前结论：

- decode 外壳不是主要瓶颈
- 后续性能问题主要位于 graph 内模型 kernel

## Baseline 与 SOTA 的关系

`src/baseline.py`：

- 用于保留 HF decode-only CUDA graph baseline
- 尽量少改模型内部算子
- 作为 correctness / 性能对比参考

`src/sota.py`：

- 用于接入 Triton kernels
- 代表当前性能优化路径
- 允许针对 DeepSeek-V2-Lite 固定形状做专用化

后续新增优化应默认先接入 `src/sota.py`，与 `src/baseline.py` 对比。

## 当前结论

- 当前 baseline 是 `src/baseline.py`。
- 当前优化路径是 `src/sota.py`。
- decode graph 外 GPU kernel 已基本收敛。
- graph 内主要耗时来自 MoE、GEMV/GEMM 和少量 torch elementwise/copy。
- 旧 HF graph 性能问题主要来自 MoE 全 experts 误算，不能据此否定 CUDA graph。
