# DeepSeek-V2-Lite Triton Inference Optimization

<div align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</div>

## English

This repository is an independent Triton inference optimization library for
`DeepSeek-V2-Lite-Chat` decode on `A100 / SM80` with `bf16`. The active
development path is:

- Hugging Face model structure
- prefill in eager mode
- single-token decode with CUDA graph replay
- DeepSeek-V2-Lite-specific Triton kernels for hot decode operators
- `nsys` / `ncu` driven optimization

## Current Scope

Target configuration:

- Model: `/data/models/DeepSeek-V2-Lite-Chat`
- GPU: NVIDIA A100 80GB, SM80
- dtype: `bf16`
- batch regime: batched decode sweep over `1 2 4 8 16 32 64 128 256`
- decode step shape: `q_len=1`
- default output length: `100` new tokens

The goal is not general inference serving. Prefill is still run eagerly to
populate the KV cache; the optimized path is the CUDA-graph captured single-step
decode loop across the requested batch size. The active runner exposes two
explicit q_len=1 Triton decode kernel families:

- `--kernel-family small`: GEMV-oriented kernels derived from the `bsz=1`
  execution template. This family is the low-batch / `bsz=1` optimized path and
  a diagnostic curve for how far route-local GEMV-style kernels scale.
- `--kernel-family batch`: grouped batching kernels optimized for larger batch
  throughput. This family is the high-batch path and is designed to amortize
  expert routing and weight reads across more tokens/routes.

Serving baselines under `baselines/` are capped at batch size 256 for comparable
plots.

The kernel optimization goal is to iterate quickly on fixed-shape decode kernels
and measure whether they improve real decode latency.

## Main Entrypoints

Run the small-GEMV kernel family:

```bash
CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family small --device cuda:0 --max-new-tokens 100
```

Run the grouped batching kernel family:

```bash
CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100
```

Run the two reporting sweeps over common batch sizes:

```bash
CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family small --device cuda:0 --max-new-tokens 100 \
  --batch-sizes "1 2 4 8 16 32 64 128 256"

CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100 \
  --batch-sizes "1 2 4 8 16 32 64 128 256"
```

Interpretation: the two `--kernel-family` values intentionally use different
kernel implementations. `small` represents the bsz=1-oriented GEMV design;
`batch` represents the grouped batching design for larger batches. Sweeps show
the tradeoff directly rather than hiding it behind an automatic dispatch policy.

For performance runs, check `nvidia-smi` first and do not run two GPU workloads
on the same card. Use a single idle A100 for final numbers; another idle A100 is
acceptable for relative micro comparisons.

## Results Visualization

The current A100 batch-size sweep compares the custom `src/run.py` small and
batch kernel families with SGLang, vLLM, and llama.cpp baselines.

<div align="center">
  <img src="docs/figures/batch_scaling.svg" alt="Batch-size throughput scaling" width="760">
</div>

GitHub or browser dark-mode processing may reduce the SVG contrast. If the plot
looks unclear in the README, download `docs/figures/batch_scaling.svg` and view
it locally.

Detailed numeric tables and reproduction commands are maintained in
`docs/third_party_baselines.md`.

## Profiling

Node-level `nsys` profile:

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --output=<nsys-output-prefix> \
  /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100
```

MoE kernel `ncu` micro profile:

```bash
CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --kernel-name regex:_fused_gate_up_swiglu_kernel \
  --launch-skip 1 \
  --launch-count 1 \
  --set full \
  --export <ncu-output-prefix> \
  --force-overwrite \
  /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  tools/profile_moe_ncu.py
```

## Performance Snapshot

Representative batch-kernel snapshot on an A100 80GB PCIe GPU:

- command: `CUDA_VISIBLE_DEVICES=3 ... src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100`
- decode throughput: about `195+ TPS`
- hardware: `NVIDIA A100 80GB PCIe`, `sm80`, `80 GB`, `INTEL(R) XEON(R) PLATINUM 8558P`, `96C/192T`, `503.53 GiB RAM`
- software: `Python 3.10.20`, `torch 2.10.0+cu130`, `torch CUDA 13.0`, `triton 3.6.0`, `transformers 4.57.6`
- node-level graph: `651 kernels/step`
- graph GPU time: about `5.145 ms/step`

Exact numbers vary with the selected physical GPU and clock state. The committed
tables use one idle A100 per run and report hardware context with the result.

## Active Directory Layout

- `src/`: runnable DeepSeek-V2-Lite decode scripts.
- `src/run.py`: CUDA graph runner with `--kernel-family small|batch`.
- `triton_kernels/`: custom Triton kernel implementations only.
- `tools/`: micro profile and validation scripts.
- `baselines/`: third-party baseline wrappers, including llama.cpp, SGLang, and vLLM.
- `docs/`: profiling summaries, Triton optimization notes, and performance reporting rules.

## Dependencies

The root `requirements.txt` tracks the active optimization path. The existing
conda environment used during development is:

```bash
/data/home/tianjianyang/.conda/envs/flashmla
```

Install Python dependencies as needed:

```bash
pip install -r requirements.txt
```

## Documentation

- `docs/hf_profile_summary.md`: archived profile conclusions from early CUDA graph experiments.
- `docs/triton_kernel_optimizations.md`: effective Triton optimizations and failed attempts.
- `docs/performance_reporting.md`: TPS report format with hardware and software context.

## 中文

本仓库是面向 `DeepSeek-V2-Lite-Chat` decode 推理的独立 Triton 优化库，
当前主要优化目标是 `A100 / SM80 / bf16`。当前主线包括：

- Hugging Face 模型结构
- eager prefill
- 单 token decode CUDA graph replay
- DeepSeek-V2-Lite 热点 decode 算子的专用 Triton kernel
- 基于 `nsys` / `ncu` 的性能分析和优化

## 当前范围

目标配置：

- 模型：`/data/models/DeepSeek-V2-Lite-Chat`
- GPU：NVIDIA A100 80GB，SM80
- dtype：`bf16`
- batch 口径：`1 2 4 8 16 32 64 128 256` 的 batched decode sweep
- decode step shape：`q_len=1`
- 默认输出长度：`100` new tokens

本项目目标不是通用推理服务，而是固定 decode 场景下的 kernel 优化研究。
prefill 仍然使用 eager 路径填充 KV cache；优化主体是对指定 batch size 的
单步 decode 循环做 CUDA graph capture 和 Triton kernel 替换。当前 runner
显式暴露两套 q_len=1 Triton decode kernel family：

- `--kernel-family small`：GEMV-oriented kernel，来自 `bsz=1` 执行模板。
  这一路径主要代表 low-batch / `bsz=1` 优化方向，也用于诊断
  route-local GEMV 风格 kernel 能扩展到什么程度。
- `--kernel-family batch`：面向更大 batch 吞吐的 grouped batching kernel。
  这一路径主要代表 high-batch 优化方向，通过更多 token / route 分摊
  expert routing 和权重读取开销。

`baselines/` 下的第三方 serving baseline 统一截断到 batch size 256，便于
结果可比。

kernel 优化目标是快速迭代固定形状 decode kernel，并用真实 decode latency
验证改动是否有效。

## 主要入口

运行 small-GEMV kernel family：

```bash
CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family small --device cuda:0 --max-new-tokens 100
```

运行 grouped batching kernel family：

```bash
CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100
```

运行两套 common batch-size sweep：

```bash
CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family small --device cuda:0 --max-new-tokens 100 \
  --batch-sizes "1 2 4 8 16 32 64 128 256"

CUDA_VISIBLE_DEVICES=0 /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100 \
  --batch-sizes "1 2 4 8 16 32 64 128 256"
```

解释：两个 `--kernel-family` 使用不同的 kernel 实现。`small` 代表
bsz=1-oriented GEMV 设计，`batch` 代表面向更大 batch 的 grouped batching
设计。sweep 直接展示二者的实现取舍，而不是隐藏在自动 dispatch policy 后面。

性能测试前先检查 `nvidia-smi`，不要在同一张卡上同时运行两个 GPU workload。
最终结果使用一张空闲 A100；做相对 micro comparison 时可以使用另一张空闲
A100。

## 结果可视化

当前 A100 batch-size sweep 比较了 `src/run.py` 的 small 和 batch kernel
family，以及 SGLang、vLLM、llama.cpp baseline。

<div align="center">
  <img src="docs/figures/batch_scaling.svg" alt="Batch-size throughput scaling" width="760">
</div>

GitHub 或浏览器暗色模式处理可能降低 SVG 对比度。如果 README 中图像看不清，
请下载 `docs/figures/batch_scaling.svg` 后本地查看。

详细数字表格和复现命令维护在 `docs/third_party_baselines.md`。

## Profiling

node-level `nsys` profile：

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --output=<nsys-output-prefix> \
  /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100
```

MoE kernel `ncu` micro profile：

```bash
CUDA_VISIBLE_DEVICES=0 ncu \
  --target-processes all \
  --kernel-name regex:_fused_gate_up_swiglu_kernel \
  --launch-skip 1 \
  --launch-count 1 \
  --set full \
  --export <ncu-output-prefix> \
  --force-overwrite \
  /data/home/tianjianyang/.conda/envs/flashmla/bin/python \
  tools/profile_moe_ncu.py
```

## 性能快照

A100 80GB PCIe GPU 上的代表性 batch-kernel 快照：

- 命令：`CUDA_VISIBLE_DEVICES=3 ... src/run.py --kernel-family batch --device cuda:0 --max-new-tokens 100`
- decode throughput：约 `195+ TPS`
- 硬件：`NVIDIA A100 80GB PCIe`, `sm80`, `80 GB`, `INTEL(R) XEON(R) PLATINUM 8558P`, `96C/192T`, `503.53 GiB RAM`
- 软件：`Python 3.10.20`, `torch 2.10.0+cu130`, `torch CUDA 13.0`, `triton 3.6.0`, `transformers 4.57.6`
- node-level graph：`651 kernels/step`
- graph GPU time：约 `5.145 ms/step`

精确数字会随物理 GPU 和 clock state 波动。提交到文档中的表格使用单张空闲
A100 运行，并随结果记录硬件上下文。

## 目录结构

- `src/`：可运行的 DeepSeek-V2-Lite decode 脚本。
- `src/run.py`：支持 `--kernel-family small|batch` 的 CUDA graph runner。
- `triton_kernels/`：自定义 Triton kernel 实现。
- `tools/`：micro profile 和 validation 脚本。
- `baselines/`：第三方 baseline wrapper，包括 llama.cpp、SGLang、vLLM。
- `docs/`：profile 总结、Triton 优化记录和性能报告规范。

## 依赖

根目录 `requirements.txt` 跟踪当前活跃优化路径需要的 Python 依赖。开发中使用的
conda 环境是：

```bash
/data/home/tianjianyang/.conda/envs/flashmla
```

按需安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 文档

- `docs/hf_profile_summary.md`：早期 CUDA graph 实验的归档 profile 结论。
- `docs/triton_kernel_optimizations.md`：有效 Triton 优化和失败尝试。
- `docs/performance_reporting.md`：TPS 报告格式、硬件和软件上下文要求。
