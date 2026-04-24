# Performance Reporting

所有 decode TPS、profile 或 baseline 数字都必须同时记录硬件和软件上下文。只写裸 `TPS` 不够，因为 GPU 频率、可见设备、driver、PyTorch/Triton 版本都会影响结果。

## 自动输出

`src/baseline.py` 和 `src/sota.py` 的 JSON 输出包含：

- `batch_size`：请求 batch size。
- `path`：实际运行路径。`src/sota.py` 中 `triton_sota_graph` 表示 batch=1 手写 Triton CUDA graph 路径；`batched_eager_fallback` 表示 batch>1 的 batched eager fallback。
- `ttft_ms`：prefill 到首 token 的时间。
- `tps`：decode 阶段 throughput，计算口径是 `decode_tokens / decode_seconds`。
- `decode_tokens`：decode 阶段统计的 token 数，不包含 TTFT 对应的首 token。
- `hardware`：CPU、GPU、Python、PyTorch、CUDA、Triton、Transformers 和 `CUDA_VISIBLE_DEVICES`。

示例结构：

```json
{
  "batch_size": 1,
  "path": "triton_sota_graph",
  "ttft_ms": 123.45,
  "tps": 195.0,
  "decode_tokens": 99,
  "hardware": {
    "cpu": {
      "model": "INTEL(R) XEON(R) PLATINUM 8558P",
      "physical_cores": 96,
      "logical_cores": 192,
      "memory_gb": 503.53
    },
    "gpu": {
      "visible_index": 0,
      "name": "NVIDIA A100 80GB PCIe",
      "capability": "sm80",
      "sm_count": 108,
      "memory_gb": 79.25
    },
    "software": {
      "python": "3.10.20",
      "torch": "2.10.0+cu130",
      "torch_cuda": "13.0",
      "nvidia_driver": "590.44.01",
      "psutil": "7.2.2",
      "triton": "3.6.0",
      "transformers": "4.57.6"
    },
    "env": {
      "cuda_visible_devices": "3"
    }
  }
}
```

`visible_index` 是 PyTorch 看到的 CUDA device index。如果设置了 `CUDA_VISIBLE_DEVICES=3`，脚本内通常仍显示 `visible_index=0`，真实物理卡号应看 `env.cuda_visible_devices`。

## 手动记录规则

写入 `README.md`、`docs/*.md` 或 `baselines/*/README.md` 的性能数据至少包含：

- 命令：完整到 `CUDA_VISIBLE_DEVICES` 或 baseline wrapper 的 `GPU` 参数。
- 模型：例如 `/data/models/DeepSeek-V2-Lite-Chat`。
- 场景：batch、prompt/input tokens、generated tokens、dtype。
- 指标：decode TPS；如果是 profile，还要写每步 decode graph kernel 数和 GPU 时间。
- 硬件：GPU 型号、SM、显存、CPU 型号、CPU 核数、系统内存。
- 软件：Python、PyTorch、CUDA、Triton、Transformers 或对应 baseline 框架版本。

## 当前机器上下文

当前开发机的主要硬件上下文：

- CPU：`INTEL(R) XEON(R) PLATINUM 8558P`
- CPU cores：`96 physical / 192 logical`
- System memory：约 `503.53 GiB`
- GPU：`NVIDIA A100 80GB PCIe`
- GPU capability：`sm80`
- GPU SM count：`108`
- GPU memory from PyTorch：约 `79.25 GiB`
- Driver/CUDA from `nvidia-smi`：`driver 590.44.01`, `CUDA 13.1`

当前 `flashmla` 环境的主要软件上下文：

- Python：`3.10.20`
- PyTorch：`2.10.0+cu130`
- PyTorch CUDA：`13.0`
- NVIDIA driver：`590.44.01`
- Triton：`3.6.0`
- Transformers：`4.57.6`
- psutil：`7.2.2`
