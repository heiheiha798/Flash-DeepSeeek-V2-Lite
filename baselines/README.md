# Baselines

This directory stores wrappers for third-party inference baselines. These
scripts keep external projects outside this repository and only pin the exact
commands used for comparison.

## Available Baselines

- `llama_cpp/`: llama.cpp CUDA baseline for `DeepSeek-V2-Lite-Chat` GGUF, including `llama-bench -b/-ub` sweep `1..512`.
- `sglang/`: official SGLang baseline for `DeepSeek-V2-Lite-Chat` HF weights, including batch sweep `1..512`.
- `vllm/`: official vLLM baseline for `DeepSeek-V2-Lite-Chat` HF weights, including batch sweep `1..512`.

## Results

See `docs/third_party_baselines.md` for the current third-party decode TPS table and reproduction notes.
