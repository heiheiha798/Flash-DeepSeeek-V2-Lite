# Baselines

This directory stores wrappers for third-party inference baselines. These
scripts keep external projects outside this repository and only pin the exact
commands used for comparison.

## Available Baselines

- `llama_cpp/`: llama.cpp CUDA baseline for `DeepSeek-V2-Lite-Chat` GGUF, including real parallel `llama-batched-bench -npl` sweep `1..256`.
- `sglang/`: official SGLang baseline for `DeepSeek-V2-Lite-Chat` HF weights, including batch sweep `1..256`.
- `vllm/`: official vLLM baseline for `DeepSeek-V2-Lite-Chat` HF weights, including batch sweep `1..256`.

## Batch Sweep Range

All baseline wrappers use the common valid sweep range `1 2 4 8 16 32 64 128 256`; 512-point measurements are omitted from the committed third-party baselines.

## Results

See `docs/third_party_baselines.md` for the current third-party decode TPS table and reproduction notes.
