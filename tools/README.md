# Tools

This directory stores runnable micro profile and validation scripts. Kernel
implementations stay in `triton_kernels/`; scripts here import them via the repo
root on `sys.path`.

## Scripts

- `profile_moe_ncu.py`: Nsight Compute micro profile for routed MoE kernels.
- `profile_attention_o_ncu.py`: Nsight Compute micro profile for attention `o_proj` GEMV.
- `profile_attention_qkv_ncu.py`: Nsight Compute micro profile for q/kv-a GEMV.
- `profile_attention_kvb_ncu.py`: Nsight Compute micro profile for kv-b GEMV.
- `test_moe_grouped_gemv.py`: correctness check for routed MoE against HF experts.
