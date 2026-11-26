# GB10 NVFP4 GEMM

The 4096³ comparison starts with packed E2M1 inputs, one UE4M3 scale per 16 values, FP32 accumulation and BF16 output. Both baselines stage the packed values and their scales. Checkpoint numbers follow the optimization order within each branch.

| Checkpoint | Parent | Change |
| --- | --- | --- |
| [6A](kernel_nvfp4_6A_scale_pipeline.py) | BF16 5A | Port the rolling pipeline, including shared scale stages. Eight compute warps, group width 4. |
| [7A](kernel_nvfp4_7A_tile_traversal.py) | 6A | Change only the CTA tile-order group width from 4 to 1. |
| [6M](kernel_nvfp4_6M_port.py) | BF16 5M | Port the cooperative TMA pipeline, including scale transfers. Eight compute warps and one producer, swizzle size 8. |
| [7M](kernel_nvfp4_7M_tile_traversal.py) | 6M | Change only the persistent scheduler’s swizzle size from 8 to 1. |
| [8M](kernel_nvfp4_8M_local_tma.py) | 7M | Issue the four loads with explicit CTA-local TMA, keeping swizzle size 1. |

The Ampere-style CTA grid and Modern persistent scheduler implement tile order differently. Group size 1 removes grouping in each scheduler. The experiment compares that intervention within each branch, rather than requiring identical physical launch schedules.

On GB10 with the CUDA toolkit installed at /usr/local/cuda:

```sh
uv sync --locked
uv run --locked python evaluate_worklog.py --rounds 5 --output ../../artifacts/nvfp4-matched
uv run --locked python tools/profile_kernel_resources.py --output ../../artifacts/nvfp4-resources
```

The evaluator shuffles kernels and cuBLAS in each round, checks correctness, and records commands, source hashes, and raw process timings. Each process runs 100 conditioning launches, 20 warmups and 50 timed launches. Custom kernels check all output values against a BF16-rounded FP32 reference on seeded random inputs; timing uses all-one values and scales for both custom kernels and cuBLAS. Resource profiling runs separately.

## Results and size sweep

The [4096³ results](validation/2026-11-25-matched/README.md) summarize the five shuffled benchmark rounds. To reproduce the separate rasterization comparison in Appendix 1 of the upcoming NVFP4 article, run the following from this directory.

```sh
for size in 4096 6144 8192; do
    uv run --locked python benchmark_tile_order.py "$size"
done
```
