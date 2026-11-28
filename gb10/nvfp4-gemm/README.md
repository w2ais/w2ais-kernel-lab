# GB10 NVFP4 GEMM

The 4096³ comparison starts with packed E2M1 inputs, one UE4M3 scale per 16 values, FP32 accumulation and BF16 output. Both baselines stage the packed values and their scales. Checkpoint numbers follow the optimization order within each branch.

| Checkpoint | Parent | Change |
| --- | --- | --- |
| [6A](kernel_nvfp4_6A_scale_pipeline.py) | BF16 5A | Port the rolling pipeline, including shared scale stages. Eight compute warps and a four-column L2 swizzle. |
| [6A+](kernel_nvfp4_6A_plus_delayed_b.py) | 6A | Delay the next shared-to-register B load until after the current MMA to eliminate spills. |
| [7A](kernel_nvfp4_7A_tile_traversal.py) | 6A+ | Tune L2 Swizzling. Switch from four-column to two-row bands on the spill-free schedule. |
| [6M](kernel_nvfp4_6M_port.py) | BF16 5M | Port the cooperative TMA pipeline, including scale transfers. Eight compute warps and one producer, swizzle size 8. |
| [7M](kernel_nvfp4_7M_tile_traversal.py) | 6M | Tune L2 Swizzling. Switch the persistent scheduler from eight-column bands to natural M-first order. |
| [8M](kernel_nvfp4_8M_local_tma.py) | 7M | Issue the four loads with explicit CTA-local TMA, keeping swizzle size 1. |

The Ampere-style CTA grid and Modern persistent scheduler implement tile order differently. Setting the swizzle width to one gives M-first order in each scheduler. Each comparison changes only the tile order within its branch.

On GB10 with the CUDA toolkit installed at /usr/local/cuda:

```sh
uv sync --locked
uv run --locked python evaluate_worklog.py --rounds 5 --output ../../artifacts/nvfp4-matched
uv run --locked python tools/profile_kernel_resources.py --output ../../artifacts/nvfp4-resources
```

The evaluator shuffles kernels and cuBLAS in each round, checks correctness, and records commands, source hashes, and raw process timings. Each process runs 100 conditioning launches, 20 warmups and 50 timed launches. Custom kernels check all output values against a BF16-rounded FP32 reference on seeded random inputs; timing uses all-one values and scales for both custom kernels and cuBLAS. Resource profiling runs separately.

## Results and size sweep

The [4096³ results](validation/2026-11-28-matched/README.md) summarize the five shuffled benchmark rounds. To reproduce the separate rasterization comparison in Appendix 1 of the upcoming NVFP4 article, run the following from this directory.

```sh
for size in 4096 6144 8192; do
    uv run --locked python benchmark_tile_order.py "$size"
done
```

To try other L2 swizzles, pass `--group` for the band width in output columns or `--group-m` for the height in output rows. Kernel 7A defaults to two-row bands. Passing `--group 1` runs the natural M-first control. The [swizzle comparison](validation/2026-11-28-matched/README.md#other-l2-swizzles-on-the-spill-free-ampere-style-kernel) covers throughput, correctness and cache measurements.
