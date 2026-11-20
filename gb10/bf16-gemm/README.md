# GB10 BF16 GEMM worklog

Self-contained uv project for the GB10 BF16 GEMM kernels and benchmarks.

The current contract is `C[M,N] = A[M,K] @ B[N,K].T`: **K-major A/B**
(strides `(K,1)`), **N-major C** (strides `(N,1)`), BF16 inputs/output and
FP32 accumulation. All published rows use 4096³, including the upstream
Ampere and cuBLAS controls.

## Kernel map

Active kernel filenames start with `kernel_bf16_` to distinguish them from
the NVFP4 checkpoints. Each branch uses the same progression: input pipeline, register pipeline, output staging.

| Kernel | Source | Retained idea |
|---|---|---|
| 0 | [kernel_bf16_0_scalar_fma.py](kernel_bf16_0_scalar_fma.py) | One scalar dot product per thread |
| 1 | [kernel_bf16_1_warp_mma.py](kernel_bf16_1_warp_mma.py) | Coalesced K-tile loads, shared reuse and synchronous Tensor Core MMA |
| 2 | [kernel_bf16_2_l2_scheduler.py](kernel_bf16_2_l2_scheduler.py) | Add L2-aware group-N4 block ordering |
| 3A | [kernel_bf16_3A_cpasync.py](kernel_bf16_3A_cpasync.py) | Cooperative cp.async pipeline |
| 4A | [kernel_bf16_4A_register_pipeline.py](kernel_bf16_4A_register_pipeline.py) | Add rolling register prefetch; keep the shared layout fixed |
| 5A | [kernel_bf16_5A_readable_stmatrix.py](kernel_bf16_5A_readable_stmatrix.py) | Add the stmatrix epilogue |
| 3M | [kernel_bf16_3M_tma_persistent.py](kernel_bf16_3M_tma_persistent.py) | TMA producer, eight compute warps, four stages and 48 persistent workers |
| 4M | [kernel_bf16_4M_register_pipeline.py](kernel_bf16_4M_register_pipeline.py) | Add rolling register prefetch; keep direct output |
| 5M | [kernel_bf16_5M_tma_tuned.py](kernel_bf16_5M_tma_tuned.py) | Add stmatrix and two-slot TMA output staging |

Kernel 2 has no asynchronous copies, dedicated producer or persistence.
Both branches start there. The numbered sources expose their complete
implementations. Input preparation lives in `problem.py`; correctness and
CUDA-event timing run through `benchmark.py`. Per-file commands remain as
thin compatibility entry points.
The separate `kernel_bf16_cheating.py` replaces the pinned Ampere kernel's epilogue
with 128×32 stmatrix subtiles. Its five-round comparison
uses the same common cuBLAS mean as every numbered kernel. The tuning sweep is reproducible with
`uv run --locked python tools/tune_cheating_epilogue.py --output results/cheating-tuning`;
its screening numbers are not substituted for validation averages.

Kernel 0 launches `(M, ceil_div(N,256), 1)` CTAs covering 1×256 output
positions. Lanes advance along N: C stores are coalesced, B loads are strided,
and A is shared across the warp. Kernel 1 introduces cooperative copies
along the contiguous K dimension. Each copy moves eight BF16 values.

## Reproduce the comparison

Requires Linux, Python 3.12, NVIDIA GB10, a CUDA 13-compatible driver, and
CUDA 13 `nvcc` for the cuBLAS control. Dependencies are pinned in `uv.lock`.

```bash
uv sync --locked
mkdir -p results
nvcc -O3 -std=c++17 -arch=sm_121a benchmarks/cublaslt_bf16_bench.cu -lcublasLt -lcublas -o results/cublaslt_bf16_bench
uv run --locked python evaluate_worklog.py --rounds 5 --size 4096 \
  --kernels 0 1 2 3A 4A 5A 3M 4M 5M ampere cublaslt cheating \
  --output results/bf16-five-rounds
```

This command compares nine numbered kernels, upstream Ampere, the Cheating
Kernel and cuBLAS. It runs five fresh processes per row, serially, with correctness
checks, 20 warmups and 50 timed launches per process. Order is shuffled
reproducibly between rounds. The published value is the arithmetic mean of
five throughputs, and every percentage uses the same measured cuBLAS mean.
Kernel 0 is included. Compilation, allocation, reference computation and
profiling are outside the timed intervals. GPU clocks are unlocked.
