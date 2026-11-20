"""Deliberately naive BF16 GEMM with K-major inputs and scalar CUDA-core FMA.

Each CUDA thread computes one output element with a scalar FP32 reduction.
The logical tensors are ``A[M,K]``, ``B[N,K]``, and ``C[M,N]`` with their
K dimension contiguous in A/B and N contiguous in C. Neighboring threads
advance along N; each CTA covers one row and 256 columns of C. B loads are
strided across the warp; cooperative K-tile loading begins in Kernel 1.
"""

from __future__ import annotations


from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute


THREADS = 256


@cute.kernel
def naive_bf16_gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    row = bidx
    col = bidy * THREADS + tidx

    if row < cute.size(mC, mode=[0]) and col < cute.size(mC, mode=[1]):
        accum = cutlass.Float32(0.0)
        for k in range(cute.size(mA, mode=[1])):
            a = mA[row, k].to(cutlass.Float32)
            b = mB[col, k].to(cutlass.Float32)
            accum += a * b
        mC[row, col] = accum.to(mC.element_type)


@cute.jit
def naive_bf16_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: cuda.CUstream,
):
    grid = (
        cute.size(mC, mode=[0]),
        cute.ceil_div(cute.size(mC, mode=[1]), THREADS),
        # Each thread reduces the full K dimension; no split-K grid.
        1,
    )
    naive_bf16_gemm_kernel(mA, mB, mC).launch(
        grid=grid,
        block=(THREADS, 1, 1),
        stream=stream,
    )


if __name__ == "__main__":
    from benchmark import main

    main("0", legacy=True)
