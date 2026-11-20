#!/usr/bin/env python3
"""Kernel 2: synchronous Tensor Cores plus group-N4 L2 tile ordering.

128x128x64, four compute warps, one shared stage. Global loads pass through
registers before shared memory; CTA barriers separate load, MMA and reuse.
No cp.async, TMA, dedicated producer warp, or persistent scheduling.
The copy and MMA schedules match Kernel 1; only tile traversal changes.
"""

from __future__ import annotations


from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warp
from helpers.l2_order import group_n_tile


CTA_TILER = (128, 128, 64)
MMA_INSTRUCTION_MNK = (16, 8, 16)
ATOM_LAYOUT_MNK = (2, 2, 1)
NUM_THREADS = 128
NUM_STAGES = 1
COPY_BITS = 128
SMEM_ALIGNMENT_BYTES = 1024

SOURCE_METADATA = {
    "variant": "sync_tensor_core_group_n4",
    "contract": "A[M,K] @ B[N,K].T -> C[M,N]",
    "layouts": "K-major A/B, N-major C",
    "input_dtype": "bfloat16",
    "accumulator_dtype": "float32",
    "output_dtype": "bfloat16",
    "cta_tile_mnk": CTA_TILER,
    "mma_instruction_mnk": MMA_INSTRUCTION_MNK,
    "atom_layout_mnk": ATOM_LAYOUT_MNK,
    "threads": NUM_THREADS,
    "stages": NUM_STAGES,
    "gmem_to_smem": "synchronous global -> registers -> shared",
    "smem_to_register": "ldmatrix.x4",
    "epilogue": "direct",
    "raster": "group-n4",
}


@cute.kernel
def tensor_core_sync_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_copy_s2r_A: cute.TiledCopy,
    tiled_copy_s2r_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    m_tiles = cute.ceil_div(cute.size(mC, mode=[0]), CTA_TILER[0])
    n_tiles = cute.ceil_div(cute.size(mC, mode=[1]), CTA_TILER[1])
    tile_m, tile_n = group_n_tile(bidx, m_tiles, n_tiles, 4)

    cta_coord = (tile_m, tile_n, None)
    gA = cute.local_tile(mA, CTA_TILER, cta_coord, proj=(1, None, 1))
    gB = cute.local_tile(mB, CTA_TILER, cta_coord, proj=(None, 1, 1))
    gC = cute.local_tile(mC, CTA_TILER, cta_coord, proj=(1, 1, None))

    smem = utils.SmemAllocator()
    sA = smem.allocate_tensor(
        mA.element_type,
        sA_layout.outer,
        byte_alignment=SMEM_ALIGNMENT_BYTES,
        swizzle=sA_layout.inner,
    )
    sB = smem.allocate_tensor(
        mB.element_type,
        sB_layout.outer,
        byte_alignment=SMEM_ALIGNMENT_BYTES,
        swizzle=sB_layout.inner,
    )

    # All four warps cooperatively load A and B through registers.
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)

    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0.0)

    thr_copy_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
    tCsA_copy = thr_copy_s2r_A.partition_S(sA)
    tCrA_copy = thr_copy_s2r_A.retile(tCrA)
    thr_copy_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
    tCsB_copy = thr_copy_s2r_B.partition_S(sB)
    tCrB_copy = thr_copy_s2r_B.retile(tCrB)

    k_tile_count = cute.size(tAgA, mode=[3])
    k_tile_slice_count = cute.size(tCrA, mode=[2])

    # The register buffers make the global -> register -> shared path explicit.
    rA = cute.make_fragment_like(tAsA[None, None, None, 0])
    rB = cute.make_fragment_like(tBsB[None, None, None, 0])
    for k_tile in range(k_tile_count):
        cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile], rA)
        cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile], rB)
        cute.copy(tiled_copy_A, rA, tAsA[None, None, None, 0])
        cute.copy(tiled_copy_B, rB, tBsB[None, None, None, 0])
        cute.arch.sync_threads()
        for k_tile_slice in range(k_tile_slice_count, unroll_full=True):
            cute.copy(
                tiled_copy_s2r_A,
                tCsA_copy[None, None, k_tile_slice, 0],
                tCrA_copy[None, None, k_tile_slice],
            )
            cute.copy(
                tiled_copy_s2r_B,
                tCsB_copy[None, None, k_tile_slice, 0],
                tCrB_copy[None, None, k_tile_slice],
            )
            # Per-thread shapes: MMA values, then M/N repetition counts.
            cute.gemm(
                tiled_mma,
                # D: (MMA, M, N) = ((2, 2), 4, 8); 128 FP32 values.
                tCrC,
                # A: (MMA, M) = ((2, 2, 2), 4); 32 BF16 values.
                tCrA[None, None, k_tile_slice],
                # B: (MMA, N) = ((2, 2), 8); 32 BF16 values.
                tCrB[None, None, k_tile_slice],
                # C: ((2, 2), 4, 8); same accumulator as D, updated in place.
                tCrC,
            )
        cute.arch.sync_threads()

    for i in range(cute.size(tCrC), unroll_full=True):
        tCgC[i] = tCrC[i].to(cutlass.BFloat16)


@cute.jit
def tensor_core_sync_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: cuda.CUstream,
):
    a_layout = utils.LayoutEnum.from_tensor(mA)
    b_layout = utils.LayoutEnum.from_tensor(mB)

    sA_layout = sm90_utils.make_smem_layout_a(
        a_layout, CTA_TILER, mA.element_type, NUM_STAGES
    )
    sB_layout = sm90_utils.make_smem_layout_b(
        b_layout, CTA_TILER, mB.element_type, NUM_STAGES
    )

    copy_atom_A = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        mA.element_type,
        num_bits_per_copy=COPY_BITS,
    )
    copy_atom_B = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        mB.element_type,
        num_bits_per_copy=COPY_BITS,
    )
    # Each lane copies eight consecutive BF16 values along K (128 bits).
    # Eight lanes fill one K row; NUM_THREADS / 8 rows are copied at a time.
    tiled_copy_A = cute.make_tiled_copy_tv(
        copy_atom_A,
        cute.make_layout((NUM_THREADS // 8, 8), stride=(8, 1)),
        cute.make_layout((1, 8)),
    )
    tiled_copy_B = cute.make_tiled_copy_tv(
        copy_atom_B,
        cute.make_layout((NUM_THREADS // 8, 8), stride=(8, 1)),
        cute.make_layout((1, 8)),
    )

    mma_op = warp.MmaF16BF16Op(mA.element_type, cutlass.Float32, MMA_INSTRUCTION_MNK)
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        atom_layout_mnk=cute.make_layout(ATOM_LAYOUT_MNK),
        permutation_mnk=(
            ATOM_LAYOUT_MNK[0] * MMA_INSTRUCTION_MNK[0],
            ATOM_LAYOUT_MNK[1] * MMA_INSTRUCTION_MNK[1] * 2,
            ATOM_LAYOUT_MNK[2] * MMA_INSTRUCTION_MNK[2],
        ),
    )
    tiled_copy_s2r_A = cute.make_tiled_copy_A(
        cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(a_layout.is_m_major_a(), 4),
            mA.element_type,
        ),
        tiled_mma,
    )
    tiled_copy_s2r_B = cute.make_tiled_copy_B(
        cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(b_layout.is_n_major_b(), 4),
            mB.element_type,
        ),
        tiled_mma,
    )

    tile_grid = cute.ceil_div(mC.shape, CTA_TILER[:2])
    grid = (tile_grid[0] * tile_grid[1], 1, 1)

    tensor_core_sync_kernel(
        mA,
        mB,
        mC,
        sA_layout,
        sB_layout,
        tiled_copy_A,
        tiled_copy_B,
        tiled_copy_s2r_A,
        tiled_copy_s2r_B,
        tiled_mma,
    ).launch(grid=grid, block=[NUM_THREADS, 1, 1], stream=stream)


if __name__ == "__main__":
    from benchmark import main

    main("2", legacy=True)
