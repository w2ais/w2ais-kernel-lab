#!/usr/bin/env python3
"""Kernel 3A: cp.async input pipeline with serial register loads.

Explicit checkpoint source; shared formatting makes changes easy to compare.
Input preparation and timing live in problem.py and benchmark.py.
"""

from __future__ import annotations


from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import warp


CTA_TILER = (128, 256, 32)
MMA_INSTRUCTION_MNK = (16, 8, 16)
ATOM_LAYOUT_MNK = (2, 4, 1)
NUM_THREADS = 256
NUM_STAGES = 3
COPY_BITS = 128
SMEM_ALIGNMENT_BYTES = 1024

SOURCE_METADATA = {
    "variant": "matched_3A_input_pipeline_direct_output",
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
    "gmem_to_smem": "128-bit cp.async",
    "smem_to_register": "ldmatrix.x4",
    "epilogue": "direct",
    "raster": "group-n4",
}


@cute.kernel
def gb10_cpasync_kernel(
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
    grid_x, _, _ = cute.arch.grid_dim()
    linear_cta = bidx + grid_x * bidy
    # Visit four neighboring N tiles before moving along M.
    group_span = 4 * m_tiles
    group_idx = linear_cta // group_span
    in_group = linear_cta - group_idx * group_span
    tile_n = group_idx * 4 + in_group % 4
    tile_m = in_group // 4

    cta_coord = (tile_m, tile_n, None)
    gA = cute.local_tile(mA, CTA_TILER, cta_coord, proj=(1, None, 1))
    gB = cute.local_tile(mB, CTA_TILER, cta_coord, proj=(None, 1, 1))
    gC = cute.local_tile(mC, CTA_TILER, cta_coord, proj=(1, 1, None))

    smem = utils.SmemAllocator()
    sA = smem.allocate_tensor(
        mA.element_type,
        sA_layout,
        byte_alignment=SMEM_ALIGNMENT_BYTES,
    )
    sB = smem.allocate_tensor(
        mB.element_type,
        sB_layout,
        byte_alignment=SMEM_ALIGNMENT_BYTES,
    )

    # All eight compute warps cooperatively execute the asynchronous global
    # loads.  A uses a 128x16 copy tile; B uses a 256x8 copy tile.  Repetition
    # modes cover the full K=32 stage.
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

    # Prime two groups.  For very small K, the fallback reload is deliberate:
    # it keeps the group schedule uniform without ever consuming that copy.
    prime_tile_0 = 0
    prime_tile_1 = 1 if k_tile_count > 1 else 0
    cute.copy(
        tiled_copy_A,
        tAgA[None, None, None, prime_tile_0],
        tAsA[None, None, None, 0],
    )
    cute.copy(
        tiled_copy_B,
        tBgB[None, None, None, prime_tile_0],
        tBsB[None, None, None, 0],
    )
    cute.arch.cp_async_commit_group()
    cute.copy(
        tiled_copy_A,
        tAgA[None, None, None, prime_tile_1],
        tAsA[None, None, None, 1],
    )
    cute.copy(
        tiled_copy_B,
        tBgB[None, None, None, prime_tile_1],
        tBsB[None, None, None, 1],
    )
    cute.arch.cp_async_commit_group()

    for k_tile in range(k_tile_count):
        read_stage = k_tile % NUM_STAGES
        write_stage = (k_tile + 2) % NUM_STAGES
        next_tile = k_tile + 2 if k_tile + 2 < k_tile_count else k_tile

        # Retire the current stage while leaving one newer group in flight.
        # This single CTA barrier also proves that all warps finished reading
        # the previous stage, so the newly freed ring slot can be overwritten.
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

        # Add the third group only after the wait.  It remains in flight while
        # ldmatrix and MMA consume the current stage.
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, next_tile],
            tAsA[None, None, None, write_stage],
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, next_tile],
            tBsB[None, None, None, write_stage],
        )
        cute.arch.cp_async_commit_group()

        # This checkpoint pipelines global loads only. Load the current
        # register fragments immediately before their MMA.
        for k_tile_slice in range(k_tile_slice_count, unroll_full=True):
            cute.copy(
                tiled_copy_s2r_A,
                tCsA_copy[None, None, k_tile_slice, read_stage],
                tCrA_copy[None, None, k_tile_slice],
            )
            cute.copy(
                tiled_copy_s2r_B,
                tCsB_copy[None, None, k_tile_slice, read_stage],
                tCrB_copy[None, None, k_tile_slice],
            )
            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, k_tile_slice],
                tCrB[None, None, k_tile_slice],
                tCrC,
            )

    cute.arch.cp_async_wait_group(0)
    # Direct output: output staging is introduced in Kernel 5A.
    for i in range(cute.size(tCrC), unroll_full=True):
        tCgC[i] = tCrC[i].to(cutlass.BFloat16)


@cute.jit
def gb10_cpasync_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: cuda.CUstream,
):
    a_layout = utils.LayoutEnum.from_tensor(mA)
    b_layout = utils.LayoutEnum.from_tensor(mB)

    # NVIDIA's Ampere K-major atom: eight rows of 32 contiguous BF16s.
    # XOR the row bits into the 16-byte sectors to distribute shared banks.
    smem_layout_atom = cute.make_composed_layout(
        cute.make_swizzle(2, 3, 3),
        0,
        cute.make_layout((8, 32), stride=(32, 1)),
    )
    sA_layout = cute.tile_to_shape(
        smem_layout_atom,
        (CTA_TILER[0], CTA_TILER[2], NUM_STAGES),
        (0, 1, 2),
    )
    sB_layout = cute.tile_to_shape(
        smem_layout_atom,
        (CTA_TILER[1], CTA_TILER[2], NUM_STAGES),
        (0, 1, 2),
    )

    copy_atom_A = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        mA.element_type,
        num_bits_per_copy=COPY_BITS,
    )
    copy_atom_B = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        mB.element_type,
        num_bits_per_copy=COPY_BITS,
    )
    # Each lane copies eight consecutive BF16 values along K (128 bits).
    # Neighboring lanes fill one K row before advancing along M or N.
    tiled_copy_A = cute.make_tiled_copy_tv(
        copy_atom_A,
        cute.make_layout((64, 4), stride=(4, 1)),
        cute.make_layout((1, 8)),
    )
    tiled_copy_B = cute.make_tiled_copy_tv(
        copy_atom_B,
        cute.make_layout((64, 4), stride=(4, 1)),
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
    total_ctas = tile_grid[0] * tile_grid[1]
    # Fixed physical grid from the published group-N4 configuration.
    # The runner requires a positive multiple of 128 logical CTAs.
    grid = (128, cute.ceil_div(total_ctas, 128), 1)

    gb10_cpasync_kernel(
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

    main("3A", legacy=True)
