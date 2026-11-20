#!/usr/bin/env python3
"""Kernel 5A: Kernel 4A plus a 64x64 stmatrix output epilogue.

Explicit checkpoint source; shared formatting makes changes easy to compare.
Input preparation and timing live in problem.py and benchmark.py.
"""

from __future__ import annotations


from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warp


CTA_TILER = (128, 256, 32)
MMA_INSTRUCTION_MNK = (16, 8, 16)
ATOM_LAYOUT_MNK = (2, 4, 1)
NUM_THREADS = 256
NUM_STAGES = 3
COPY_BITS = 128
SMEM_ALIGNMENT_BYTES = 1024
EPI_TILE = (64, 64)
EPI_COUNT_M = CTA_TILER[0] // EPI_TILE[0]
EPI_COUNT_N = CTA_TILER[1] // EPI_TILE[1]
EPI_COUNT = EPI_COUNT_M * EPI_COUNT_N

SOURCE_METADATA = {
    "variant": "matched_5A_register_pipeline_stmatrix",
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
    "epilogue": "stmatrix",
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
    sC_layout: cute.ComposedLayout,
    tiled_copy_r2s_C: cute.TiledCopy,
    tiled_copy_stmatrix_s2g_C: cute.TiledCopy,
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

    # CUTLASS's Ampere tensorop loop combines a three-stage
    # gmem->smem pipeline with a two-fragment smem->register pipeline.  Prime
    # stages zero and one, then make tile zero visible before entering the
    # steady state.  K=32 is two m16n8k16 register slices.
    cute.copy(
        tiled_copy_A,
        tAgA[None, None, None, 0],
        tAsA[None, None, None, 0],
    )
    cute.copy(
        tiled_copy_B,
        tBgB[None, None, None, 0],
        tBsB[None, None, None, 0],
    )
    cute.arch.cp_async_commit_group()

    if k_tile_count > 1:
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, 1],
            tAsA[None, None, None, 1],
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, 1],
            tBsB[None, None, None, 1],
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(NUM_STAGES - 2)
    else:
        cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    smem_pipe_read = 0
    smem_pipe_write = NUM_STAGES - 1
    tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
    tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]

    # Register-pipeline prologue.  Fragment zero is consumed by the first MMA;
    # fragment one is filled at the head of the first steady-state iteration.
    cute.copy(
        tiled_copy_s2r_A,
        tCsA_p[None, None, 0],
        tCrA_copy[None, None, 0],
    )
    cute.copy(
        tiled_copy_s2r_B,
        tCsB_p[None, None, 0],
        tCrB_copy[None, None, 0],
    )

    # Every iteration computes one current tile and leaves slice zero of the
    # following tile in the just-consumed slice-zero registers.  The last tile
    # is peeled below, so this loop never performs a useless next-tile load.
    for k_tile in range(k_tile_count - 1):
        future_tile = k_tile + NUM_STAGES - 1
        has_future_copy = k_tile + NUM_STAGES - 1 < k_tile_count

        for k_tile_slice in cutlass.range(k_tile_slice_count, unroll_full=True):
            if k_tile_slice == k_tile_slice_count - 1:
                # The previous slice advanced the ring.  Retire the tile we
                # are about to load while leaving the newer future tile in
                # flight.  On the penultimate tile there is no future group,
                # so the single remaining group must be drained completely.
                tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
                tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]
                if has_future_copy:
                    cute.arch.cp_async_wait_group(NUM_STAGES - 2)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

            # Produce register fragment i+1 before consuming fragment i.  For
            # the final slice this wraps to fragment zero of the next K-tile,
            # whose old value was consumed by slice-zero MMA below.
            k_tile_slice_next = (k_tile_slice + 1) % k_tile_slice_count
            cute.copy(
                tiled_copy_s2r_A,
                tCsA_p[None, None, k_tile_slice_next],
                tCrA_copy[None, None, k_tile_slice_next],
            )
            cute.copy(
                tiled_copy_s2r_B,
                tCsB_p[None, None, k_tile_slice_next],
                tCrB_copy[None, None, k_tile_slice_next],
            )

            # Match the CUTLASS instruction-order hint: issue future A,
            # compute from the current register fragment, then issue future B.
            if k_tile_slice == 0 and has_future_copy:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, future_tile],
                    tAsA[None, None, None, smem_pipe_write],
                )

            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, k_tile_slice],
                tCrB[None, None, k_tile_slice],
                tCrC,
            )

            if k_tile_slice == 0:
                if has_future_copy:
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, future_tile],
                        tBsB[None, None, None, smem_pipe_write],
                    )
                    cute.arch.cp_async_commit_group()

                # The wait/barrier in slice one makes the old read stage safe
                # before it becomes this producer's next write stage.
                smem_pipe_write = smem_pipe_read
                smem_pipe_read = smem_pipe_read + 1
                if smem_pipe_read == NUM_STAGES:
                    smem_pipe_read = 0

    # Peeling the final tile removes both the tail predicate from its compute
    # path and the official generic loop's empty cp.async commit / unused next
    # register load.  Fragment zero was prepared by the previous iteration (or
    # by the register prologue when K contains a single CTA tile).
    tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
    tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]
    cute.copy(
        tiled_copy_s2r_A,
        tCsA_p[None, None, 1],
        tCrA_copy[None, None, 1],
    )
    cute.copy(
        tiled_copy_s2r_B,
        tCsB_p[None, None, 1],
        tCrB_copy[None, None, 1],
    )
    cute.gemm(
        tiled_mma,
        tCrC,
        tCrA[None, None, 0],
        tCrB[None, None, 0],
        tCrC,
    )
    cute.gemm(
        tiled_mma,
        tCrC,
        tCrA[None, None, 1],
        tCrB[None, None, 1],
        tCrC,
    )

    cute.arch.cp_async_wait_group(0)
    # wait_group is per-thread.  The CTA barrier is required before the A/B
    # arena is repurposed as the staged C tile, otherwise another warp's last
    # redundant cp.async can overwrite a few freshly stored accumulator values.
    cute.arch.sync_threads()

    # Reuse the dead A/B arena for one swizzled 64x64 output subtile.
    sC_ptr = cute.recast_ptr(sA.iterator, sC_layout.inner, dtype=mC.element_type)
    sC = cute.make_tensor(sC_ptr, sC_layout.outer)
    thr_r2s = tiled_copy_r2s_C.get_slice(tidx)
    tRS_sC = thr_r2s.partition_D(sC)
    tRS_rAcc = tiled_copy_r2s_C.retile(tCrC)

    rD_shape = cute.shape(thr_r2s.partition_S(sC))
    rD_layout = cute.make_layout(rD_shape[:3])
    rD = cute.make_rmem_tensor(rD_layout.shape, cutlass.Float32)
    rD_out = cute.make_rmem_tensor(rD_layout.shape, cutlass.BFloat16)
    rD_size = cute.size(rD)

    thr_s2g = tiled_copy_stmatrix_s2g_C.get_slice(tidx)
    tCsC_vec = thr_s2g.partition_S(sC[None, None, 0])

    for epi_idx in cutlass.range_constexpr(EPI_COUNT):
        for epi_v in cutlass.range_constexpr(rD_size):
            rD[epi_v] = tRS_rAcc[epi_idx * rD_size + epi_v]
        rD_out.store(rD.load().to(cutlass.BFloat16))
        cute.copy(tiled_copy_r2s_C, rD_out, tRS_sC[(None, None, None, 0)])

        # stmatrix is warp-scoped; the vector drain consumes all warps' data.
        cute.arch.sync_threads()
        epi_m = epi_idx % EPI_COUNT_M
        epi_n = epi_idx // EPI_COUNT_M
        gC_epi = cute.local_tile(gC, EPI_TILE, (epi_m, epi_n))
        tCgC_vec = thr_s2g.partition_D(gC_epi)
        cute.copy(tiled_copy_stmatrix_s2g_C, tCsC_vec, tCgC_vec)
        cute.arch.sync_threads()


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

    c_layout = utils.LayoutEnum.ROW_MAJOR
    sC_layout = sm90_utils.make_smem_layout_epi(mC.element_type, c_layout, EPI_TILE, 1)

    # Compose the MMA C mapping with FP32->BF16 stmatrix stores.
    r2s_atom = sm90_utils.sm90_get_smem_store_op(
        c_layout,
        elem_ty_d=mC.element_type,
        elem_ty_acc=cutlass.Float32,
    )
    stmatrix_atom = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(False, 4), mC.element_type
    )
    copy_C_atom = cute.make_tiled_copy_C_atom(stmatrix_atom, tiled_mma)
    tiled_copy_r2s_C = cute.make_tiled_copy_S(r2s_atom, copy_C_atom)

    s2g_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        mC.element_type,
        num_bits_per_copy=COPY_BITS,
    )
    tiled_copy_stmatrix_s2g_C = cute.make_tiled_copy_tv(
        s2g_atom,
        cute.make_layout((32, 8), stride=(8, 1)),
        cute.make_layout((1, 8)),
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
        sC_layout,
        tiled_copy_r2s_C,
        tiled_copy_stmatrix_s2g_C,
    ).launch(grid=grid, block=[NUM_THREADS, 1, 1], stream=stream)


if __name__ == "__main__":
    from benchmark import main

    main("5A", legacy=True)
