#!/usr/bin/env python3
"""Kernel 4M: Kernel 3M plus a rolling shared-to-register pipeline.

Explicit checkpoint source; shared formatting makes changes easy to compare.
Input preparation and timing live in problem.py and benchmark.py.
"""

from __future__ import annotations


from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warp


CTA_TILER = (128, 128, 32)
MMA_INSTRUCTION_MNK = (16, 8, 16)
NUM_MMA_WARPS = 8
NUM_STAGES = 4
L2_GROUP = 8
RASTER_ALONG_M = True
SMEM_ALIGNMENT_BYTES = 1024
LOAD_REGISTER_REQUIREMENT = 40
MMA_REGISTER_REQUIREMENT = 232

SOURCE_METADATA = {
    "variant": "matched_4M_tma_group_m8_128x128x32_4stage_direct_epilogue",
    "derived_from": "cutedsl-examples/fp16bf16_gemm_marimo.py",
    "contract": "A[M,K] @ B[N,K].T -> C[M,N]",
    "layouts": "K-major A/B, N-major C",
    "input_dtype": "bfloat16",
    "accumulator_dtype": "float32",
    "output_dtype": "bfloat16",
    "alpha": 1.0,
    "beta": 0.0,
    "cta_tile_mnk": CTA_TILER,
    "mma_instruction_mnk": MMA_INSTRUCTION_MNK,
    "mma_warps": NUM_MMA_WARPS,
    "producer_warps": 1,
    "threads": 32 * (NUM_MMA_WARPS + 1),
    "stages": NUM_STAGES,
    "gmem_to_smem": "TMA bulk tensor",
    "smem_to_register": "ldmatrix.x4",
    "register_pipeline": "rolling next-fragment prefetch across K-tiles",
    "epilogue": "direct scalar FP32 accumulator to BF16 C; no C read",
}


@cute.kernel
def bf16_tma_baseline_kernel(
    tma_atom_A: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_B: cute.CopyAtom,
    mB: cute.Tensor,
    mC: cute.Tensor,
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    tiled_copy_s2r_A: cute.TiledCopy,
    tiled_copy_s2r_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    tile_sched_params: utils.PersistentTileSchedulerParams,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # TMA tensors keep explicit M/N tile coordinates until the persistent
    # scheduler assigns an output tile to this CTA.
    gA = cute.local_tile(mA, (CTA_TILER[0], CTA_TILER[2]), (None, None))
    gB = cute.local_tile(mB, (CTA_TILER[1], CTA_TILER[2]), (None, None))
    gC = cute.local_tile(mC, CTA_TILER[:2], (None, None))

    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_A)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_B)

    smem = utils.SmemAllocator()
    pipeline_barriers = smem.allocate_array(
        cutlass.Int64,
        NUM_STAGES * 2,
        byte_alignment=8,
    )
    sA = smem.allocate_tensor(
        tma_atom_A.value_type,
        sA_layout.outer,
        byte_alignment=SMEM_ALIGNMENT_BYTES,
        swizzle=sA_layout.inner,
    )
    sB = smem.allocate_tensor(
        tma_atom_B.value_type,
        sB_layout.outer,
        byte_alignment=SMEM_ALIGNMENT_BYTES,
        swizzle=sB_layout.inner,
    )

    one_stage_sA = cute.slice_(sA_layout, (None, None, 0))
    one_stage_sB = cute.slice_(sB_layout, (None, None, 0))
    tma_copy_bytes = cute.size_in_bytes(
        tma_atom_A.value_type,
        one_stage_sA,
    ) + cute.size_in_bytes(
        tma_atom_B.value_type,
        one_stage_sB,
    )

    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        num_stages=NUM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            NUM_MMA_WARPS,
        ),
        tx_count=tma_copy_bytes,
        barrier_storage=pipeline_barriers,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )

    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_A,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA, 0, 2),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_B,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 2),
        cute.group_modes(gB, 0, 2),
    )

    k_tile_count = cute.size(gA, mode=[3])
    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params,
        cute.arch.block_idx(),
        cute.arch.grid_dim(),
    )
    work_tile = tile_sched.initial_work_tile_info()

    # PipelineTmaAsync initializes its mbarriers collectively.  All threads
    # converge once more before producer and consumer control flow diverges.
    pipeline.sync(barrier_id=1)

    if warp_idx < NUM_MMA_WARPS:
        cute.arch.setmaxregister_increase(MMA_REGISTER_REQUIREMENT)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        thr_copy_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
        tCsA_copy_view = thr_copy_s2r_A.partition_S(sA)
        tCrA_copy_view = thr_copy_s2r_A.retile(tCrA)
        thr_copy_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsB_copy_view = thr_copy_s2r_B.partition_S(sB)
        tCrB_copy_view = thr_copy_s2r_B.retile(tCrB)
        k_tile_slice_count = cute.size(tCrA, mode=[2])

        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            NUM_STAGES,
        )
        while work_tile.is_valid_tile:
            tile_coord = work_tile.tile_idx
            gC_tile = gC[None, None, tile_coord[0], tile_coord[1]]
            tCgC = thr_mma.partition_C(gC_tile)
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            consumer_state.reset_count()

            peek_ab_full_status = cutlass.Boolean(1)
            if consumer_state.count < k_tile_count:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    consumer_state
                )

            #  Wait for TMA copies to complete
            mainloop_pipeline.consumer_wait(consumer_state, peek_ab_full_status)
            # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsA_p: (MMA, (4, MMA_N / 4), MMA_K)
            tCsA_p = tCsA_copy_view[None, None, None, consumer_state.index]
            tCsB_p = tCsB_copy_view[None, None, None, consumer_state.index]
            cute.copy(
                tiled_copy_s2r_A,
                tCsA_p[None, None, 0],
                tCrA_copy_view[None, None, 0],
            )
            cute.copy(
                tiled_copy_s2r_B,
                tCsB_p[None, None, 0],
                tCrB_copy_view[None, None, 0],
            )

            for k_tile in range(0, k_tile_count - 1, 1, unroll=1):
                # unroll the loop
                for k_tile_slice in cutlass.range_constexpr(k_tile_slice_count):
                    k_tile_slice_next = (
                        0 if k_tile_slice + 1 == k_tile_slice_count else k_tile_slice + 1
                    )

                    if k_tile_slice == k_tile_slice_count - 1:
                        # All compute warps finish reading before the slot is reused.
                        pipeline.NamedBarrier(
                            barrier_id=2, num_threads=NUM_MMA_WARPS * 32
                        ).arrive_and_wait()
                        mainloop_pipeline.consumer_release(consumer_state)
                        consumer_state.advance()

                        peek_ab_full_status = cutlass.Boolean(1)
                        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                            consumer_state
                        )

                        # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsA_p: (MMA, (4, MMA_N / 4), MMA_K)
                        tCsA_p = tCsA_copy_view[None, None, None, consumer_state.index]
                        tCsB_p = tCsB_copy_view[None, None, None, consumer_state.index]
                        mainloop_pipeline.consumer_wait(
                            consumer_state, peek_ab_full_status
                        )

                    # Copy data from smem to tCrA/tCrB for the next k_tile_slice
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_p[None, None, k_tile_slice_next],
                        tCrA_copy_view[None, None, k_tile_slice_next],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_p[None, None, k_tile_slice_next],
                        tCrB_copy_view[None, None, k_tile_slice_next],
                    )
                    # Gemm of the current k_tile_slice
                    cute.gemm(
                        tiled_mma,
                        tCrC,
                        tCrA[None, None, k_tile_slice],
                        tCrB[None, None, k_tile_slice],
                        tCrC,
                    )
            # end of for loop
            # Hoist out last k_tile
            for k_tile_slice in cutlass.range_constexpr(k_tile_slice_count):
                k_tile_slice_next = (
                    0 if k_tile_slice + 1 == k_tile_slice_count else k_tile_slice + 1
                )

                if k_tile_slice == k_tile_slice_count - 1:
                    # All compute warps finish reading before the slot is reused.
                    pipeline.NamedBarrier(
                        barrier_id=2, num_threads=NUM_MMA_WARPS * 32
                    ).arrive_and_wait()
                    mainloop_pipeline.consumer_release(consumer_state)
                    consumer_state.advance()

                if k_tile_slice_next > 0:
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_p[None, None, k_tile_slice_next],
                        tCrA_copy_view[None, None, k_tile_slice_next],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_p[None, None, k_tile_slice_next],
                        tCrB_copy_view[None, None, k_tile_slice_next],
                    )
                # Gemm of the current k_tile_slice
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_tile_slice],
                    tCrB[None, None, k_tile_slice],
                    tCrC,
                )

            # alpha=1 and beta=0 are fixed by this variant.  In particular,
            # the destination C tensor is never loaded by the epilogue.
            for i in range(cute.size(tCrC), unroll_full=True):
                tCgC[i] = tCrC[i].to(cutlass.BFloat16)

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    elif warp_idx == NUM_MMA_WARPS:
        cute.arch.setmaxregister_decrease(LOAD_REGISTER_REQUIREMENT)

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            NUM_STAGES,
        )
        while work_tile.is_valid_tile:
            tile_coord = work_tile.tile_idx
            producer_state.reset_count()

            for k_tile in range(k_tile_count):
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tma_atom_A,
                    tAgA[None, tile_coord[0], k_tile],
                    tAsA[None, producer_state.index],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                )
                cute.copy(
                    tma_atom_B,
                    tBgB[None, tile_coord[1], k_tile],
                    tBsB[None, producer_state.index],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                )
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        mainloop_pipeline.producer_tail(producer_state)


@cute.jit
def bf16_tma_baseline(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    a_layout = utils.LayoutEnum.from_tensor(mA)
    b_layout = utils.LayoutEnum.from_tensor(mB)

    sA_layout = sm90_utils.make_smem_layout_a(
        a_layout,
        CTA_TILER,
        mA.element_type,
        NUM_STAGES,
    )
    sB_layout = sm90_utils.make_smem_layout_b(
        b_layout,
        CTA_TILER,
        mB.element_type,
        NUM_STAGES,
    )

    tma_atom_A, tma_tensor_A = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mA,
        cute.slice_(sA_layout, (None, None, 0)),
        (CTA_TILER[0], CTA_TILER[2]),
    )
    tma_atom_B, tma_tensor_B = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mB,
        cute.slice_(sB_layout, (None, None, 0)),
        (CTA_TILER[1], CTA_TILER[2]),
    )

    mma_op = warp.MmaF16BF16Op(
        mA.element_type,
        cutlass.Float32,
        MMA_INSTRUCTION_MNK,
    )
    atom_layout_mnk = (2, NUM_MMA_WARPS // 2, 1)
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        atom_layout_mnk=cute.make_layout(atom_layout_mnk),
        permutation_mnk=(
            atom_layout_mnk[0] * MMA_INSTRUCTION_MNK[0],
            atom_layout_mnk[1] * MMA_INSTRUCTION_MNK[1] * 2,
            atom_layout_mnk[2] * MMA_INSTRUCTION_MNK[2],
        ),
    )

    ldmatrix_atom_A = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(a_layout.is_m_major_a(), 4),
        mA.element_type,
    )
    ldmatrix_atom_B = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(b_layout.is_n_major_b(), 4),
        mB.element_type,
    )
    tiled_copy_s2r_A = cute.make_tiled_copy_A(ldmatrix_atom_A, tiled_mma)
    tiled_copy_s2r_B = cute.make_tiled_copy_B(ldmatrix_atom_B, tiled_mma)

    problem_shape_ntile_mnl = (
        *cute.ceil_div(mC.shape, CTA_TILER[:2]),
        1,
    )
    tile_sched_params = utils.PersistentTileSchedulerParams(
        problem_shape_ntile_mnl,
        (1, 1, 1),
        swizzle_size=L2_GROUP,
        raster_along_m=RASTER_ALONG_M,
    )
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params,
        max_active_clusters,
    )

    bf16_tma_baseline_kernel(
        tma_atom_A,
        tma_tensor_A,
        tma_atom_B,
        tma_tensor_B,
        mC,
        sA_layout,
        sB_layout,
        tiled_copy_s2r_A,
        tiled_copy_s2r_B,
        tiled_mma,
        tile_sched_params,
    ).launch(
        grid=grid,
        block=[32 * (NUM_MMA_WARPS + 1), 1, 1],
        cluster=[1, 1, 1],
        stream=stream,
    )


if __name__ == "__main__":
    from benchmark import main

    main("4M", legacy=True)
