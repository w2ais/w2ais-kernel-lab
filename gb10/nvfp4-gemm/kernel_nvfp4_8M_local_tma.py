# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from references import sm120_helpers as sm120_utils
from references import scale_layout as blockscaled_utils
from references.tma_load import load_local

TILE = (128, 128, 128)
AB_STAGES = 4
EPILOGUE_STAGES = 2
L2_GROUP = 1
RASTER_ALONG_M = True


class BF16DerivedNVFP4:
    """Direct port of Part 1 Kernel 5M; see docs/bf16_to_nvfp4.patch.

    Keep its cooperative mainloop, persistent scheduler and epilogue lifecycle.
    Kernel 8M retains the group-one traversal and uses CTA-local TMA loads.
    """

    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
    ):
        self.acc_dtype = acc_dtype
        self.sf_vec_size = 16
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.tiled_mma = None

        self.occupancy = 1
        self.atom_layout = (2, 4, 1)
        self.num_mma_warps = (
            self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        )
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        self.mma_inst_mnk = (16, 8, 64)
        op = cute.nvgpu.warp.MmaMXF4NVF4Op(self.a_dtype, self.acc_dtype, self.sf_dtype)
        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, False
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=False
        )

        self.epi_tile = (64, 64)

        # Budget the four packed-input stages and two output stages.
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

        self.sfa_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfa(
            self.tiled_mma, self.tile_shape_mnk, self.sf_vec_size, self.ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma, self.tile_shape_mnk, self.sf_vec_size, self.ab_stage
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        assert self.a_dtype == self.b_dtype == cutlass.Float4E2M1FN
        assert self.sf_dtype == sfb.element_type == cutlass.Float8E4M3FN
        assert self.c_dtype == cutlass.BFloat16

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
        )

        sfa_tensor = cute.make_tensor(
            sfa.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size),
        )
        sfb_tensor = cute.make_tensor(
            sfb.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size),
        )
        tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            internal_type=cutlass.Int16,
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):

        #  Get cta/warp/thread idx
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #  Prefetch Tma desc
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfb)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        tma_copy_bytes += cute.size_in_bytes(
            self.sf_dtype, cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        )
        tma_copy_bytes += cute.size_in_bytes(
            self.sf_dtype, cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        )

        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each compute warp contributes one consumer arrival.
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

        #  Generate smem tensor A/B
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        #  Local_tile partition global tensors
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            (None, None, None),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            (None, None, None),
        )

        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        #  Partition global tensor for TiledMMA_A/B/C
        thr_mma = tiled_mma.get_slice(tidx)

        #  Partition shared tensor for TMA load A/B
        #  TMA load A partition_S/D
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA load B partition_S/D
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA_mkl, 0, 2),
        )
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_nkl, 0, 2),
        )
        tAsSFA, tAgSFA = cute.filter_zeros(tAsSFA), cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cute.filter_zeros(tBsSFB), cute.filter_zeros(tBgSFB)
        rSFA = sm120_utils.partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
        rSFB = sm120_utils.partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)
        rSFA = cute.group_modes(rSFA, 2, cute.rank(rSFA))
        # Four N warps leave (N repetitions, K slices) in one grouped mode.
        rSFB = cute.make_tensor(
            rSFB.iterator,
            cute.make_layout(
                (rSFB.shape[0], *rSFB.shape[1]),
                stride=(rSFB.stride[0], *rSFB.stride[1]),
            ),
        )

        #  Make frangments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # All CTA threads observe initialized pipeline barriers.
        pipeline.sync(barrier_id=1)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Create the tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Create the pipeline states for producer and consumer
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            k_tile_slice_count = cute.size(tCrA, mode=[2])

            # Copy Atom A/B retiling for TMA load A/B
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)

            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)

            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            scale_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.sf_dtype
            )
            copy_sfa = cute.make_tiled_copy(
                scale_copy_atom,
                sm120_utils.get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            copy_sfb = cute.make_tiled_copy(
                scale_copy_atom,
                sm120_utils.get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            lane_sfa = copy_sfa.get_slice(tidx)
            lane_sfb = copy_sfb.get_slice(tidx)
            shared_sfa = cute.filter_zeros(lane_sfa.partition_S(sSFA))
            shared_sfb = cute.filter_zeros(lane_sfb.partition_S(sSFB))
            register_sfa = cute.filter_zeros(lane_sfa.retile(rSFA))
            register_sfb = cute.filter_zeros(lane_sfb.retile(rSFB))

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                # Clear the accumulator
                accumulators.fill(0.0)

                #  Pipelined MAINLOOP

                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                #  Wait for TMA copies to complete
                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsA_p: (MMA, (4, MMA_N / 4), MMA_K)
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )
                cute.copy(
                    copy_sfa,
                    shared_sfa[None, None, 0, mainloop_consumer_state.index],
                    register_sfa[None, None, 0],
                )
                cute.copy(
                    copy_sfb,
                    shared_sfb[None, None, 0, mainloop_consumer_state.index],
                    register_sfb[None, None, 0],
                )

                for k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    # unroll the loop
                    for k_tile_slice in cutlass.range_constexpr(k_tile_slice_count):
                        k_tile_slice_next = (
                            0 if k_tile_slice + 1 == k_tile_slice_count else k_tile_slice + 1
                        )

                        if k_tile_slice == k_tile_slice_count - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            # tCsA_p: (MMA, (4, MMA_M / 4), MMA_K), tCsA_p: (MMA, (4, MMA_N / 4), MMA_K)
                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        # Copy data from smem to tCrA/tCrB for the next k_tile_slice
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_tile_slice_next],
                            tCrA_copy_view[None, None, k_tile_slice_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_tile_slice_next],
                            tCrB_copy_view[None, None, k_tile_slice_next],
                        )
                        cute.copy(
                            copy_sfa,
                            shared_sfa[
                                None, None, k_tile_slice_next, mainloop_consumer_state.index
                            ],
                            register_sfa[None, None, k_tile_slice_next],
                        )
                        cute.copy(
                            copy_sfb,
                            shared_sfb[
                                None, None, k_tile_slice_next, mainloop_consumer_state.index
                            ],
                            register_sfb[None, None, k_tile_slice_next],
                        )
                        # Gemm of the current k_tile_slice
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            [
                                tCrA[None, None, k_tile_slice],
                                rSFA[None, None, k_tile_slice],
                            ],
                            [
                                tCrB[None, None, k_tile_slice],
                                rSFB[None, None, k_tile_slice],
                            ],
                            accumulators,
                        )
                # end of for loop
                # Hoist out last k_tile
                for k_tile_slice in cutlass.range_constexpr(k_tile_slice_count):
                    k_tile_slice_next = (
                        0 if k_tile_slice + 1 == k_tile_slice_count else k_tile_slice + 1
                    )

                    if k_tile_slice == k_tile_slice_count - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_tile_slice_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_tile_slice_next],
                            tCrA_copy_view[None, None, k_tile_slice_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_tile_slice_next],
                            tCrB_copy_view[None, None, k_tile_slice_next],
                        )
                        cute.copy(
                            copy_sfa,
                            shared_sfa[
                                None, None, k_tile_slice_next, mainloop_consumer_state.index
                            ],
                            register_sfa[None, None, k_tile_slice_next],
                        )
                        cute.copy(
                            copy_sfb,
                            shared_sfb[
                                None, None, k_tile_slice_next, mainloop_consumer_state.index
                            ],
                            register_sfb[None, None, k_tile_slice_next],
                        )
                    # Gemm of the current k_tile_slice
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        [tCrA[None, None, k_tile_slice], rSFA[None, None, k_tile_slice]],
                        [tCrB[None, None, k_tile_slice], rSFB[None, None, k_tile_slice]],
                        accumulators,
                    )

                #  EPILOG

                copy_atom_r2s = sm120_utils.sm120_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )

                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        2,
                    ),
                    self.c_dtype,
                )

                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                # (R2S, R2S_M, R2S_N, PIPE_D)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                # (R2S, R2S_M, R2S_N)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                # Allocate D registers.
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )

                epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
                epi_tile_shape = tcgc_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(1, epi_tile_shape[0])
                )

                # Initialize tma store pipeline
                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps * self.num_threads_per_warp,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=tma_store_producer_group,
                )

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    # Copy from accumulators to D registers
                    epi_m, epi_n = epi_tile_layout.get_hier_coord(epi_idx)
                    fragments_m = cute.size(tRS_rD, mode=[1])
                    fragments_n = cute.size(tRS_rD, mode=[2])
                    for frag_n in cutlass.range_constexpr(fragments_n):
                        for frag_m in cutlass.range_constexpr(fragments_m):
                            values = tRS_rAcc[
                                None,
                                epi_m * fragments_m + frag_m,
                                epi_n * fragments_n + frag_n,
                            ]
                            output = tRS_rD[None, frag_m, frag_n]
                            for v in cutlass.range_constexpr(cute.size(output)):
                                output[v] = values[v]

                    # Type conversion
                    tRS_rD_out = cute.make_rmem_tensor(
                        tRS_rD_layout.shape, self.c_dtype
                    )
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    # Warp 0 completed the preceding TMA-store acquire wait.
                    # Join all compute warps before any can overwrite the next slot.
                    self.epilog_sync_barrier.arrive_and_wait()
                    # Register to shared memory
                    epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    # barrier for sync
                    self.epilog_sync_barrier.arrive_and_wait()

                    # Get the global memory coordinate for the current epi tile.
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    # Copy from shared memory to global memory
                    if warp_idx == 0:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

                # Advance to the next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                tma_store_pipeline.producer_tail()
                # End of for k_tile loop
            # End of while loop
        # End of MMA warp group
        # Start of DMA warp group
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]

                tAgSFA_mkl = tAgSFA[None, tile_coord_mnl[0], None, tile_coord_mnl[2]]
                tBgSFB_nkl = tBgSFB[None, tile_coord_mnl[1], None, tile_coord_mnl[2]]
                mainloop_producer_state.reset_count()

                for k_tile in range(0, k_tile_cnt, 1, unroll=1):
                    #  Wait for A/B buffers to be empty before loading into them
                    #  Also sets the transaction barrier for the A/B buffers
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    #  Slice to global/shared memref to current k_tile
                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    #  TMA load A/B
                    load_local(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    load_local(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    load_local(
                        tma_atom_sfa,
                        tAgSFA_mkl[None, mainloop_producer_state.count],
                        tAsSFA[None, mainloop_producer_state.index],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    load_local(
                        tma_atom_sfb,
                        tBgSFB_nkl[None, mainloop_producer_state.count],
                        tBsSFB[None, mainloop_producer_state.index],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    # Mainloop pipeline's producer commit is a NOP
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # end of while loop

            # Wait A/B buffer empty
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk, a_dtype, b_dtype, epi_tile, c_dtype, smem_capacity, occupancy
    ):
        ab_bytes = (
            (tile_shape_mnk[0] + tile_shape_mnk[1])
            * tile_shape_mnk[2]
            * a_dtype.width
            // 8
            * AB_STAGES
        )
        c_bytes = cute.size(epi_tile) * c_dtype.width // 8 * EPILOGUE_STAGES
        scale_bytes = (
            (tile_shape_mnk[0] + tile_shape_mnk[1])
            * tile_shape_mnk[2]
            // 16
            * AB_STAGES
        )
        if ab_bytes + scale_bytes + c_bytes + 1024 > smem_capacity:
            raise ValueError(
                f"Shared-memory budget exceeded: AB={ab_bytes}, C={c_bytes}, capacity={smem_capacity}"
            )
        return AB_STAGES, EPILOGUE_STAGES

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: cute.Layout,
        b_dtype: type[cutlass.Numeric],
        b_layout: cute.Layout,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: cute.Layout,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )

        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size=L2_GROUP,
            raster_along_m=RASTER_ALONG_M,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params,
            max_active_clusters,
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        internal_type: Optional[Type[cutlass.Numeric]] = None,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            internal_type=internal_type,
        )
        return tma_atom, tma_tensor


def make_kernel():
    return BF16DerivedNVFP4(cutlass.Float32, TILE)
