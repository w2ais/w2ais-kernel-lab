"""NVFP4 port of Part 1's cooperative cp.async / rolling-register strategy.

Packed A/B use three shared stages. Scale factors join A/B in the asynchronous input ring. The epilogue reuses the A/B arena for stmatrix + vectors.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from references import sm120_helpers as sm120_utils
from references import scale_layout
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import warp
from cuda.bindings import driver as cuda

TILE = (128, 256, 128)
STAGES = 3
GROUP_N = 4
WARPS = 8


@cute.kernel
def nvfp4_cpasync_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mSFA: cute.Tensor,
    mSFB: cute.Tensor,
    mC: cute.Tensor,
    mma: cute.TiledMma,
    slA: cute.ComposedLayout,
    slB: cute.ComposedLayout,
    slC: cute.ComposedLayout,
    cp: cute.TiledCopy,
    ldA: cute.TiledCopy,
    ldB: cute.TiledCopy,
    r2s: cute.TiledCopy,
    s2g: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    m_tiles = cute.size(mC, mode=[0]) // TILE[0]
    n_tiles = cute.size(mC, mode=[1]) // TILE[1]
    group_id = bid // (GROUP_N * m_tiles)
    width = min(GROUP_N, n_tiles - group_id * GROUP_N)
    in_group = bid - group_id * GROUP_N * m_tiles
    bm = in_group // width
    bn = group_id * GROUP_N + in_group % width
    gA = cute.local_tile(mA, (TILE[0], TILE[2]), (bm, None, 0))
    gB = cute.local_tile(mB, (TILE[1], TILE[2]), (bn, None, 0))
    gSFA = cute.local_tile(mSFA, (TILE[0], TILE[2]), (bm, None, 0))
    gSFB = cute.local_tile(mSFB, (TILE[1], TILE[2]), (bn, None, 0))
    gC = cute.local_tile(mC, TILE[:2], (bm, bn, 0))
    smem = utils.SmemAllocator()
    sA = smem.allocate_tensor(
        mA.element_type, slA.outer, byte_alignment=1024, swizzle=slA.inner
    )
    sB = smem.allocate_tensor(
        mB.element_type, slB.outer, byte_alignment=1024, swizzle=slB.inner
    )
    tc = cp.get_slice(tidx)
    ga, sa = tc.partition_S(gA), tc.partition_D(sA)
    gb, sb = tc.partition_S(gB), tc.partition_D(sB)
    tiled_mma = mma
    thr_mma = mma.get_slice(tidx)
    tCgC = thr_mma.partition_C(gC)
    acc = cute.make_rmem_tensor(tCgC.shape, cutlass.Float32)
    acc.fill(0.0)
    # Copy each physical scale byte once per CTA, then share it across warps.
    sfla = scale_layout.sm120_make_smem_layout_sfa(mma, TILE, 16, STAGES)
    sflb = scale_layout.sm120_make_smem_layout_sfb(mma, TILE, 16, STAGES)
    sSFA = smem.allocate_tensor(cutlass.Float8E4M3FN, sfla, byte_alignment=128)
    sSFB = smem.allocate_tensor(cutlass.Float8E4M3FN, sflb, byte_alignment=128)
    sf_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(), cutlass.Float8E4M3FN, num_bits_per_copy=32
    )
    sf_copy = cute.make_tiled_copy_tv(
        sf_atom, cute.make_layout((128, 2), stride=(1, 128)), cute.make_layout((1, 4))
    )
    sf_thread = sf_copy.get_slice(tidx)
    gsfa = sf_thread.partition_S(cute.filter_zeros(gSFA))
    gsfb = sf_thread.partition_S(cute.filter_zeros(gSFB))
    ssfa = sf_thread.partition_D(cute.filter_zeros(sSFA))
    ssfb = sf_thread.partition_D(cute.filter_zeros(sSFB))
    gSFA, gSFB = sSFA, sSFB

    sfa_thr_layout = sm120_utils.thrfrg_SFA(gSFA.layout, tiled_mma)
    sfa_thr_tensor = cute.make_tensor(gSFA.iterator, sfa_thr_layout)
    thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
    thr_vmk = (thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3]))
    tAgSFA = sfa_thr_tensor[thr_vmk, (None, None)]
    tAgSFA = cute.group_modes(cute.flatten(tAgSFA), 0, 2)

    sfb_thr_layout = sm120_utils.thrfrg_SFB(gSFB.layout, tiled_mma)
    sfb_thr_tensor = cute.make_tensor(gSFB.iterator, sfb_thr_layout)
    thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
    tBgSFB = sfb_thr_tensor[thr_vnk, (None, None)]
    tBgSFB = cute.group_modes(cute.flatten(tBgSFB), 0, 2)
    tBgSFB = cute.group_modes(tBgSFB, 1, cute.rank(tBgSFB) - 2)

    rA = mma.make_fragment_A(thr_mma.partition_A(sA)[None, None, None, 0])
    rB = mma.make_fragment_B(thr_mma.partition_B(sB)[None, None, None, 0])
    rSFA = cute.make_fragment_like(tAgSFA[None, None, None, 0])
    rSFB = cute.make_fragment_like(tBgSFB[None, None, None, 0])
    la, lb = ldA.get_slice(tidx), ldB.get_slice(tidx)
    sla, slb = la.partition_S(sA), lb.partition_S(sB)
    rla, rlb = la.retile(rA), lb.retile(rB)
    kt = cute.size(ga, mode=[3])
    kb = cute.size(rA, mode=[2])
    for prime in cutlass.range_constexpr(STAGES - 1):
        cute.copy(cp, ga[None, None, None, prime], sa[None, None, None, prime])
        cute.copy(cp, gb[None, None, None, prime], sb[None, None, None, prime])
        cute.copy(sf_copy, gsfa[None, None, None, prime], ssfa[None, None, None, prime])
        cute.copy(sf_copy, gsfb[None, None, None, prime], ssfb[None, None, None, prime])
        cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(STAGES - 2)
    cute.arch.sync_threads()
    read = cutlass.Int32(0)
    write = cutlass.Int32(STAGES - 1)
    pa, pb = sla[None, None, None, read], slb[None, None, None, read]
    cute.copy(ldA, pa[None, None, 0], rla[None, None, 0])
    cute.copy(ldB, pb[None, None, 0], rlb[None, None, 0])
    cute.autovec_copy(tAgSFA[None, None, 0, 0], rSFA[None, None, 0])
    cute.autovec_copy(tBgSFB[None, None, 0, 0], rSFB[None, None, 0])
    for tile in range(kt - 1):
        future = tile + STAGES - 1
        has_future = future < kt
        for k_tile_slice in cutlass.range(kb, unroll_full=True):
            nxt = (k_tile_slice + 1) % kb
            sf_tile = read
            if k_tile_slice == kb - 1:
                if has_future:
                    cute.arch.cp_async_wait_group(STAGES - 2)
                else:
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                pa, pb = sla[None, None, None, read], slb[None, None, None, read]
                sf_tile = read
            cute.copy(ldA, pa[None, None, nxt], rla[None, None, nxt])
            cute.copy(ldB, pb[None, None, nxt], rlb[None, None, nxt])
            cute.autovec_copy(tAgSFA[None, None, nxt, sf_tile], rSFA[None, None, nxt])
            cute.autovec_copy(tBgSFB[None, None, nxt, sf_tile], rSFB[None, None, nxt])
            if k_tile_slice == 0 and has_future:
                cute.copy(cp, ga[None, None, None, future], sa[None, None, None, write])
            cute.gemm(
                mma,
                acc,
                [rA[None, None, k_tile_slice], rSFA[None, None, k_tile_slice]],
                [rB[None, None, k_tile_slice], rSFB[None, None, k_tile_slice]],
                acc,
            )
            if k_tile_slice == 0:
                if has_future:
                    cute.copy(
                        cp, gb[None, None, None, future], sb[None, None, None, write]
                    )
                    cute.copy(
                        sf_copy,
                        gsfa[None, None, None, future],
                        ssfa[None, None, None, write],
                    )
                    cute.copy(
                        sf_copy,
                        gsfb[None, None, None, future],
                        ssfb[None, None, None, write],
                    )
                    cute.arch.cp_async_commit_group()
                write = read
                read = (read + 1) % STAGES
    pa, pb = sla[None, None, None, read], slb[None, None, None, read]
    for k_tile_slice in cutlass.range(kb, unroll_full=True):
        if k_tile_slice + 1 < kb:
            nxt = k_tile_slice + 1
            cute.copy(ldA, pa[None, None, nxt], rla[None, None, nxt])
            cute.copy(ldB, pb[None, None, nxt], rlb[None, None, nxt])
            cute.autovec_copy(tAgSFA[None, None, nxt, read], rSFA[None, None, nxt])
            cute.autovec_copy(tBgSFB[None, None, nxt, read], rSFB[None, None, nxt])
        cute.gemm(
            mma,
            acc,
            [rA[None, None, k_tile_slice], rSFA[None, None, k_tile_slice]],
            [rB[None, None, k_tile_slice], rSFB[None, None, k_tile_slice]],
            acc,
        )
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # The 64x64 epilogue aliases the drained input arena.
    sC = cute.make_tensor(
        cute.recast_ptr(sA.iterator, slC.inner, dtype=cutlass.BFloat16), slC.outer
    )
    rs = r2s.get_slice(tidx)
    dst = rs.partition_D(sC)
    ra = r2s.retile(acc)
    shape = cute.shape(rs.partition_S(sC))[:3]
    tmp = cute.make_rmem_tensor(shape, cutlass.Float32)
    half = cute.make_rmem_tensor(shape, cutlass.BFloat16)
    drain = s2g.get_slice(tidx)
    src = drain.partition_S(sC[None, None, 0])
    mm = 64 // (TILE[0] // cute.size(ra, mode=[1]))
    nn = 64 // (TILE[1] // cute.size(ra, mode=[2]))
    for em in cutlass.range_constexpr(TILE[0] // 64):
        for en in cutlass.range_constexpr(TILE[1] // 64):
            for j in cutlass.range_constexpr(nn):
                for i in cutlass.range_constexpr(mm):
                    part = tmp[None, i, j]
                    values = ra[None, em * mm + i, en * nn + j]
                    for v in cutlass.range_constexpr(cute.size(part)):
                        part[v] = values[v]
            half.store(tmp.load().to(cutlass.BFloat16))
            cute.copy(r2s, half, dst[None, None, None, 0])
            cute.arch.sync_threads()
            output = cute.local_tile(gC, (64, 64), (em, en))
            cute.copy(s2g, src, drain.partition_D(output))
            cute.arch.sync_threads()


@cute.jit
def gemm(
    a: cute.Tensor,
    b: cute.Tensor,
    sfA: cute.Tensor,
    sfB: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
):
    sfa = cute.make_tensor(
        sfA.iterator, blockscaled_utils.tile_atom_to_shape_SF(a.shape, 16)
    )
    sfb = cute.make_tensor(
        sfB.iterator, blockscaled_utils.tile_atom_to_shape_SF(b.shape, 16)
    )
    mma = cute.make_tiled_mma(
        warp.MmaMXF4NVF4Op(cutlass.Float4E2M1FN, cutlass.Float32, cutlass.Float8E4M3FN),
        cute.make_layout((2, 4, 1)),
        permutation_mnk=sm120_utils.get_permutation_mnk(TILE, 16, False),
    )
    slA = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, TILE, a.element_type, STAGES
    )
    slB = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, TILE, b.element_type, STAGES
    )
    atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(), a.element_type, num_bits_per_copy=128
    )
    cp = cute.make_tiled_copy_tv(
        atom,
        cute.make_layout(
            (WARPS * 32 // (TILE[2] // 32), TILE[2] // 32), stride=(TILE[2] // 32, 1)
        ),
        cute.make_layout((1, 32)),
    )
    ldA = cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), a.element_type), mma
    )
    ldB = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), b.element_type), mma
    )
    slC = sm90_utils.make_smem_layout_epi(
        cutlass.BFloat16, utils.LayoutEnum.ROW_MAJOR, (64, 64), 1
    )
    atomC = cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), cutlass.BFloat16)
    r2s = cute.make_tiled_copy_S(
        sm120_utils.sm120_get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR,
            elem_ty_d=cutlass.BFloat16,
            elem_ty_acc=cutlass.Float32,
        ),
        cute.make_tiled_copy_C_atom(atomC, mma),
    )
    s2g = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        ),
        cute.make_layout((32, 8), stride=(8, 1)),
        cute.make_layout((1, 8)),
    )
    grid = (
        cute.ceil_div(c.shape[0], TILE[0]) * cute.ceil_div(c.shape[1], TILE[1]),
        1,
        1,
    )
    nvfp4_cpasync_kernel(
        a, b, sfa, sfb, c, mma, slA, slB, slC, cp, ldA, ldB, r2s, s2g
    ).launch(grid=grid, block=(WARPS * 32, 1, 1), stream=stream)
