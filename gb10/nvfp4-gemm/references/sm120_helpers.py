# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# This software contains source code provided by NVIDIA Corporation.
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.


from typing import Optional, Type, Tuple
import cutlass.cute as cute
from cutlass.cutlass_dsl import Numeric, NumericMeta, dsl_user_op
from cutlass._mlir import ir
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu.warp import StMatrix8x8x16bOp
from cutlass.cute.nvgpu.common import CopyUniversalOp


@dsl_user_op
def get_permutation_mnk(
    tile_shape_mnk: cute.Shape,
    sf_vec_size: int,
    use_mxf8f6f4: bool,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[int, int, int]:
    """
    Get the permutation of M, N, K for the tiled MMA.

    :param tile_shape_mnk: The shape of the tile
    :type tile_shape_mnk: cute.Shape
    :param sf_vec_size: The vector size of the Scale Factor.
    :type sf_vec_size: int
    :param use_mxf8f6f4: Whether to use MXF8F6F4 or MXF4NVF4.
    :type use_mxf8f6f4: bool

    :return: The permutation of M, N, K
    :rtype: Tuple[int, int, int]

    :raise ValueError: If the tile shape is not divisible by the sf_vec_size
    """
    perm_m = min(tile_shape_mnk[0], 128)  # type: ignore[index]
    # refer to C++ code:
    # /include/cutlass/gemm/collective/builders/sm120_common.inl?ref_type=heads#L158
    if sf_vec_size == 32 or sf_vec_size == 16:
        perm_n_shape = (8, 2, 2)
        perm_n_stride = (1, 16, 8)
    else:
        raise ValueError(f"Unsupported sf_vec_size, got {sf_vec_size}")

    perm_n_layout = cute.make_layout(perm_n_shape, stride=perm_n_stride)
    perm_k = 32 if use_mxf8f6f4 else 64
    permutation_mnk = (
        perm_m,
        perm_n_layout,
        perm_k,
    )

    return permutation_mnk  # type: ignore[return-value]


@dsl_user_op  # type: ignore[no-redef]
def sm120_get_smem_store_op(
    layout_d: LayoutEnum,
    elem_ty_d: Type[Numeric],
    elem_ty_acc: Type[Numeric],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.CopyAtom:
    """
    Selects the largest vectorized smem store atom available subject to constraint of gmem layout.

    Parameters:
    -----------
    layout_d : LayoutEnum
        The layout enum of the output tensor D.

    elem_ty_d : Type[Numeric]
        The element type for output tensor D.

    elem_ty_acc : Type[Numeric]
        The element type for accumulator.

    Returns:
    --------
    Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.
    """

    def validate_type(ty: Type[Numeric], ty_name: str) -> None:
        if not isinstance(ty, NumericMeta):
            raise TypeError(f"{ty_name} must be a Numeric, but got {ty}")

    validate_type(elem_ty_d, "elem_ty_d")
    validate_type(elem_ty_acc, "elem_ty_acc")

    is_m_major = layout_d.is_m_major_c()

    if elem_ty_d.width == 16:
        return cute.make_copy_atom(
            StMatrix8x8x16bOp(is_m_major, 2), elem_ty_d, loc=loc, ip=ip
        )
    else:
        return cute.make_copy_atom(CopyUniversalOp(), elem_ty_d, loc=loc, ip=ip)


def thrfrg_SFA(
    sfa_tensor: cute.Tensor,
    tiled_mma: cute.TiledMma,
) -> cute.Tensor:
    """Thread-fragment scale factor A tensor for SM120 block-scaled MMA.

    Implements the ThrFrg partitioning for scale factor A according to the
    corresponding C++ code in cutlass/include/cute/atom/mma_traits_sm120.hpp:
    SFALayout for SM120 MXF4 16x8x64 uses K=64, SM120 MXF8F6F4 16x8x32 uses
    K=32; the stride pattern ``((_8,_0,_1), _16)`` is shared.
    """
    assert cute.rank(sfa_tensor) >= 2

    atom_shape_mnk = tiled_mma.shape_mnk
    # K-dim of the warp-MMA atom: FP4 -> 64, FP8 -> 32 (per mma_traits_sm120.hpp).
    # For FP8 (atom_K=32) where mma_nsf=1, wrap K in a 2-tuple ``(atom_K, 1)``
    # so the layout's K mode keeps its 2D structure and the resulting fragment
    # has the same rank as the FP4 path. For FP4 (atom_K=64) the original 1D
    # layout already produces a 2D K decomposition through SMEM-layout
    # composition, so we keep the original shape.
    atom_K = atom_shape_mnk[2]
    if atom_K == 32:
        atom_sfa_layout = cute.make_layout(
            shape=((2, 2, 8), (atom_K, 1)),
            stride=((8, 0, 1), (16, 0)),
        )
    elif atom_K == 64:
        atom_sfa_layout = cute.make_layout(
            shape=((2, 2, 8), atom_K),
            stride=((8, 0, 1), 16),
        )
    else:
        raise ValueError(
            f"thrfrg_SFA: unsupported atom_K={atom_K}; SM120 block-scaled atoms "
            f"use atom_K=32 (mxf8/mxf8f6f4) or atom_K=64 (mxf4/mxf4nvf4)"
        )
    permutation_mnk = tiled_mma.permutation_mnk
    thr_layout_vmnk = tiled_mma.thr_layout_vmnk

    # Reorder the tensor for TiledAtom
    t_tile = (permutation_mnk[0], permutation_mnk[2])
    t_tensor = cute.logical_divide(sfa_tensor, t_tile)

    # Tile the tensor for the Atom
    a_tile = (
        cute.make_layout((atom_shape_mnk[0])),
        cute.make_layout((atom_shape_mnk[2])),
    )
    a_tensor = cute.zipped_divide(t_tensor, a_tile)

    # Transform the Atom mode from (M,K) to (Thr,Val)
    tv_tensor = cute.composition(a_tensor, (atom_sfa_layout, None))

    # Tile the tensor for the Thread
    thr_tile = (
        None,
        (
            cute.make_layout(cute.size(thr_layout_vmnk[1])),
            cute.make_layout(cute.size(thr_layout_vmnk[3])),
        ),
    )

    thr_tensor = cute.zipped_divide(tv_tensor, thr_tile)

    return thr_tensor


def thrfrg_SFB(
    sfb_tensor: cute.Tensor,
    tiled_mma: cute.TiledMma,
) -> cute.Tensor:
    """Thread-fragment scale factor B tensor for SM120 block-scaled MMA.

    Implements the ThrFrg partitioning for scale factor B according to the
    corresponding C++ code in cutlass/include/cute/atom/mma_traits_sm120.hpp:
    SFBLayout for SM120 MXF4 16x8x64 uses K=64, SM120 MXF8F6F4 16x8x32 uses
    K=32; the stride pattern ``((_0,_1), _8)`` is shared.
    """
    assert cute.rank(sfb_tensor) >= 2

    atom_shape_mnk = tiled_mma.shape_mnk
    # K-dim of the warp-MMA atom: FP4 -> 64, FP8 -> 32 (per mma_traits_sm120.hpp).
    # See :func:`thrfrg_SFA` for the rationale behind the FP8-only
    # ``(atom_K, 1)`` wrapping.
    atom_K = atom_shape_mnk[2]
    if atom_K == 32:
        atom_sfb_layout = cute.make_layout(
            shape=((4, 8), (atom_K, 1)),
            stride=((0, 1), (8, 0)),
        )
    elif atom_K == 64:
        atom_sfb_layout = cute.make_layout(
            shape=((4, 8), atom_K),
            stride=((0, 1), 8),
        )
    else:
        raise ValueError(
            f"thrfrg_SFB: unsupported atom_K={atom_K}; SM120 block-scaled atoms "
            f"use atom_K=32 (mxf8/mxf8f6f4) or atom_K=64 (mxf4/mxf4nvf4)"
        )
    permutation_mnk = tiled_mma.permutation_mnk
    thr_layout_vmnk = tiled_mma.thr_layout_vmnk

    # Reorder the tensor for TiledAtom
    t_tile = (permutation_mnk[1], permutation_mnk[2])
    t_tensor = cute.logical_divide(sfb_tensor, t_tile)

    # Tile the tensor for the Atom
    a_tile = (
        cute.make_layout((atom_shape_mnk[1])),
        cute.make_layout((atom_shape_mnk[2])),
    )
    a_tensor = cute.zipped_divide(t_tensor, a_tile)

    # Transform the Atom mode from (M,K) to (Thr,Val)
    tv_tensor = cute.composition(a_tensor, (atom_sfb_layout, None))

    # Tile the tensor for the Thread
    thr_tile = (
        None,
        (
            cute.make_layout(cute.size(thr_layout_vmnk[2])),
            cute.make_layout(cute.size(thr_layout_vmnk[3])),
        ),
    )

    thr_tensor = cute.zipped_divide(tv_tensor, thr_tile)

    return thr_tensor


def partition_fragment_SFA(
    sfa_tensor: cute.Tensor,
    thr_mma: cute.ThrMma,
    tidx: int,
) -> cute.Tensor:
    """Partition and create a register fragment for scale factor A."""
    thrfrg_sfa_layout = thrfrg_SFA(sfa_tensor.layout, thr_mma)  # type: ignore[arg-type]
    thr_tensor = cute.make_tensor(sfa_tensor.iterator, thrfrg_sfa_layout)
    thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
    thr_vmk = (thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3]))
    partitioned_sfa = thr_tensor[thr_vmk, (None, None)]
    partitioned_sfa = cute.group_modes(cute.flatten(partitioned_sfa), 0, 2)
    return cute.make_fragment_like(partitioned_sfa)


def partition_fragment_SFB(
    sfb_tensor: cute.Tensor,
    thr_mma: cute.ThrMma,
    tidx: int,
) -> cute.Tensor:
    """Partition and create a register fragment for scale factor B."""
    thrfrg_sfb_layout = thrfrg_SFB(sfb_tensor.layout, thr_mma)  # type: ignore[arg-type]
    thr_tensor = cute.make_tensor(sfb_tensor.iterator, thrfrg_sfb_layout)
    thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
    thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
    partitioned_sfb = thr_tensor[thr_vnk, (None, None)]
    partitioned_sfb = cute.group_modes(cute.flatten(partitioned_sfb), 0, 2)
    partitioned_sfb = cute.group_modes(partitioned_sfb, 1, 3)
    return cute.make_fragment_like(partitioned_sfb)


def get_layoutSFA_TV(tiled_mma: cute.TiledMma) -> cute.Layout:
    """Get the Thread-Value layout for scale factor A."""
    if tiled_mma.permutation_mnk is not None:
        perm_m = tiled_mma.permutation_mnk[0]
        perm_k = tiled_mma.permutation_mnk[2]
        tile_m = cute.size(perm_m)
        tile_k = cute.size(perm_k)
    else:
        tile_shape_mnk = tiled_mma.shape_mnk * tiled_mma.thr_layout_vmnk
        tile_m = cute.size(tile_shape_mnk[0])
        tile_k = cute.size(tile_shape_mnk[2])

    ref_A = cute.make_layout((tile_m, tile_k))
    thr_layout_vmnk = tiled_mma.thr_layout_vmnk

    # (ThrV, (ThrM, ThrK)) -> (ThrV, (ThrM, ThrN, ThrK))
    atile = (
        None,
        (
            cute.make_layout(
                shape=(
                    cute.size(thr_layout_vmnk[1]),
                    cute.size(thr_layout_vmnk[2]),
                ),
                stride=(1, 0),
            ),
            None,
        ),
    )

    # thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    thridx_2_thrid = cute.right_inverse(thr_layout_vmnk)
    thrfrg_sfa = thrfrg_SFA(ref_A, tiled_mma)
    layout_tv_1 = cute.composition(thrfrg_sfa, (atile, None))
    layout_tv = cute.composition(layout_tv_1, (thridx_2_thrid, None))

    return layout_tv  # type: ignore[return-value]


def get_layoutSFB_TV(tiled_mma: cute.TiledMma) -> cute.Layout:
    """Get the Thread-Value layout for scale factor B."""
    if tiled_mma.permutation_mnk is not None:
        perm_n_layout = tiled_mma.permutation_mnk[1]
        perm_k = tiled_mma.permutation_mnk[2]
        tile_n = cute.size(perm_n_layout)
        tile_k = cute.size(perm_k)
    else:
        tile_shape_mnk = tiled_mma.shape_mnk * tiled_mma.thr_layout_vmnk
        tile_n = cute.size(tile_shape_mnk[1])
        tile_k = cute.size(tile_shape_mnk[2])

    ref_B = cute.make_layout((tile_n, tile_k))
    thr_layout_vmnk = tiled_mma.thr_layout_vmnk

    # (ThrV, (ThrM, ThrK)) -> (ThrV, (ThrM, ThrN, ThrK))
    atile = (
        None,
        (
            cute.make_layout(
                shape=(
                    cute.size(thr_layout_vmnk[1]),
                    cute.size(thr_layout_vmnk[2]),
                ),
                stride=(0, 1),
            ),
            None,
        ),
    )

    # thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    thridx_2_thrid = cute.right_inverse(thr_layout_vmnk)
    thrfrg_sfb = thrfrg_SFB(ref_B, tiled_mma)
    layout_tv = cute.composition(thrfrg_sfb, (atile, None))
    layout_tv = cute.composition(layout_tv, (thridx_2_thrid, None))
    return layout_tv  # type: ignore[return-value]
