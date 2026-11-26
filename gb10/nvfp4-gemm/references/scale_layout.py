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


from typing import Optional
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF


@dsl_user_op
def sm120_make_smem_layout_sfa(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """
    Make smem layout for SFA based on:
    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[0] % blk_mn == 0, (  # type: ignore[index, operator]
        "tile_shape_mnk[0] must be divisible by blk_mn"
    )

    sSFA_shapeM = (mn_basic_block_shape, tile_shape_mnk[0] // blk_mn)  # type: ignore[index, operator]
    sSF_strideM = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (  # type: ignore[index]
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,  # type: ignore[index, operator]
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        tile_shape_mnk[0] // blk_mn * blk_elems,  # type: ignore[index, operator]
    )

    sSFA_shape = (sSFA_shapeM, sSFA_shapeK)
    sSFA_stride = (sSF_strideM, sSF_strideK)

    smem_layout = cute.make_layout(sSFA_shape, stride=sSFA_stride)

    # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
    sfa_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfa_smem_layout_staged


@dsl_user_op
def sm120_make_smem_layout_sfb(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cute.Layout:
    """
    Make smem layout for SFB based on:
    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """

    # A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
    # 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row(col).
    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    assert tile_shape_mnk[1] % blk_mn == 0, (  # type: ignore[index, operator]
        "tile_shape_mnk[1] must be divisible by blk_mn"
    )

    assert tile_shape_mnk[2] % sf_vec_size == 0, (  # type: ignore[index, operator]
        "tile_shape_mnk[2] must be divisible by sf_vec_size"
    )

    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[1] % blk_mn == 0, (  # type: ignore[index, operator]
        "tile_shape_mnk[1] must be divisible by blk_mn"
    )

    sSFA_shapeN = (mn_basic_block_shape, tile_shape_mnk[1] // blk_mn)  # type: ignore[index, operator]
    sSF_strideN = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (  # type: ignore[index]
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,  # type: ignore[index, operator]
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        tile_shape_mnk[1] // blk_mn * blk_elems,  # type: ignore[index, operator]
    )

    sSFA_shape = (sSFA_shapeN, sSFA_shapeK)
    sSFA_stride = (sSF_strideN, sSF_strideK)

    smem_layout = cute.make_layout(sSFA_shape, stride=sSFA_stride)

    # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
    sfb_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfb_smem_layout_staged
