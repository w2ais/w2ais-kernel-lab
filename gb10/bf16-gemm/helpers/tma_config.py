"""TMA resource and persistent-grid configuration used by benchmark runners."""

from __future__ import annotations
from types import ModuleType

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils


SUPPORTED_TILES = {
    (64, 64, 64),
    (64, 128, 64),
    (128, 64, 64),
    (128, 128, 64),
    (128, 256, 64),
    (128, 128, 128),
}


def make_tuned_kernel_class(
    upstream: ModuleType,
    epilogue_stages: int,
    swizzle_size: int,
    raster_along_m: bool,
):
    class OccupancyTunedSm120Gemm(upstream.Sm120GemmKernel):
        @staticmethod
        def _compute_stages(
            tile_shape_mnk,
            a_dtype,
            b_dtype,
            epi_tile,
            c_dtype,
            smem_capacity,
            occupancy,
        ):
            c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
            epilogue_bytes = c_bytes_per_stage * epilogue_stages
            a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
            b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
            ab_bytes_per_stage = (
                cute.size(a_shape) * a_dtype.width // 8
                + cute.size(b_shape) * b_dtype.width // 8
            )
            barrier_and_alignment_bytes = 1024
            bytes_per_cta = (smem_capacity - occupancy * 1024) // occupancy
            ab_stages = (
                bytes_per_cta - barrier_and_alignment_bytes - epilogue_bytes
            ) // ab_bytes_per_stage
            if ab_stages < 1:
                raise ValueError(
                    "Configuration leaves no room for an A/B pipeline stage"
                )
            return ab_stages, epilogue_stages

        @staticmethod
        def _compute_grid(
            c: cute.Tensor,
            tile_shape_mnk: tuple[int, int, int],
            max_active_clusters: cutlass.Constexpr,
        ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
            """Build the persistent schedule with an explicit L2 tile swizzle."""
            c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
            gc = cute.zipped_divide(c, tiler=c_shape)
            num_ctas_mnl = gc[(0, (None, None, None))].shape
            cluster_shape_mnl = (1, 1, 1)
            tile_sched_params = utils.PersistentTileSchedulerParams(
                num_ctas_mnl,
                cluster_shape_mnl,
                swizzle_size=swizzle_size,
                raster_along_m=raster_along_m,
            )
            grid = utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params,
                max_active_clusters,
            )
            return tile_sched_params, grid

    return OccupancyTunedSm120Gemm


def make_op(module, name, tile, stages, epi_stages, group, raster):
    # Base scheduler factory expects this exported class name.
    if name == "candidate":
        module.Sm120GemmKernel = module.TunedSm120GemmKernel
    parent = make_tuned_kernel_class(module, epi_stages, group, raster == "m")

    class ConfiguredTma(parent):
        @staticmethod
        def _compute_stages(
            tile_shape_mnk,
            a_dtype,
            b_dtype,
            epi_tile,
            c_dtype,
            smem_capacity,
            occupancy,
        ):
            ab_bytes = (
                (tile_shape_mnk[0] + tile_shape_mnk[1])
                * tile_shape_mnk[2]
                * a_dtype.width
                // 8
                * stages
            )
            c_bytes = cute.size(epi_tile) * c_dtype.width // 8 * epi_stages
            if ab_bytes + c_bytes + 1024 > smem_capacity:
                raise ValueError(
                    f"Shared-memory budget exceeded: AB={ab_bytes}, C={c_bytes}, capacity={smem_capacity}"
                )
            return stages, epi_stages

    return ConfiguredTma(cutlass.Float32, tile)
