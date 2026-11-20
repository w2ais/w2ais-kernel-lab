#!/usr/bin/env python3
"""Exact NVIDIA Ampere TensorOp kernel with only its C epilogue replaced.

This source-transform experiment reads the byte-exact vendored NVIDIA kernel
from ``nvidia_ampere_tensorop_gemm.py``.  Its prologue and mainloop remain
unchanged.  The full-CTA scalar/shared accumulator scatter is replaced by the
four-warp CUTLASS Blackwell-GeForce epilogue construction:

``sm90_get_smem_store_op -> make_tiled_copy_C_atom -> make_tiled_copy_S``.

The compatible four-warp ``(2,2,1)`` MMA has a 32x32 C tiler.  A 128x32 BF16
epilogue subtile therefore covers complete C atoms, and four subtiles cover
the 128x128 CTA result.  This variant is intentionally isolated from the
byte-exact vendor file so the comparison changes no upstream mainloop source.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE / "references" / "nvidia_ampere_tensorop_gemm.py"
EXPECTED_BASE_SHA256 = (
    "6ae4a6bb8214f865a1d8d9bef6c6da5abc621f0b6a1fa6ea9deba3bac7fb215b"
)
base_bytes = BASE.read_bytes()
BASE_SHA256 = hashlib.sha256(base_bytes).hexdigest()
if BASE_SHA256 != EXPECTED_BASE_SHA256:
    raise RuntimeError(
        "exact NVIDIA base changed; refusing a marker-based transform: "
        f"expected {EXPECTED_BASE_SHA256}, got {BASE_SHA256}"
    )

source = base_bytes.decode("utf-8")
mainloop_start_marker = "            # Mainloop\n"
mainloop_end_marker = "            # Sync before epilogue\n"
mainloop_start = source.index(mainloop_start_marker)
mainloop_end = source.index(mainloop_end_marker, mainloop_start)
ORIGINAL_MAINLOOP_SHA256 = hashlib.sha256(
    source[mainloop_start:mainloop_end].encode("utf-8")
).hexdigest()

# Add only the helper and static geometry required by the replacement
# epilogue.  No upstream MMA/mainloop constants are changed.
source = source.replace(
    "import cutlass.utils as utils\n",
    "import cutlass.utils as utils\n"
    "import cutlass.utils.hopper_helpers as sm90_utils\n",
)
constant_marker = "from cutlass.cute.runtime import from_dlpack\n\n"
constants = '''from cutlass.cute.runtime import from_dlpack

EPI_TILE = (128, 32)
EPI_TILE_COUNT_M = 128 // EPI_TILE[0]
EPI_TILE_COUNT_N = 128 // EPI_TILE[1]
EPI_TILE_COUNT = EPI_TILE_COUNT_M * EPI_TILE_COUNT_N

'''
if source.count(constant_marker) != 1:
    raise RuntimeError("upstream import marker changed")
source = source.replace(constant_marker, constants)

# Use CUTLASS's swizzled, single-stage 128x32 epilogue buffer instead of the
# upstream full 128x128 unswizzled staging tensor.
old_sC_layout = '''        sC_layout = self._make_smem_layout_C(
            mC.element_type,
            self.c_major_mode,
            ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[1]),
        )
'''
new_sC_layout = '''        sC_layout = sm90_utils.make_smem_layout_epi(
            mC.element_type,
            self.c_major_mode,
            EPI_TILE,
            1,
        )
'''
if source.count(old_sC_layout) != 1:
    raise RuntimeError("upstream C shared-layout marker changed")
source = source.replace(old_sC_layout, new_sC_layout)

# A 128x32 subtile uses a 32x32 base copy tile: 128 threads each own one
# contiguous eight-BF16 vector along N, repeated four times along M.
old_copy_C = '''        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, mC.element_type, self.c_major_mode, c_copy_bits
        )
'''
new_copy_C = '''        tiled_copy_C = cute.make_tiled_copy_tv(
            atom_sync_copy,
            cute.make_layout((32, 4), stride=(4, 1)),
            cute.make_layout((1, 8)),
        )
'''
if source.count(old_copy_C) != 1:
    raise RuntimeError("upstream C global-copy marker changed")
source = source.replace(old_copy_C, new_copy_C)

# Compose stmatrix with the exact upstream four-warp MMA C mapping.
mma_end_marker = "\n\n        # grid_dim:"
mma_end = source.index(mma_end_marker, source.index("        tiled_mma ="))
r2s_setup = '''

        copy_atom_r2s_C = sm90_utils.sm90_get_smem_store_op(
            self.c_major_mode,
            elem_ty_d=mC.element_type,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C_mapping = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_major_mode.is_m_major_c(), 4
            ),
            mC.element_type,
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(
            copy_atom_C_mapping,
            tiled_mma,
        )
        tiled_copy_r2s_C = cute.make_tiled_copy_S(
            copy_atom_r2s_C,
            tiled_copy_C_atom,
        )
'''
source = source[:mma_end] + r2s_setup + source[mma_end:]

# Thread the derived R2S copy through the launch and kernel signature.
call_args = (
    "            tiled_copy_B,\n"
    "            tiled_copy_C,\n"
    "            tiled_mma,\n"
)
call_args_r2s = (
    "            tiled_copy_B,\n"
    "            tiled_copy_C,\n"
    "            tiled_copy_r2s_C,\n"
    "            tiled_mma,\n"
)
if source.count(call_args) != 1:
    raise RuntimeError("upstream kernel-call marker changed")
source = source.replace(call_args, call_args_r2s)

signature_args = (
    "        tiled_copy_B: cute.TiledCopy,\n"
    "        tiled_copy_C: cute.TiledCopy,\n"
    "        tiled_mma: cute.TiledMma,\n"
)
signature_args_r2s = (
    "        tiled_copy_B: cute.TiledCopy,\n"
    "        tiled_copy_C: cute.TiledCopy,\n"
    "        tiled_copy_r2s_C: cute.TiledCopy,\n"
    "        tiled_mma: cute.TiledMma,\n"
)
if source.count(signature_args) != 1:
    raise RuntimeError("upstream kernel-signature marker changed")
source = source.replace(signature_args, signature_args_r2s)

# Recast the dead A/B arena with C's pointer-side swizzle.  Using C's outer
# layout separately avoids applying A's and C's XOR swizzles together.
old_alias = '''            sC = cute.make_tensor(
                cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout
            )
'''
new_alias = '''            sC_ptr = cute.recast_ptr(
                sA.iterator,
                sC_layout.inner,
                dtype=self.c_dtype,
            )
            sC = cute.make_tensor(sC_ptr, sC_layout.outer)
'''
if source.count(old_alias) != 1:
    raise RuntimeError("upstream C alias marker changed")
source = source.replace(old_alias, new_alias)

# C is partitioned per 128x32 subtile in the replacement epilogue.  Preserve
# the exact upstream A/B copy setup around these removed lines.
old_c_copy_partitions = '''            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            thr_copy_C = tiled_copy_C.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCgC_epilogue = thr_copy_C.partition_D(gC)
'''
new_c_copy_partitions = '''            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
'''
if source.count(old_c_copy_partitions) != 1:
    raise RuntimeError("upstream C copy-partition marker changed")
source = source.replace(old_c_copy_partitions, new_c_copy_partitions)

old_mma_c_partition = "            tCsC = thr_mma.partition_C(sC)\n"
if source.count(old_mma_c_partition) != 1:
    raise RuntimeError("upstream MMA C-partition marker changed")
source = source.replace(old_mma_c_partition, "")

# Replace the epilogue region only.  The wait/barrier immediately before it is
# still upstream source and the method's return immediately after it remains
# upstream source.
epilogue_start_marker = (
    "            # " + "/" * 79 + "\n"
    "            # Epilogue with fusion\n"
)
epilogue_start = source.index(epilogue_start_marker)
epilogue_end = source.index("        return\n\n    def _make_smem_layout_AB", epilogue_start)
stmatrix_epilogue = '''            # /////////////////////////////////////////////////////////////////////////////
            # CUTLASS four-warp, 128x32 subtiled StMatrix epilogue
            # /////////////////////////////////////////////////////////////////////////////
            thr_copy_r2s = tiled_copy_r2s_C.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s_C.retile(tCrC)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(
                tRS_rD_layout.shape, self.acc_dtype
            )
            tRS_rD_out = cute.make_rmem_tensor(
                tRS_rD_layout.shape, self.c_dtype
            )
            rD_size = cute.size(tRS_rD)

            thr_copy_C = tiled_copy_C.get_slice(tidx)
            tCsC_epilogue = thr_copy_C.partition_S(sC[None, None, 0])

            for epi_idx in cutlass.range_constexpr(EPI_TILE_COUNT):
                for epi_v in cutlass.range_constexpr(rD_size):
                    tRS_rD[epi_v] = tRS_rAcc[
                        epi_idx * rD_size + epi_v
                    ]
                tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))
                cute.copy(
                    tiled_copy_r2s_C,
                    tRS_rD_out,
                    tRS_sC[(None, None, None, 0)],
                )

                # StMatrix is warp-scoped, while the vector drain consumes
                # values written by all four warps.  The second CTA barrier
                # prevents the next subtile from overwriting shared memory.
                cute.arch.sync_threads()
                epi_m = epi_idx % EPI_TILE_COUNT_M
                epi_n = epi_idx // EPI_TILE_COUNT_M
                gC_epi = cute.local_tile(gC, EPI_TILE, (epi_m, epi_n))
                tCgC_epilogue = thr_copy_C.partition_D(gC_epi)
                cute.copy(tiled_copy_C, tCsC_epilogue, tCgC_epilogue)
                cute.arch.sync_threads()
'''
source = source[:epilogue_start] + stmatrix_epilogue + source[epilogue_end:]

# Prove that the text spanning NVIDIA's Mainloop section was untouched.
new_mainloop_start = source.index(mainloop_start_marker)
new_mainloop_end = source.index(mainloop_end_marker, new_mainloop_start)
TRANSFORMED_MAINLOOP_SHA256 = hashlib.sha256(
    source[new_mainloop_start:new_mainloop_end].encode("utf-8")
).hexdigest()
if TRANSFORMED_MAINLOOP_SHA256 != ORIGINAL_MAINLOOP_SHA256:
    raise RuntimeError("source transform unexpectedly changed NVIDIA's mainloop")

generated_path = Path("/tmp/gb10_nvidia_ampere_stmatrix_generated.py")
generated_path.write_text(source, encoding="utf-8")
GENERATED_SOURCE_SHA256 = hashlib.sha256(source.encode("utf-8")).hexdigest()

namespace = {
    "__name__": "_gb10_nvidia_ampere_stmatrix_generated",
    "__file__": str(generated_path),
}
exec(compile(source, str(generated_path), "exec"), namespace)

TensorOpGemm = namespace["TensorOpGemm"]
run = namespace["run"]


__all__ = [
    "BASE_SHA256",
    "EXPECTED_BASE_SHA256",
    "GENERATED_SOURCE_SHA256",
    "ORIGINAL_MAINLOOP_SHA256",
    "TRANSFORMED_MAINLOOP_SHA256",
    "TensorOpGemm",
    "run",
]


if __name__ == "__main__":
    from benchmark import main

    main("cheating", legacy=True)
