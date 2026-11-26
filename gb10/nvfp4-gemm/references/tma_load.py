"""CTA-local TMA load for SM121; destination and barrier belong to this CTA."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu, llvm


@dsl_user_op
def _load_local(atom, source, destination, *, tma_bar_ptr, loc=None, ip=None):
    descriptor = cute_nvgpu.get_tma_desc_addr(
        ir.Type.parse("!cute.ptr<i8, generic, align<64>>"),
        atom._trait.unpack(tma_bar_ptr=tma_bar_ptr, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    descriptor = descriptor.toint(loc=loc, ip=ip)
    shared = destination.iterator.toint(loc=loc, ip=ip)
    barrier = tma_bar_ptr.toint(loc=loc, ip=ip)
    coords = source.iterator
    rank = len(coords)
    operands = [cutlass.Int32(shared).ir_value(), descriptor.ir_value()]
    operands += [cutlass.Int32(c).ir_value() for c in coords]
    operands += [cutlass.Int32(barrier).ir_value()]
    coordinate_args = ", ".join(f"${i + 2}" for i in range(rank))
    instruction = f"cp.async.bulk.tensor.{rank}d.shared::cta.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {{{coordinate_args}}}], [${rank + 2}];"
    # cute.copy elects one lane; keep the same issuer rule here.
    llvm.inline_asm(
        None,
        operands,
        instruction,
        ",".join(["r", "l"] + ["r"] * (rank + 1)),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def load_local(
    atom: cute.CopyAtom,
    source: cute.Tensor,
    destination: cute.Tensor,
    *,
    tma_bar_ptr: cute.Pointer,
):
    with cute.arch.elect_one():
        _load_local(atom, source, destination, tma_bar_ptr=tma_bar_ptr)
