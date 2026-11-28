"""Small PTX primitives shared by the CuTe DSL probes.

Keep instruction spelling explicit: these benchmarks measure instructions, not
matrix layouts. The Python kernels own loops, launch geometry and timing.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float64, Int32, Int64, Uint32, Uint64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


@dsl_user_op
def asm(code, constraints, args=(), returns=(), *, loc=None, ip=None):
    """Emit side-effecting PTX; memory clobbers keep timing boundaries ordered."""
    values = [x.ir_value(loc=loc, ip=ip) for x in args]
    if len(returns) == 0:
        result_type = ir.Type.parse('!llvm.void')
    elif len(returns) == 1:
        result_type = returns[0].mlir_type
    else:
        result_type = ir.Type.parse('!llvm.struct<(' + ', '.join(str(t.mlir_type) for t in returns) + ')>')
    value = llvm.inline_asm(result_type, values, code, constraints + ',~{memory}' if constraints else '~{memory}',
                            has_side_effects=True, is_align_stack=False,
                            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    if len(returns) == 1:
        return returns[0](value)
    if len(returns) > 1:
        return tuple(t(llvm.extractvalue(t.mlir_type, value, [i], loc=loc, ip=ip)) for i, t in enumerate(returns))


@cute.jit
def record(timing: cute.Pointer, start: Int64):
    timing[0] = cute.arch.clock64() - start
    timing[1] = asm('mov.u32 $0, %smid;', '=r', returns=(Uint32,))
    timing[3] = cute.arch.dynamic_smem_size()


@cute.jit
def drain_f32(ptr: cute.Pointer, value: Float32):
    asm('st.volatile.shared.f32 [$0], $1;', 'r,f', (ptr.toint(), value))


@cute.jit
def drain_f64(ptr: cute.Pointer, value: Float64):
    asm('st.volatile.shared.f64 [$0], $1;', 'r,d', (ptr.toint(), value))


@cute.jit
def barrier_init(address: Uint32):
    asm('mbarrier.init.shared::cta.b64 [$0], 1;', 'r', (address,))


@cute.jit
def barrier_wait(address: Uint32, phase: Int32):
    asm('{ .reg .pred ready; wait_again: mbarrier.try_wait.parity.shared::cta.b64 ready, [$0], $1; @!ready bra wait_again; }',
        'r,r', (address, phase))


@cute.jit
def proxy_fence():
    asm('fence.proxy.async.shared::cta;', '')
