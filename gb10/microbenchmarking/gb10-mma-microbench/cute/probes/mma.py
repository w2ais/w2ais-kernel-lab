"""Seven warp-wide MMA forms, with runtime nonzero operands in registers."""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from .common import asm, drain_f32, record

# K, packed representation of 1, packed representation of 0.5.
SPECS = {
    'fp16_k8': (8, 0x3c003c00, 0x38003800),
    'fp16': (16, 0x3c003c00, 0x38003800),
    'bf16': (16, 0x3f803f80, 0x3f003f00),
    'tf32': (8, 0x3f800000, 0x3f000000),
    'fp4_k32': (32, 0x08080808, 0x04040404),
    'mxfp4': (64, 0x22222222, 0x11111111),
    'nvfp4': (64, 0x22222222, 0x11111111),
}


def instruction(op):
    """One PTX instruction; $0–3 outputs, $4–7 accumulators, $8 input."""
    prefix = 'mma.sync.aligned.'
    suffix = ''
    a, b = '{$8,$8,$8,$8}', '{$8,$8}'
    if op == 'fp16_k8':
        shape = 'm16n8k8.row.col.f32.f16.f16.f32'
        a, b = '{$8,$8}', '{$8}'
    elif op in ('fp16', 'bf16', 'tf32'):
        kind = {'fp16': 'f16', 'bf16': 'bf16', 'tf32': 'tf32'}[op]
        shape = f'm16n8k{SPECS[op][0]}.row.col.f32.{kind}.{kind}.f32'
    else:
        kind, vector, scale = ('mxf8f6f4', '1X', 'ue8m0') if op == 'fp4_k32' else (
            'mxf4nvf4', '4X' if op == 'nvfp4' else '2X', 'ue4m3' if op == 'nvfp4' else 'ue8m0')
        prefix += f'kind::{kind}.block_scale.scale_vec::{vector}.'
        shape = f'm16n8k{SPECS[op][0]}.row.col.f32.e2m1.e2m1.f32.{scale}'
        suffix = ', $9, {0,0}, $9, {0,0}'
    return f'{prefix}{shape} {{$0,$1,$2,$3}}, {a}, {b}, {{$4,$5,$6,$7}}{suffix};'


@cute.jit
def mma_step(op: cutlass.Constexpr, d0: Float32, d1: Float32, d2: Float32, d3: Float32, x: Uint32):
    if cutlass.const_expr(op in ('fp4_k32', 'mxfp4', 'nvfp4')):
        scale = Uint32(0x38383838 if op == 'nvfp4' else 0x7f7f7f7f)
        result = asm(instruction(op), '=f,=f,=f,=f,f,f,f,f,r,r',
                     (d0, d1, d2, d3, x, scale), (Float32,)*4)
    else:
        result = asm(instruction(op), '=f,=f,=f,=f,f,f,f,f,r',
                     (d0, d1, d2, d3, x), (Float32,)*4)
    return result


@cute.kernel
def measure(op: cutlass.Constexpr, chains: cutlass.Constexpr, unroll: cutlass.Constexpr,
            iterations: Int32, x: Uint32, timing: cute.Pointer, output: cute.Pointer):
    tid, _, _ = cute.arch.thread_idx()
    allocator = cutlass.utils.SmemAllocator()
    drain = allocator.allocate_array(Float32, 1024)
    start_shared = allocator.allocate_array(cutlass.Int64, 1)
    accum = cute.make_rmem_tensor((chains, 4), Float32)
    for j in cutlass.range_constexpr(chains):
        for k in cutlass.range_constexpr(4):
            accum[j, k] = Float32(j + k)
    cute.arch.barrier()
    if tid == 0:
        start_shared[0] = cute.arch.clock64()
    cute.arch.barrier()
    for _ in cutlass.range(iterations, unroll=1):
        for u in cutlass.range_constexpr(unroll):
            for j in cutlass.range_constexpr(chains):
                values = mma_step(op, accum[j,0], accum[j,1], accum[j,2], accum[j,3], x)
                for k in cutlass.range_constexpr(4):
                    accum[j,k] = values[k]
    total = Float32(0)
    for j in cutlass.range_constexpr(chains):
        for k in cutlass.range_constexpr(4):
            total += accum[j,k]
    drain_f32(drain + tid, total)
    cute.arch.barrier()
    if tid == 0:
        record(timing, start_shared[0])
    output[tid] = drain[tid]
