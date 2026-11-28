"""Scalar recurrence/throughput and independent MMA–scalar overlap probes."""
import cutlass
import cutlass.cute as cute
from cutlass import Float16, Float32, Float64, Int32, Uint32
from .common import asm, drain_f64, record
from .mma import mma_step


@cute.jit
def initial(op: cutlass.Constexpr, lane: Int32, chain: cutlass.Constexpr):
    if cutlass.const_expr(op == 'rcp'):
        value = Float32(1.25) + Float32((lane + chain) % 4)
    elif cutlass.const_expr(op == 'hfma2'):
        x = Float32(0.25) + Float32(0.125) * Float32((lane + chain) % 4)
        value = asm('cvt.rn.f16x2.f32 $0, $1, $1;', '=r,f', (x,), (Uint32,))
    elif cutlass.const_expr(op == 'dfma'):
        value = Float64(0.25) + Float64(0.125) * Float64((lane + chain) % 4)
    else:
        value = Float32(0.25) + Float32(0.125) * Float32((lane + chain) % 4)
    return value


@cute.jit
def scalar_step(op: cutlass.Constexpr, x, coefficient):
    if cutlass.const_expr(op == 'ffma'):
        result = asm('fma.rn.f32 $0, $2, $1, 0f3f800000;', '=f,f,f',
                     (x, Float32(coefficient)), (Float32,))
    elif cutlass.const_expr(op == 'dfma'):
        result = asm('fma.rn.f64 $0, $2, $1, 0d3ff0000000000000;', '=d,d,d',
                     (x, coefficient), (Float64,))
    elif cutlass.const_expr(op == 'hfma2'):
        result = asm('fma.rn.f16x2 $0, $2, $1, $3;', '=r,r,r,r',
                     (x,coefficient,Uint32(0x3c003c00)), (Uint32,))
    elif cutlass.const_expr(op == 'rcp'):
        result = asm('{ .reg .f32 v; add.f32 v, $1, 0f3f800000; rcp.approx.ftz.f32 $0, v; }',
                     '=f,f', (x,), (Float32,))
    else:
        result = asm('{ .reg .f32 v; neg.f32 v, $1; ex2.approx.ftz.f32 $0, v; }',
                     '=f,f', (x,), (Float32,))
    return result


@cute.jit
def decoded(op: cutlass.Constexpr, x):
    if cutlass.const_expr(op == 'hfma2'):
        lo,hi = asm('{ .reg .b16 a,b; mov.b32 {a,b}, $2; cvt.f32.f16 $0, a; cvt.f32.f16 $1, b; }',
                    '=f,=f,r', (x,), (Float32,Float32))
        result = Float64(lo) + Float64(hi)
    else:
        result = Float64(x)
    return result


@cute.kernel
def arithmetic(op: cutlass.Constexpr, chains: cutlass.Constexpr,
               iterations: Int32, coefficient: Float64, timing: cute.Pointer, output: cute.Pointer):
    tid,_,_ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    drain = smem.allocate_array(Float64,1024)
    start = smem.allocate_array(cutlass.Int64,1)
    dtype = Float64 if op == 'dfma' else Uint32 if op == 'hfma2' else Float32
    if cutlass.const_expr(op == 'hfma2'):
        step_coefficient = asm('cvt.rn.f16x2.f32 $0,$1,$1;', '=r,f', (Float32(coefficient),), (Uint32,))
    else:
        step_coefficient = coefficient
    values = cute.make_rmem_tensor((chains,),dtype)
    for j in cutlass.range_constexpr(chains):
        values[j] = initial(op,tid,j)
    cute.arch.barrier()
    if tid == 0:
        start[0] = cute.arch.clock64()
    cute.arch.barrier()
    for _ in cutlass.range(iterations,unroll=1):
        for u in cutlass.range_constexpr(16):
            for j in cutlass.range_constexpr(chains):
                values[j] = scalar_step(op,values[j],step_coefficient)
    total = Float64(0)
    for j in cutlass.range_constexpr(chains):
        total += decoded(op,values[j])
    drain_f64(drain+tid,total)
    cute.arch.barrier()
    if tid == 0:
        record(timing,start[0])
    output[tid] = drain[tid]


@cute.kernel
def mixed(op: cutlass.Constexpr, ratio: cutlass.Constexpr, mode: cutlass.Constexpr,
          iterations: Int32, coefficient: Float64, timing: cute.Pointer, output: cute.Pointer):
    """Mode 0=MMA only, 1=scalar only, 2=both; operands are independent."""
    tid,_,_ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    drain = smem.allocate_array(Float64,512)
    start = smem.allocate_array(cutlass.Int64,1)
    accum = cute.make_rmem_tensor((4,4),Float32)
    scalar = cute.make_rmem_tensor((4,ratio),Float32)
    for j in cutlass.range_constexpr(4):
        for k in cutlass.range_constexpr(4):
            accum[j,k] = Float32(j+k)
        for r in cutlass.range_constexpr(ratio):
            scalar[j,r] = initial(op,tid,j+r)
    cute.arch.barrier()
    if tid == 0:
        start[0] = cute.arch.clock64()
    cute.arch.barrier()
    for _ in cutlass.range(iterations,unroll=1):
        for u in cutlass.range_constexpr(16):
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(mode != 1):
                    values = mma_step('fp16',accum[j,0],accum[j,1],accum[j,2],accum[j,3],Uint32(0x3c003c00))
                    for k in cutlass.range_constexpr(4):
                        accum[j,k] = values[k]
                if cutlass.const_expr(mode != 0):
                    for r in cutlass.range_constexpr(ratio):
                        scalar[j,r] = scalar_step(op,scalar[j,r],coefficient)
    sum_mma, sum_scalar = Float64(0),Float64(0)
    for j in cutlass.range_constexpr(4):
        if cutlass.const_expr(mode != 1):
            for k in cutlass.range_constexpr(4):
                sum_mma += Float64(accum[j,k])
        if cutlass.const_expr(mode != 0):
            for r in cutlass.range_constexpr(ratio):
                sum_scalar += Float64(scalar[j,r])
    drain_f64(drain+tid,sum_mma+sum_scalar)
    cute.arch.barrier()
    if tid == 0:
        record(timing,start[0])
    output[2*tid] = sum_mma
    output[2*tid+1] = sum_scalar
