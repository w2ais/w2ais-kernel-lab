"""Dependent global/shared loads and synchronization round trips."""
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Uint64
from .common import asm, barrier_init, barrier_wait, proxy_fence, record


@cute.kernel
def pointer_chase(policy: cutlass.Constexpr, first: Uint64, iterations: Int32,
                  timing: cute.Pointer, output: cute.Pointer):
    tid,_,_ = cute.arch.thread_idx()
    if tid == 0:
        pointer = first
        start_ns = cute.arch.globaltimer()
        start = cute.arch.clock64()
        for _ in cutlass.range(iterations,unroll=1):
            for u in cutlass.range_constexpr(32):
                pointer = asm(f'ld.global.{policy}.u64 $0, [$1];', '=l,l', (pointer,), (Uint64,))
        # The end timestamps depend on the final load; otherwise the compiler
        # could issue the timer while that load is still outstanding.
        end,end_ns = asm('{ .reg .pred ready; mov.u64 $0,0; mov.u64 $1,0; setp.ne.u64 ready,$2,0; '
                         '@ready mov.u64 $0,%clock64; @ready mov.u64 $1,%globaltimer; }',
                         '=&l,=&l,l',(pointer,),(Uint64,Uint64))
        timing[0] = end - Uint64(start)
        timing[1] = asm('mov.u32 $0,%smid;','=r',returns=(Uint32,))
        timing[2] = end_ns - Uint64(start_ns)
        timing[3] = cute.arch.dynamic_smem_size()
        output[0] = pointer


@cute.kernel
def shared_chase(stride: Int32, iterations: Int32, timing: cute.Pointer, output: cute.Pointer):
    tid,_,_ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    nodes = smem.allocate_array(Uint32,1024)
    drain = smem.allocate_array(Uint32,32)
    start = smem.allocate_array(Int64,1)
    base = nodes.toint()
    for i in cutlass.range(tid,1024,32):
        nodes[i] = base + Uint32(4*i)
    cute.arch.barrier()
    pointer = base + Uint32(4*tid*stride)
    if tid == 0:
        start[0] = cute.arch.clock64()
    cute.arch.barrier()
    for _ in cutlass.range(iterations,unroll=1):
        for u in cutlass.range_constexpr(32):
            pointer = asm('ld.shared.u32 $0,[$1];','=r,r',(pointer,),(Uint32,))
    asm('st.volatile.shared.u32 [$0],$1;','r,r',((drain+tid).toint(),pointer))
    cute.arch.barrier()
    if tid == 0:
        record(timing,start[0])
    output[tid] = drain[tid] - base


@cute.kernel
def synchronization(op: cutlass.Constexpr, iterations: Int32, timing: cute.Pointer, output: cute.Pointer):
    tid,_,_ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    barrier = smem.allocate_array(Uint64,1)
    start = smem.allocate_array(Int64,1)
    address = barrier.toint()
    if tid == 0:
        barrier_init(address)
    cute.arch.barrier()
    if tid == 0:
        start[0] = cute.arch.clock64()
    cute.arch.barrier()
    phase = Int32(0)
    for _ in cutlass.range(iterations,unroll=1):
        for u in cutlass.range_constexpr(16):
            if cutlass.const_expr(op == 'cta'):
                cute.arch.barrier()
            elif tid == 0:
                if cutlass.const_expr(op == 'mbarrier'):
                    asm('mbarrier.arrive.shared::cta.b64 _,[$0];','r',(address,))
                    barrier_wait(address,phase)
                    phase ^= 1
                else:
                    proxy_fence()
    cute.arch.barrier()
    if tid == 0:
        record(timing,start[0])
        output[0] = phase
