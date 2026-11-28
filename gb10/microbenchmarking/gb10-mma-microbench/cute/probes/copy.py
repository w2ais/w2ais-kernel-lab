"""Fully drained copy batches: vector, cp.async, bulk, 2D TMA, and store.

All modes use 128 threads. Bulk/TMA issue on thread zero; ordinary copies
use the whole CTA. Depth counts outstanding transfers within one batch.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Uint64
from .common import asm, barrier_init, barrier_wait, proxy_fence, record


@cute.kernel
def copies(op: cutlass.Constexpr, total_bytes: cutlass.Constexpr,
           tensor_map: cute.Pointer, source: cute.Pointer, destination: cute.Pointer,
           tile_bytes: Int32, depth: Int32, batches: Int32, timing: cute.Pointer):
    tid,_,_ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    barriers = smem.allocate_array(Uint64,16,byte_alignment=128)
    tile = smem.allocate_array(Uint32,total_bytes//4,byte_alignment=128)
    start = smem.allocate_array(Int64,1)
    total_words = tile_bytes*depth//4
    for i in cutlass.range(tid,total_words,128):
        tile[i] = Uint32(0x12340000) + Uint32(i) if op == 'store' else Uint32(0)
    if tid == 0:
        for d in cutlass.range(depth,unroll=1):
            barrier_init((barriers+d).toint())
    proxy_fence()
    cute.arch.barrier()
    if tid == 0:
        start[0] = cute.arch.clock64()
    cute.arch.barrier()
    for batch in cutlass.range(batches,unroll=1):
        if cutlass.const_expr(op in ('vector','cp_async')):
            for d in cutlass.range(depth,unroll=1):
                for offset in cutlass.range(tid*16,tile_bytes,128*16,unroll=1):
                    dst = (tile+(d*tile_bytes+offset)//4).toint()
                    src = (source+(d*tile_bytes+offset)//4).toint()
                    if cutlass.const_expr(op == 'vector'):
                        asm('{ .reg .b32 a,b,c,e; ld.global.cg.v4.u32 {a,b,c,e},[$1]; '
                            'st.shared.v4.u32 [$0],{a,b,c,e}; }','r,l',(dst,src))
                    else:
                        asm('cp.async.cg.shared.global [$0],[$1],16;','r,l',(dst,src))
                if cutlass.const_expr(op == 'cp_async'):
                    asm('cp.async.commit_group;','')
            if cutlass.const_expr(op == 'cp_async'):
                asm('cp.async.wait_group 0;','')
        elif tid == 0:
            for d in cutlass.range(depth,unroll=1):
                shared = (tile+d*tile_bytes//4).toint()
                bar = (barriers+d).toint()
                if cutlass.const_expr(op == 'store'):
                    dst = (destination+d*tile_bytes//4).toint()
                    asm('cp.async.bulk.global.shared::cta.bulk_group [$0],[$1],$2;',
                        'l,r,r',(dst,shared,tile_bytes))
                else:
                    asm('mbarrier.arrive.expect_tx.shared::cta.b64 _,[$0],$1;',
                        'r,r',(bar,tile_bytes))
                    if cutlass.const_expr(op == 'bulk'):
                        src = (source+d*tile_bytes//4).toint()
                        asm('cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0],[$1],$2,[$3];',
                            'r,l,r,r',(shared,src,tile_bytes,bar))
                    else:
                        asm('cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [$0],[$1,{0,$2}],[$3];',
                            'r,l,r,r',(shared,tensor_map.toint(),d*(tile_bytes//256),bar))
            if cutlass.const_expr(op == 'store'):
                # Full completion wait: a .read wait only permits SMEM reuse.
                asm('cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;','')
            else:
                for d in cutlass.range(depth,unroll=1):
                    barrier_wait((barriers+d).toint(),batch & 1)
        cute.arch.barrier()
    if tid == 0:
        record(timing,start[0])
    if cutlass.const_expr(op != 'store'):
        for i in cutlass.range(tid,total_words,128):
            destination[i] = tile[i]
