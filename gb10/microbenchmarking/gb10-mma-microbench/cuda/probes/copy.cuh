#pragma once
#include "common.cuh"

// 128 threads in every mode. Scalar/vector copies use the whole CTA; TMA and
// bulk transfers use its leader. Depth is a batch of outstanding transfers,
// not a GEMM producer/consumer pipeline. Every batch is fully drained.
// Mode: 0=vector load/store, 1=cp.async, 2=bulk load, 3=tensor-2D load, 4=bulk store.
template<int Mode>
__global__ void copies(const __grid_constant__ CUtensorMap map,const uint32_t* source,
                       uint32_t* destination,int bytes,int depth,int batches,
                       Timing* timing) {
  extern __shared__ __align__(128) unsigned char storage[];
  __shared__ unsigned long long start;
  auto* barriers=reinterpret_cast<uint64_t*>(storage);
  auto* tile=reinterpret_cast<uint32_t*>(storage+128);
  const int total_words=bytes*depth/4;
  for(int i=threadIdx.x;i<total_words;i+=128)
    tile[i]=Mode==4 ? (0x12340000u+uint32_t(i)) : 0;
  if(threadIdx.x==0) for(int d=0;d<depth;d++) barrier_init(shared_address(barriers+d));
  proxy_fence();
  __syncthreads();
  if(threadIdx.x==0) start=clock64();
  __syncthreads();
  #pragma unroll 1
  for(int batch=0;batch<batches;batch++) {
    if constexpr(Mode==0 || Mode==1) {
      for(int d=0;d<depth;d++) {
        for(int offset=threadIdx.x*16;offset<bytes;offset+=128*16) {
          auto* dst=reinterpret_cast<unsigned char*>(tile)+d*bytes+offset;
          auto* src=reinterpret_cast<const unsigned char*>(source)+d*bytes+offset;
          if constexpr(Mode==0) {
            uint32_t a,b,c,e;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" :
              "=r"(a),"=r"(b),"=r"(c),"=r"(e) : "l"(src) : "memory");
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::
              "r"(shared_address(dst)),"r"(a),"r"(b),"r"(c),"r"(e) : "memory");
          } else asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                             "r"(shared_address(dst)), "l"(src) : "memory");
        }
        if constexpr(Mode==1) asm volatile("cp.async.commit_group;" ::: "memory");
      }
      if constexpr(Mode==1) asm volatile("cp.async.wait_group 0;" ::: "memory");
    } else if(threadIdx.x==0) {
      for(int d=0;d<depth;d++) {
        const unsigned dst=shared_address(tile)+d*bytes;
        const unsigned bar=shared_address(barriers+d);
        if constexpr(Mode==4) {
          asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::
            "l"(destination+d*bytes/4),"r"(dst),"r"(bytes) : "memory");
        } else {
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
            "r"(bar),"r"(bytes) : "memory");
          if constexpr(Mode==2)
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
              "[%0], [%1], %2, [%3];" :: "r"(dst),"l"(source+d*bytes/4),"r"(bytes),"r"(bar) : "memory");
          else
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
              "[%0], [%1, {0,%2}], [%3];" :: "r"(dst),"l"(&map),"r"(d*(bytes/256)),"r"(bar) : "memory");
        }
      }
      if constexpr(Mode==4) {
        asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;" ::: "memory");
      } else for(int d=0;d<depth;d++) barrier_wait(shared_address(barriers+d),batch&1);
    }
    __syncthreads(); // All writers completed before any buffer is reused.
  }
  if(threadIdx.x==0) record(timing,start);
  if constexpr(Mode!=4)
    for(int i=threadIdx.x;i<total_words;i+=128) destination[i]=tile[i];
}
