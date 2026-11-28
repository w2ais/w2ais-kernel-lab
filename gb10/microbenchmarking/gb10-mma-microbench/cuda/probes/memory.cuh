#pragma once
#include "common.cuh"

template<bool CG>
__global__ void pointer_chase(uint64_t first,int iterations,Timing* timing,uint64_t* output) {
  if(threadIdx.x!=0) return;
  uint64_t p=first;
  uint64_t start_ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_ns));
  const auto start=clock64();
  #pragma unroll 1
  for(int i=0;i<iterations;i++) {
    #pragma unroll
    for(int u=0;u<32;u++) {
      if constexpr(CG) asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p) :: "memory");
      else asm volatile("ld.global.ca.u64 %0, [%0];" : "+l"(p) :: "memory");
    }
  }
  // The final load must finish before the end timestamp. A predicate dependent
  // on its non-null pointer prevents the timer from passing it in the schedule.
  uint64_t end,end_ns;
  asm volatile("{ .reg .pred ready; mov.u64 %0, 0; mov.u64 %1, 0; setp.ne.u64 ready, %2, 0; "
               "@ready mov.u64 %0, %%clock64; @ready mov.u64 %1, %%globaltimer; }" :
               "=&l"(end),"=&l"(end_ns) : "l"(p) : "memory");
  timing->cycles=end-start;
  timing->nanoseconds=end_ns-start_ns;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(timing->sm));
  output[0]=p;
}

__global__ void shared_chase(int stride,int iterations,Timing* timing,uint64_t* output) {
  __shared__ uint32_t nodes[1024],drain[32];
  __shared__ unsigned long long start;
  const auto base=shared_address(nodes);
  for(int i=threadIdx.x;i<1024;i+=32) nodes[i]=base+4*i;
  __syncthreads();
  uint32_t p=base+4*threadIdx.x*stride;
  if(threadIdx.x==0) start=clock64();
  __syncthreads();
  #pragma unroll 1
  for(int i=0;i<iterations;i++) {
    #pragma unroll
    for(int u=0;u<32;u++) asm volatile("ld.shared.u32 %0, [%0];" : "+r"(p) :: "memory");
  }
  drain[threadIdx.x]=p;
  __syncthreads();
  if(threadIdx.x==0) record(timing,start);
  output[threadIdx.x]=drain[threadIdx.x]-base;
}

// The mbarrier loop is a one-arrival / wait round trip. The fence loop is idle:
// it has no pending stores to drain. Neither is isolated instruction latency.
template<int Mode>
__global__ void synchronization(int iterations,Timing* timing,uint64_t* output) {
  __shared__ uint64_t barrier,start;
  unsigned b=shared_address(&barrier);
  if(threadIdx.x==0) barrier_init(b);
  __syncthreads();
  if(threadIdx.x==0) start=clock64();
  __syncthreads();
  int phase=0;
  #pragma unroll 1
  for(int i=0;i<iterations;i++) {
    #pragma unroll
    for(int u=0;u<16;u++) {
      if constexpr(Mode==0) __syncthreads();
      else if(threadIdx.x==0) {
        if constexpr(Mode==1) {
          asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
          barrier_wait(b,phase); phase^=1;
        } else proxy_fence();
      }
    }
  }
  __syncthreads();
  if(threadIdx.x==0) { record(timing,start); output[0]=phase; }
}
