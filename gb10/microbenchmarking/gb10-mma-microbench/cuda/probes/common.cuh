#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1); } } while(0)
#define DRIVER(call) do { auto e=(call); if(e!=CUDA_SUCCESS) { \
  const char* s=nullptr; cuGetErrorString(e,&s); fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,s); std::exit(1); } } while(0)

struct Timing { unsigned long long cycles; unsigned sm; unsigned long long nanoseconds; };
__device__ __forceinline__ unsigned shared_address(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void record(Timing* out, unsigned long long start) {
  out->cycles=clock64()-start;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(out->sm));
}
__device__ __forceinline__ void barrier_init(unsigned addr) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void barrier_wait(unsigned addr, int phase) {
  asm volatile("{ .reg .pred p; wait_again: "
    "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
    "@!p bra wait_again; }" :: "r"(addr), "r"(phase) : "memory");
}
__device__ __forceinline__ void proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
