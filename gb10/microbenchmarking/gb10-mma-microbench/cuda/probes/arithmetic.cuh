#pragma once
#include "common.cuh"
#include "../ops.cuh"
#include <cuda_fp16.h>
#include <type_traits>

enum class Alu { FFMA, DFMA, HFMA2, RCP, EXP2 };
template<Alu op> using Value = std::conditional_t<op==Alu::DFMA, double,
  std::conditional_t<op==Alu::HFMA2, uint32_t, float>>;

template<Alu op> __device__ __forceinline__ void step(Value<op>& x, double coefficient) {
  if constexpr(op==Alu::FFMA)
    asm volatile("fma.rn.f32 %0, %1, %0, 0f3f800000;" : "+f"(x) : "f"(float(coefficient)));
  else if constexpr(op==Alu::DFMA)
    asm volatile("fma.rn.f64 %0, %1, %0, 0d3ff0000000000000;" : "+d"(x) : "d"(coefficient));
  else if constexpr(op==Alu::HFMA2) {
    uint32_t bits=__half_as_ushort(__float2half_rn(float(coefficient)));
    const uint32_t minus_one=bits | (bits<<16);
    asm volatile("fma.rn.f16x2 %0, %1, %0, %2;" : "+r"(x) : "r"(minus_one),"r"(0x3c003c00u));
  } else if constexpr(op==Alu::RCP)
    // ptxas cancels adjacent reciprocal pairs, even in volatile PTX. Use a
    // bounded, non-involutive recurrence and report BOTH instructions.
    asm volatile("add.f32 %0, %0, 0f3f800000; rcp.approx.ftz.f32 %0, %0;" : "+f"(x));
  else if constexpr(op==Alu::EXP2)
    // Bounded recurrence. Inspect SASS: NEG should be an operand modifier.
    asm volatile("{ .reg .f32 negated; neg.f32 negated, %0; "
                 "ex2.approx.ftz.f32 %0, negated; }" : "+f"(x));
}

template<Alu op> __device__ __forceinline__ Value<op> initial(int chain) {
  float x=.25f + .125f*((threadIdx.x+chain)%4);
  if constexpr(op==Alu::HFMA2) {
    uint32_t bits=__half_as_ushort(__float2half_rn(x));
    return bits | (bits<<16);
  }
  else if constexpr(op==Alu::RCP) return 1.25f+float((threadIdx.x+chain)%4);
  else return x;
}
template<Alu op> __device__ __forceinline__ double decoded(Value<op> x) {
  if constexpr(op==Alu::HFMA2) {
    return double(__half2float(__ushort_as_half(uint16_t(x))))+
           double(__half2float(__ushort_as_half(uint16_t(x>>16))));
  } else return double(x);
}

template<Alu op,int N>
__global__ void arithmetic(int iterations,double coefficient,Timing* timing,double* output) {
  __shared__ double drain[1024];
  __shared__ unsigned long long start;
  Value<op> x[N];
  #pragma unroll
  for(int j=0;j<N;j++) x[j]=initial<op>(j);
  __syncthreads();
  if(threadIdx.x==0) start=clock64();
  __syncthreads();
  #pragma unroll 1
  for(int i=0;i<iterations;i++) {
    #pragma unroll
    for(int u=0;u<16;u++)
      #pragma unroll
      for(int j=0;j<N;j++) step<op>(x[j],coefficient);
  }
  double sum=0;
  #pragma unroll
  for(int j=0;j<N;j++) sum+=decoded<op>(x[j]);
  drain[threadIdx.x]=sum;
  __syncthreads();
  if(threadIdx.x==0) record(timing,start);
  output[threadIdx.x]=drain[threadIdx.x];
}

// Fixed 16-warps/CTA, four independent MMA fragments. Scalar chains are also
// independent of each other and of MMA. Mode 0=MMA, 1=scalar, 2=both.
template<Alu op,int R,int Mode>
__global__ void mixed(int iterations,float coefficient,Timing* timing,double* output) {
  __shared__ volatile double drain[512];
  __shared__ unsigned long long start;
  float d[4][4], scalar[4][R];
  #pragma unroll
  for(int j=0;j<4;j++) {
    #pragma unroll
    for(int k=0;k<4;k++) d[j][k]=float(j+k);
    #pragma unroll
    for(int r=0;r<R;r++) scalar[j][r]=initial<op>(j+r);
  }
  __syncthreads();
  if(threadIdx.x==0) start=clock64();
  __syncthreads();
  #pragma unroll 1
  for(int i=0;i<iterations;i++) {
    #pragma unroll
    for(int u=0;u<16;u++)
      #pragma unroll
      for(int j=0;j<4;j++) {
        if constexpr(Mode!=1) mma<Op::FP16_K16>(d[j],0x3c003c00);
        if constexpr(Mode!=0) {
          #pragma unroll
          for(int r=0;r<R;r++) step<op>(scalar[j][r],coefficient);
        }
      }
  }
  double sum_mma=0, sum_scalar=0;
  #pragma unroll
  for(int j=0;j<4;j++) {
    if constexpr(Mode!=1) {
      #pragma unroll
      for(int k=0;k<4;k++) sum_mma+=d[j][k];
    }
    if constexpr(Mode!=0) {
      #pragma unroll
      for(int r=0;r<R;r++) sum_scalar+=scalar[j][r];
    }
  }
  drain[threadIdx.x]=sum_mma+sum_scalar;
  __syncthreads();
  if(threadIdx.x==0) record(timing,start);
  output[2*threadIdx.x]=sum_mma;
  output[2*threadIdx.x+1]=sum_scalar;
}
