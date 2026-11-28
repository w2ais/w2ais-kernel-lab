#pragma once
#include <cstdint>

// Each invocation is one warp-wide PTX MMA, with four FP32 outputs per lane.
// Inputs are uniform, finite, nonzero values, supplied at runtime. Thus every
// output should increase by K * a * b, regardless of the fragment mapping.
enum class Op { FP16_K8, FP16_K16, BF16, TF32, FP4_K32, MXFP4, NVFP4 };

template<Op op> __device__ __forceinline__ void mma(float (&d)[4], uint32_t x) {
  if constexpr (op == Op::FP16_K8) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%4}, {%4}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(x));
  } else if constexpr (op == Op::FP16_K16) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(x));
  } else if constexpr (op == Op::BF16) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(x));
  } else if constexpr (op == Op::TF32) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(x));
  } else if constexpr (op == Op::FP4_K32) {
    uint32_t scale = 0x7f7f7f7f; // UE8M0: all scales = 1.
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
      "m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3}, "
      "%5, {0,0}, %5, {0,0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(x), "r"(scale));
  } else if constexpr (op == Op::MXFP4) {
    uint32_t scale = 0x7f7f7f7f;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3}, "
      "%5, {0,0}, %5, {0,0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(x), "r"(scale));
  } else if constexpr (op == Op::NVFP4) {
    uint32_t scale = 0x38383838; // UE4M3: all scales = 1.
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3}, "
      "%5, {0,0}, %5, {0,0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(x), "r"(scale));
  }
}
