#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <initializer_list>
#define CHECK(call)                                                            \
  do {                                                                         \
    auto error = (call);                                                       \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "%s\n", cudaGetErrorString(error));                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
constexpr int threads = 512, groups = 4;
__global__ void probe(int loops, unsigned long long *elapsed, unsigned *check) {
  __shared__ uint4 words[threads * groups];
  int t = threadIdx.x;
  for (int j = 0; j < groups; j++) {
    unsigned i = t + j * threads;
    words[i] = make_uint4(i, i + 1, i + 2, i + 3);
  }
  unsigned base = __cvta_generic_to_shared(words + t);
  uint4 value[groups];
  __syncthreads();
  unsigned long long start = clock64();
#pragma unroll 1
  for (int i = 0; i < loops; i++) {
#pragma unroll
    for (int j = 0; j < groups; j++) {
      asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(value[j].x), "=r"(value[j].y), "=r"(value[j].z),
                     "=r"(value[j].w)
                   : "r"(base + j * threads * 16)
                   : "memory");
    }
  }
  unsigned sum = 0;
  for (int j = 0; j < groups; j++)
    sum += value[j].x + value[j].y + value[j].z + value[j].w;
  check[t] = sum;
  __syncthreads();
  unsigned long long end = clock64();
  if (t == 0)
    *elapsed = end - start;
}
int main() {
  unsigned long long *e, h;
  unsigned *c, out[threads];
  CHECK(cudaMalloc(&e, sizeof(h)));
  CHECK(cudaMalloc(&c, sizeof(out)));
  for (int r = -2; r < 7; r++) {
    for (int k : {256, 1024}) {
      probe<<<1, threads>>>(k, e, c);
      CHECK(cudaMemcpy(&h, e, sizeof(h), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(out, c, sizeof(out), cudaMemcpyDeviceToHost));
      for (int t = 0; t < threads; t++) {
        unsigned expected = 0;
        for (int j = 0; j < groups; j++)
          expected += 4 * (t + j * threads) + 6;
        if (out[t] != expected) {
          printf("thread %d got %u expected %u\n", t, out[t], expected);
          return 2;
        }
      }
      if (r >= 0)
        printf("%d %d %llu\n", r, k, h);
    }
  }
}
