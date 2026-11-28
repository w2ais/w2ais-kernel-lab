#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <utility>
void chk(cudaError_t s) {
  if (s != cudaSuccess) {
    puts(cudaGetErrorString(s));
    std::exit(1);
  }
}
__global__ void progress(unsigned long long *trace, float *output, int stride,
                         float f) {
  __shared__ unsigned long long stamps[4][65];
  int t = threadIdx.x, w = t / 32, j = w / stride;
  bool active = (w % stride == 0 && j < 4);
  float v = t;
  f += float(t & 1);
  __syncthreads();
  if (active) {
    for (int block = 0; block <= 64; ++block) {
      unsigned long long at = clock64();
      if ((t & 31) == 0)
        stamps[j][block] = at;
      if (block == 64)
        break;
#pragma unroll
      for (int i = 0; i < 64; ++i)
        asm volatile("fma.rn.f32 %0,%1,%1,%0;" : "+f"(v) : "f"(f));
    }
  }
  output[t] = v;
  __syncthreads();
  if (t < 260)
    trace[t] = stamps[t / 65][t % 65];
}
int main() {
  unsigned long long *s;
  float *v;
  chk(cudaMallocManaged(&s, 260 * sizeof(*s)));
  chk(cudaMallocManaged(&v, 512 * sizeof(*v)));
  for (int r = -2; r < 20; r++)
    for (int i = 0; i < 2; i++) {
      int stride = ((r + i) % 2 == 0) ? 1 : 4;
      progress<<<1, 512>>>(s, v, stride, 1.f);
      chk(cudaDeviceSynchronize());
      for (int t = 0; t < 512; t++) {
        int w = t / 32;
        float target =
            t +
            (w % stride == 0 && w / stride < 4 ? 4096 * ((t & 1) ? 4 : 1) : 0);
        if (v[t] != target)
          return 2;
      }
      if (r >= 0)
        for (int j = 0; j < 4; j++)
          for (int c = 0; c <= 64; c++)
            printf("%d,%d,%d,%d,%llu\n", r, stride, j, c, s[j * 65 + c]);
    }
}
