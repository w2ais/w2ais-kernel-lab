#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "ops.cuh"

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

struct Timing { unsigned long long cycles; unsigned sm; };

// Exactly ONE CTA: every active warp runs on the same SM. No division by the
// GPU's SM count, no assumption about distribution of a multi-CTA grid.
// UNROLL=16 amortizes loop control. N independent accumulator fragments allow
// ILP; N=1 keeps every MMA dependent on the previous MMA's C output.
template<Op op, int N, int UNROLL=16>
__global__ void measure(int iterations, uint32_t input, Timing* timing,
                        float* output) {
  __shared__ float drain[1024];
  __shared__ unsigned long long start;
  float d[N][4];
  #pragma unroll
  for (int j=0; j<N; ++j)
    #pragma unroll
    for (int k=0; k<4; ++k) d[j][k] = float(j + k);
  __syncthreads();
  if (threadIdx.x == 0) start = clock64();
  __syncthreads();
  #pragma unroll 1
  for (int i=0; i<iterations; ++i) {
    #pragma unroll
    for (int u=0; u<UNROLL; ++u)
      #pragma unroll
      for (int j=0; j<N; ++j) mma<op>(d[j], input);
  }
  // Depend on ALL accumulators before the end timestamp. The shared store
  // and CTA barrier drain the outstanding arithmetic of every active warp.
  // These fixed costs are exposed by the length sweep, not assumed to be zero.
  float sum = 0;
  #pragma unroll
  for (int j=0; j<N; ++j)
    #pragma unroll
    for (int k=0; k<4; ++k) sum += d[j][k];
  drain[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    timing->cycles = clock64() - start;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(timing->sm));
  }
  output[threadIdx.x] = drain[threadIdx.x];
}

struct Spec { const char* name; Op op; int k; uint32_t one, half; };
const Spec specs[] = {
  {"fp16_k8", Op::FP16_K8, 8, 0x3c003c00, 0x38003800},
  {"fp16", Op::FP16_K16, 16, 0x3c003c00, 0x38003800},
  {"bf16", Op::BF16, 16, 0x3f803f80, 0x3f003f00},
  {"tf32", Op::TF32, 8, 0x3f800000, 0x3f000000},
  // mxf8f6f4 places E2M1 in bits [5:2] of each byte, not the low nibble.
  {"fp4_k32", Op::FP4_K32, 32, 0x08080808, 0x04040404},
  {"mxfp4", Op::MXFP4, 64, 0x22222222, 0x11111111},
  {"nvfp4", Op::NVFP4, 64, 0x22222222, 0x11111111},
};

template<Op op, int N, int U>
void run(const Spec& spec, int warps, int iterations, int samples, int variant,
         Timing* dt, float* dout) {
  cudaFuncAttributes attr{};
  CUDA(cudaFuncGetAttributes(&attr, measure<op,N,U>));
  int resident=0;
  CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, measure<op,N,U>, warps*32, 0));
  auto launch = [&] {
    measure<op,N,U><<<1, warps*32>>>(iterations, variant ? spec.half : spec.one, dt, dout);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
  };
  for (int i=0; i<3; ++i) launch();
  double expected=0;
  for (int j=0; j<N; ++j)
    for (int k=0; k<4; ++k)
      expected += j+k+double(iterations)*U*spec.k*(variant ? .25 : 1.);
  std::vector<float> values(warps*32);
  for (int sample=0; sample<samples; ++sample) {
    launch();
    Timing t{};
    CUDA(cudaMemcpy(&t, dt, sizeof(t), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(values.data(), dout, values.size()*sizeof(float), cudaMemcpyDeviceToHost));
    double max_error=0;
    for (float value : values) {
      if (!std::isfinite(value)) { fprintf(stderr,"Non-finite output\n"); std::exit(2); }
      max_error = std::max(max_error, std::abs(double(value)-expected));
    }
    if (max_error > std::max(1., std::abs(expected))*2e-6) {
      fprintf(stderr, "Validation failed: %s N=%d U=%d expected=%.0f got=%.0f error=%g\n",
        spec.name, N, U, expected, values[0], max_error);
      std::exit(2);
    }
    unsigned long long mmas = (unsigned long long)warps*iterations*U*N;
    printf("%s,%d,%d,%d,%d,%d,%d,%llu,%u,%llu,%d,%.9f,%.9f,%d,%zu,%zu,%d,%.9g\n",
      spec.name, warps, N, U, iterations, variant, sample, t.cycles, t.sm,
      mmas, 2*16*8*spec.k, double(mmas)/t.cycles,
      double(mmas)*2*16*8*spec.k/t.cycles, attr.numRegs,
      attr.sharedSizeBytes, attr.localSizeBytes, resident, max_error);
    fflush(stdout);
  }
}

template<Op op, int U>
void dispatch_chains(const Spec& spec, int warps, int n, int iters, int samples,
                     int variant, Timing* dt, float* out) {
  switch(n) {
    case 1: return run<op,1,U>(spec,warps,iters,samples,variant,dt,out);
    case 2: return run<op,2,U>(spec,warps,iters,samples,variant,dt,out);
    case 4: return run<op,4,U>(spec,warps,iters,samples,variant,dt,out);
    case 8: return run<op,8,U>(spec,warps,iters,samples,variant,dt,out);
    default: fprintf(stderr,"Invalid independent chain count\n"); std::exit(1);
  }
}

template<Op op>
void dispatch_unroll(const Spec& spec, int warps, int n, int u, int iters,
                     int samples, int variant, Timing* dt, float* out) {
  if (u==16) return dispatch_chains<op,16>(spec,warps,n,iters,samples,variant,dt,out);
  if (u==1) return dispatch_chains<op,1>(spec,warps,n,iters,samples,variant,dt,out);
  fprintf(stderr,"Invalid unroll count\n"); std::exit(1);
}

int main(int argc, char** argv) {
  cudaDeviceProp p{};
  CUDA(cudaGetDeviceProperties(&p,0));
  if (argc==2 && std::string(argv[1])=="--info") {
    int driver=0, runtime=0;
    CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
    printf("{\"name\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,"
      "\"warp_size\":%d,\"threads_per_sm\":%d,\"registers_per_sm\":%d,"
      "\"shared_bytes_per_sm\":%zu,\"l2_bytes\":%d,\"driver_api\":%d,\"runtime\":%d}\n",
      p.name,p.major,p.minor,p.multiProcessorCount,p.warpSize,p.maxThreadsPerMultiProcessor,
      p.regsPerMultiprocessor,p.sharedMemPerMultiprocessor,p.l2CacheSize,driver,runtime);
    return 0;
  }
  if (argc!=2) { fprintf(stderr,"Usage: bench PLAN.csv | --info\n"); return 1; }
  if (p.major!=12) { fprintf(stderr,"This suite targets sm_12x; recompile/adapt for other architectures.\n"); return 1; }
  Timing* dt; float* out;
  CUDA(cudaMalloc(&dt,sizeof(Timing))); CUDA(cudaMalloc(&out,1024*sizeof(float)));
  std::ifstream input(argv[1]);
  if (!input) { fprintf(stderr,"Cannot open plan\n"); return 1; }
  printf("op,warps,chains,unroll,iterations,variant,sample,cycles,sm_id,mmas,flops_per_mma,mma_per_sm_cycle,flops_per_sm_cycle,registers_per_thread,shared_bytes,local_bytes,resident_cta_capacity,max_abs_error\n");
  std::string line;
  while (std::getline(input,line)) {
    if (line.empty() || line.rfind("op,",0)==0) continue;
    std::replace(line.begin(),line.end(),',',' ');
    std::istringstream fields(line);
    std::string name; int w,n,u,iters,samples,variant;
    if (!(fields>>name>>w>>n>>u>>iters>>samples>>variant) || w<1 || w>32 ||
        iters<1 || samples<1 || (variant!=0 && variant!=1)) return 1;
    const Spec* s=nullptr;
    for (const auto& candidate: specs) if (name==candidate.name) s=&candidate;
    if (!s) { fprintf(stderr,"Unknown op %s\n",name.c_str()); return 1; }
    switch(s->op) {
#define CASE(OP) case Op::OP: dispatch_unroll<Op::OP>(*s,w,n,u,iters,samples,variant,dt,out); break
      CASE(FP16_K8); CASE(FP16_K16); CASE(BF16); CASE(TF32);
      CASE(FP4_K32); CASE(MXFP4); CASE(NVFP4);
#undef CASE
    }
  }
  CUDA(cudaFree(dt)); CUDA(cudaFree(out));
}
