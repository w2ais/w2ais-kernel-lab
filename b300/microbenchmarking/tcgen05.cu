// One CTA, one SM. Operands stay in shared memory and all MMAs update one TMEM tile.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int M = 128, N = 128, issues = 256;
enum Format { fp16, bf16, tf32, fp4_k32, mxfp4, nvfp4 };
struct Operand {
    const char *name;
    int k;
    unsigned one, half, scale_one, scale_two;
};
constexpr Operand formats[] = {
    {"fp16", 16, 0x3c003c00, 0x38003800, 0, 0},
    {"bf16", 16, 0x3f803f80, 0x3f003f00, 0, 0},
    {"tf32", 8, 0x3f800000, 0x3f000000, 0, 0},
    {"fp4_k32", 32, 0x22222222, 0x11111111, 0x7f7f7f7f, 0x80808080},
    {"mxfp4", 64, 0x22222222, 0x11111111, 0x7f7f7f7f, 0x80808080},
    {"nvfp4", 64, 0x22222222, 0x11111111, 0x38383838, 0x40404040},
};

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

// K-major, unswizzled 16-byte strips. Eight rows per atom, 128 rows per K strip.
__device__ unsigned long long descriptor(const void *ptr) {
    unsigned addr = __cvta_generic_to_shared(ptr);
    return (addr >> 4) | (128ull << 16) | (8ull << 32) | (1ull << 46);
}

__device__ void commit_wait(unsigned barrier, unsigned phase) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(barrier) : "memory");
    asm volatile("{ .reg .pred done; wait_here: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;"
                 " @!done bra wait_here; }" :: "r"(barrier), "r"(phase) : "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

template<Format F>
__device__ __forceinline__ void issue(unsigned dest, unsigned long long a, unsigned long long b,
                                      unsigned sfa, unsigned sfb, int accumulate) {
    constexpr unsigned type = F == fp4_k32 ? 5 : F == tf32 ? 2 : F == fp16 ? 0 : 1;
    constexpr unsigned shape = ((M / 16) << 24) | ((N / 8) << 17) | (type << 7) | (type << 10);
    if constexpr (F <= tf32) {
        constexpr unsigned idesc = shape | (1 << 4); // FP32 output
#define DENSE(KIND) asm volatile("{ .reg .pred p; setp.ne.u32 p,%4,0; tcgen05.mma.cta_group::1.kind::" KIND \
    " [%0], %1, %2, %3, {%5,%5,%5,%5}, p; }" :: "r"(dest),"l"(a),"l"(b),"r"(idesc),"r"(accumulate),"r"(0u) : "memory")
        if constexpr (F == tf32) { DENSE("tf32"); } else { DENSE("f16"); }
#undef DENSE
    } else {
        constexpr unsigned idesc = shape | ((F == nvfp4 ? 0 : 1) << 23);
#define SCALED(KIND, VEC) asm volatile("{ .reg .pred p; setp.ne.u32 p,%6,0; tcgen05.mma.cta_group::1.kind::" KIND \
    ".block_scale.scale_vec::" VEC " [%0], %1, %2, %3, [%4], [%5], p; }" :: \
    "r"(dest),"l"(a),"l"(b),"r"(idesc),"r"(sfa),"r"(sfb),"r"(accumulate) : "memory")
        if constexpr (F == nvfp4) { SCALED("mxf4nvf4", "4X"); }
        else if constexpr (F == mxfp4) { SCALED("mxf4", "2X"); }
        else { SCALED("mxf8f6f4", "1X"); }
#undef SCALED
    }
}

template<Format F>
__global__ void measure(unsigned a_bits, unsigned b_bits, unsigned scale_a, unsigned scale_b,
                        int batch, unsigned long long *elapsed, float *tile) {
    __shared__ __align__(128) unsigned a[M * 32 / 4], b[N * 32 / 4];
    __shared__ __align__(8) unsigned long long barrier;
    __shared__ unsigned base;
    const int tid = threadIdx.x;
    unsigned bar = __cvta_generic_to_shared(&barrier);
    if (tid == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
    if (tid < 32) asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::
                               "r"(unsigned(__cvta_generic_to_shared(&base))) : "memory");
    for (int i = tid; i < M * 32 / 4; i += blockDim.x) {
        // K32 E2M1 in SMEM: 16 packed nibbles and 8 padding bytes per 16-byte strip.
        const bool padding = F == fp4_k32 && (i % 4 >= 2);
        a[i] = padding ? 0 : a_bits;
        b[i] = padding ? 0 : b_bits;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    unsigned row_offset = (tid / 32 * 32) << 16;
    unsigned sfa = base + N, sfb = sfa + 4;
    if constexpr (F >= fp4_k32) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%1,%1,%1};" ::
                     "r"(sfa + row_offset), "r"(scale_a) : "memory");
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%1,%1,%1};" ::
                     "r"(sfb + row_offset), "r"(scale_b) : "memory");
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (tid == 0) {
        auto da = descriptor(a), db = descriptor(b);
        issue<F>(base, da, db, sfa, sfb, 0);
        commit_wait(bar, 0);
        unsigned phase = 1;
        auto start = clock64();
        for (int offset = 0; offset < issues; offset += batch) {
            #pragma unroll 1
            for (int i = 0; i < batch; ++i) issue<F>(base, da, db, sfa, sfb, 1);
            commit_wait(bar, phase);
            phase ^= 1;
        }
        *elapsed = clock64() - start;
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    for (int col = 0; col < N; ++col) {
        unsigned word;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" :
                     "=r"(word) : "r"(base + row_offset + col) : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        tile[tid * N + col] = __uint_as_float(word);
    }
    __syncthreads();
    if (tid < 32) {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(base) : "memory");
    }
}

using Kernel = decltype(&measure<fp16>);
int main(int argc, char **argv) {
    if (argc != 4) return 1;
    int rounds = std::atoi(argv[1]), samples = std::atoi(argv[2]);
    FILE *out = std::fopen(argv[3], "w");
    if (!out || rounds < 1 || samples < 1) return 1;
    std::fprintf(out, "round,sample,op,m,n,k,batch,instructions,cycles\n");
    unsigned long long *ticks;
    float *tile;
    check(cudaMalloc(&ticks, sizeof(*ticks)));
    check(cudaMalloc(&tile, M*N*sizeof(float)));
    const Kernel kernels[] = {measure<fp16>,measure<bf16>,measure<tf32>,measure<fp4_k32>,measure<mxfp4>,measure<nvfp4>};
    std::vector<float> result(M*N);
    std::vector<std::pair<int,int>> cases;
    for (int f = fp16; f <= nvfp4; ++f) for (int batch : {1, 4, 16, 64}) cases.emplace_back(f, batch);
    std::mt19937 rng(103);
    for (int r = 0; r < rounds; ++r) {
        std::shuffle(cases.begin(), cases.end(), rng);
        for (auto [f,batch] : cases) {
            auto spec = formats[f];
            for (int sample = -2; sample < samples; ++sample) {
                bool alternate = sample == -2;
                unsigned a_bits = alternate ? spec.half : spec.one;
                unsigned b_bits = alternate ? spec.half : spec.one;
                unsigned sfa = alternate ? spec.scale_two : spec.scale_one, sfb = spec.scale_one;
                void *args[] = {&a_bits, &b_bits, &sfa, &sfb, &batch, &ticks, &tile};
                check(cudaLaunchKernel(reinterpret_cast<const void *>(kernels[f]), dim3(1), dim3(M), args, 0, nullptr));
                unsigned long long cycles;
                check(cudaMemcpy(&cycles, ticks, sizeof(cycles), cudaMemcpyDeviceToHost));
                check(cudaMemcpy(result.data(), tile, M*N*sizeof(float), cudaMemcpyDeviceToHost));
                float expected = (issues + 1) * spec.k * (alternate ? (f >= fp4_k32 ? .5f : .25f) : 1.f);
                for (int i = 0; i < M*N; ++i) if (result[i] != expected) {
                    std::fprintf(stderr, "%s batch %d sample %d output %d: expected %g, got %g\n",
                                 spec.name, batch, sample, i, expected, result[i]);
                    return 2;
                }
                if (sample >= 0) std::fprintf(out, "%d,%d,%s,%d,%d,%d,%d,%d,%llu\n",
                                              r,sample,spec.name,M,N,spec.k,batch,issues,cycles);
            }
            std::fflush(out);
        }
    }
    std::fclose(out);
    check(cudaFree(tile)); check(cudaFree(ticks));
}
