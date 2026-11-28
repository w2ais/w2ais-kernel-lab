#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

enum Op { idle, fma32, warp_exp2, mma };
constexpr int streams = 8, warps = 16, threads = 32 * warps;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

template<Op op>
__device__ __noinline__ float run_streams(int trips, float factor, unsigned packed) {
    float values[streams][op == mma ? 4 : 1] = {};
    #pragma unroll 1
    for (int i = 0; i < trips; ++i) {
        #pragma unroll
        for (int j = 0; j < streams; ++j) {
            float *v = values[j];
            if constexpr (op == fma32) {
                asm volatile("fma.rn.f32 %0, %1, %1, %0;" : "+f"(v[0]) : "f"(factor));
            } else if constexpr (op == warp_exp2) {
                asm volatile("{ .reg .f32 t; neg.f32 t, %0; ex2.approx.ftz.f32 %0,t; }" : "+f"(v[0]));
            } else {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};"
                    : "+f"(v[0]), "+f"(v[1]), "+f"(v[2]), "+f"(v[3]) : "r"(packed));
            }
        }
    }
    float sum = 0;
    for (auto &fragment : values) for (float x : fragment) sum += x;
    return sum;
}

__global__ void pair_probe(int first, int second, Op a, Op b, int trips,
                           float factor, unsigned packed, unsigned long long *ticks,
                           float *output) {
    int lane = threadIdx.x;
    int warp = lane / 32;
    Op role = warp == first ? a : (warp == second ? b : idle);
    __syncthreads();
    unsigned long long start = clock64();
    __syncthreads();
    float sum = 0;
    switch (role) {
        case fma32:   sum = run_streams<fma32>(trips * 8, factor + float(lane & 1), packed); break;
        case warp_exp2: sum = run_streams<warp_exp2>(trips * 2, factor, packed); break;
        case mma:     sum = run_streams<mma>(trips, factor, packed); break;
        default: break;
    }
    output[lane] = sum;
    __syncthreads();
    if (lane == 0) *ticks = clock64() - start;
}

struct Trial { int first, gap, trips; Op a, b; };

float expected(Op op, int trips) {
    if (op == fma32) return float(trips * 8 * streams);
    if (op == mma) return float(trips * streams * 4 * 16);
    if (op == warp_exp2) {
        float x = 0;
        for (int i = 0; i < 2 * trips; ++i) x = std::exp2(-x);
        return x * streams;
    }
    return 0;
}

int main(int argc, char **argv) {
    const bool placement = argc == 2 && std::strcmp(argv[1], "--placement") == 0;
    if (argc != 1 && !placement) {
        std::fprintf(stderr, "Usage: %s [--placement]\n", argv[0]);
        return 1;
    }
    unsigned long long *ticks;
    float *output;
    check(cudaMallocManaged(&ticks, sizeof(*ticks)));
    check(cudaMallocManaged(&output, threads * sizeof(*output)));
    // Three launches for the separate Nsight Compute SMSP placement check.
    if (placement) {
        for (int gap : {1, 4, 8}) {
            pair_probe<<<1, threads>>>(0, gap, mma, mma, 1024, 1.0f, 0x3f803f80u, ticks, output);
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
        }
        check(cudaFree(output));
        check(cudaFree(ticks));
        return 0;
    }

    std::vector<Trial> trials;
    const Op ops[] = {fma32, warp_exp2, mma};
    for (int first = 0; first < warps; ++first) {
        for (int trips : {256, 1024}) {
            for (Op a : ops) {
                trials.push_back({first, 0, trips, a, idle});
                for (Op b : ops) {
                    if (b < a) continue;
                    for (int gap = 1; gap <= 8; ++gap) {
                        trials.push_back({first, gap, trips, a, b});
                    }
                }
            }
        }
    }
    std::mt19937 rng(20261126);
    std::puts("round,first,gap,a,b,trips,cycles");
    for (int round = -1; round < 5; ++round) {
        std::shuffle(trials.begin(), trials.end(), rng);
        for (const auto &t : trials) {
            int second = (t.first + t.gap) % warps;
            pair_probe<<<1,threads>>>(t.first, second, t.a, t.b, t.trips,
                                       1.0f, 0x3f803f80u, ticks, output);
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
            float va = expected(t.a, t.trips), vb = expected(t.b, t.trips);
            for (int i = 0; i < threads; ++i) {
                float target = i / 32 == t.first ? va : (i / 32 == second ? vb : 0);
                if ((i & 1) && ((i / 32 == t.first && t.a == fma32) || (i / 32 != t.first && i / 32 == second && t.b == fma32))) target *= 4;
                if (!(std::fabs(output[i] - target) <= 0.00001f)) {
                    std::fprintf(stderr, "Checksum failed in lane %d\n", i);
                    return 2;
                }
            }
            if (round >= 0) std::printf("%d,%d,%d,%d,%d,%d,%llu\n",round,t.first,t.gap,
                                        int(t.a),int(t.b),t.trips,*ticks);
        }
    }
    check(cudaFree(output));
    check(cudaFree(ticks));
}
