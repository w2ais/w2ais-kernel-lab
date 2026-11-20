// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Direct cuBLASLt reference for C(M,N) = A(M,K) @ B(N,K)^T.
// A, B, and C may independently use first-mode-major (column-major) or
// second-mode-major (row-major) storage.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status_ = (expr);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status_));                                 \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLAS_CHECK(expr)                                                       \
  do {                                                                           \
    cublasStatus_t status_ = (expr);                                              \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLASLt error %s:%d: status %d\n", __FILE__,       \
                   __LINE__, int(status_));                                       \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

enum class Major { First, Second };

struct Options {
  int m = 4096;
  int n = 4096;
  int k = 4096;
  int warmup = 5;
  int iterations = 20;
  int trials = 5;
  int heuristic_count = 64;
  size_t workspace_bytes = 256ull << 20;
  Major a_major = Major::Second;  // K-major for A(M,K)
  Major b_major = Major::Second;  // K-major for B(N,K)
  Major c_major = Major::Second;  // N-major for C(M,N)
};

static void usage(const char* program) {
  std::printf(
      "Usage: %s [options]\n"
      "  --m INT --n INT --k INT\n"
      "  --warmup INT --iterations INT --trials INT --heuristics INT\n"
      "  --workspace-mib INT\n"
      "  --a-major M|K     A is logically (M,K); M=column, K=row\n"
      "  --b-major N|K     B is logically (N,K); N=column, K=row\n"
      "  --c-major M|N     C is logically (M,N); M=column, N=row\n"
      "Aliases first/column/col and second/row are also accepted.\n",
      program);
}

static std::string lowercase(const char* value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  return result;
}

static Major parse_major(const char* value, char first_mode,
                         char second_mode) {
  const std::string token = lowercase(value);
  const std::string first(1, char(std::tolower(first_mode)));
  const std::string second(1, char(std::tolower(second_mode)));
  if (token == first || token == "first" || token == "column" ||
      token == "col") {
    return Major::First;
  }
  if (token == second || token == "second" || token == "row") {
    return Major::Second;
  }
  std::fprintf(stderr, "Invalid major mode '%s' (expected %c or %c)\n",
               value, first_mode, second_mode);
  std::exit(EXIT_FAILURE);
}

static int parse_positive(const char* option, const char* value) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "%s requires a positive integer, got '%s'\n",
                 option, value);
    std::exit(EXIT_FAILURE);
  }
  return int(parsed);
}

static Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const char* option = argv[i];
    if (!std::strcmp(option, "--help") || !std::strcmp(option, "-h")) {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "Missing value after %s\n", option);
      std::exit(EXIT_FAILURE);
    }
    const char* value = argv[++i];
    if (!std::strcmp(option, "--m")) {
      options.m = parse_positive(option, value);
    } else if (!std::strcmp(option, "--n")) {
      options.n = parse_positive(option, value);
    } else if (!std::strcmp(option, "--k")) {
      options.k = parse_positive(option, value);
    } else if (!std::strcmp(option, "--warmup")) {
      options.warmup = parse_positive(option, value);
    } else if (!std::strcmp(option, "--iterations")) {
      options.iterations = parse_positive(option, value);
    } else if (!std::strcmp(option, "--trials")) {
      options.trials = parse_positive(option, value);
    } else if (!std::strcmp(option, "--heuristics")) {
      options.heuristic_count = parse_positive(option, value);
    } else if (!std::strcmp(option, "--workspace-mib")) {
      options.workspace_bytes = size_t(parse_positive(option, value)) << 20;
    } else if (!std::strcmp(option, "--a-major")) {
      options.a_major = parse_major(value, 'M', 'K');
    } else if (!std::strcmp(option, "--b-major")) {
      options.b_major = parse_major(value, 'N', 'K');
    } else if (!std::strcmp(option, "--c-major")) {
      options.c_major = parse_major(value, 'M', 'N');
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", option);
      usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  return options;
}

static std::string major_name(Major major, char first_mode,
                              char second_mode) {
  char name[32];
  if (major == Major::First) {
    std::snprintf(name, sizeof(name), "%c-major (column)", first_mode);
  } else {
    std::snprintf(name, sizeof(name), "%c-major (row)", second_mode);
  }
  return name;
}

__device__ __forceinline__ size_t matrix_offset(int row, int column,
                                                 int rows, int columns,
                                                 int second_major) {
  return second_major ? size_t(row) * columns + column
                      : size_t(row) + size_t(column) * rows;
}

__global__ void initialize_inputs(__nv_bfloat16* a, __nv_bfloat16* b, __nv_bfloat16* c, int m,
                                  int n, int k, int a_second_major,
                                  int b_second_major, int c_second_major) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  for (size_t linear = index; linear < size_t(m) * k; linear += stride) {
    const int row = int(linear / k);
    const int column = int(linear % k);
    // Dense +/-1 rows from a Walsh-Hadamard matrix.  For power-of-two K,
    // different rows are orthogonal, so correctness has a closed form without
    // making the benchmark's A operand artificially sparse.
    const float value = (__popc((row % k) & column) & 1) ? -1.0f : 1.0f;
    a[matrix_offset(row, column, m, k, a_second_major)] =
        __float2bfloat16(value);
  }
  for (size_t linear = index; linear < size_t(n) * k; linear += stride) {
    const int row = int(linear / k);
    const int column = int(linear % k);
    const float value = (__popc((row % k) & column) & 1) ? -1.0f : 1.0f;
    b[matrix_offset(row, column, n, k, b_second_major)] =
        __float2bfloat16(value);
  }
  for (size_t linear = index; linear < size_t(m) * n; linear += stride) {
    const int row = int(linear / n);
    const int column = int(linear % n);
    c[matrix_offset(row, column, m, n, c_second_major)] =
        __float2bfloat16(17.0f);
  }
}

__global__ void verify_output(const __nv_bfloat16* d, int m, int n, int k,
                              int d_second_major, unsigned long long* errors,
                              unsigned int* max_error_bits) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  for (size_t linear = index; linear < size_t(m) * n; linear += stride) {
    const int row = int(linear / n);
    const int column = int(linear % n);
    const float expected = (row % k) == (column % k) ? float(k) : 0.0f;
    const float actual = __bfloat162float(
        d[matrix_offset(row, column, m, n, d_second_major)]);
    const float error = fabsf(actual - expected);
    if (!(error == 0.0f)) atomicAdd(errors, 1ull);
    atomicMax(max_error_bits, __float_as_uint(error));
  }
}

static cublasLtMatrixLayout_t make_layout(int rows, int columns, Major major) {
  cublasLtMatrixLayout_t layout;
  const int64_t leading_dimension =
      major == Major::First ? rows : columns;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layout, CUDA_R_16BF, rows, columns,
                                          leading_dimension));
  const cublasLtOrder_t order = major == Major::First ? CUBLASLT_ORDER_COL
                                                       : CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  return layout;
}

static float time_algorithm(cublasLtHandle_t handle,
                            cublasLtMatmulDesc_t operation,
                            cublasLtMatrixLayout_t a_layout,
                            cublasLtMatrixLayout_t b_layout,
                            cublasLtMatrixLayout_t c_layout,
                            cublasLtMatrixLayout_t d_layout,
                            const cublasLtMatmulAlgo_t* algorithm,
                            const void* a, const void* b, const void* c,
                            void* d, void* workspace, size_t workspace_bytes,
                            cudaStream_t stream, int warmup, int iterations) {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  for (int i = 0; i < warmup; ++i) {
    const cublasStatus_t status = cublasLtMatmul(
        handle, operation, &alpha, a, a_layout, b, b_layout, &beta, c,
        c_layout, d, d_layout, algorithm, workspace, workspace_bytes, stream);
    if (status != CUBLAS_STATUS_SUCCESS) return -1.0f;
  }
  const cudaError_t warmup_status = cudaStreamSynchronize(stream);
  if (warmup_status != cudaSuccess) {
    cudaGetLastError();
    return -1.0f;
  }

  cudaEvent_t start;
  cudaEvent_t end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) {
    const cublasStatus_t status = cublasLtMatmul(
        handle, operation, &alpha, a, a_layout, b, b_layout, &beta, c,
        c_layout, d, d_layout, algorithm, workspace, workspace_bytes, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cudaEventDestroy(start);
      cudaEventDestroy(end);
      return -1.0f;
    }
  }
  CUDA_CHECK(cudaEventRecord(end, stream));
  const cudaError_t execution_status = cudaEventSynchronize(end);
  if (execution_status != cudaSuccess) {
    cudaGetLastError();
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return -1.0f;
  }
  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  return total_ms / iterations;
}

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if ((options.k & (options.k - 1)) != 0 || options.k > 65504) {
    std::fprintf(stderr,
                 "The dense Walsh-Hadamard correctness pattern requires a "
                 "power-of-two K representable in BF16\n");
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::printf("GPU: %s (sm_%d%d)\n", properties.name, properties.major,
              properties.minor);
  std::printf("Problem: C(%d,%d) = A(%d,%d) @ B(%d,%d)^T\n", options.m,
              options.n, options.m, options.k, options.n, options.k);
  const std::string a_major_name = major_name(options.a_major, 'M', 'K');
  const std::string b_major_name = major_name(options.b_major, 'N', 'K');
  const std::string c_major_name = major_name(options.c_major, 'M', 'N');
  std::printf("Layouts: A=%s, B=%s, C/D=%s\n", a_major_name.c_str(),
              b_major_name.c_str(), c_major_name.c_str());
  std::printf("Math: BF16 inputs/output, FP32 compute; alpha=1, beta=0\n");

  const size_t a_elements = size_t(options.m) * options.k;
  const size_t b_elements = size_t(options.n) * options.k;
  const size_t c_elements = size_t(options.m) * options.n;
  __nv_bfloat16* a = nullptr;
  __nv_bfloat16* b = nullptr;
  __nv_bfloat16* c = nullptr;
  __nv_bfloat16* d = nullptr;
  void* workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&a, a_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&b, b_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&c, c_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d, c_elements * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&workspace, options.workspace_bytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  initialize_inputs<<<std::min<size_t>(4096, (std::max(
                          {a_elements, b_elements, c_elements}) + 255) /
                                             256),
                      256, 0, stream>>>(
      a, b, c, options.m, options.n, options.k,
      options.a_major == Major::Second,
      options.b_major == Major::Second,
      options.c_major == Major::Second);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t operation;
  cublasLtMatmulPreference_t preference;
  CUBLAS_CHECK(cublasLtCreate(&handle));
  CUBLAS_CHECK(
      cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans_a = CUBLAS_OP_N;
  const cublasOperation_t trans_b = CUBLAS_OP_T;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

  cublasLtMatrixLayout_t a_layout =
      make_layout(options.m, options.k, options.a_major);
  cublasLtMatrixLayout_t b_layout =
      make_layout(options.n, options.k, options.b_major);
  cublasLtMatrixLayout_t c_layout =
      make_layout(options.m, options.n, options.c_major);
  cublasLtMatrixLayout_t d_layout =
      make_layout(options.m, options.n, options.c_major);

  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &options.workspace_bytes, sizeof(options.workspace_bytes)));
  std::vector<cublasLtMatmulHeuristicResult_t> candidates(
      options.heuristic_count);
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      handle, operation, a_layout, b_layout, c_layout, d_layout, preference,
      options.heuristic_count, candidates.data(), &returned));
  std::printf("Heuristics returned: %d (requested %d)\n", returned,
              options.heuristic_count);

  int best_candidate = -1;
  int best_algorithm_id = -1;
  float best_ms = std::numeric_limits<float>::infinity();
  for (int i = 0; i < returned; ++i) {
    if (candidates[i].state != CUBLAS_STATUS_SUCCESS ||
        candidates[i].workspaceSize > options.workspace_bytes) {
      continue;
    }
    const float milliseconds = time_algorithm(
        handle, operation, a_layout, b_layout, c_layout, d_layout,
        &candidates[i].algo, a, b, c, d, workspace, options.workspace_bytes,
        stream, options.warmup, options.iterations);
    if (milliseconds <= 0.0f) continue;
    int algorithm_id = -1;
    size_t bytes_written = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
        &candidates[i].algo, CUBLASLT_ALGO_CONFIG_ID, &algorithm_id,
        sizeof(algorithm_id), &bytes_written));
    const double tflops = 2.0 * double(options.m) * options.n * options.k /
                          (double(milliseconds) * 1.0e9);
    std::printf("ALGORITHM candidate=%d id=%d time_ms=%.4f tflops=%.3f "
                "workspace_bytes=%zu\n",
                i, algorithm_id, milliseconds, tflops,
                candidates[i].workspaceSize);
    if (milliseconds < best_ms) {
      best_ms = milliseconds;
      best_candidate = i;
      best_algorithm_id = algorithm_id;
    }
  }
  if (best_candidate < 0) {
    std::fprintf(stderr, "No heuristic algorithm executed successfully\n");
    return EXIT_FAILURE;
  }

  // Remeasure the winner in independent CUDA-event trials after the sweep so
  // the reported reference is neither biased by heuristic-list position nor
  // by one unusually favorable timing sample.
  std::vector<float> trial_milliseconds;
  trial_milliseconds.reserve(options.trials);
  for (int trial = 0; trial < options.trials; ++trial) {
    const float milliseconds = time_algorithm(
        handle, operation, a_layout, b_layout, c_layout, d_layout,
        &candidates[best_candidate].algo, a, b, c, d, workspace,
        options.workspace_bytes, stream, options.warmup, options.iterations);
    if (milliseconds <= 0.0f) {
      std::fprintf(stderr,
                   "Winning heuristic failed during final timing trial %d\n",
                   trial);
      return EXIT_FAILURE;
    }
    trial_milliseconds.push_back(milliseconds);
    const double trial_tflops =
        2.0 * double(options.m) * options.n * options.k /
        (double(milliseconds) * 1.0e9);
    std::printf("TRIAL index=%d time_ms=%.6f tflops=%.6f\n", trial,
                milliseconds, trial_tflops);
  }
  std::vector<float> ordered_milliseconds = trial_milliseconds;
  std::sort(ordered_milliseconds.begin(), ordered_milliseconds.end());
  const size_t middle = ordered_milliseconds.size() / 2;
  const float median_ms = ordered_milliseconds.size() % 2
                              ? ordered_milliseconds[middle]
                              : 0.5f * (ordered_milliseconds[middle - 1] +
                                        ordered_milliseconds[middle]);
  const float fastest_ms = ordered_milliseconds.front();

  unsigned long long* device_errors = nullptr;
  unsigned int* device_max_error = nullptr;
  CUDA_CHECK(cudaMalloc(&device_errors, sizeof(*device_errors)));
  CUDA_CHECK(cudaMalloc(&device_max_error, sizeof(*device_max_error)));
  CUDA_CHECK(cudaMemsetAsync(device_errors, 0, sizeof(*device_errors), stream));
  CUDA_CHECK(
      cudaMemsetAsync(device_max_error, 0, sizeof(*device_max_error), stream));
  verify_output<<<std::min<size_t>(4096, (c_elements + 255) / 256), 256, 0,
                  stream>>>(d, options.m, options.n, options.k,
                            options.c_major == Major::Second, device_errors,
                            device_max_error);
  CUDA_CHECK(cudaGetLastError());
  unsigned long long errors = 0;
  unsigned int max_error_bits = 0;
  CUDA_CHECK(cudaMemcpyAsync(&errors, device_errors, sizeof(errors),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&max_error_bits, device_max_error,
                             sizeof(max_error_bits), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float max_error = 0.0f;
  std::memcpy(&max_error, &max_error_bits, sizeof(max_error));
  const double median_tflops =
      2.0 * double(options.m) * options.n * options.k /
      (double(median_ms) * 1.0e9);
  const double fastest_tflops =
      2.0 * double(options.m) * options.n * options.k /
      (double(fastest_ms) * 1.0e9);
  std::printf("CORRECTNESS mismatches=%llu max_abs_error=%.9g\n", errors,
              max_error);
  std::printf("RESULT algo_id=%d time_ms=%.6f tflops=%.6f "
              "best_time_ms=%.6f best_tflops=%.6f trials=%d\n",
              best_algorithm_id, median_ms, median_tflops, fastest_ms,
              fastest_tflops, options.trials);

  CUDA_CHECK(cudaFree(device_max_error));
  CUDA_CHECK(cudaFree(device_errors));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d_layout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_layout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_layout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_layout));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
  CUBLAS_CHECK(cublasLtDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFree(d));
  CUDA_CHECK(cudaFree(c));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(a));
  return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
