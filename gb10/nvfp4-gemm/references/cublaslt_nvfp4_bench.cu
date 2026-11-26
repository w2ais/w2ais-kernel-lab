// Copyright (c) 2026.
// SPDX-License-Identifier: BSD-3-Clause
//
// Reproducible cuBLASLt control for dense TN NVFP4 GEMM on GB10 / SM 12.1.
//
// The benchmark deliberately uses an exact, easily validated contract:
//   A storage: K x M column-major packed E2M1, op(A) = transpose
//   B storage: K x N column-major packed E2M1, op(B) = identity
//   C/D:       M x N row-major BF16 (N contiguous, matching the CuTe kernel)
//   scales:    VEC16 unsigned E4M3, all equal to 1
//   alpha/beta: 1/0, FP32 compute
//
// A and B are filled with packed E2M1 1.0 values, so every D element must be K.

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    const cudaError_t status_ = (expr);                                          \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status_));                                 \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CUBLAS_CHECK(expr)                                                       \
  do {                                                                          \
    const cublasStatus_t status_ = (expr);                                       \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,      \
                   static_cast<int>(status_));                                   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

namespace {

constexpr std::size_t kWorkspaceBytes = 256ull << 20;
constexpr int kRequestedAlgorithms = 64;

struct Options {
  int m = 4096;
  int n = 4096;
  int k = 4096;
  int warmups = 50;
  int iterations = 0;  // 0 selects a size-dependent default.
  int trials = 7;
  int device_warmups = 1000;
  int search_warmups = 1;
  int search_iterations = 0;  // 0 selects the same default as iterations.
  int search_sweeps = 2;
  std::string json_path;
};

struct AlgoConfig {
  std::int32_t id = -1;
  std::uint32_t tile_id = 0;
  std::int32_t split_k = 1;
  std::uint32_t reduction_scheme = 0;
  std::uint32_t cta_swizzling = 0;
  std::uint32_t custom_option = 0;
  std::uint32_t stages_id = 0;
  std::uint16_t inner_shape_id = 0;
  std::uint16_t cluster_shape_id = 0;
};

struct Candidate {
  int heuristic_index = -1;
  cublasLtMatmulAlgo_t algo{};
  std::size_t workspace_bytes = 0;
  float waves_count = 0.0f;
  AlgoConfig config;
  std::vector<float> search_trial_ms;
  float search_median_ms = 0.0f;
};

struct MatmulContext {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  const void* a = nullptr;
  const void* b = nullptr;
  const void* c = nullptr;
  void* d = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  cudaStream_t stream = nullptr;
};

void usage(const char* program) {
  std::printf(
      "Usage: %s [M N K] [options]\n"
      "\n"
      "Options:\n"
      "  --warmups N           Winner warmup launches (default: 50)\n"
      "  --iterations N        Launches per winner trial (default: adaptive)\n"
      "  --trials N            Independent winner trials (default: 7)\n"
      "  --device-warmups N    Initial clock-conditioning launches (default: 1000)\n"
      "  --search-warmups N    Warmups before each heuristic sample (default: 1)\n"
      "  --search-iterations N Launches per heuristic sample (default: adaptive)\n"
      "  --search-sweeps N     Times each candidate occupies each order position\n"
      "                        (default: 2)\n"
      "  --json PATH           Also write the final JSON object to PATH\n"
      "  --help                Show this help\n",
      program);
}

bool parse_nonnegative_int(const char* text, int* value) {
  if (text == nullptr || *text == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool parse_options(int argc, char** argv, Options* options) {
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    if (argument == "--help" || argument == "-h") {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    auto parse_value = [&](int* destination) -> bool {
      if (++i >= argc || !parse_nonnegative_int(argv[i], destination)) {
        std::fprintf(stderr, "Invalid or missing value after %s\n", argument.c_str());
        return false;
      }
      return true;
    };
    if (argument == "--warmups") {
      if (!parse_value(&options->warmups)) return false;
    } else if (argument == "--iterations") {
      if (!parse_value(&options->iterations)) return false;
    } else if (argument == "--trials") {
      if (!parse_value(&options->trials)) return false;
    } else if (argument == "--device-warmups") {
      if (!parse_value(&options->device_warmups)) return false;
    } else if (argument == "--search-warmups") {
      if (!parse_value(&options->search_warmups)) return false;
    } else if (argument == "--search-iterations") {
      if (!parse_value(&options->search_iterations)) return false;
    } else if (argument == "--search-sweeps") {
      if (!parse_value(&options->search_sweeps)) return false;
    } else if (argument == "--json") {
      if (++i >= argc) {
        std::fprintf(stderr, "Missing path after --json\n");
        return false;
      }
      options->json_path = argv[i];
    } else if (!argument.empty() && argument[0] == '-') {
      std::fprintf(stderr, "Unknown option: %s\n", argument.c_str());
      return false;
    } else {
      positional.push_back(argument);
    }
  }

  if (!positional.empty() && positional.size() != 3) {
    std::fprintf(stderr, "Provide either no dimensions or exactly M N K.\n");
    return false;
  }
  if (positional.size() == 3) {
    if (!parse_nonnegative_int(positional[0].c_str(), &options->m) ||
        !parse_nonnegative_int(positional[1].c_str(), &options->n) ||
        !parse_nonnegative_int(positional[2].c_str(), &options->k)) {
      std::fprintf(stderr, "M, N, and K must be nonnegative integers.\n");
      return false;
    }
  }

  if (options->m <= 0 || options->n <= 0 || options->k <= 0 ||
      options->m % 128 != 0 || options->n % 128 != 0 ||
      options->k % 64 != 0) {
    std::fprintf(stderr,
                 "This control requires positive M/N multiples of 128 and a "
                 "positive K multiple of 64.\n");
    return false;
  }
  if (options->k > 65504) {
    std::fprintf(stderr,
                 "This benchmark restricts K to at most 65504.\n");
    return false;
  }
  if (options->iterations < 0 || options->search_iterations < 0 ||
      options->warmups < 0 || options->device_warmups < 0 ||
      options->search_warmups < 0 || options->trials <= 0 ||
      options->search_sweeps <= 0) {
    std::fprintf(stderr, "Iteration counts must be nonnegative; trials and sweeps must be positive.\n");
    return false;
  }
  return true;
}

int adaptive_iterations(const Options& options) {
  const int largest = std::max(options.m, std::max(options.n, options.k));
  if (largest <= 1024) return 2000;
  if (largest <= 2048) return 500;
  return 100;
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs,
                            const char* name) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    std::fprintf(stderr, "Size overflow while computing %s.\n", name);
    std::exit(EXIT_FAILURE);
  }
  return lhs * rhs;
}

std::size_t vec16_scale_elements(int inner, int outer) {
  // cuBLASLt VEC16_UE4M3 stores one 128 x 4 scale tile for each 128 x 64
  // matrix tile. Inputs are restricted to full tiles, so no OOB scale values
  // exist and every allocated byte represents one E4M3 scale factor.
  const std::size_t padded_inner =
      (static_cast<std::size_t>(inner) + 63) / 64 * 64;
  const std::size_t padded_outer =
      (static_cast<std::size_t>(outer) + 127) / 128 * 128;
  return checked_product(padded_inner / 16, padded_outer, "scale buffer");
}

template <typename T>
T algo_attribute(const cublasLtMatmulAlgo_t& algo,
                 cublasLtMatmulAlgoConfigAttributes_t attribute,
                 T fallback) {
  T value = fallback;
  std::size_t written = 0;
  const cublasStatus_t status = cublasLtMatmulAlgoConfigGetAttribute(
      &algo, attribute, &value, sizeof(value), &written);
  return status == CUBLAS_STATUS_SUCCESS && written == sizeof(value) ? value
                                                                     : fallback;
}

AlgoConfig read_algo_config(const cublasLtMatmulAlgo_t& algo) {
  AlgoConfig config;
  config.id = algo_attribute<std::int32_t>(
      algo, CUBLASLT_ALGO_CONFIG_ID, -1);
  config.tile_id = algo_attribute<std::uint32_t>(
      algo, CUBLASLT_ALGO_CONFIG_TILE_ID, 0);
  config.split_k = algo_attribute<std::int32_t>(
      algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1);
  config.reduction_scheme = algo_attribute<std::uint32_t>(
      algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0);
  config.cta_swizzling = algo_attribute<std::uint32_t>(
      algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0);
  config.custom_option = algo_attribute<std::uint32_t>(
      algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 0);
  config.stages_id = algo_attribute<std::uint32_t>(
      algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, 0);
  config.inner_shape_id = algo_attribute<std::uint16_t>(
      algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, 0);
  config.cluster_shape_id = algo_attribute<std::uint16_t>(
      algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, 0);
  return config;
}

float median(std::vector<float> values) {
  if (values.empty()) return std::numeric_limits<float>::infinity();
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) return values[middle];
  return 0.5f * (values[middle - 1] + values[middle]);
}

double tflops(int m, int n, int k, float milliseconds) {
  return 2.0 * static_cast<double>(m) * static_cast<double>(n) *
         static_cast<double>(k) / (static_cast<double>(milliseconds) * 1.0e9);
}

float time_matmul(const MatmulContext& context,
                  const cublasLtMatmulAlgo_t& algo, int warmups,
                  int iterations, cudaEvent_t start, cudaEvent_t stop) {
  if (iterations <= 0) return -1.0f;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  auto launch = [&]() {
    return cublasLtMatmul(
        context.handle, context.operation, &alpha, context.a, context.a_desc,
        context.b, context.b_desc, &beta, context.c, context.c_desc, context.d,
        context.d_desc, &algo, context.workspace, context.workspace_bytes,
        context.stream);
  };

  for (int i = 0; i < warmups; ++i) {
    if (launch() != CUBLAS_STATUS_SUCCESS) return -1.0f;
  }
  CUDA_CHECK(cudaStreamSynchronize(context.stream));

  CUDA_CHECK(cudaEventRecord(start, context.stream));
  for (int i = 0; i < iterations; ++i) {
    if (launch() != CUBLAS_STATUS_SUCCESS) return -1.0f;
  }
  CUDA_CHECK(cudaEventRecord(stop, context.stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  return total_ms / static_cast<float>(iterations);
}

std::string json_string(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(character) << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
  return output.str();
}

std::string cublas_version_string(std::size_t encoded_version) {
  const std::size_t major = encoded_version / 10000;
  const std::size_t minor = (encoded_version % 10000) / 100;
  const std::size_t patch = encoded_version % 100;
  return std::to_string(major) + "." + std::to_string(minor) + "." +
         std::to_string(patch);
}

void append_float_array(std::ostringstream& output,
                        const std::vector<float>& values) {
  output << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) output << ',';
    output << values[i];
  }
  output << ']';
}

void append_algo_config(std::ostringstream& output, const AlgoConfig& config) {
  output << "{\"id\":" << config.id
         << ",\"tile_id\":" << config.tile_id
         << ",\"split_k\":" << config.split_k
         << ",\"reduction_scheme\":" << config.reduction_scheme
         << ",\"cta_swizzling\":" << config.cta_swizzling
         << ",\"custom_option\":" << config.custom_option
         << ",\"stages_id\":" << config.stages_id
         << ",\"inner_shape_id\":" << config.inner_shape_id
         << ",\"cluster_shape_id\":" << config.cluster_shape_id << '}';
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }
  if (options.iterations == 0) options.iterations = adaptive_iterations(options);
  if (options.search_iterations == 0) {
    options.search_iterations = adaptive_iterations(options);
  }

  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  const std::size_t cublas_runtime_version = cublasLtGetVersion();
  std::printf("GPU: %s sm_%d%d, problem %d x %d x %d\n", properties.name,
              properties.major, properties.minor, options.m, options.n,
              options.k);

  const std::size_t mk = checked_product(static_cast<std::size_t>(options.m),
                                         static_cast<std::size_t>(options.k),
                                         "M*K");
  const std::size_t nk = checked_product(static_cast<std::size_t>(options.n),
                                         static_cast<std::size_t>(options.k),
                                         "N*K");
  const std::size_t mn = checked_product(static_cast<std::size_t>(options.m),
                                         static_cast<std::size_t>(options.n),
                                         "M*N");
  const std::size_t a_bytes = mk / 2;
  const std::size_t b_bytes = nk / 2;
  const std::size_t output_bytes =
      checked_product(mn, sizeof(__nv_bfloat16), "output buffer");
  const std::size_t a_scale_bytes =
      vec16_scale_elements(options.k, options.m);
  const std::size_t b_scale_bytes =
      vec16_scale_elements(options.k, options.n);

  void* a = nullptr;
  void* b = nullptr;
  void* c = nullptr;
  void* d = nullptr;
  void* a_scale = nullptr;
  void* b_scale = nullptr;
  void* workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&a, a_bytes));
  CUDA_CHECK(cudaMalloc(&b, b_bytes));
  CUDA_CHECK(cudaMalloc(&c, output_bytes));
  CUDA_CHECK(cudaMalloc(&d, output_bytes));
  CUDA_CHECK(cudaMalloc(&a_scale, a_scale_bytes));
  CUDA_CHECK(cudaMalloc(&b_scale, b_scale_bytes));
  CUDA_CHECK(cudaMalloc(&workspace, kWorkspaceBytes));

  // E2M1 nibble 0x2 is +1.0; 0x22 packs two +1.0 values per byte.
  CUDA_CHECK(cudaMemset(a, 0x22, a_bytes));
  CUDA_CHECK(cudaMemset(b, 0x22, b_bytes));
  CUDA_CHECK(cudaMemset(c, 0, output_bytes));
  CUDA_CHECK(cudaMemset(d, 0, output_bytes));
  // Unsigned E4M3 bit pattern 0x38 is 1.0.
  CUDA_CHECK(cudaMemset(a_scale, 0x38, a_scale_bytes));
  CUDA_CHECK(cudaMemset(b_scale, 0x38, b_scale_bytes));

  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtCreate(&handle));
  CUBLAS_CHECK(
      cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  const cublasOperation_t trans_a = CUBLAS_OP_T;
  const cublasOperation_t trans_b = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

  const cublasLtMatmulMatrixScale_t scale_mode =
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode,
      sizeof(scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode,
      sizeof(scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale,
      sizeof(a_scale)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale,
      sizeof(b_scale)));

  // A/B use the TN column-major contract required by block-scaled NVFP4:
  // op(A[K,M]) * B[K,N] -> D[M,N]. C/D are row-major so the output's
  // contiguous mode exactly matches the CuTe kernel's C-major-N contract.
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &a_desc, CUDA_R_4F_E2M1, options.k, options.m, options.k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &b_desc, CUDA_R_4F_E2M1, options.k, options.n, options.k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &c_desc, CUDA_R_16BF, options.m, options.n, options.n));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &d_desc, CUDA_R_16BF, options.m, options.n, options.n));
  const cublasLtOrder_t output_order = CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &output_order,
      sizeof(output_order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &output_order,
      sizeof(output_order)));

  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes,
      sizeof(kWorkspaceBytes)));

  std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
      kRequestedAlgorithms);
  int heuristic_count = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      handle, operation, a_desc, b_desc, c_desc, d_desc, preference,
      kRequestedAlgorithms, heuristic_results.data(), &heuristic_count));
  std::printf("cuBLASLt returned %d heuristic entries\n", heuristic_count);

  std::vector<Candidate> candidates;
  for (int i = 0; i < heuristic_count; ++i) {
    const auto& result = heuristic_results[i];
    if (result.state != CUBLAS_STATUS_SUCCESS ||
        result.workspaceSize > kWorkspaceBytes) {
      continue;
    }
    Candidate candidate;
    candidate.heuristic_index = i;
    candidate.algo = result.algo;
    candidate.workspace_bytes = result.workspaceSize;
    candidate.waves_count = result.wavesCount;
    candidate.config = read_algo_config(candidate.algo);
    candidates.push_back(candidate);
  }
  if (candidates.empty()) {
    std::fprintf(stderr, "No usable cuBLASLt heuristic algorithm was returned.\n");
    return 2;
  }

  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const MatmulContext context{handle,
                              operation,
                              a_desc,
                              b_desc,
                              c_desc,
                              d_desc,
                              a,
                              b,
                              c,
                              d,
                              workspace,
                              kWorkspaceBytes,
                              stream};

  // Condition clocks/caches before comparing candidates. This launch block is
  // not used as a measurement and is configurable for reproducible experiments.
  if (options.device_warmups > 0) {
    const float ignored = time_matmul(
        context, candidates.front().algo, 0, options.device_warmups, start, stop);
    if (!(ignored > 0.0f)) {
      std::fprintf(stderr, "Initial cuBLASLt warmup failed.\n");
      return 3;
    }
  }

  // A complete rotation makes every candidate occupy every order position
  // exactly once. Repeating it search_sweeps times avoids favoring the first or
  // last candidate as clocks and temperature move during the search.
  const int candidate_count = static_cast<int>(candidates.size());
  if (options.search_sweeps >
      std::numeric_limits<int>::max() / candidate_count) {
    std::fprintf(stderr, "The requested search sweep count is too large.\n");
    return 4;
  }
  const int rotations = options.search_sweeps * candidate_count;
  for (int rotation = 0; rotation < rotations; ++rotation) {
    const int first = rotation % candidate_count;
    for (int offset = 0; offset < candidate_count; ++offset) {
      Candidate& candidate = candidates[(first + offset) % candidate_count];
      const float ms = time_matmul(
          context, candidate.algo, options.search_warmups,
          options.search_iterations, start, stop);
      if (!(ms > 0.0f) || !std::isfinite(ms)) {
        std::fprintf(stderr, "Heuristic %d failed during timed search.\n",
                     candidate.heuristic_index);
        return 4;
      }
      candidate.search_trial_ms.push_back(ms);
    }
  }

  int selected_index = -1;
  float selected_search_ms = std::numeric_limits<float>::infinity();
  for (int i = 0; i < candidate_count; ++i) {
    Candidate& candidate = candidates[i];
    candidate.search_median_ms = median(candidate.search_trial_ms);
    std::printf(
        "candidate %2d heuristic %2d algo %d tile %u stages %u cluster %u: "
        "median %.6f ms, %.3f TFLOP/s, workspace %zu\n",
        i, candidate.heuristic_index, candidate.config.id,
        candidate.config.tile_id, candidate.config.stages_id,
        static_cast<unsigned>(candidate.config.cluster_shape_id),
        candidate.search_median_ms,
        tflops(options.m, options.n, options.k, candidate.search_median_ms),
        candidate.workspace_bytes);
    if (candidate.search_median_ms < selected_search_ms) {
      selected_search_ms = candidate.search_median_ms;
      selected_index = i;
    }
  }
  if (selected_index < 0) {
    std::fprintf(stderr, "No heuristic survived the timed search.\n");
    return 5;
  }

  Candidate& selected = candidates[selected_index];
  std::vector<float> final_trial_ms;
  final_trial_ms.reserve(options.trials);
  for (int trial = 0; trial < options.trials; ++trial) {
    const float ms = time_matmul(
        context, selected.algo, trial == 0 ? options.warmups : 0,
        options.iterations, start, stop);
    if (!(ms > 0.0f) || !std::isfinite(ms)) {
      std::fprintf(stderr, "Selected algorithm failed during final trial %d.\n",
                   trial + 1);
      return 6;
    }
    final_trial_ms.push_back(ms);
    std::printf("selected trial %d/%d: %.6f ms, %.3f TFLOP/s\n", trial + 1,
                options.trials, ms,
                tflops(options.m, options.n, options.k, ms));
  }
  const float final_median_ms = median(final_trial_ms);
  const float final_min_ms =
      *std::min_element(final_trial_ms.begin(), final_trial_ms.end());

  // Validate a fresh, untimed launch. Comparing the expected BF16 value is
  // exact here: the reference uses the same BF16 rounding as the output.
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  CUBLAS_CHECK(cublasLtMatmul(
      handle, operation, &alpha, a, a_desc, b, b_desc, &beta, c, c_desc, d,
      d_desc, &selected.algo, workspace, kWorkspaceBytes, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> host_output(mn);
  CUDA_CHECK(cudaMemcpy(host_output.data(), d, output_bytes,
                        cudaMemcpyDeviceToHost));
  const float expected_value =
      __bfloat162float(__float2bfloat16_rn(static_cast<float>(options.k)));
  std::size_t mismatch_count = 0;
  std::size_t nonfinite_count = 0;
  float max_abs_error = 0.0f;
  for (const __nv_bfloat16 value : host_output) {
    const float actual = __bfloat162float(value);
    if (!std::isfinite(actual)) {
      ++mismatch_count;
      ++nonfinite_count;
      max_abs_error = std::numeric_limits<float>::infinity();
      continue;
    }
    const float error = std::fabs(actual - expected_value);
    max_abs_error = std::max(max_abs_error, error);
    if (actual != expected_value) ++mismatch_count;
  }
  const bool correct = mismatch_count == 0;
  std::printf("validation: %zu/%zu mismatches, max abs error %.9g\n",
              mismatch_count, mn, max_abs_error);

  std::ostringstream json;
  json << std::setprecision(9);
  json << '{'
       << "\"schema_version\":1"
       << ",\"gpu\":" << json_string(properties.name)
       << ",\"compute_capability\":"
       << json_string("sm_" + std::to_string(properties.major) +
                      std::to_string(properties.minor))
       << ",\"cublas_version\":"
       << json_string(cublas_version_string(cublas_runtime_version))
       << ",\"problem\":{\"m\":" << options.m << ",\"n\":" << options.n
       << ",\"k\":" << options.k << '}'
       << ",\"contract\":{"
       << "\"layout\":\"TN\",\"a_type\":\"CUDA_R_4F_E2M1\","
       << "\"b_type\":\"CUDA_R_4F_E2M1\",\"output_type\":\"CUDA_R_16BF\","
       << "\"compute_type\":\"CUBLAS_COMPUTE_32F\","
       << "\"scale_mode\":\"CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3\","
       << "\"alpha\":1,\"beta\":0,\"a_storage\":{\"rows\":"
       << options.k << ",\"cols\":" << options.m << ",\"ld\":"
       << options.k << "},\"b_storage\":{\"rows\":" << options.k
       << ",\"cols\":" << options.n << ",\"ld\":" << options.k
       << "},\"output_storage\":{\"rows\":" << options.m
       << ",\"cols\":" << options.n << ",\"ld\":" << options.n
       << ",\"order\":\"row-major\"}}"
       << ",\"heuristic_count\":" << heuristic_count
       << ",\"valid_candidate_count\":" << candidate_count
       << ",\"search\":{\"device_warmups\":" << options.device_warmups
       << ",\"sample_warmups\":" << options.search_warmups
       << ",\"iterations\":" << options.search_iterations
       << ",\"sweeps\":" << options.search_sweeps
       << ",\"rotations\":" << rotations << ",\"candidates\":[";
  for (int i = 0; i < candidate_count; ++i) {
    if (i != 0) json << ',';
    const Candidate& candidate = candidates[i];
    json << "{\"candidate_index\":" << i
         << ",\"heuristic_index\":" << candidate.heuristic_index
         << ",\"workspace_bytes\":" << candidate.workspace_bytes
         << ",\"waves_count\":" << candidate.waves_count
         << ",\"config\":";
    append_algo_config(json, candidate.config);
    json << ",\"trial_ms\":";
    append_float_array(json, candidate.search_trial_ms);
    json << ",\"median_ms\":" << candidate.search_median_ms << '}';
  }
  json << "]}"
       << ",\"selected_candidate_index\":" << selected_index
       << ",\"selected_heuristic_index\":" << selected.heuristic_index
       << ",\"selected_algo\":";
  append_algo_config(json, selected.config);
  json << ",\"selected_workspace_bytes\":" << selected.workspace_bytes
       << ",\"timing\":{\"warmups\":" << options.warmups
       << ",\"iterations\":" << options.iterations
       << ",\"trials\":" << options.trials << ",\"trial_ms\":";
  append_float_array(json, final_trial_ms);
  json << ",\"median_ms\":" << final_median_ms
       << ",\"min_ms\":" << final_min_ms
       << ",\"median_tflops\":"
       << tflops(options.m, options.n, options.k, final_median_ms)
       << ",\"min_time_tflops\":"
       << tflops(options.m, options.n, options.k, final_min_ms) << '}'
       << ",\"validation\":{\"correct\":"
       << (correct ? "true" : "false")
       << ",\"checked_elements\":" << mn
       << ",\"expected_value\":" << expected_value
       << ",\"mismatch_count\":" << mismatch_count
       << ",\"nonfinite_count\":" << nonfinite_count
       << ",\"max_abs_error\":";
  if (std::isfinite(max_abs_error)) {
    json << max_abs_error;
  } else {
    json << "null";
  }
  json << "}}";

  const std::string json_text = json.str();
  std::printf(
      "RESULT median %.6f ms %.3f TFLOP/s; min %.6f ms %.3f TFLOP/s; "
      "algo %d; correct %s\n",
      final_median_ms, tflops(options.m, options.n, options.k, final_median_ms),
      final_min_ms, tflops(options.m, options.n, options.k, final_min_ms),
      selected.config.id, correct ? "true" : "false");
  std::printf("RESULT_JSON %s\n", json_text.c_str());

  bool json_written = true;
  if (!options.json_path.empty()) {
    std::ofstream json_file(options.json_path);
    if (!json_file) {
      std::fprintf(stderr, "Could not open JSON output path: %s\n",
                   options.json_path.c_str());
      json_written = false;
    } else {
      json_file << json_text << '\n';
      if (!json_file) {
        std::fprintf(stderr, "Could not write JSON output path: %s\n",
                     options.json_path.c_str());
        json_written = false;
      }
    }
  }

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
  CUBLAS_CHECK(cublasLtDestroy(handle));
  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFree(b_scale));
  CUDA_CHECK(cudaFree(a_scale));
  CUDA_CHECK(cudaFree(d));
  CUDA_CHECK(cudaFree(c));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(a));

  if (!json_written) return 7;
  return correct ? EXIT_SUCCESS : 8;
}
