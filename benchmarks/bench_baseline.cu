// Two-pass baseline benchmark: dequant kernel and cuBLAS FP16 GEMM timed
// separately (event-bracketed) plus a sync-bracketed end-to-end wall time.
// Raw per-rep timings are preserved; medians and bandwidth figures are
// derived from them. Bandwidth is analytic bytes moved / time, labeled per
// phase; the GEMM figure is the analytic minimum (W + X in, C out) and does
// not account for cuBLAS-internal reuse or overhead.
//
// Decode-class shapes (small M) are the point: the two-pass pays 4x the
// weight bytes a fused kernel would. Bench-size correctness is covered only
// by the dequant bitwise gate; the FP64 oracle is gated at the test sizes
// in test_gpu, so no precision claim is made at bench scale.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "w4/dequant.cuh"

namespace {

// FP64 streaming triad (c = a + b) for a same-session bandwidth reference;
// every GB/s figure in the output is judged against this, not a spec sheet.
__global__ void triad_kernel(const double* __restrict__ a,
                             const double* __restrict__ b,
                             double* __restrict__ c, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    c[i] = a[i] + b[i];
}

// Returns raw per-rep milliseconds; bytes moved per rep are 3*n*8.
std::vector<double> run_triad(int reps) {
  const size_t n = size_t{1} << 25;  // 256 MiB per array
  double *a, *b, *c;
  w4::gpu::cuda_check(cudaMalloc(&a, n * sizeof(double)), "triad a");
  w4::gpu::cuda_check(cudaMalloc(&b, n * sizeof(double)), "triad b");
  w4::gpu::cuda_check(cudaMalloc(&c, n * sizeof(double)), "triad c");
  w4::gpu::cuda_check(cudaMemset(a, 0x3C, n * sizeof(double)), "triad init a");
  w4::gpu::cuda_check(cudaMemset(b, 0x3D, n * sizeof(double)), "triad init b");
  int sms = 0;
  w4::gpu::cuda_check(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0),
      "sm count");
  const int block = 256;
  const int grid = sms * 8;  // a few waves; the grid-stride loop covers
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  std::vector<double> ms;
  for (int rep = -2; rep < reps; ++rep) {
    cudaEventRecord(e0);
    triad_kernel<<<grid, block>>>(a, b, c, n);
    cudaEventRecord(e1);
    w4::gpu::cuda_check(cudaEventSynchronize(e1), "triad sync");
    if (rep < 0) continue;
    float t = 0.0f;
    w4::gpu::cuda_check(cudaEventElapsedTime(&t, e0, e1), "triad time");
    ms.push_back(t);
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  return ms;
}

std::string timestamp_utc() {
  const std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Case {
  int M, K, N, gs;
  bool sym;
};

struct Result {
  Case c;
  std::vector<double> dequant_ms, gemm_ms, e2e_ms;
  bool dequant_bitwise;
};

Result run_case(const Case& c, int reps, uint64_t seed) {
  using namespace w4;
  using namespace w4::gpu;

  const auto w_src = normal_weights(c.N * c.K, seed);
  const PackedW p = quantize(w_src, c.N, c.K, c.gs, c.sym);
  const auto x = normal_activations(c.M * c.K, seed + 1);

  uint32_t* q_d = nullptr;
  uint32_t* z_d = nullptr;
  __half *s_d = nullptr, *w_d = nullptr, *x_d = nullptr, *c_d = nullptr;
  cuda_check(cudaMalloc(&q_d, p.q.size() * sizeof(uint32_t)), "malloc q");
  cuda_check(cudaMalloc(&s_d, p.scales.size() * sizeof(__half)), "malloc s");
  if (!c.sym)
    cuda_check(cudaMalloc(&z_d, p.zeros.size() * sizeof(uint32_t)),
               "malloc z");
  cuda_check(cudaMalloc(&w_d, (size_t)c.N * c.K * sizeof(__half)),
             "malloc w");
  cuda_check(cudaMalloc(&x_d, (size_t)c.M * c.K * sizeof(__half)),
             "malloc x");
  cuda_check(cudaMalloc(&c_d, (size_t)c.M * c.N * sizeof(__half)),
             "malloc c");
  cuda_check(cudaMemcpy(q_d, p.q.data(), p.q.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice),
             "up q");
  cuda_check(cudaMemcpy(s_d, p.scales.data(), p.scales.size() * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "up s");
  if (!c.sym)
    cuda_check(cudaMemcpy(z_d, p.zeros.data(), p.zeros.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "up z");
  cuda_check(cudaMemcpy(x_d, x.data(), (size_t)c.M * c.K * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "up x");

  const DevicePackedW dp{c.N, c.K, c.gs, p.G, c.sym ? 1 : 0, q_d, s_d, z_d};
  cublasHandle_t cb = make_cublas();

  // Bitwise dequant gate at bench size, checked once per case.
  dequant(dp, w_d);
  cuda_check(cudaDeviceSynchronize(), "dequant sync");
  std::vector<__half> w_gpu((size_t)c.N * c.K);
  cuda_check(cudaMemcpy(w_gpu.data(), w_d, (size_t)c.N * c.K * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "down w");
  const auto w_ref = dequant_all(p);
  const bool bitwise =
      std::memcmp(w_gpu.data(), w_ref.data(),
                  (size_t)c.N * c.K * sizeof(__half)) == 0;

  cudaEvent_t e0, e1, e2;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);

  Result r;
  r.c = c;
  r.dequant_bitwise = bitwise;

  for (int rep = -2; rep < reps; ++rep) {  // two warmup reps, unrecorded
    const auto t0 = std::chrono::steady_clock::now();
    cudaEventRecord(e0);
    dequant(dp, w_d);
    cudaEventRecord(e1);
    gemm_fp16(cb, w_d, x_d, c_d, c.M, c.N, c.K);
    cudaEventRecord(e2);
    cuda_check(cudaEventSynchronize(e2), "event sync");
    const auto t1 = std::chrono::steady_clock::now();
    if (rep < 0) continue;
    float dq = 0.0f, gm = 0.0f;
    cuda_check(cudaEventElapsedTime(&dq, e0, e1), "dequant time");
    cuda_check(cudaEventElapsedTime(&gm, e1, e2), "gemm time");
    r.dequant_ms.push_back(dq);
    r.gemm_ms.push_back(gm);
    r.e2e_ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
  cublasDestroy(cb);
  cudaFree(q_d);
  cudaFree(s_d);
  cudaFree(z_d);
  cudaFree(w_d);
  cudaFree(x_d);
  cudaFree(c_d);
  return r;
}

void print_ms_array(std::FILE* f, const std::vector<double>& v) {
  std::fprintf(f, "[");
  for (size_t i = 0; i < v.size(); ++i)
    std::fprintf(f, "%s%.4f", i ? ", " : "", v[i]);
  std::fprintf(f, "]");
}

}  // namespace

int main(int argc, char** argv) {
  int reps = 9;
  uint64_t seed = 0xB46EULL;
  std::string out = std::string(W4_DATA_DIR) + "/w4a16_baseline_bench.json";
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--reps=", 7) == 0)
      reps = std::atoi(argv[i] + 7);
    else if (std::strncmp(argv[i], "--seed=", 7) == 0)
      seed = std::stoull(argv[i] + 7, nullptr, 0);
    else if (std::strncmp(argv[i], "--out=", 6) == 0)
      out = argv[i] + 6;
    else {
      std::fprintf(stderr, "usage: bench_baseline [--reps=9] [--seed=0x...] "
                           "[--out=path]\n");
      return 2;
    }
  }
  if (reps < 1) {
    std::fprintf(stderr, "bench_baseline: --reps must be >= 1\n");
    return 2;
  }

  // Decode-class M; (K, N) pairs are Llama-shaped hidden and MLP sizes.
  const std::vector<Case> cases = {
      {1, 4096, 4096, 128, true},   {8, 4096, 4096, 128, true},
      {16, 4096, 4096, 128, true},  {64, 4096, 4096, 128, true},
      {1, 4096, 14336, 128, true},  {8, 4096, 14336, 128, true},
      {16, 4096, 14336, 128, true}, {64, 4096, 14336, 128, true},
  };

  cudaDeviceProp prop;
  w4::gpu::cuda_check(cudaGetDeviceProperties(&prop, 0), "props");
  int driver_v = 0, runtime_v = 0;
  w4::gpu::cuda_check(cudaDriverGetVersion(&driver_v), "driver");
  w4::gpu::cuda_check(cudaRuntimeGetVersion(&runtime_v), "runtime");
  int core_clk = 0, mem_clk = 0;
  w4::gpu::cuda_check(
      cudaDeviceGetAttribute(&core_clk, cudaDevAttrClockRate, 0), "core clk");
  w4::gpu::cuda_check(
      cudaDeviceGetAttribute(&mem_clk, cudaDevAttrMemoryClockRate, 0),
      "memory clock");
  std::printf("gpu: %s sm_%d%d, %d SMs, %zu MiB, driver %d, runtime %d\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              prop.totalGlobalMem >> 20, driver_v, runtime_v);

  // Same-session bandwidth reference, run before the cases (doubles as the
  // machine warm-up).
  const std::vector<double> triad_ms = run_triad(reps);
  const double triad_bytes = 3.0 * (double)(size_t{1} << 25) * 8.0;
  const double triad_gbs = triad_bytes / median(triad_ms) * 1e-6;
  std::printf("triad: %.1f GB/s (median of %d)\n", triad_gbs, reps);

  std::FILE* f = std::fopen(out.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "bench_baseline: cannot open %s\n", out.c_str());
    return 1;
  }

  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"benchmark\": \"w4a16 two-pass baseline (dequant kernel "
                  "+ cuBLAS FP16 GEMM, FP32 accumulate)\",\n");
  std::fprintf(f, "  \"timestamp_utc\": \"%s\",\n", timestamp_utc().c_str());
  std::fprintf(f, "  \"seed\": \"%" PRIu64 "\",\n", seed);
  std::fprintf(f, "  \"reps\": %d,\n", reps);
  std::fprintf(f, "  \"build_type\": \"%s\",\n", W4_BUILD_TYPE);
  std::fprintf(f, "  \"device\": \"%s\",\n", prop.name);
  std::fprintf(f, "  \"sm\": %d%d,\n", prop.major, prop.minor);
  std::fprintf(f, "  \"sms\": %d,\n", prop.multiProcessorCount);
  std::fprintf(f, "  \"total_mem_mib\": %zu,\n", prop.totalGlobalMem >> 20);
  std::fprintf(f, "  \"core_clock_khz\": %d,\n", core_clk);
  std::fprintf(f, "  \"memory_clock_khz\": %d,\n", mem_clk);
  std::fprintf(f, "  \"driver_version\": %d,\n", driver_v);
  std::fprintf(f, "  \"runtime_version\": %d,\n", runtime_v);
  std::fprintf(f, "  \"nvcc\": \"%d.%d.%d\",\n", __CUDACC_VER_MAJOR__,
               __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
  std::fprintf(f, "  \"bytes_note\": \"analytic bytes moved; gemm term is "
                  "the minimum W+X-in/C-out, not measured DRAM traffic\",\n");
  std::fprintf(f, "  \"triad_fp64_ms\": ");
  print_ms_array(f, triad_ms);
  std::fprintf(f, ",\n  \"triad_fp64_gbs\": %.1f,\n", triad_gbs);
  std::fprintf(f, "  \"cases\": [\n");

  for (size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    const uint64_t cseed = seed + (uint64_t)i * 7919;
    const Result r = run_case(c, reps, cseed);

    const double w_bytes = (double)c.N * c.K * 2.0;
    const double dq_bytes = (double)c.N * c.K * 0.5 +
                            (double)c.N * ((c.K + c.gs - 1) / c.gs) * 2.0 +
                            (c.sym ? 0.0
                                   : (double)c.N *
                                         ((c.K + c.gs - 1) / c.gs) * 0.5) +
                            w_bytes;
    const double gemm_bytes = w_bytes + (double)c.M * c.K * 2.0 +
                              (double)c.M * c.N * 2.0;
    const double dq_med = median(r.dequant_ms);
    const double gm_med = median(r.gemm_ms);
    const double e2e_med = median(r.e2e_ms);

    std::printf("M=%d K=%d N=%d gs=%d %s: dequant %.3f ms (%.1f GB/s), gemm "
                "%.3f ms (%.1f GB/s), e2e wall %.3f ms%s\n",
                c.M, c.K, c.N, c.gs, c.sym ? "sym" : "asym", dq_med,
                dq_bytes / dq_med * 1e-6, gm_med, gemm_bytes / gm_med * 1e-6,
                e2e_med, r.dequant_bitwise ? "" : "  [BITWISE GATE FAILED]");

    std::fprintf(f, "    {\"M\": %d, \"K\": %d, \"N\": %d, \"gs\": %d, "
                    "\"symmetric\": %s,\n", c.M, c.K, c.N, c.gs,
                    c.sym ? "true" : "false");
    std::fprintf(f, "     \"dequant_bitwise\": %s,\n",
                 r.dequant_bitwise ? "true" : "false");
    std::fprintf(f, "     \"dequant_ms\": ");
    print_ms_array(f, r.dequant_ms);
    std::fprintf(f, ",\n     \"gemm_ms\": ");
    print_ms_array(f, r.gemm_ms);
    std::fprintf(f, ",\n     \"e2e_wall_ms\": ");
    print_ms_array(f, r.e2e_ms);
    std::fprintf(f, ",\n     \"dequant_median_ms\": %.4f, \"gemm_median_ms\": "
                    "%.4f, \"e2e_median_ms\": %.4f,\n", dq_med, gm_med,
                 e2e_med);
    std::fprintf(f, "     \"dequant_bytes\": %.0f, \"gemm_bytes_min\": "
                    "%.0f,\n", dq_bytes, gemm_bytes);
    std::fprintf(f, "     \"dequant_gbs\": %.1f, \"gemm_gbs_min\": %.1f}%s\n",
                 dq_bytes / dq_med * 1e-6, gemm_bytes / gm_med * 1e-6,
                 i + 1 < cases.size() ? "," : "");
  }

  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
