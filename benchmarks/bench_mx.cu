// MXFP4 benchmark (R4): MX two-pass baseline (dequant_mx + cuBLAS FP16
// GEMM) vs fused SIMT dequant+GEMM vs fused WMMA dequant+GEMM, same
// session, same shapes as bench_fused. The baseline is re-measured here so
// ratios are immune to machine state; the FP64 triad is the same-session
// bandwidth reference. Raw per-rep timings are preserved; medians and
// bandwidth figures are derived. Byte counts are analytic minima, not
// measured DRAM traffic.
//
// Each case also records the format-accuracy figure: rel L2 of
// dequant(quant(W)) vs the FP64 source weights, for MXFP4 and for W4A16
// gs=128 symmetric — the two formats' price of admission on identical
// data. This is a quantization-error figure, not a GEMM gate; the GEMM
// gates (bitwise dequant, oracle tolerances) live in test_gpu.
//
// Bench-size correctness: both fused dequant paths are bitwise-gated per
// case (w_out hook vs dequant_all_mx); the oracle is gated at test sizes.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "w4/dequant.cuh"
#include "w4/mxfused.cuh"

namespace {

__global__ void triad_kernel(const double* __restrict__ a,
                             const double* __restrict__ b,
                             double* __restrict__ c, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    c[i] = a[i] + b[i];
}

std::vector<double> run_triad(int reps) {
  const size_t n = size_t{1} << 25;  // 256 MiB per array
  double *a, *b, *c;
  w4::gpu::cuda_check(cudaMalloc(&a, n * sizeof(double)), "triad a");
  w4::gpu::cuda_check(cudaMalloc(&b, n * sizeof(double)), "triad b");
  w4::gpu::cuda_check(cudaMalloc(&c, n * sizeof(double)), "triad c");
  cudaMemset(a, 0x3C, n * sizeof(double));
  cudaMemset(b, 0x3D, n * sizeof(double));
  int sms = 0;
  w4::gpu::cuda_check(
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0),
      "sm count");
  const int block = 256;
  const int grid = sms * 8;
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

std::string cpu_governor() {
  std::ifstream f("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  std::string g;
  return (f >> g) ? g : "unknown";
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int mtile_of(int M) {
  int mtile = 1;
  while (mtile < M && mtile < 4) mtile <<= 1;
  return mtile;
}

// rel L2 of dequantized weights vs the FP64 source.
double quant_rel_l2(const std::vector<double>& src,
                    const std::vector<__half>& dq) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < src.size(); ++i) {
    const double d = (double)__half2float(dq[i]) - src[i];
    num += d * d;
    den += src[i] * src[i];
  }
  return std::sqrt(num / (den > 0.0 ? den : 1.0));
}

struct Case {
  int M, K, N;
};

struct Result {
  Case c;
  std::vector<double> dequant_ms, gemm_ms, base_e2e_ms, fused_ms, mma_ms;
  bool fused_dequant_bitwise, mma_dequant_bitwise;
  double fused_vs_cublas_rel, mma_vs_cublas_rel;
  double quant_rel_mx, quant_rel_w4;
};

Result run_case(const Case& c, int reps, uint64_t seed) {
  using namespace w4;
  using namespace w4::gpu;

  const auto w_src = normal_weights(c.N * c.K, seed);
  const mx::PackedWm p = mx::quantize_mx(w_src, c.N, c.K);
  const auto w_ref = mx::dequant_all_mx(p);
  const auto x = normal_activations(c.M * c.K, seed + 1);

  // Format accuracy on identical data: MXFP4 vs W4A16 gs=128 symmetric.
  const PackedW pw4 = quantize(w_src, c.N, c.K, 128, true);
  const auto w_ref_w4 = dequant_all(pw4);

  uint32_t* q_d = nullptr;
  uint8_t* se_d = nullptr;
  __half *w_d = nullptr, *x_d = nullptr, *c_d = nullptr, *cf_d = nullptr;
  cuda_check(cudaMalloc(&q_d, p.q.size() * sizeof(uint32_t)), "malloc q");
  cuda_check(cudaMalloc(&se_d, p.scales.size() * sizeof(uint8_t)),
             "malloc scales");
  cuda_check(cudaMalloc(&w_d, (size_t)c.N * c.K * sizeof(__half)),
             "malloc w");
  cuda_check(cudaMalloc(&x_d, (size_t)c.M * c.K * sizeof(__half)),
             "malloc x");
  cuda_check(cudaMalloc(&c_d, (size_t)c.M * c.N * sizeof(__half)),
             "malloc c");
  cuda_check(cudaMalloc(&cf_d, (size_t)c.M * c.N * sizeof(__half)),
             "malloc cf");
  cuda_check(cudaMemcpy(q_d, p.q.data(), p.q.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice),
             "up q");
  cuda_check(cudaMemcpy(se_d, p.scales.data(),
                        p.scales.size() * sizeof(uint8_t),
                        cudaMemcpyHostToDevice),
             "up scales");
  cuda_check(cudaMemcpy(x_d, x.data(), (size_t)c.M * c.K * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "up x");

  const DevicePackedWm dp{c.N, c.K, p.B, q_d, se_d};
  cublasHandle_t cb = make_cublas();

  // Bitwise gate on both fused dequant paths, once per case.
  gemm_fused_mx(dp, x_d, cf_d, c.M, 0, w_d);
  cuda_check(cudaDeviceSynchronize(), "mx fused w_out sync");
  std::vector<__half> w_gpu((size_t)c.N * c.K);
  cuda_check(cudaMemcpy(w_gpu.data(), w_d, (size_t)c.N * c.K * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "down w fused");
  const bool bitwise =
      std::memcmp(w_gpu.data(), w_ref.data(),
                  (size_t)c.N * c.K * sizeof(__half)) == 0;
  gemm_mma_mx(dp, x_d, cf_d, c.M, 0, w_d);
  cuda_check(cudaDeviceSynchronize(), "mx mma w_out sync");
  cuda_check(cudaMemcpy(w_gpu.data(), w_d, (size_t)c.N * c.K * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "down w mma");
  const bool bitwise_mma =
      std::memcmp(w_gpu.data(), w_ref.data(),
                  (size_t)c.N * c.K * sizeof(__half)) == 0;

  // One fused/mma-vs-cuBLAS agreement figure per case (informational; the
  // gate lives in test_gpu).
  dequant_mx(dp, w_d);
  gemm_fp16(cb, w_d, x_d, c_d, c.M, c.N, c.K);
  gemm_fused_mx(dp, x_d, cf_d, c.M);
  cuda_check(cudaDeviceSynchronize(), "mx agree sync");
  std::vector<__half> c_base((size_t)c.M * c.N), c_fused((size_t)c.M * c.N);
  cuda_check(cudaMemcpy(c_base.data(), c_d, (size_t)c.M * c.N * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "down c base");
  cuda_check(cudaMemcpy(c_fused.data(), cf_d,
                        (size_t)c.M * c.N * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "down c fused");
  double num = 0.0, den = 0.0;
  for (size_t j = 0; j < c_base.size(); ++j) {
    const double d =
        (double)__half2float(c_fused[j]) - (double)__half2float(c_base[j]);
    num += d * d;
    const double b = (double)__half2float(c_base[j]);
    den += b * b;
  }
  const double fused_rel = std::sqrt(num / (den > 0.0 ? den : 1.0));
  gemm_mma_mx(dp, x_d, cf_d, c.M);
  cuda_check(cudaDeviceSynchronize(), "mx mma agree sync");
  cuda_check(cudaMemcpy(c_fused.data(), cf_d,
                        (size_t)c.M * c.N * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "down c mma");
  num = 0.0;
  for (size_t j = 0; j < c_base.size(); ++j) {
    const double d =
        (double)__half2float(c_fused[j]) - (double)__half2float(c_base[j]);
    num += d * d;
  }
  const double mma_rel = std::sqrt(num / (den > 0.0 ? den : 1.0));

  cudaEvent_t e0, e1, e2;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);

  Result r;
  r.c = c;
  r.fused_dequant_bitwise = bitwise;
  r.mma_dequant_bitwise = bitwise_mma;
  r.fused_vs_cublas_rel = fused_rel;
  r.mma_vs_cublas_rel = mma_rel;
  r.quant_rel_mx = quant_rel_l2(w_src, w_ref);
  r.quant_rel_w4 = quant_rel_l2(w_src, w_ref_w4);

  for (int rep = -2; rep < reps; ++rep) {  // two warmup reps, unrecorded
    const auto t0 = std::chrono::steady_clock::now();
    cudaEventRecord(e0);
    dequant_mx(dp, w_d);
    cudaEventRecord(e1);
    gemm_fp16(cb, w_d, x_d, c_d, c.M, c.N, c.K);
    cudaEventRecord(e2);
    cuda_check(cudaEventSynchronize(e2), "event sync");
    const auto t1 = std::chrono::steady_clock::now();
    const auto t2 = std::chrono::steady_clock::now();
    gemm_fused_mx(dp, x_d, cf_d, c.M);
    cuda_check(cudaDeviceSynchronize(), "mx fused sync");
    const auto t3 = std::chrono::steady_clock::now();
    gemm_mma_mx(dp, x_d, cf_d, c.M);
    cuda_check(cudaDeviceSynchronize(), "mx mma sync");
    const auto t4 = std::chrono::steady_clock::now();
    if (rep < 0) continue;
    float dq = 0.0f, gm = 0.0f;
    cuda_check(cudaEventElapsedTime(&dq, e0, e1), "dequant time");
    cuda_check(cudaEventElapsedTime(&gm, e1, e2), "gemm time");
    r.dequant_ms.push_back(dq);
    r.gemm_ms.push_back(gm);
    r.base_e2e_ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
    r.fused_ms.push_back(
        std::chrono::duration<double, std::milli>(t3 - t2).count());
    r.mma_ms.push_back(
        std::chrono::duration<double, std::milli>(t4 - t3).count());
  }

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
  cublasDestroy(cb);
  cudaFree(q_d);
  cudaFree(se_d);
  cudaFree(w_d);
  cudaFree(x_d);
  cudaFree(c_d);
  cudaFree(cf_d);
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
  uint64_t seed = 0xE40FULL;
  std::string out = std::string(W4_DATA_DIR) + "/w4a16_mx_bench.json";
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--reps=", 7) == 0)
      reps = std::atoi(argv[i] + 7);
    else if (std::strncmp(argv[i], "--seed=", 7) == 0)
      seed = std::stoull(argv[i] + 7, nullptr, 0);
    else if (std::strncmp(argv[i], "--out=", 6) == 0)
      out = argv[i] + 6;
    else {
      std::fprintf(stderr, "usage: bench_mx [--reps=9] [--seed=0x...] "
                           "[--out=path]\n");
      return 2;
    }
  }
  if (reps < 1) {
    std::fprintf(stderr, "bench_mx: --reps must be >= 1\n");
    return 2;
  }

  // Same decode-class shapes as bench_fused.
  const std::vector<Case> cases = {
      {1, 4096, 4096},   {8, 4096, 4096},   {16, 4096, 4096},
      {64, 4096, 4096},  {1, 4096, 14336},  {8, 4096, 14336},
      {16, 4096, 14336}, {64, 4096, 14336},
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

  const std::vector<double> triad_ms = run_triad(reps);
  const double triad_bytes = 3.0 * (double)(size_t{1} << 25) * 8.0;
  const double triad_gbs = triad_bytes / median(triad_ms) * 1e-6;
  std::printf("triad: %.1f GB/s (median of %d)\n", triad_gbs, reps);

  std::FILE* f = std::fopen(out.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "bench_mx: cannot open %s\n", out.c_str());
    return 1;
  }

  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"benchmark\": \"mxfp4 two-pass vs fused SIMT vs fused "
                  "WMMA, same session\",\n");
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
  std::fprintf(f, "  \"cpu_governor\": \"%s\",\n", cpu_governor().c_str());
  std::fprintf(f, "  \"nvcc\": \"%d.%d.%d\",\n", __CUDACC_VER_MAJOR__,
               __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
  std::fprintf(f, "  \"bytes_note\": \"analytic bytes moved; fused term is "
                  "packed W x m-tiles + E8M0 scales + x FP32 pre-pass "
                  "traffic + xf re-reads + C-out, not measured DRAM traffic; "
                  "ratio is baseline_e2e_median / fused_median\",\n");
  std::fprintf(f, "  \"quant_note\": \"quant_rel_l2 is ||dequant(quant(W)) "
                  "- W|| / ||W|| on FP64 source weights, MXFP4 vs W4A16 "
                  "gs=128 symmetric; quantization error, not a GEMM gate\",\n");
  std::fprintf(f, "  \"triad_fp64_ms\": ");
  print_ms_array(f, triad_ms);
  std::fprintf(f, ",\n  \"triad_fp64_gbs\": %.1f,\n", triad_gbs);
  std::fprintf(f, "  \"cases\": [\n");

  for (size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    const uint64_t cseed = seed + (uint64_t)i * 7919;
    const Result r = run_case(c, reps, cseed);

    const int mtiles = (c.M + mtile_of(c.M) - 1) / mtile_of(c.M);
    const double packed_bytes =
        (double)c.N * c.K * 0.5 + (double)c.N * ((c.K + 31) / 32);
    const double fused_bytes =
        packed_bytes * mtiles + (double)c.M * c.K * (2.0 + 4.0) +
        (double)c.M * c.K * 4.0 * mtiles + (double)c.M * c.N * 2.0;
    const int mma_mtiles = (c.M + 63) / 64;
    const double mma_bytes = packed_bytes * mma_mtiles +
                             (double)c.M * c.K * 2.0 +
                             (double)c.M * c.N * 2.0;
    const double base_med = median(r.base_e2e_ms);
    const double fused_med = median(r.fused_ms);
    const double mma_med = median(r.mma_ms);

    std::printf(
        "M=%d K=%d N=%d: baseline %.3f ms, fused %.3f ms (%.2fx), "
        "mma %.3f ms (%.2fx, %.1f GB/s eff), quant rel L2 mx %.2e w4 %.2e"
        "%s%s\n",
        c.M, c.K, c.N, base_med, fused_med, base_med / fused_med, mma_med,
        base_med / mma_med, mma_bytes / mma_med * 1e-6, r.quant_rel_mx,
        r.quant_rel_w4,
        r.fused_dequant_bitwise ? "" : "  [FUSED BITWISE FAILED]",
        r.mma_dequant_bitwise ? "" : "  [MMA BITWISE FAILED]");

    std::fprintf(f, "    {\"M\": %d, \"K\": %d, \"N\": %d,\n", c.M, c.K, c.N);
    std::fprintf(f, "     \"fused_dequant_bitwise\": %s,\n",
                 r.fused_dequant_bitwise ? "true" : "false");
    std::fprintf(f, "     \"mma_dequant_bitwise\": %s,\n",
                 r.mma_dequant_bitwise ? "true" : "false");
    std::fprintf(f, "     \"fused_vs_cublas_rel_l2\": %.3e,\n",
                 r.fused_vs_cublas_rel);
    std::fprintf(f, "     \"mma_vs_cublas_rel_l2\": %.3e,\n",
                 r.mma_vs_cublas_rel);
    std::fprintf(f, "     \"quant_rel_l2_mxfp4\": %.3e, "
                    "\"quant_rel_l2_w4a16_gs128_sym\": %.3e,\n",
                 r.quant_rel_mx, r.quant_rel_w4);
    std::fprintf(f, "     \"mtiles\": %d,\n", mtiles);
    std::fprintf(f, "     \"dequant_ms\": ");
    print_ms_array(f, r.dequant_ms);
    std::fprintf(f, ",\n     \"gemm_ms\": ");
    print_ms_array(f, r.gemm_ms);
    std::fprintf(f, ",\n     \"baseline_e2e_wall_ms\": ");
    print_ms_array(f, r.base_e2e_ms);
    std::fprintf(f, ",\n     \"fused_wall_ms\": ");
    print_ms_array(f, r.fused_ms);
    std::fprintf(f, ",\n     \"mma_wall_ms\": ");
    print_ms_array(f, r.mma_ms);
    std::fprintf(f,
                 ",\n     \"baseline_e2e_median_ms\": %.4f, "
                 "\"fused_median_ms\": %.4f, \"mma_median_ms\": %.4f,\n",
                 base_med, fused_med, mma_med);
    std::fprintf(f, "     \"ratio_fused\": %.3f, \"ratio_mma\": %.3f,\n",
                 base_med / fused_med, base_med / mma_med);
    std::fprintf(f, "     \"fused_bytes_min\": %.0f, \"fused_gbs_eff\": "
                    "%.1f,\n", fused_bytes, fused_bytes / fused_med * 1e-6);
    std::fprintf(f, "     \"mma_bytes_min\": %.0f, \"mma_gbs_eff\": %.1f}%s\n",
                 mma_bytes, mma_bytes / mma_med * 1e-6,
                 i + 1 < cases.size() ? "," : "");
  }

  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
