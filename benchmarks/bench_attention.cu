// KV-cache decode attention benchmark (R5/R5.5): single-block FP16-cache
// fused vs single-block W4 fused vs the two-pass (dequant both caches to
// FP16, then FP16 attention), plus the split-S flash paths for both cache
// types. The single-block kernels reject S > 4096 by contract (the shared
// scores bound); those cases time the flash paths alone, which is the
// point of R5.5. The two-pass is the fusion denominator; the FP16 paths
// isolate the cache-bandwidth win — same kernel shape, 4x the K/V bytes.
// Same session, raw per-rep timings preserved, FP64 triad as the bandwidth
// reference. Byte counts are analytic minima, not measured DRAM traffic.
//
// Correctness at bench size: the k_out/v_out hooks are checked bitwise
// against dequant_head per case; the output oracle gate lives in test_gpu
// at test sizes, so no precision claim is made at bench scale.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "w4/kvattention.cuh"
#include "w4/kvflash.cuh"
#include "w4/kvpaged.hpp"

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
  const size_t n = size_t{1} << 25;
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
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  std::vector<double> ms;
  for (int rep = -2; rep < reps; ++rep) {
    cudaEventRecord(e0);
    triad_kernel<<<sms * 8, 256>>>(a, b, c, n);
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

struct Case {
  int H, S, D, gs;
};

struct Result {
  Case c;
  std::vector<double> fp16_ms, w4_ms, twopass_ms, flash16_ms, flashw4_ms,
      flashf8_ms, pagedw4_ms;
  bool hooks_bitwise;
  double vs_reference_rel;
  double quant_rel_w4, quant_rel_f8;
};

Result run_case(const Case& c, int reps, uint64_t seed) {
  using namespace w4;
  using namespace w4::gpu;

  const auto k_src = normal_weights(c.H * c.S * c.D, seed);
  const auto v_src = normal_weights(c.H * c.S * c.D, seed + 1);
  const auto cc = kv::quantize_cache(k_src, v_src, c.H, c.S, c.D, c.gs, true);
  const auto cc8 = kv8::quantize_cache_f8(k_src, v_src, c.H, c.S, c.D, c.gs);
  const auto q = normal_activations(c.H * c.D, seed + 2);

  const int wd = words(c.D);
  const size_t nq = (size_t)c.H * c.S * wd;
  const size_t ns = (size_t)c.H * c.S * cc.G;
  const size_t ne = (size_t)c.H * c.S * c.D;
  uint32_t *qk_d, *qv_d;
  uint8_t *qk8_d, *qv8_d;
  __half *sk_d, *sv_d, *sk8_d, *sv8_d, *q_d, *o_d, *k16_d, *v16_d, *ko_d, *vo_d;
  cuda_check(cudaMalloc(&qk_d, nq * sizeof(uint32_t)), "qk");
  cuda_check(cudaMalloc(&qv_d, nq * sizeof(uint32_t)), "qv");
  cuda_check(cudaMalloc(&sk_d, ns * sizeof(__half)), "sk");
  cuda_check(cudaMalloc(&sv_d, ns * sizeof(__half)), "sv");
  cuda_check(cudaMalloc(&qk8_d, ne), "qk8");
  cuda_check(cudaMalloc(&qv8_d, ne), "qv8");
  cuda_check(cudaMalloc(&sk8_d, ns * sizeof(__half)), "sk8");
  cuda_check(cudaMalloc(&sv8_d, ns * sizeof(__half)), "sv8");
  cuda_check(cudaMalloc(&q_d, (size_t)c.H * c.D * sizeof(__half)), "q");
  cuda_check(cudaMalloc(&o_d, (size_t)c.H * c.D * sizeof(__half)), "o");
  cuda_check(cudaMalloc(&k16_d, ne * sizeof(__half)), "k16");
  cuda_check(cudaMalloc(&v16_d, ne * sizeof(__half)), "v16");
  cuda_check(cudaMalloc(&ko_d, ne * sizeof(__half)), "ko");
  cuda_check(cudaMalloc(&vo_d, ne * sizeof(__half)), "vo");
  cuda_check(cudaMemcpy(qk_d, cc.q_k.data(), nq * sizeof(uint32_t),
                        cudaMemcpyHostToDevice),
             "up qk");
  cuda_check(cudaMemcpy(qv_d, cc.q_v.data(), nq * sizeof(uint32_t),
                        cudaMemcpyHostToDevice),
             "up qv");
  cuda_check(cudaMemcpy(sk_d, cc.s_k.data(), ns * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "up sk");
  cuda_check(cudaMemcpy(sv_d, cc.s_v.data(), ns * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "up sv");
  cuda_check(cudaMemcpy(qk8_d, cc8.q_k.data(), ne, cudaMemcpyHostToDevice),
             "up qk8");
  cuda_check(cudaMemcpy(qv8_d, cc8.q_v.data(), ne, cudaMemcpyHostToDevice),
             "up qv8");
  cuda_check(cudaMemcpy(sk8_d, cc8.s_k.data(), ns * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "up sk8");
  cuda_check(cudaMemcpy(sv8_d, cc8.s_v.data(), ns * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "up sv8");
  cuda_check(cudaMemcpy(q_d, q.data(), (size_t)c.H * c.D * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "up q");

  const DeviceKVW4 dp{c.H, c.S, c.D, c.gs, cc.G, 1,
                      qk_d, sk_d, nullptr, qv_d, sv_d, nullptr};
  const DeviceKVF8 dp8{c.H, c.S, c.D, c.gs, cc8.G, qk8_d, sk8_d,
                       qv8_d, sv8_d};

  // Paged pool: page 16, shuffled placement (the worst case for the
  // indirection). Reuses the q_k/q_v etc. buffers — the pool is the same
  // byte count as the contiguous cache here (npages*page >= S rounds up
  // slightly), so fresh allocations sized to the pool.
  const int page = 16;
  const int npages = (c.S + page - 1) / page;
  std::vector<int> bt(npages);
  std::iota(bt.begin(), bt.end(), 0);
  std::mt19937_64 shuf(seed ^ 0x9A6EULL);
  std::shuffle(bt.begin(), bt.end(), shuf);
  const auto pc = kv::paged_from_contiguous(cc, page, bt);
  uint32_t *pqk_d, *pqv_d;
  __half* psk_d;
  __half* psv_d;
  int* bt_d;
  const size_t pool_q = (size_t)c.H * npages * page * wd;
  const size_t pool_s = (size_t)c.H * npages * page * cc.G;
  cuda_check(cudaMalloc(&pqk_d, pool_q * sizeof(uint32_t)), "pqk");
  cuda_check(cudaMalloc(&pqv_d, pool_q * sizeof(uint32_t)), "pqv");
  cuda_check(cudaMalloc(&psk_d, pool_s * sizeof(__half)), "psk");
  cuda_check(cudaMalloc(&psv_d, pool_s * sizeof(__half)), "psv");
  cuda_check(cudaMalloc(&bt_d, npages * sizeof(int)), "bt");
  cuda_check(cudaMemcpy(pqk_d, pc.q_k.data(), pool_q * sizeof(uint32_t),
                        cudaMemcpyHostToDevice),
             "up pqk");
  cuda_check(cudaMemcpy(pqv_d, pc.q_v.data(), pool_q * sizeof(uint32_t),
                        cudaMemcpyHostToDevice),
             "up pqv");
  cuda_check(cudaMemcpy(psk_d, pc.s_k.data(), pool_s * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "up psk");
  cuda_check(cudaMemcpy(psv_d, pc.s_v.data(), pool_s * sizeof(__half),
                        cudaMemcpyHostToDevice),
             "up psv");
  cuda_check(cudaMemcpy(bt_d, bt.data(), npages * sizeof(int),
                        cudaMemcpyHostToDevice),
             "up bt");
  const DeviceKVW4 dpp{c.H,  c.S,   c.D,   c.gs, cc.G, 1,
                       pqk_d, psk_d, nullptr, pqv_d, psv_d, nullptr};

  // Bitwise hook check plus one agreement figure vs the host reference.
  // The W4 flash path runs the hooks everywhere; the single-block path
  // rejects S > 4096 by contract, so it runs the hooks only when in
  // bounds. Each variant is checked against its own hook output.
  const bool old_ok = c.S <= KV_MAX_S;
  std::vector<__half> buf(ne);
  bool hooks = true;
  const int nvar = old_ok ? 3 : 2;
  for (int variant = 0; variant < nvar; ++variant) {
    if (variant == 0 && old_ok) {
      attention_w4(dp, q_d, o_d, 0, ko_d, vo_d);
      cuda_check(cudaDeviceSynchronize(), "hooks sync");
    } else if (variant <= 1) {
      attention_w4_flash(dp, q_d, o_d, 0, ko_d, vo_d);
      cuda_check(cudaDeviceSynchronize(), "flash hooks sync");
    } else {
      attention_f8_flash(dp8, q_d, o_d, 0, ko_d, vo_d);
      cuda_check(cudaDeviceSynchronize(), "f8 hooks sync");
    }
    for (int h = 0; h < c.H; ++h) {
      const auto kref = variant == 2 ? kv8::dequant_head_f8(cc8, h, false)
                                     : kv::dequant_head(cc, h, false);
      const auto vref = variant == 2 ? kv8::dequant_head_f8(cc8, h, true)
                                     : kv::dequant_head(cc, h, true);
      cuda_check(cudaMemcpy(buf.data(), ko_d + (size_t)h * c.S * c.D,
                            (size_t)c.S * c.D * sizeof(__half),
                            cudaMemcpyDeviceToHost),
                 "down ko");
      if (std::memcmp(buf.data(), kref.data(),
                      (size_t)c.S * c.D * sizeof(__half)) != 0)
        hooks = false;
      cuda_check(cudaMemcpy(buf.data(), vo_d + (size_t)h * c.S * c.D,
                            (size_t)c.S * c.D * sizeof(__half),
                            cudaMemcpyDeviceToHost),
                 "down vo");
      if (std::memcmp(buf.data(), vref.data(),
                      (size_t)c.S * c.D * sizeof(__half)) != 0)
        hooks = false;
    }
  }

  // Format accuracy on identical data: rel L2 of dequant(quant(cache)) vs
  // the FP64 source, W4 vs FP8, K and V together.
  double qnum4 = 0.0, qnum8 = 0.0, qden = 0.0;
  for (int h = 0; h < c.H; ++h) {
    const auto k4 = kv::dequant_head(cc, h, false);
    const auto v4 = kv::dequant_head(cc, h, true);
    const auto k8 = kv8::dequant_head_f8(cc8, h, false);
    const auto v8 = kv8::dequant_head_f8(cc8, h, true);
    const size_t off = (size_t)h * c.S * c.D;
    for (size_t j = 0; j < (size_t)c.S * c.D; ++j) {
      const double sk = k_src[off + j], sv_ = v_src[off + j];
      const double dk4 = (double)__half2float(k4[j]) - sk;
      const double dv4 = (double)__half2float(v4[j]) - sv_;
      const double dk8 = (double)__half2float(k8[j]) - sk;
      const double dv8 = (double)__half2float(v8[j]) - sv_;
      qnum4 += dk4 * dk4 + dv4 * dv4;
      qnum8 += dk8 * dk8 + dv8 * dv8;
      qden += sk * sk + sv_ * sv_;
    }
  }
  // The reference figure is the single-block W4 output vs the host
  // reference where that path is in bounds, the flash output past it (the
  // hooks loop left the flash output in o_d for the large-S cases).
  if (old_ok) {
    attention_w4(dp, q_d, o_d);
    cuda_check(cudaDeviceSynchronize(), "w4 rel sync");
  }
  std::vector<__half> o_gpu((size_t)c.H * c.D);
  cuda_check(cudaMemcpy(o_gpu.data(), o_d, (size_t)c.H * c.D * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "down o");
  const auto ref = kv::attention_reference(q, cc);
  double num = 0.0, den = 0.0;
  for (size_t j = 0; j < ref.size(); ++j) {
    const double d = (double)__half2float(o_gpu[j]) - ref[j];
    num += d * d;
    den += ref[j] * ref[j];
  }

  // The FP16 cache the baseline reads: dequantized once up front (not
  // timed — a real FP16 cache simply exists).
  for (int h = 0; h < c.H; ++h) {
    const DevicePackedW dk{c.S, c.D, c.gs, cc.G, 1,
                           qk_d + (size_t)h * c.S * wd,
                           sk_d + (size_t)h * c.S * cc.G, nullptr};
    dequant(dk, k16_d + (size_t)h * c.S * c.D);
    const DevicePackedW dv{c.S, c.D, c.gs, cc.G, 1,
                           qv_d + (size_t)h * c.S * wd,
                           sv_d + (size_t)h * c.S * cc.G, nullptr};
    dequant(dv, v16_d + (size_t)h * c.S * c.D);
  }
  cuda_check(cudaDeviceSynchronize(), "pre-dequant sync");

  Result r;
  r.c = c;
  r.hooks_bitwise = hooks;
  r.vs_reference_rel = std::sqrt(num / (den > 0.0 ? den : 1.0));
  r.quant_rel_w4 = std::sqrt(qnum4 / (qden > 0.0 ? qden : 1.0));
  r.quant_rel_f8 = std::sqrt(qnum8 / (qden > 0.0 ? qden : 1.0));

  // The single-block kernels reject S > 4096 by contract (the shared
  // scores bound); the flash paths carry those cases alone.
  for (int rep = -2; rep < reps; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    if (old_ok) {
      attention_fp16(q_d, k16_d, v16_d, o_d, c.H, c.S, c.D);
      cuda_check(cudaDeviceSynchronize(), "fp16 sync");
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (old_ok) {
      attention_w4(dp, q_d, o_d);
      cuda_check(cudaDeviceSynchronize(), "w4 sync");
    }
    const auto t2 = std::chrono::steady_clock::now();
    if (old_ok) {
      // Two-pass e2e: dequant both caches (timed), then FP16 attention.
      for (int h = 0; h < c.H; ++h) {
        const DevicePackedW dk{c.S, c.D, c.gs, cc.G, 1,
                               qk_d + (size_t)h * c.S * wd,
                               sk_d + (size_t)h * c.S * cc.G, nullptr};
        dequant(dk, k16_d + (size_t)h * c.S * c.D);
        const DevicePackedW dv{c.S, c.D, c.gs, cc.G, 1,
                               qv_d + (size_t)h * c.S * wd,
                               sv_d + (size_t)h * c.S * cc.G, nullptr};
        dequant(dv, v16_d + (size_t)h * c.S * c.D);
      }
      attention_fp16(q_d, k16_d, v16_d, o_d, c.H, c.S, c.D);
      cuda_check(cudaDeviceSynchronize(), "twopass sync");
    }
    const auto t3 = std::chrono::steady_clock::now();
    attention_fp16_flash(q_d, k16_d, v16_d, o_d, c.H, c.S, c.D);
    cuda_check(cudaDeviceSynchronize(), "fp16 flash sync");
    const auto t4 = std::chrono::steady_clock::now();
    attention_w4_flash(dp, q_d, o_d);
    cuda_check(cudaDeviceSynchronize(), "w4 flash sync");
    const auto t5 = std::chrono::steady_clock::now();
    attention_f8_flash(dp8, q_d, o_d);
    cuda_check(cudaDeviceSynchronize(), "f8 flash sync");
    const auto t6 = std::chrono::steady_clock::now();
    attention_w4_flash_paged(dpp, bt_d, page, npages, q_d, o_d);
    cuda_check(cudaDeviceSynchronize(), "paged w4 sync");
    const auto t7 = std::chrono::steady_clock::now();
    if (rep < 0) continue;
    if (old_ok) {
      r.fp16_ms.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
      r.w4_ms.push_back(
          std::chrono::duration<double, std::milli>(t2 - t1).count());
      r.twopass_ms.push_back(
          std::chrono::duration<double, std::milli>(t3 - t2).count());
    }
    r.flash16_ms.push_back(
        std::chrono::duration<double, std::milli>(t4 - t3).count());
    r.flashw4_ms.push_back(
        std::chrono::duration<double, std::milli>(t5 - t4).count());
    r.flashf8_ms.push_back(
        std::chrono::duration<double, std::milli>(t6 - t5).count());
    r.pagedw4_ms.push_back(
        std::chrono::duration<double, std::milli>(t7 - t6).count());
  }

  cudaFree(qk_d);
  cudaFree(qv_d);
  cudaFree(qk8_d);
  cudaFree(qv8_d);
  cudaFree(sk_d);
  cudaFree(sv_d);
  cudaFree(sk8_d);
  cudaFree(sv8_d);
  cudaFree(pqk_d);
  cudaFree(pqv_d);
  cudaFree(psk_d);
  cudaFree(psv_d);
  cudaFree(bt_d);
  cudaFree(q_d);
  cudaFree(o_d);
  cudaFree(k16_d);
  cudaFree(v16_d);
  cudaFree(ko_d);
  cudaFree(vo_d);
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
  uint64_t seed = 0xA77EULL;
  std::string out = std::string(W4_DATA_DIR) + "/w4a16_kv_bench.json";
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--reps=", 7) == 0)
      reps = std::atoi(argv[i] + 7);
    else if (std::strncmp(argv[i], "--seed=", 7) == 0)
      seed = std::stoull(argv[i] + 7, nullptr, 0);
    else if (std::strncmp(argv[i], "--out=", 6) == 0)
      out = argv[i] + 6;
    else {
      std::fprintf(stderr, "usage: bench_attention [--reps=9] [--seed=0x...] "
                           "[--out=path]\n");
      return 2;
    }
  }
  if (reps < 1) {
    std::fprintf(stderr, "bench_attention: --reps must be >= 1\n");
    return 2;
  }

  // Decode-class shapes: MHA (H=32) and GQA (H=8) head counts, D=128,
  // cache lengths from a context page past the single-block S=4096 shared
  // bound (the flash paths carry the S=8192/32768 cases alone).
  const std::vector<Case> cases = {
      {32, 512, 128, 128},   {32, 2048, 128, 128},  {32, 4096, 128, 128},
      {32, 8192, 128, 128},  {32, 32768, 128, 128},
      {8, 512, 128, 128},    {8, 2048, 128, 128},   {8, 4096, 128, 128},
      {8, 8192, 128, 128},   {8, 32768, 128, 128},
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
    std::fprintf(stderr, "bench_attention: cannot open %s\n", out.c_str());
    return 1;
  }

  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"benchmark\": \"kv-cache decode attention: fp16 fused "
                  "vs w4 fused vs two-pass vs split-S flash (both cache "
                  "types), same session\",\n");
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
  std::fprintf(f, "  \"bytes_note\": \"analytic bytes moved per decode token: "
                  "K+V cache reads (+ FP16 write-back and re-read for the "
                  "two-pass), q and out negligible; not measured DRAM "
                  "traffic\",\n");
  std::fprintf(f, "  \"triad_fp64_ms\": ");
  print_ms_array(f, triad_ms);
  std::fprintf(f, ",\n  \"triad_fp64_gbs\": %.1f,\n", triad_gbs);
  std::fprintf(f, "  \"cases\": [\n");

  for (size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    const Result r = run_case(c, reps, seed + (uint64_t)i * 7919);

    const bool old_ok = c.S <= w4::gpu::KV_MAX_S;
    const double cache_elems = 2.0 * c.H * c.S * c.D;  // K and V
    const double fp16_bytes = cache_elems * 2.0;
    const double w4_bytes = cache_elems * 0.5 +
                            2.0 * c.H * c.S * ((c.D + c.gs - 1) / c.gs) * 2.0;
    const double f8_bytes = cache_elems * 1.0 +
                            2.0 * c.H * c.S * ((c.D + c.gs - 1) / c.gs) * 2.0;
    const double twopass_bytes = w4_bytes + 2.0 * cache_elems * 2.0;
    const double f16_med = median(r.flash16_ms);
    const double fw4_med = median(r.flashw4_ms);
    const double ff8_med = median(r.flashf8_ms);
    const double pw4_med = median(r.pagedw4_ms);
    if (old_ok) {
      const double fp16_med = median(r.fp16_ms);
      const double w4_med = median(r.w4_ms);
      const double tp_med = median(r.twopass_ms);
      std::printf(
          "H=%d S=%d D=%d: fp16 %.3f (%.1f GB/s), w4 %.3f (%.2fx vs fp16), "
          "two-pass %.3f (%.2fx), flash16 %.3f, flashw4 %.3f (%.2fx vs "
          "flash16, %.1f GB/s), flashf8 %.3f (%.2fx vs flash16), "
          "quant w4 %.2e f8 %.2e%s\n",
          c.H, c.S, c.D, fp16_med, fp16_bytes / fp16_med * 1e-6, w4_med,
          fp16_med / w4_med, tp_med, tp_med / w4_med, f16_med, fw4_med,
          f16_med / fw4_med, w4_bytes / fw4_med * 1e-6, ff8_med,
          f16_med / ff8_med, r.quant_rel_w4, r.quant_rel_f8,
          r.hooks_bitwise ? "" : "  [HOOKS BITWISE FAILED]");
    } else {
      std::printf(
          "H=%d S=%d D=%d: flash16 %.3f (%.1f GB/s), flashw4 %.3f (%.2fx vs "
          "flash16, %.1f GB/s), flashf8 %.3f (%.2fx vs flash16), "
          "quant w4 %.2e f8 %.2e%s  [single-block paths skip S>4096]\n",
          c.H, c.S, c.D, f16_med, fp16_bytes / f16_med * 1e-6, fw4_med,
          f16_med / fw4_med, w4_bytes / fw4_med * 1e-6, ff8_med,
          f16_med / ff8_med, r.quant_rel_w4, r.quant_rel_f8,
          r.hooks_bitwise ? "" : "  [HOOKS BITWISE FAILED]");
    }

    std::fprintf(f, "    {\"H\": %d, \"S\": %d, \"D\": %d, \"gs\": %d,\n",
                 c.H, c.S, c.D, c.gs);
    std::fprintf(f, "     \"old_paths_run\": %s,\n",
                 old_ok ? "true" : "false");
    std::fprintf(f, "     \"hooks_bitwise\": %s,\n",
                 r.hooks_bitwise ? "true" : "false");
    std::fprintf(f, "     \"vs_reference_rel_l2\": %.3e,\n",
                 r.vs_reference_rel);
    if (old_ok) {
      std::fprintf(f, "     \"fp16_wall_ms\": ");
      print_ms_array(f, r.fp16_ms);
      std::fprintf(f, ",\n     \"w4_wall_ms\": ");
      print_ms_array(f, r.w4_ms);
      std::fprintf(f, ",\n     \"twopass_wall_ms\": ");
      print_ms_array(f, r.twopass_ms);
      std::fprintf(f,
                   ",\n     \"fp16_median_ms\": %.4f, \"w4_median_ms\": %.4f, "
                   "\"twopass_median_ms\": %.4f,\n",
                   median(r.fp16_ms), median(r.w4_ms), median(r.twopass_ms));
      std::fprintf(f,
                   "     \"ratio_w4_vs_fp16\": %.3f, \"ratio_w4_vs_twopass\": "
                   "%.3f,\n",
                   median(r.fp16_ms) / median(r.w4_ms),
                   median(r.twopass_ms) / median(r.w4_ms));
    }
    std::fprintf(f, "     \"flash16_wall_ms\": ");
    print_ms_array(f, r.flash16_ms);
    std::fprintf(f, ",\n     \"flashw4_wall_ms\": ");
    print_ms_array(f, r.flashw4_ms);
    std::fprintf(f, ",\n     \"flashf8_wall_ms\": ");
    print_ms_array(f, r.flashf8_ms);
    std::fprintf(f, ",\n     \"pagedw4_wall_ms\": ");
    print_ms_array(f, r.pagedw4_ms);
    std::fprintf(f,
                 ",\n     \"flash16_median_ms\": %.4f, \"flashw4_median_ms\": "
                 "%.4f, \"flashf8_median_ms\": %.4f, \"pagedw4_median_ms\": "
                 "%.4f,\n",
                 f16_med, fw4_med, ff8_med, pw4_med);
    std::fprintf(f,
                 "     \"ratio_flashw4_vs_flash16\": %.3f, "
                 "\"ratio_flashf8_vs_flash16\": %.3f, "
                 "\"ratio_pagedw4_vs_flashw4\": %.3f,\n",
                 f16_med / fw4_med, f16_med / ff8_med, fw4_med / pw4_med);
    std::fprintf(f,
                 "     \"quant_rel_l2_w4\": %.4e, \"quant_rel_l2_f8\": %.4e,\n",
                 r.quant_rel_w4, r.quant_rel_f8);
    std::fprintf(f,
                 "     \"fp16_bytes_min\": %.0f, \"w4_bytes_min\": %.0f, "
                 "\"f8_bytes_min\": %.0f, \"twopass_bytes_min\": %.0f,\n",
                 fp16_bytes, w4_bytes, f8_bytes, twopass_bytes);
    std::fprintf(f,
                 "     \"flash16_gbs_eff\": %.1f, \"flashw4_gbs_eff\": %.1f, "
                 "\"flashf8_gbs_eff\": %.1f}%s\n",
                 fp16_bytes / f16_med * 1e-6, w4_bytes / fw4_med * 1e-6,
                 f8_bytes / ff8_med * 1e-6, i + 1 < cases.size() ? "," : "");
  }

  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
  std::printf("wrote %s\n", out.c_str());
  return 0;
}
