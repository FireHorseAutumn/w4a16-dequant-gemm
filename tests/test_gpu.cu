// GPU gates for the two-pass baseline and the fused kernel:
//   1. dequant_kernel output is bitwise identical to the host reference
//      (both sides call dequant_one; any drift here is a real bug).
//   2. cuBLAS FP16/FP32-accum GEMM vs the FP64 oracle at rel L2 <= 1e-3.
//      Inputs are identical FP16 values on both sides, so the only legal
//      deviation is accumulation order and the final half rounding;
//      measured worst case is printed. The gate exists to catch silent
//      wrong output, not to grade rounding.
//   3. fused_kernel with w_out set reproduces dequant_all bitwise (gate 1
//      applied to the fused dequant path).
//   4. fused_kernel GEMM vs the FP64 oracle at rel L2 <= 1e-3, and vs the
//      cuBLAS output at rel L2 <= 2e-3 (each side is within 1e-3 of the
//      oracle, so they are within 2e-3 of each other).
//      Configs with M > 16 exercise the gridDim.y m-tiling and its guards.

#include <cstdio>
#include <cstring>
#include <vector>

#include "w4/dequant.cuh"
#include "w4/fused.cuh"

struct Cfg {
  int M, N, K, gs;
};

int main() {
  const std::vector<Cfg> cfgs = {
      {1, 64, 64, 8},      {8, 128, 96, 16},  {16, 256, 100, 32},
      {4, 96, 72, 48},     {2, 32, 128, 64},  {3, 40, 256, 128},
      {5, 72, 40, 8},
      {17, 64, 128, 16},   // M > 16: m-tiling and partial-tile guards
      {33, 48, 104, 24},   // m-tiles with gs % 32 != 0
      {20, 32, 100, 8},    // m-tiles with ragged K
  };
  int checks = 0, fails = 0;
  double worst_rel = 0.0;

  cublasHandle_t cb = w4::gpu::make_cublas();

  for (const bool sym : {true, false}) {
    for (size_t i = 0; i < cfgs.size(); ++i) {
      const Cfg c = cfgs[i];
      const uint64_t seed = 0x44A1ULL ^ (uint64_t)(i * 977 + (sym ? 0 : 1));
      const auto w_src = w4::normal_weights(c.N * c.K, seed);
      const w4::PackedW p = w4::quantize(w_src, c.N, c.K, c.gs, sym);
      const auto w_ref = w4::dequant_all(p);
      const auto x = w4::normal_activations(c.M * c.K, seed + 1);

      uint32_t* q_d = nullptr;
      uint32_t* z_d = nullptr;
      __half *s_d = nullptr, *w_d = nullptr, *x_d = nullptr, *c_d = nullptr;
      w4::gpu::cuda_check(
          cudaMalloc(&q_d, p.q.size() * sizeof(uint32_t)), "malloc q");
      w4::gpu::cuda_check(
          cudaMalloc(&s_d, p.scales.size() * sizeof(__half)), "malloc s");
      if (!sym)
        w4::gpu::cuda_check(
            cudaMalloc(&z_d, p.zeros.size() * sizeof(uint32_t)),
            "malloc z");
      w4::gpu::cuda_check(
          cudaMalloc(&w_d, (size_t)c.N * c.K * sizeof(__half)), "malloc w");
      w4::gpu::cuda_check(
          cudaMalloc(&x_d, (size_t)c.M * c.K * sizeof(__half)), "malloc x");
      w4::gpu::cuda_check(
          cudaMalloc(&c_d, (size_t)c.M * c.N * sizeof(__half)), "malloc c");
      __half* c_f = nullptr;
      w4::gpu::cuda_check(
          cudaMalloc(&c_f, (size_t)c.M * c.N * sizeof(__half)),
          "malloc c fused");

      w4::gpu::cuda_check(cudaMemcpy(q_d, p.q.data(),
                                     p.q.size() * sizeof(uint32_t),
                                     cudaMemcpyHostToDevice),
                          "up q");
      w4::gpu::cuda_check(cudaMemcpy(s_d, p.scales.data(),
                                     p.scales.size() * sizeof(__half),
                                     cudaMemcpyHostToDevice),
                          "up s");
      if (!sym)
        w4::gpu::cuda_check(cudaMemcpy(z_d, p.zeros.data(),
                                       p.zeros.size() * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice),
                            "up z");
      w4::gpu::cuda_check(cudaMemcpy(x_d, x.data(),
                                     (size_t)c.M * c.K * sizeof(__half),
                                     cudaMemcpyHostToDevice),
                          "up x");

      const w4::gpu::DevicePackedW dp{c.N, c.K, c.gs, p.G,
                                      sym ? 1 : 0, q_d, s_d, z_d};
      w4::gpu::dequant(dp, w_d);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "dequant sync");

      std::vector<__half> w_gpu((size_t)c.N * c.K);
      w4::gpu::cuda_check(cudaMemcpy(w_gpu.data(), w_d,
                                     (size_t)c.N * c.K * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down w");

      ++checks;
      if (std::memcmp(w_gpu.data(), w_ref.data(),
                      (size_t)c.N * c.K * sizeof(__half)) != 0) {
        ++fails;
        std::printf("FAIL dequant bitwise %s N=%d K=%d gs=%d\n",
                    sym ? "sym" : "asym", c.N, c.K, c.gs);
      }

      w4::gpu::gemm_fp16(cb, w_d, x_d, c_d, c.M, c.N, c.K);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "gemm sync");

      std::vector<__half> c_gpu((size_t)c.M * c.N);
      w4::gpu::cuda_check(cudaMemcpy(c_gpu.data(), c_d,
                                     (size_t)c.M * c.N * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down c");
      const auto c_ref = w4::gemm_oracle(x, c.M, c.K, w_ref, c.N);
      double num = 0.0, den = 0.0;
      for (size_t j = 0; j < c_ref.size(); ++j) {
        const double d = (double)__half2float(c_gpu[j]) - c_ref[j];
        num += d * d;
        den += c_ref[j] * c_ref[j];
      }
      const double rel = std::sqrt(num / (den > 0.0 ? den : 1.0));
      worst_rel = std::max(worst_rel, rel);
      ++checks;
      if (rel > 1e-3) {
        ++fails;
        std::printf("FAIL gemm %s N=%d K=%d gs=%d rel=%.3e\n",
                    sym ? "sym" : "asym", c.N, c.K, c.gs, rel);
      }

      // Fused kernel, dequant path: w_out must reproduce dequant_all
      // bitwise (w_d is free to reuse; the two-pass no longer needs it).
      w4::gpu::gemm_fused(dp, x_d, c_f, c.M, 0, w_d);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "fused w_out sync");
      w4::gpu::cuda_check(cudaMemcpy(w_gpu.data(), w_d,
                                     (size_t)c.N * c.K * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down w fused");
      ++checks;
      if (std::memcmp(w_gpu.data(), w_ref.data(),
                      (size_t)c.N * c.K * sizeof(__half)) != 0) {
        ++fails;
        std::printf("FAIL fused dequant bitwise %s M=%d N=%d K=%d gs=%d\n",
                    sym ? "sym" : "asym", c.M, c.N, c.K, c.gs);
      }

      // Fused kernel, GEMM path: vs oracle at 1e-3, vs cuBLAS at 2e-3.
      w4::gpu::gemm_fused(dp, x_d, c_f, c.M);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "fused sync");
      std::vector<__half> c_fused((size_t)c.M * c.N);
      w4::gpu::cuda_check(cudaMemcpy(c_fused.data(), c_f,
                                     (size_t)c.M * c.N * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down c fused");
      double num_f = 0.0, num_x = 0.0;
      for (size_t j = 0; j < c_ref.size(); ++j) {
        const double d = (double)__half2float(c_fused[j]) - c_ref[j];
        num_f += d * d;
        const double dx =
            (double)__half2float(c_fused[j]) - (double)__half2float(c_gpu[j]);
        num_x += dx * dx;
      }
      const double rel_f = std::sqrt(num_f / (den > 0.0 ? den : 1.0));
      const double rel_x = std::sqrt(num_x / (den > 0.0 ? den : 1.0));
      worst_rel = std::max(worst_rel, rel_f);
      ++checks;
      if (rel_f > 1e-3) {
        ++fails;
        std::printf("FAIL fused gemm %s M=%d N=%d K=%d gs=%d rel=%.3e\n",
                    sym ? "sym" : "asym", c.M, c.N, c.K, c.gs, rel_f);
      }
      ++checks;
      if (rel_x > 2e-3) {
        ++fails;
        std::printf(
            "FAIL fused-vs-cublas %s M=%d N=%d K=%d gs=%d rel=%.3e\n",
            sym ? "sym" : "asym", c.M, c.N, c.K, c.gs, rel_x);
      }

      cudaFree(c_f);

      cudaFree(q_d);
      cudaFree(s_d);
      cudaFree(z_d);
      cudaFree(w_d);
      cudaFree(x_d);
      cudaFree(c_d);
    }
  }

  cublasDestroy(cb);

  // Host-side entry contracts: every malformed launch must throw before any
  // kernel or cuBLAS call. Dummy pointers are never dereferenced.
  {
    auto throws_invalid = [](auto&& f) {
      try {
        f();
      } catch (const std::invalid_argument&) {
        return true;
      } catch (...) {
      }
      return false;
    };
    auto expect = [&](bool ok, const char* name) {
      ++checks;
      if (!ok) {
        ++fails;
        std::printf("FAIL no-throw: %s\n", name);
      }
    };
    const w4::gpu::DevicePackedW good{2, 32, 8, 4, 1,
                                      (const uint32_t*)1, (const __half*)1,
                                      nullptr};
    __half* out = (__half*)1;
    auto bad_gs0 = good; bad_gs0.gs = 0;
    expect(throws_invalid([&] { w4::gpu::dequant(bad_gs0, out); }), "gs=0");
    auto bad_gs7 = good; bad_gs7.gs = 7;
    expect(throws_invalid([&] { w4::gpu::dequant(bad_gs7, out); }), "gs=7");
    auto bad_g = good; bad_g.G = 3;
    expect(throws_invalid([&] { w4::gpu::dequant(bad_g, out); }), "G wrong");
    auto bad_q = good; bad_q.q = nullptr;
    expect(throws_invalid([&] { w4::gpu::dequant(bad_q, out); }), "q null");
    auto bad_z = good; bad_z.symmetric = 0;
    expect(throws_invalid([&] { w4::gpu::dequant(bad_z, out); }),
           "asym zeros null");
    auto sym_z = good; sym_z.zeros = (const uint32_t*)1;
    expect(throws_invalid([&] { w4::gpu::dequant(sym_z, out); }),
           "sym carries zeros");
    // 2^21 rows x K=8192 -> words(K)=1024 -> total hits 2^31 exactly.
    auto big = good; big.N = 1 << 21; big.K = 8192; big.G = 1024;
    expect(throws_invalid([&] { w4::gpu::dequant(big, out); }),
           "total >= 2^31");
    cublasHandle_t h = w4::gpu::make_cublas();
    expect(throws_invalid(
               [&] { w4::gpu::gemm_fp16(h, out, out, out, 0, 2, 2); }),
           "gemm M=0");
    expect(throws_invalid(
               [&] { w4::gpu::gemm_fp16(h, nullptr, out, out, 2, 2, 2); }),
           "gemm w null");
    cublasDestroy(h);

    // Fused entry contracts: same structural rules as dequant, plus the
    // activation/output pointers and M.
    expect(throws_invalid([&] { w4::gpu::gemm_fused(bad_gs0, out, out, 2); }),
           "fused gs=0");
    expect(throws_invalid([&] { w4::gpu::gemm_fused(bad_gs7, out, out, 2); }),
           "fused gs=7");
    expect(throws_invalid([&] { w4::gpu::gemm_fused(bad_g, out, out, 2); }),
           "fused G wrong");
    expect(throws_invalid([&] { w4::gpu::gemm_fused(bad_q, out, out, 2); }),
           "fused q null");
    expect(throws_invalid([&] { w4::gpu::gemm_fused(bad_z, out, out, 2); }),
           "fused asym zeros null");
    expect(throws_invalid([&] { w4::gpu::gemm_fused(sym_z, out, out, 2); }),
           "fused sym carries zeros");
    expect(throws_invalid([&] { w4::gpu::gemm_fused(good, out, out, 0); }),
           "fused M=0");
    expect(
        throws_invalid([&] { w4::gpu::gemm_fused(good, nullptr, out, 2); }),
        "fused x null");
    expect(
        throws_invalid([&] { w4::gpu::gemm_fused(good, out, nullptr, 2); }),
        "fused c null");
    {
      auto bad_n = good;
      bad_n.N = 0;
      expect(throws_invalid([&] { w4::gpu::gemm_fused(bad_n, out, out, 2); }),
             "fused N=0");
    }
    // gridDim.y caps at 65535: with mtile 4 the launcher must reject
    // M > 262140 before any allocation or launch. Only the throw side is
    // testable here — a passing M would launch against the dummy pointers.
    expect(throws_invalid([&] { w4::gpu::gemm_fused(good, out, out, 262141); }),
           "fused M past grid.y bound");
  }

  std::printf("gpu: %d checks, %d failures, worst gemm rel L2 = %.3e\n",
              checks, fails, worst_rel);
  return fails ? 1 : 0;
}
