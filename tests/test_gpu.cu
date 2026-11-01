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

//   5. mma_kernel (WMMA/tensor-core path): same gates as the fused kernel —
//      bitwise dequant through w_out, oracle at 1e-3, cuBLAS cross-check at
//      2e-3. Ragged-K and partial-tile configs exercise the shared-memory
//      zero padding.
//   6. MXFP4 kernels (dequant_mx, fused_kernel_mx, mma_kernel_mx): the same
//      gate set against the MXFP4 host reference — bitwise dequant, oracle
//      at 1e-3, cuBLAS at 2e-3, entry contracts including the 2^31 bound on
//      the naive dequant launch.
//   7. KV-cache decode attention (attention_w4): k_out/v_out hooks reproduce
//      dequant_head bitwise per head, output vs the FP64 host reference at
//      rel L2 <= 1e-3, vs the device two-pass (dequant + FP16 attention) at
//      2e-3. Configs cover S=1, ragged words (D=96), both sign modes, and
//      S=4096 exactly on the shared-memory bound; entry contracts throw
//      past it.

#include <cstdio>
#include <cstring>
#include <vector>

#include "w4/dequant.cuh"
#include "w4/fused.cuh"
#include "w4/kvattention.cuh"
#include "w4/mma.cuh"
#include "w4/mxfused.cuh"

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

      // MMA kernel: same gates as the fused path (bitwise dequant through
      // w_out, oracle at 1e-3, cuBLAS cross-check at 2e-3).
      w4::gpu::gemm_mma(dp, x_d, c_f, c.M, 0, w_d);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "mma w_out sync");
      w4::gpu::cuda_check(cudaMemcpy(w_gpu.data(), w_d,
                                     (size_t)c.N * c.K * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down w mma");
      ++checks;
      if (std::memcmp(w_gpu.data(), w_ref.data(),
                      (size_t)c.N * c.K * sizeof(__half)) != 0) {
        ++fails;
        std::printf("FAIL mma dequant bitwise %s M=%d N=%d K=%d gs=%d\n",
                    sym ? "sym" : "asym", c.M, c.N, c.K, c.gs);
      }
      w4::gpu::gemm_mma(dp, x_d, c_f, c.M);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "mma sync");
      w4::gpu::cuda_check(cudaMemcpy(c_fused.data(), c_f,
                                     (size_t)c.M * c.N * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down c mma");
      double num_m = 0.0, num_mx = 0.0;
      for (size_t j = 0; j < c_ref.size(); ++j) {
        const double d = (double)__half2float(c_fused[j]) - c_ref[j];
        num_m += d * d;
        const double dx =
            (double)__half2float(c_fused[j]) - (double)__half2float(c_gpu[j]);
        num_mx += dx * dx;
      }
      const double rel_m = std::sqrt(num_m / (den > 0.0 ? den : 1.0));
      const double rel_mx = std::sqrt(num_mx / (den > 0.0 ? den : 1.0));
      worst_rel = std::max(worst_rel, rel_m);
      ++checks;
      if (rel_m > 1e-3) {
        ++fails;
        std::printf("FAIL mma gemm %s M=%d N=%d K=%d gs=%d rel=%.3e\n",
                    sym ? "sym" : "asym", c.M, c.N, c.K, c.gs, rel_m);
      }
      ++checks;
      if (rel_mx > 2e-3) {
        ++fails;
        std::printf("FAIL mma-vs-cublas %s M=%d N=%d K=%d gs=%d rel=%.3e\n",
                    sym ? "sym" : "asym", c.M, c.N, c.K, c.gs, rel_mx);
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

  // MXFP4 (E2M1 + E8M0 block scale): the same gate set applied to the MX
  // kernels — dequant_mx bitwise vs the host reference, fused/MMA dequant
  // bitwise through w_out, GEMM vs the FP64 oracle at 1e-3, vs cuBLAS at
  // 2e-3. The gs column of cfgs is unused here: the MX block is fixed at
  // 32, and K in {72, 100, 104, 40} exercises partial blocks and ragged
  // word tails.
  for (size_t i = 0; i < cfgs.size(); ++i) {
    const Cfg c = cfgs[i];
    const uint64_t seed = 0x1E4AULL ^ (uint64_t)(i * 977);
    const auto w_src = w4::normal_weights(c.N * c.K, seed);
    const w4::mx::PackedWm p = w4::mx::quantize_mx(w_src, c.N, c.K);
    const auto w_ref = w4::mx::dequant_all_mx(p);
    const auto x = w4::normal_activations(c.M * c.K, seed + 1);

    uint32_t* q_d = nullptr;
    uint8_t* se_d = nullptr;
    __half *w_d = nullptr, *x_d = nullptr, *c_d = nullptr, *c_f = nullptr;
    w4::gpu::cuda_check(
        cudaMalloc(&q_d, p.q.size() * sizeof(uint32_t)), "malloc mx q");
    w4::gpu::cuda_check(
        cudaMalloc(&se_d, p.scales.size() * sizeof(uint8_t)),
        "malloc mx scales");
    w4::gpu::cuda_check(
        cudaMalloc(&w_d, (size_t)c.N * c.K * sizeof(__half)), "malloc mx w");
    w4::gpu::cuda_check(
        cudaMalloc(&x_d, (size_t)c.M * c.K * sizeof(__half)), "malloc mx x");
    w4::gpu::cuda_check(
        cudaMalloc(&c_d, (size_t)c.M * c.N * sizeof(__half)), "malloc mx c");
    w4::gpu::cuda_check(
        cudaMalloc(&c_f, (size_t)c.M * c.N * sizeof(__half)), "malloc mx cf");

    w4::gpu::cuda_check(cudaMemcpy(q_d, p.q.data(),
                                   p.q.size() * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice),
                        "up mx q");
    w4::gpu::cuda_check(cudaMemcpy(se_d, p.scales.data(),
                                   p.scales.size() * sizeof(uint8_t),
                                   cudaMemcpyHostToDevice),
                        "up mx scales");
    w4::gpu::cuda_check(cudaMemcpy(x_d, x.data(),
                                   (size_t)c.M * c.K * sizeof(__half),
                                   cudaMemcpyHostToDevice),
                        "up mx x");

    const w4::gpu::DevicePackedWm dp{c.N, c.K, p.B, q_d, se_d};
    w4::gpu::dequant_mx(dp, w_d);
    w4::gpu::cuda_check(cudaDeviceSynchronize(), "mx dequant sync");

    std::vector<__half> w_gpu((size_t)c.N * c.K);
    w4::gpu::cuda_check(cudaMemcpy(w_gpu.data(), w_d,
                                   (size_t)c.N * c.K * sizeof(__half),
                                   cudaMemcpyDeviceToHost),
                        "down mx w");
    ++checks;
    if (std::memcmp(w_gpu.data(), w_ref.data(),
                    (size_t)c.N * c.K * sizeof(__half)) != 0) {
      ++fails;
      std::printf("FAIL mx dequant bitwise N=%d K=%d\n", c.N, c.K);
    }

    w4::gpu::gemm_fp16(cb, w_d, x_d, c_d, c.M, c.N, c.K);
    w4::gpu::cuda_check(cudaDeviceSynchronize(), "mx gemm sync");
    std::vector<__half> c_gpu((size_t)c.M * c.N);
    w4::gpu::cuda_check(cudaMemcpy(c_gpu.data(), c_d,
                                   (size_t)c.M * c.N * sizeof(__half),
                                   cudaMemcpyDeviceToHost),
                        "down mx c");
    const auto c_ref = w4::gemm_oracle(x, c.M, c.K, w_ref, c.N);
    double den = 0.0;
    for (size_t j = 0; j < c_ref.size(); ++j) den += c_ref[j] * c_ref[j];

    // Fused MX: w_out bitwise, then GEMM gates.
    w4::gpu::gemm_fused_mx(dp, x_d, c_f, c.M, 0, w_d);
    w4::gpu::cuda_check(cudaDeviceSynchronize(), "mx fused w_out sync");
    w4::gpu::cuda_check(cudaMemcpy(w_gpu.data(), w_d,
                                   (size_t)c.N * c.K * sizeof(__half),
                                   cudaMemcpyDeviceToHost),
                        "down mx w fused");
    ++checks;
    if (std::memcmp(w_gpu.data(), w_ref.data(),
                    (size_t)c.N * c.K * sizeof(__half)) != 0) {
      ++fails;
      std::printf("FAIL mx fused dequant bitwise M=%d N=%d K=%d\n", c.M, c.N,
                  c.K);
    }
    w4::gpu::gemm_fused_mx(dp, x_d, c_f, c.M);
    w4::gpu::cuda_check(cudaDeviceSynchronize(), "mx fused sync");
    std::vector<__half> c_fused((size_t)c.M * c.N);
    w4::gpu::cuda_check(cudaMemcpy(c_fused.data(), c_f,
                                   (size_t)c.M * c.N * sizeof(__half),
                                   cudaMemcpyDeviceToHost),
                        "down mx c fused");
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
      std::printf("FAIL mx fused gemm M=%d N=%d K=%d rel=%.3e\n", c.M, c.N,
                  c.K, rel_f);
    }
    ++checks;
    if (rel_x > 2e-3) {
      ++fails;
      std::printf("FAIL mx fused-vs-cublas M=%d N=%d K=%d rel=%.3e\n", c.M,
                  c.N, c.K, rel_x);
    }

    // MMA MX: same gates.
    w4::gpu::gemm_mma_mx(dp, x_d, c_f, c.M, 0, w_d);
    w4::gpu::cuda_check(cudaDeviceSynchronize(), "mx mma w_out sync");
    w4::gpu::cuda_check(cudaMemcpy(w_gpu.data(), w_d,
                                   (size_t)c.N * c.K * sizeof(__half),
                                   cudaMemcpyDeviceToHost),
                        "down mx w mma");
    ++checks;
    if (std::memcmp(w_gpu.data(), w_ref.data(),
                    (size_t)c.N * c.K * sizeof(__half)) != 0) {
      ++fails;
      std::printf("FAIL mx mma dequant bitwise M=%d N=%d K=%d\n", c.M, c.N,
                  c.K);
    }
    w4::gpu::gemm_mma_mx(dp, x_d, c_f, c.M);
    w4::gpu::cuda_check(cudaDeviceSynchronize(), "mx mma sync");
    w4::gpu::cuda_check(cudaMemcpy(c_fused.data(), c_f,
                                   (size_t)c.M * c.N * sizeof(__half),
                                   cudaMemcpyDeviceToHost),
                        "down mx c mma");
    double num_m = 0.0, num_mx = 0.0;
    for (size_t j = 0; j < c_ref.size(); ++j) {
      const double d = (double)__half2float(c_fused[j]) - c_ref[j];
      num_m += d * d;
      const double dx =
          (double)__half2float(c_fused[j]) - (double)__half2float(c_gpu[j]);
      num_mx += dx * dx;
    }
    const double rel_m = std::sqrt(num_m / (den > 0.0 ? den : 1.0));
    const double rel_mx = std::sqrt(num_mx / (den > 0.0 ? den : 1.0));
    worst_rel = std::max(worst_rel, rel_m);
    ++checks;
    if (rel_m > 1e-3) {
      ++fails;
      std::printf("FAIL mx mma gemm M=%d N=%d K=%d rel=%.3e\n", c.M, c.N,
                  c.K, rel_m);
    }
    ++checks;
    if (rel_mx > 2e-3) {
      ++fails;
      std::printf("FAIL mx mma-vs-cublas M=%d N=%d K=%d rel=%.3e\n", c.M, c.N,
                  c.K, rel_mx);
    }

    cudaFree(q_d);
    cudaFree(se_d);
    cudaFree(w_d);
    cudaFree(x_d);
    cudaFree(c_d);
    cudaFree(c_f);
  }

  // KV-cache decode attention (R5): fused dequant-at-load vs the FP64 host
  // reference at rel L2 <= 1e-3 and vs the device two-pass (per-head
  // dequant -> FP16 attention) at 2e-3; the k_out/v_out hooks must
  // reproduce dequant_head bitwise. S=1 is the single-position edge;
  // D=96/gs=32 exercises ragged words; S=4096 sits exactly on the shared
  // bound.
  {
    struct KvCfg {
      int H, S, D, gs;
    };
    const std::vector<KvCfg> kcfgs = {
        {1, 1, 64, 32},      {2, 7, 64, 8},      {4, 128, 128, 128},
        {2, 300, 96, 32},    {3, 65, 40, 8},     {1, 4096, 64, 64},
    };
    for (const bool sym : {true, false}) {
      for (size_t i = 0; i < kcfgs.size(); ++i) {
        const KvCfg c = kcfgs[i];
        const uint64_t seed = 0x77B5ULL ^ (uint64_t)(i * 977 + (sym ? 0 : 1));
        const auto k_src = w4::normal_weights(c.H * c.S * c.D, seed);
        const auto v_src = w4::normal_weights(c.H * c.S * c.D, seed + 1);
        const auto cc =
            w4::kv::quantize_cache(k_src, v_src, c.H, c.S, c.D, c.gs, sym);
        const auto q = w4::normal_activations(c.H * c.D, seed + 2);
        const auto ref = w4::kv::attention_reference(q, cc);

        const int wd = w4::words(c.D);
        const size_t nq = (size_t)c.H * c.S * wd;
        const size_t ns = (size_t)c.H * c.S * cc.G;
        const size_t nz = (size_t)c.H * c.S * w4::words(cc.G);
        const size_t ne = (size_t)c.H * c.S * c.D;
        uint32_t *qk_d = nullptr, *qv_d = nullptr, *zk_d = nullptr,
                 *zv_d = nullptr;
        __half *sk_d = nullptr, *sv_d = nullptr, *q_d = nullptr,
               *o_d = nullptr, *o16_d = nullptr, *k16_d = nullptr,
               *v16_d = nullptr, *ko_d = nullptr, *vo_d = nullptr;
        w4::gpu::cuda_check(cudaMalloc(&qk_d, nq * sizeof(uint32_t)), "kv qk");
        w4::gpu::cuda_check(cudaMalloc(&qv_d, nq * sizeof(uint32_t)), "kv qv");
        w4::gpu::cuda_check(cudaMalloc(&sk_d, ns * sizeof(__half)), "kv sk");
        w4::gpu::cuda_check(cudaMalloc(&sv_d, ns * sizeof(__half)), "kv sv");
        if (!sym) {
          w4::gpu::cuda_check(cudaMalloc(&zk_d, nz * sizeof(uint32_t)),
                              "kv zk");
          w4::gpu::cuda_check(cudaMalloc(&zv_d, nz * sizeof(uint32_t)),
                              "kv zv");
        }
        w4::gpu::cuda_check(
            cudaMalloc(&q_d, (size_t)c.H * c.D * sizeof(__half)), "kv q");
        w4::gpu::cuda_check(
            cudaMalloc(&o_d, (size_t)c.H * c.D * sizeof(__half)), "kv o");
        w4::gpu::cuda_check(
            cudaMalloc(&o16_d, (size_t)c.H * c.D * sizeof(__half)), "kv o16");
        w4::gpu::cuda_check(cudaMalloc(&k16_d, ne * sizeof(__half)), "kv k16");
        w4::gpu::cuda_check(cudaMalloc(&v16_d, ne * sizeof(__half)), "kv v16");
        w4::gpu::cuda_check(cudaMalloc(&ko_d, ne * sizeof(__half)), "kv ko");
        w4::gpu::cuda_check(cudaMalloc(&vo_d, ne * sizeof(__half)), "kv vo");
        w4::gpu::cuda_check(cudaMemcpy(qk_d, cc.q_k.data(),
                                       nq * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice),
                            "up kv qk");
        w4::gpu::cuda_check(cudaMemcpy(qv_d, cc.q_v.data(),
                                       nq * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice),
                            "up kv qv");
        w4::gpu::cuda_check(cudaMemcpy(sk_d, cc.s_k.data(),
                                       ns * sizeof(__half),
                                       cudaMemcpyHostToDevice),
                            "up kv sk");
        w4::gpu::cuda_check(cudaMemcpy(sv_d, cc.s_v.data(),
                                       ns * sizeof(__half),
                                       cudaMemcpyHostToDevice),
                            "up kv sv");
        if (!sym) {
          w4::gpu::cuda_check(cudaMemcpy(zk_d, cc.z_k.data(),
                                         nz * sizeof(uint32_t),
                                         cudaMemcpyHostToDevice),
                              "up kv zk");
          w4::gpu::cuda_check(cudaMemcpy(zv_d, cc.z_v.data(),
                                         nz * sizeof(uint32_t),
                                         cudaMemcpyHostToDevice),
                              "up kv zv");
        }
        w4::gpu::cuda_check(cudaMemcpy(q_d, q.data(),
                                       (size_t)c.H * c.D * sizeof(__half),
                                       cudaMemcpyHostToDevice),
                            "up kv q");

        // Two-pass: per-head dequant through the R1 kernel (itself bitwise
        // gated), then the FP16 attention kernel.
        for (int h = 0; h < c.H; ++h) {
          const w4::gpu::DevicePackedW dk{c.S, c.D, c.gs, cc.G, sym ? 1 : 0,
                                          qk_d + (size_t)h * c.S * wd,
                                          sk_d + (size_t)h * c.S * cc.G,
                                          zk_d ? zk_d + (size_t)h * c.S *
                                                     w4::words(cc.G)
                                               : nullptr};
          w4::gpu::dequant(dk, k16_d + (size_t)h * c.S * c.D);
          const w4::gpu::DevicePackedW dv{c.S, c.D, c.gs, cc.G, sym ? 1 : 0,
                                          qv_d + (size_t)h * c.S * wd,
                                          sv_d + (size_t)h * c.S * cc.G,
                                          zv_d ? zv_d + (size_t)h * c.S *
                                                     w4::words(cc.G)
                                               : nullptr};
          w4::gpu::dequant(dv, v16_d + (size_t)h * c.S * c.D);
        }
        w4::gpu::attention_fp16(q_d, k16_d, v16_d, o16_d, c.H, c.S, c.D);

        // Fused with the dequant hooks armed.
        const w4::gpu::DeviceKVW4 dp{c.H,    c.S,    c.D, c.gs, cc.G,
                                     sym ? 1 : 0,    qk_d,  sk_d, zk_d,
                                     qv_d,   sv_d,   zv_d};
        w4::gpu::attention_w4(dp, q_d, o_d, 0, ko_d, vo_d);
        w4::gpu::cuda_check(cudaDeviceSynchronize(), "kv fused hooks");

        // Bitwise: the hooks must reproduce the host dequant, and the
        // two-pass FP16 caches must match the same reference.
        std::vector<__half> buf(ne);
        bool k_ok = true, v_ok = true;
        for (int h = 0; h < c.H; ++h) {
          const auto kref = w4::kv::dequant_head(cc, h, false);
          const auto vref = w4::kv::dequant_head(cc, h, true);
          w4::gpu::cuda_check(cudaMemcpy(buf.data(),
                                         ko_d + (size_t)h * c.S * c.D,
                                         (size_t)c.S * c.D * sizeof(__half),
                                         cudaMemcpyDeviceToHost),
                              "down kv ko");
          if (std::memcmp(buf.data(), kref.data(),
                          (size_t)c.S * c.D * sizeof(__half)) != 0)
            k_ok = false;
          w4::gpu::cuda_check(cudaMemcpy(buf.data(),
                                         vo_d + (size_t)h * c.S * c.D,
                                         (size_t)c.S * c.D * sizeof(__half),
                                         cudaMemcpyDeviceToHost),
                              "down kv vo");
          if (std::memcmp(buf.data(), vref.data(),
                          (size_t)c.S * c.D * sizeof(__half)) != 0)
            v_ok = false;
        }
        ++checks;
        if (!k_ok || !v_ok) {
          ++fails;
          std::printf("FAIL kv dequant bitwise %s H=%d S=%d D=%d gs=%d\n",
                      sym ? "sym" : "asym", c.H, c.S, c.D, c.gs);
        }

        std::vector<__half> o_gpu((size_t)c.H * c.D);
        w4::gpu::cuda_check(cudaMemcpy(o_gpu.data(), o_d,
                                       (size_t)c.H * c.D * sizeof(__half),
                                       cudaMemcpyDeviceToHost),
                            "down kv o");
        std::vector<__half> o_16((size_t)c.H * c.D);
        w4::gpu::cuda_check(cudaMemcpy(o_16.data(), o16_d,
                                       (size_t)c.H * c.D * sizeof(__half),
                                       cudaMemcpyDeviceToHost),
                            "down kv o16");
        double num_r = 0.0, num_t = 0.0, den = 0.0;
        for (size_t j = 0; j < o_gpu.size(); ++j) {
          const double d = (double)__half2float(o_gpu[j]) - ref[j];
          num_r += d * d;
          const double dt =
              (double)__half2float(o_gpu[j]) - (double)__half2float(o_16[j]);
          num_t += dt * dt;
          den += ref[j] * ref[j];
        }
        const double rel_r = std::sqrt(num_r / (den > 0.0 ? den : 1.0));
        const double rel_t = std::sqrt(num_t / (den > 0.0 ? den : 1.0));
        worst_rel = std::max(worst_rel, rel_r);
        ++checks;
        if (rel_r > 1e-3) {
          ++fails;
          std::printf("FAIL kv oracle %s H=%d S=%d D=%d gs=%d rel=%.3e\n",
                      sym ? "sym" : "asym", c.H, c.S, c.D, c.gs, rel_r);
        }
        ++checks;
        if (rel_t > 2e-3) {
          ++fails;
          std::printf("FAIL kv two-pass %s H=%d S=%d D=%d gs=%d rel=%.3e\n",
                      sym ? "sym" : "asym", c.H, c.S, c.D, c.gs, rel_t);
        }

        cudaFree(qk_d);
        cudaFree(qv_d);
        cudaFree(zk_d);
        cudaFree(zv_d);
        cudaFree(sk_d);
        cudaFree(sv_d);
        cudaFree(q_d);
        cudaFree(o_d);
        cudaFree(o16_d);
        cudaFree(k16_d);
        cudaFree(v16_d);
        cudaFree(ko_d);
        cudaFree(vo_d);
      }
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
    // MMA entry contracts: same structural rules (padding covers shapes).
    expect(throws_invalid([&] { w4::gpu::gemm_mma(bad_gs0, out, out, 2); }),
           "mma gs=0");
    expect(throws_invalid([&] { w4::gpu::gemm_mma(bad_gs7, out, out, 2); }),
           "mma gs=7");
    expect(throws_invalid([&] { w4::gpu::gemm_mma(bad_g, out, out, 2); }),
           "mma G wrong");
    expect(throws_invalid([&] { w4::gpu::gemm_mma(bad_q, out, out, 2); }),
           "mma q null");
    expect(throws_invalid([&] { w4::gpu::gemm_mma(bad_z, out, out, 2); }),
           "mma asym zeros null");
    expect(throws_invalid([&] { w4::gpu::gemm_mma(sym_z, out, out, 2); }),
           "mma sym carries zeros");
    expect(throws_invalid([&] { w4::gpu::gemm_mma(good, out, out, 0); }),
           "mma M=0");
    expect(throws_invalid([&] { w4::gpu::gemm_mma(good, nullptr, out, 2); }),
           "mma x null");
    expect(throws_invalid([&] { w4::gpu::gemm_mma(good, out, nullptr, 2); }),
           "mma c null");

    // MX entry contracts: dims, B-vs-K consistency, null pointers, M, and
    // the 2^31 launch bound on the naive dequant.
    const w4::gpu::DevicePackedWm good_mx{2, 32, 1, (const uint32_t*)1,
                                          (const uint8_t*)1};
    auto bad_mxn = good_mx; bad_mxn.N = 0;
    auto bad_mxb = good_mx; bad_mxb.B = 2;
    auto bad_mxq = good_mx; bad_mxq.q = nullptr;
    auto bad_mxs = good_mx; bad_mxs.scales = nullptr;
    auto big_mx = good_mx; big_mx.N = 1 << 21; big_mx.K = 8192;
    big_mx.B = 256;
    expect(throws_invalid([&] { w4::gpu::dequant_mx(bad_mxn, out); }),
           "mx dequant N=0");
    expect(throws_invalid([&] { w4::gpu::dequant_mx(bad_mxb, out); }),
           "mx dequant B wrong");
    expect(throws_invalid([&] { w4::gpu::dequant_mx(bad_mxq, out); }),
           "mx dequant q null");
    expect(throws_invalid([&] { w4::gpu::dequant_mx(bad_mxs, out); }),
           "mx dequant scales null");
    expect(throws_invalid([&] { w4::gpu::dequant_mx(good_mx, nullptr); }),
           "mx dequant w null");
    expect(throws_invalid([&] { w4::gpu::dequant_mx(big_mx, out); }),
           "mx dequant total >= 2^31");
    expect(throws_invalid([&] { w4::gpu::gemm_fused_mx(bad_mxb, out, out, 2); }),
           "mx fused B wrong");
    expect(throws_invalid([&] { w4::gpu::gemm_fused_mx(good_mx, nullptr, out, 2); }),
           "mx fused x null");
    expect(throws_invalid([&] { w4::gpu::gemm_fused_mx(good_mx, out, nullptr, 2); }),
           "mx fused c null");
    expect(throws_invalid([&] { w4::gpu::gemm_fused_mx(good_mx, out, out, 0); }),
           "mx fused M=0");
    expect(throws_invalid([&] { w4::gpu::gemm_mma_mx(bad_mxb, out, out, 2); }),
           "mx mma B wrong");
    expect(throws_invalid([&] { w4::gpu::gemm_mma_mx(good_mx, nullptr, out, 2); }),
           "mx mma x null");
    expect(throws_invalid([&] { w4::gpu::gemm_mma_mx(good_mx, out, out, 0); }),
           "mx mma M=0");

    // KV attention entry contracts: dims, the S/D shared-memory bounds, gs,
    // G consistency, null pointers, zeros discipline.
    const w4::gpu::DeviceKVW4 good_kv{1, 4, 8, 8, 1, 1,
                                      (const uint32_t*)1, (const __half*)1,
                                      nullptr,
                                      (const uint32_t*)1, (const __half*)1,
                                      nullptr};
    auto kv_s0 = good_kv; kv_s0.S = 0;
    auto kv_big_s = good_kv; kv_big_s.S = 4097;
    auto kv_big_d = good_kv; kv_big_d.D = 513;
    auto kv_gs = good_kv; kv_gs.gs = 7;
    auto kv_g = good_kv; kv_g.G = 2;
    auto kv_q = good_kv; kv_q.q_k = nullptr;
    auto kv_z = good_kv; kv_z.symmetric = 0;
    auto kv_sz = good_kv; kv_sz.z_k = (const uint32_t*)1;
    expect(throws_invalid([&] { w4::gpu::attention_w4(kv_s0, out, out); }),
           "kv S=0");
    expect(throws_invalid([&] { w4::gpu::attention_w4(kv_big_s, out, out); }),
           "kv S=4097");
    expect(throws_invalid([&] { w4::gpu::attention_w4(kv_big_d, out, out); }),
           "kv D=513");
    expect(throws_invalid([&] { w4::gpu::attention_w4(kv_gs, out, out); }),
           "kv gs=7");
    expect(throws_invalid([&] { w4::gpu::attention_w4(kv_g, out, out); }),
           "kv G wrong");
    expect(throws_invalid([&] { w4::gpu::attention_w4(kv_q, out, out); }),
           "kv q null");
    expect(throws_invalid([&] { w4::gpu::attention_w4(kv_z, out, out); }),
           "kv asym zeros null");
    expect(throws_invalid([&] { w4::gpu::attention_w4(kv_sz, out, out); }),
           "kv sym carries zeros");
    expect(
        throws_invalid([&] { w4::gpu::attention_w4(good_kv, nullptr, out); }),
        "kv x null");
    expect(throws_invalid([&] { w4::gpu::attention_w4(good_kv, out, nullptr); }),
           "kv out null");
    expect(throws_invalid(
               [&] { w4::gpu::attention_fp16(out, out, out, out, 1, 5000, 8); }),
           "kv fp16 S bound");
    expect(throws_invalid(
               [&] { w4::gpu::attention_fp16(nullptr, out, out, out, 1, 4, 8); }),
           "kv fp16 q null");
  }

  std::printf("gpu: %d checks, %d failures, worst gemm rel L2 = %.3e\n",
              checks, fails, worst_rel);
  return fails ? 1 : 0;
}
