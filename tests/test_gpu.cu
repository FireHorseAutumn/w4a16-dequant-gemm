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
//   8. The SIMT/MMA dispatcher (gemm/gemm_mx): output must be bitwise
//      identical to the callee the crossover selects (SIMT below
//      DISPATCH_MMA_MIN_M, MMA from there), and the w_out hook must stay
//      bitwise through the dispatch; entry contracts throw on both sides
//      of the threshold.

#include <cstdio>
#include <cstring>
#include <vector>

#include "w4/dequant.cuh"
#include "w4/dispatch.cuh"
#include "w4/fused.cuh"
#include "w4/kvattention.cuh"
#include "w4/kvflash.cuh"
#include "w4/kvpaged.hpp"
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
      {65, 96, 100, 16},   // M > 64: multi-m-tile MMA grid, ragged K
      // The classes vLLM's exllama group walk gets wrong (group switch only
      // on 32-row steps off each 128-row block: vllm-project/vllm#60921) —
      // here every word's group is k0 / gs exactly, so these must pass.
      {1, 128, 512, 16},   // gs=16 across four 128-row blocks
      {4, 96, 192, 96},    // gs=96, K = 2*gs
      {4, 96, 320, 160},   // gs=160, K = 2*gs
      {4, 96, 384, 192},   // gs=192, K = 2*gs
      {4, 96, 96, 48},     // gs=48, K = 2*gs (not a 32-step multiple)
      {2, 200, 128, 32},   // N % 128 != 0 at 2 <= M <= 8 (vLLM Humming NaN)
      {8, 197, 96, 16},    // odd N: partial warp in the SIMT fused kernel
      {9, 64, 24, 8},      // M=9: the dispatch threshold; single-chunk MMA
      {9, 64, 33, 8},      // ragged K (K % 8 == 1): one-element second chunk
      {9, 64, 8, 8},       // one packed word per row
      {4, 64, 40, 64},     // single group, K % 32 != 0, gs > K
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
      const std::vector<__half> c_simt = c_fused;

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

      // Dispatcher: the w_out hook must stay bitwise through the dispatch,
      // and the output must memcmp against the callee the crossover selects
      // — SIMT below DISPATCH_MMA_MIN_M (c_simt), MMA from there (c_fused).
      w4::gpu::gemm(dp, x_d, c_f, c.M, 0, w_d);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "dispatch w_out sync");
      w4::gpu::cuda_check(cudaMemcpy(w_gpu.data(), w_d,
                                     (size_t)c.N * c.K * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down w dispatch");
      ++checks;
      if (std::memcmp(w_gpu.data(), w_ref.data(),
                      (size_t)c.N * c.K * sizeof(__half)) != 0) {
        ++fails;
        std::printf("FAIL dispatch dequant bitwise %s M=%d N=%d K=%d gs=%d\n",
                    sym ? "sym" : "asym", c.M, c.N, c.K, c.gs);
      }
      w4::gpu::gemm(dp, x_d, c_f, c.M);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "dispatch sync");
      std::vector<__half> c_disp((size_t)c.M * c.N);
      w4::gpu::cuda_check(cudaMemcpy(c_disp.data(), c_f,
                                     (size_t)c.M * c.N * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down c dispatch");
      const std::vector<__half>& c_winner =
          c.M < w4::gpu::DISPATCH_MMA_MIN_M ? c_simt : c_fused;
      ++checks;
      if (std::memcmp(c_disp.data(), c_winner.data(),
                      (size_t)c.M * c.N * sizeof(__half)) != 0) {
        ++fails;
        std::printf("FAIL dispatch routing %s M=%d N=%d K=%d gs=%d\n",
                    sym ? "sym" : "asym", c.M, c.N, c.K, c.gs);
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
    const std::vector<__half> c_simt_mx = c_fused;

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

    // MX dispatcher: same gates as the W4 dispatch — hook bitwise, output
    // memcmp against the callee the crossover selects.
    w4::gpu::gemm_mx(dp, x_d, c_f, c.M, 0, w_d);
    w4::gpu::cuda_check(cudaDeviceSynchronize(), "mx dispatch w_out sync");
    w4::gpu::cuda_check(cudaMemcpy(w_gpu.data(), w_d,
                                   (size_t)c.N * c.K * sizeof(__half),
                                   cudaMemcpyDeviceToHost),
                        "down mx w dispatch");
    ++checks;
    if (std::memcmp(w_gpu.data(), w_ref.data(),
                    (size_t)c.N * c.K * sizeof(__half)) != 0) {
      ++fails;
      std::printf("FAIL mx dispatch dequant bitwise M=%d N=%d K=%d\n", c.M,
                  c.N, c.K);
    }
    w4::gpu::gemm_mx(dp, x_d, c_f, c.M);
    w4::gpu::cuda_check(cudaDeviceSynchronize(), "mx dispatch sync");
    std::vector<__half> c_disp_mx((size_t)c.M * c.N);
    w4::gpu::cuda_check(cudaMemcpy(c_disp_mx.data(), c_f,
                                   (size_t)c.M * c.N * sizeof(__half),
                                   cudaMemcpyDeviceToHost),
                        "down mx c dispatch");
    const std::vector<__half>& c_winner_mx =
        c.M < w4::gpu::DISPATCH_MMA_MIN_M ? c_simt_mx : c_fused;
    ++checks;
    if (std::memcmp(c_disp_mx.data(), c_winner_mx.data(),
                    (size_t)c.M * c.N * sizeof(__half)) != 0) {
      ++fails;
      std::printf("FAIL mx dispatch routing M=%d N=%d K=%d\n", c.M, c.N,
                  c.K);
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
        {2, 33, 44, 8},      // D % 8 != 0: ragged packed words
        {2, 33, 44, 64},     // gs > D: single ragged group
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

        // Flash (split-S) variants, same gates: hooks bitwise vs
        // dequant_head, output vs the FP64 reference at 1e-3, vs the
        // single-block kernels at 2e-3 (tile reduction order differs;
        // both are gated paths).
        w4::gpu::attention_w4_flash(dp, q_d, o_d, 0, ko_d, vo_d);
        w4::gpu::cuda_check(cudaDeviceSynchronize(), "kv flash hooks");
        bool fk_ok = true, fv_ok = true;
        for (int h = 0; h < c.H; ++h) {
          const auto kref = w4::kv::dequant_head(cc, h, false);
          const auto vref = w4::kv::dequant_head(cc, h, true);
          w4::gpu::cuda_check(cudaMemcpy(buf.data(),
                                         ko_d + (size_t)h * c.S * c.D,
                                         (size_t)c.S * c.D * sizeof(__half),
                                         cudaMemcpyDeviceToHost),
                              "down kv fko");
          if (std::memcmp(buf.data(), kref.data(),
                          (size_t)c.S * c.D * sizeof(__half)) != 0)
            fk_ok = false;
          w4::gpu::cuda_check(cudaMemcpy(buf.data(),
                                         vo_d + (size_t)h * c.S * c.D,
                                         (size_t)c.S * c.D * sizeof(__half),
                                         cudaMemcpyDeviceToHost),
                              "down kv fvo");
          if (std::memcmp(buf.data(), vref.data(),
                          (size_t)c.S * c.D * sizeof(__half)) != 0)
            fv_ok = false;
        }
        ++checks;
        if (!fk_ok || !fv_ok) {
          ++fails;
          std::printf("FAIL kv flash dequant bitwise %s H=%d S=%d D=%d\n",
                      sym ? "sym" : "asym", c.H, c.S, c.D);
        }
        w4::gpu::attention_fp16_flash(q_d, k16_d, v16_d, o16_d, c.H, c.S,
                                      c.D);
        w4::gpu::cuda_check(cudaDeviceSynchronize(), "kv fp16 flash sync");
        std::vector<__half> o_fw4((size_t)c.H * c.D), o_f16((size_t)c.H * c.D);
        w4::gpu::cuda_check(cudaMemcpy(o_fw4.data(), o_d,
                                       (size_t)c.H * c.D * sizeof(__half),
                                       cudaMemcpyDeviceToHost),
                            "down kv fw4");
        w4::gpu::cuda_check(cudaMemcpy(o_f16.data(), o16_d,
                                       (size_t)c.H * c.D * sizeof(__half),
                                       cudaMemcpyDeviceToHost),
                            "down kv f16");
        double num_fw = 0.0, num_fc = 0.0, num_f6r = 0.0, num_f6c = 0.0;
        for (size_t j = 0; j < o_fw4.size(); ++j) {
          const double fw = (double)__half2float(o_fw4[j]);
          const double f6 = (double)__half2float(o_f16[j]);
          const double w4b = (double)__half2float(o_gpu[j]);
          const double f16b = (double)__half2float(o_16[j]);
          num_fw += (fw - ref[j]) * (fw - ref[j]);
          num_fc += (fw - w4b) * (fw - w4b);
          num_f6r += (f6 - ref[j]) * (f6 - ref[j]);
          num_f6c += (f6 - f16b) * (f6 - f16b);
        }
        const double rel_fw = std::sqrt(num_fw / (den > 0.0 ? den : 1.0));
        const double rel_fc = std::sqrt(num_fc / (den > 0.0 ? den : 1.0));
        const double rel_f6r = std::sqrt(num_f6r / (den > 0.0 ? den : 1.0));
        const double rel_f6c = std::sqrt(num_f6c / (den > 0.0 ? den : 1.0));
        ++checks;
        if (rel_fw > 1e-3) {
          ++fails;
          std::printf("FAIL kv flash oracle %s H=%d S=%d D=%d rel=%.3e\n",
                      sym ? "sym" : "asym", c.H, c.S, c.D, rel_fw);
        }
        ++checks;
        if (rel_fc > 2e-3) {
          ++fails;
          std::printf(
              "FAIL kv flash-vs-singleblock %s H=%d S=%d D=%d rel=%.3e\n",
              sym ? "sym" : "asym", c.H, c.S, c.D, rel_fc);
        }
        ++checks;
        if (rel_f6r > 1e-3) {
          ++fails;
          std::printf("FAIL fp16 flash oracle %s H=%d S=%d D=%d rel=%.3e\n",
                      sym ? "sym" : "asym", c.H, c.S, c.D, rel_f6r);
        }
        ++checks;
        if (rel_f6c > 2e-3) {
          ++fails;
          std::printf(
              "FAIL fp16 flash-vs-singleblock %s H=%d S=%d D=%d rel=%.3e\n",
              sym ? "sym" : "asym", c.H, c.S, c.D, rel_f6c);
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

  // Flash-only configs beyond the old single-block bound: S > 4096
  // exercises the tiling (and attention_w4 rejects these by contract, so
  // the FP64 reference gates the flash path directly).
  {
    struct KvCfg {
      int H, S, D, gs;
    };
    const std::vector<KvCfg> kcfgs = {{2, 5000, 64, 32}, {1, 9000, 128, 64}};
    for (const bool sym : {true, false}) {
      for (size_t i = 0; i < kcfgs.size(); ++i) {
        const KvCfg c = kcfgs[i];
        const uint64_t seed = 0x51D0ULL ^ (uint64_t)(i * 977 + (sym ? 0 : 1));
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
               *o_d = nullptr, *ko_d = nullptr, *vo_d = nullptr;
        w4::gpu::cuda_check(cudaMalloc(&qk_d, nq * sizeof(uint32_t)),
                            "fl qk");
        w4::gpu::cuda_check(cudaMalloc(&qv_d, nq * sizeof(uint32_t)),
                            "fl qv");
        w4::gpu::cuda_check(cudaMalloc(&sk_d, ns * sizeof(__half)), "fl sk");
        w4::gpu::cuda_check(cudaMalloc(&sv_d, ns * sizeof(__half)), "fl sv");
        if (!sym) {
          w4::gpu::cuda_check(cudaMalloc(&zk_d, nz * sizeof(uint32_t)),
                              "fl zk");
          w4::gpu::cuda_check(cudaMalloc(&zv_d, nz * sizeof(uint32_t)),
                              "fl zv");
        }
        w4::gpu::cuda_check(
            cudaMalloc(&q_d, (size_t)c.H * c.D * sizeof(__half)), "fl q");
        w4::gpu::cuda_check(
            cudaMalloc(&o_d, (size_t)c.H * c.D * sizeof(__half)), "fl o");
        w4::gpu::cuda_check(cudaMalloc(&ko_d, ne * sizeof(__half)), "fl ko");
        w4::gpu::cuda_check(cudaMalloc(&vo_d, ne * sizeof(__half)), "fl vo");
        w4::gpu::cuda_check(cudaMemcpy(qk_d, cc.q_k.data(),
                                       nq * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice),
                            "up fl qk");
        w4::gpu::cuda_check(cudaMemcpy(qv_d, cc.q_v.data(),
                                       nq * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice),
                            "up fl qv");
        w4::gpu::cuda_check(cudaMemcpy(sk_d, cc.s_k.data(),
                                       ns * sizeof(__half),
                                       cudaMemcpyHostToDevice),
                            "up fl sk");
        w4::gpu::cuda_check(cudaMemcpy(sv_d, cc.s_v.data(),
                                       ns * sizeof(__half),
                                       cudaMemcpyHostToDevice),
                            "up fl sv");
        if (!sym) {
          w4::gpu::cuda_check(cudaMemcpy(zk_d, cc.z_k.data(),
                                         nz * sizeof(uint32_t),
                                         cudaMemcpyHostToDevice),
                              "up fl zk");
          w4::gpu::cuda_check(cudaMemcpy(zv_d, cc.z_v.data(),
                                         nz * sizeof(uint32_t),
                                         cudaMemcpyHostToDevice),
                              "up fl zv");
        }
        w4::gpu::cuda_check(cudaMemcpy(q_d, q.data(),
                                       (size_t)c.H * c.D * sizeof(__half),
                                       cudaMemcpyHostToDevice),
                            "up fl q");

        const w4::gpu::DeviceKVW4 dp{c.H,    c.S,    c.D, c.gs, cc.G,
                                     sym ? 1 : 0,    qk_d,  sk_d, zk_d,
                                     qv_d,   sv_d,   zv_d};
        w4::gpu::attention_w4_flash(dp, q_d, o_d, 0, ko_d, vo_d);
        w4::gpu::cuda_check(cudaDeviceSynchronize(), "fl hooks sync");
        std::vector<__half> buf(ne);
        bool k_ok = true, v_ok = true;
        for (int h = 0; h < c.H; ++h) {
          const auto kref = w4::kv::dequant_head(cc, h, false);
          const auto vref = w4::kv::dequant_head(cc, h, true);
          w4::gpu::cuda_check(cudaMemcpy(buf.data(),
                                         ko_d + (size_t)h * c.S * c.D,
                                         (size_t)c.S * c.D * sizeof(__half),
                                         cudaMemcpyDeviceToHost),
                              "down fl ko");
          if (std::memcmp(buf.data(), kref.data(),
                          (size_t)c.S * c.D * sizeof(__half)) != 0)
            k_ok = false;
          w4::gpu::cuda_check(cudaMemcpy(buf.data(),
                                         vo_d + (size_t)h * c.S * c.D,
                                         (size_t)c.S * c.D * sizeof(__half),
                                         cudaMemcpyDeviceToHost),
                              "down fl vo");
          if (std::memcmp(buf.data(), vref.data(),
                          (size_t)c.S * c.D * sizeof(__half)) != 0)
            v_ok = false;
        }
        ++checks;
        if (!k_ok || !v_ok) {
          ++fails;
          std::printf("FAIL flash big-S dequant bitwise %s H=%d S=%d D=%d\n",
                      sym ? "sym" : "asym", c.H, c.S, c.D);
        }
        std::vector<__half> o_gpu((size_t)c.H * c.D);
        w4::gpu::cuda_check(cudaMemcpy(o_gpu.data(), o_d,
                                       (size_t)c.H * c.D * sizeof(__half),
                                       cudaMemcpyDeviceToHost),
                            "down fl o");
        double num = 0.0, den = 0.0;
        for (size_t j = 0; j < o_gpu.size(); ++j) {
          const double d = (double)__half2float(o_gpu[j]) - ref[j];
          num += d * d;
          den += ref[j] * ref[j];
        }
        const double rel = std::sqrt(num / (den > 0.0 ? den : 1.0));
        worst_rel = std::max(worst_rel, rel);
        ++checks;
        if (rel > 1e-3) {
          ++fails;
          std::printf("FAIL flash big-S oracle %s H=%d S=%d D=%d rel=%.3e\n",
                      sym ? "sym" : "asym", c.H, c.S, c.D, rel);
        }
        cudaFree(qk_d);
        cudaFree(qv_d);
        cudaFree(zk_d);
        cudaFree(zv_d);
        cudaFree(sk_d);
        cudaFree(sv_d);
        cudaFree(q_d);
        cudaFree(o_d);
        cudaFree(ko_d);
        cudaFree(vo_d);
      }
    }
  }

  // FP8 (E4M3) KV cache, flash path (R6): hooks bitwise vs
  // dequant_head_f8, output vs the FP64 reference at 1e-3, vs the device
  // two-pass (per-head dequant_f8 + FP16 flash attention) at 2e-3. The
  // S=9000 config is beyond the old single-block bound on purpose.
  {
    struct KvCfg {
      int H, S, D, gs;
    };
    const std::vector<KvCfg> kcfgs = {
        {1, 1, 64, 32},    {2, 7, 64, 8},   {4, 128, 128, 128},
        {2, 300, 96, 32},  {3, 65, 40, 8},  {1, 9000, 128, 64},
        {2, 33, 44, 8},    // D % 8 != 0 (word walk is 4-wide, D % 4 == 0)
    };
    for (size_t i = 0; i < kcfgs.size(); ++i) {
      const KvCfg c = kcfgs[i];
      const uint64_t seed = 0xF80AULL ^ (uint64_t)(i * 977);
      const auto k_src = w4::normal_weights(c.H * c.S * c.D, seed);
      const auto v_src = w4::normal_weights(c.H * c.S * c.D, seed + 1);
      const auto cc =
          w4::kv8::quantize_cache_f8(k_src, v_src, c.H, c.S, c.D, c.gs);
      const auto q = w4::normal_activations(c.H * c.D, seed + 2);
      const auto ref = w4::kv8::attention_reference_f8(q, cc);

      const size_t ne = (size_t)c.H * c.S * c.D;
      const size_t ns = (size_t)c.H * c.S * cc.G;
      uint8_t *qk_d = nullptr, *qv_d = nullptr;
      __half *sk_d = nullptr, *sv_d = nullptr, *q_d = nullptr,
             *o_d = nullptr, *o16_d = nullptr, *k16_d = nullptr,
             *v16_d = nullptr, *ko_d = nullptr, *vo_d = nullptr;
      w4::gpu::cuda_check(cudaMalloc(&qk_d, ne), "f8 qk");
      w4::gpu::cuda_check(cudaMalloc(&qv_d, ne), "f8 qv");
      w4::gpu::cuda_check(cudaMalloc(&sk_d, ns * sizeof(__half)), "f8 sk");
      w4::gpu::cuda_check(cudaMalloc(&sv_d, ns * sizeof(__half)), "f8 sv");
      w4::gpu::cuda_check(
          cudaMalloc(&q_d, (size_t)c.H * c.D * sizeof(__half)), "f8 q");
      w4::gpu::cuda_check(
          cudaMalloc(&o_d, (size_t)c.H * c.D * sizeof(__half)), "f8 o");
      w4::gpu::cuda_check(
          cudaMalloc(&o16_d, (size_t)c.H * c.D * sizeof(__half)), "f8 o16");
      w4::gpu::cuda_check(cudaMalloc(&k16_d, ne * sizeof(__half)), "f8 k16");
      w4::gpu::cuda_check(cudaMalloc(&v16_d, ne * sizeof(__half)), "f8 v16");
      w4::gpu::cuda_check(cudaMalloc(&ko_d, ne * sizeof(__half)), "f8 ko");
      w4::gpu::cuda_check(cudaMalloc(&vo_d, ne * sizeof(__half)), "f8 vo");
      w4::gpu::cuda_check(cudaMemcpy(qk_d, cc.q_k.data(), ne,
                                     cudaMemcpyHostToDevice),
                          "up f8 qk");
      w4::gpu::cuda_check(cudaMemcpy(qv_d, cc.q_v.data(), ne,
                                     cudaMemcpyHostToDevice),
                          "up f8 qv");
      w4::gpu::cuda_check(cudaMemcpy(sk_d, cc.s_k.data(),
                                     ns * sizeof(__half),
                                     cudaMemcpyHostToDevice),
                          "up f8 sk");
      w4::gpu::cuda_check(cudaMemcpy(sv_d, cc.s_v.data(),
                                     ns * sizeof(__half),
                                     cudaMemcpyHostToDevice),
                          "up f8 sv");
      w4::gpu::cuda_check(cudaMemcpy(q_d, q.data(),
                                     (size_t)c.H * c.D * sizeof(__half),
                                     cudaMemcpyHostToDevice),
                          "up f8 q");

      const w4::gpu::DeviceKVF8 dp{c.H, c.S,  c.D,  c.gs, cc.G,
                                   qk_d, sk_d, qv_d, sv_d};
      w4::gpu::attention_f8_flash(dp, q_d, o_d, 0, ko_d, vo_d);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "f8 hooks sync");
      std::vector<__half> buf(ne);
      bool k_ok = true, v_ok = true;
      for (int h = 0; h < c.H; ++h) {
        const auto kref = w4::kv8::dequant_head_f8(cc, h, false);
        const auto vref = w4::kv8::dequant_head_f8(cc, h, true);
        w4::gpu::cuda_check(cudaMemcpy(buf.data(),
                                       ko_d + (size_t)h * c.S * c.D,
                                       (size_t)c.S * c.D * sizeof(__half),
                                       cudaMemcpyDeviceToHost),
                            "down f8 ko");
        if (std::memcmp(buf.data(), kref.data(),
                        (size_t)c.S * c.D * sizeof(__half)) != 0)
          k_ok = false;
        w4::gpu::cuda_check(cudaMemcpy(buf.data(),
                                       vo_d + (size_t)h * c.S * c.D,
                                       (size_t)c.S * c.D * sizeof(__half),
                                       cudaMemcpyDeviceToHost),
                            "down f8 vo");
        if (std::memcmp(buf.data(), vref.data(),
                        (size_t)c.S * c.D * sizeof(__half)) != 0)
          v_ok = false;
      }
      ++checks;
      if (!k_ok || !v_ok) {
        ++fails;
        std::printf("FAIL f8 dequant bitwise H=%d S=%d D=%d gs=%d\n", c.H,
                    c.S, c.D, c.gs);
      }

      // Two-pass: per-head FP8 dequant, then the FP16 flash kernel.
      for (int h = 0; h < c.H; ++h) {
        w4::gpu::dequant_f8(qk_d + (size_t)h * c.S * c.D,
                            sk_d + (size_t)h * c.S * cc.G,
                            k16_d + (size_t)h * c.S * c.D, c.S, c.D, c.gs);
        w4::gpu::dequant_f8(qv_d + (size_t)h * c.S * c.D,
                            sv_d + (size_t)h * c.S * cc.G,
                            v16_d + (size_t)h * c.S * c.D, c.S, c.D, c.gs);
      }
      w4::gpu::attention_fp16_flash(q_d, k16_d, v16_d, o16_d, c.H, c.S, c.D);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "f8 twopass sync");

      std::vector<__half> o_gpu((size_t)c.H * c.D), o_2p((size_t)c.H * c.D);
      w4::gpu::cuda_check(cudaMemcpy(o_gpu.data(), o_d,
                                     (size_t)c.H * c.D * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down f8 o");
      w4::gpu::cuda_check(cudaMemcpy(o_2p.data(), o16_d,
                                     (size_t)c.H * c.D * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down f8 o2p");
      double num_r = 0.0, num_t = 0.0, den = 0.0;
      for (size_t j = 0; j < o_gpu.size(); ++j) {
        const double d = (double)__half2float(o_gpu[j]) - ref[j];
        num_r += d * d;
        const double dt =
            (double)__half2float(o_gpu[j]) - (double)__half2float(o_2p[j]);
        num_t += dt * dt;
        den += ref[j] * ref[j];
      }
      const double rel_r = std::sqrt(num_r / (den > 0.0 ? den : 1.0));
      const double rel_t = std::sqrt(num_t / (den > 0.0 ? den : 1.0));
      worst_rel = std::max(worst_rel, rel_r);
      ++checks;
      if (rel_r > 1e-3) {
        ++fails;
        std::printf("FAIL f8 oracle H=%d S=%d D=%d gs=%d rel=%.3e\n", c.H,
                    c.S, c.D, c.gs, rel_r);
      }
      ++checks;
      if (rel_t > 2e-3) {
        ++fails;
        std::printf("FAIL f8 two-pass H=%d S=%d D=%d gs=%d rel=%.3e\n", c.H,
                    c.S, c.D, c.gs, rel_t);
      }

      cudaFree(qk_d);
      cudaFree(qv_d);
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

  // Paged flash (R7): positions resolve through the block table. Gates:
  // paged output bitwise vs the contiguous flash kernel on the same
  // logical cache (deterministic kernels, identical logical bytes), hooks
  // bitwise vs dequant_head in logical order, plus a prefix-shared pool
  // exhibit below.
  {
    struct PCfg {
      int H, S, D, gs, page;
    };
    const std::vector<PCfg> pcfgs = {
        {2, 300, 96, 32, 16},   // ragged last page
        {2, 520, 64, 32, 16},   {1, 1000, 128, 64, 64},
        {3, 65, 40, 8, 32},     // page < gs is fine: groups partition D
        {1, 5000, 64, 32, 256}, // past the old single-block bound
    };
    for (const bool sym : {true, false}) {
      for (size_t i = 0; i < pcfgs.size(); ++i) {
        const PCfg c = pcfgs[i];
        const uint64_t seed = 0x9A6EULL ^ (uint64_t)(i * 977 + (sym ? 0 : 1));
        const auto k_src = w4::normal_weights(c.H * c.S * c.D, seed);
        const auto v_src = w4::normal_weights(c.H * c.S * c.D, seed + 1);
        const auto cc =
            w4::kv::quantize_cache(k_src, v_src, c.H, c.S, c.D, c.gs, sym);
        const auto q = w4::normal_activations(c.H * c.D, seed + 2);

        const int npages = (c.S + c.page - 1) / c.page;
        std::vector<int> bt(npages);
        std::iota(bt.begin(), bt.end(), 0);
        std::mt19937_64 shuf(seed ^ 0x5EDULL);
        std::shuffle(bt.begin(), bt.end(), shuf);  // worst case: shuffled
        const auto pc = w4::kv::paged_from_contiguous(cc, c.page, bt);

        const int wd = w4::words(c.D);
        const size_t pool_q = (size_t)c.H * npages * c.page * wd;
        const size_t pool_s = (size_t)c.H * npages * c.page * cc.G;
        const size_t pool_z =
            (size_t)c.H * npages * c.page * w4::words(cc.G);
        const size_t ne = (size_t)c.H * c.S * c.D;
        uint32_t *qk_d = nullptr, *qv_d = nullptr, *zk_d = nullptr,
                 *zv_d = nullptr;
        __half *sk_d = nullptr, *sv_d = nullptr, *q_d = nullptr,
               *o_p = nullptr, *o_c = nullptr, *ko_d = nullptr,
               *vo_d = nullptr;
        int* bt_d = nullptr;
        w4::gpu::cuda_check(cudaMalloc(&qk_d, pool_q * sizeof(uint32_t)),
                            "pg qk");
        w4::gpu::cuda_check(cudaMalloc(&qv_d, pool_q * sizeof(uint32_t)),
                            "pg qv");
        w4::gpu::cuda_check(cudaMalloc(&sk_d, pool_s * sizeof(__half)),
                            "pg sk");
        w4::gpu::cuda_check(cudaMalloc(&sv_d, pool_s * sizeof(__half)),
                            "pg sv");
        if (!sym) {
          w4::gpu::cuda_check(cudaMalloc(&zk_d, pool_z * sizeof(uint32_t)),
                              "pg zk");
          w4::gpu::cuda_check(cudaMalloc(&zv_d, pool_z * sizeof(uint32_t)),
                              "pg zv");
        }
        w4::gpu::cuda_check(
            cudaMalloc(&q_d, (size_t)c.H * c.D * sizeof(__half)), "pg q");
        w4::gpu::cuda_check(cudaMalloc(&o_p, (size_t)c.H * c.D * sizeof(__half)),
                            "pg op");
        w4::gpu::cuda_check(cudaMalloc(&o_c, (size_t)c.H * c.D * sizeof(__half)),
                            "pg oc");
        w4::gpu::cuda_check(cudaMalloc(&ko_d, ne * sizeof(__half)), "pg ko");
        w4::gpu::cuda_check(cudaMalloc(&vo_d, ne * sizeof(__half)), "pg vo");
        w4::gpu::cuda_check(
            cudaMalloc(&bt_d, npages * sizeof(int)), "pg bt");
        w4::gpu::cuda_check(cudaMemcpy(qk_d, pc.q_k.data(),
                                       pool_q * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice),
                            "up pg qk");
        w4::gpu::cuda_check(cudaMemcpy(qv_d, pc.q_v.data(),
                                       pool_q * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice),
                            "up pg qv");
        w4::gpu::cuda_check(cudaMemcpy(sk_d, pc.s_k.data(),
                                       pool_s * sizeof(__half),
                                       cudaMemcpyHostToDevice),
                            "up pg sk");
        w4::gpu::cuda_check(cudaMemcpy(sv_d, pc.s_v.data(),
                                       pool_s * sizeof(__half),
                                       cudaMemcpyHostToDevice),
                            "up pg sv");
        if (!sym) {
          w4::gpu::cuda_check(cudaMemcpy(zk_d, pc.z_k.data(),
                                         pool_z * sizeof(uint32_t),
                                         cudaMemcpyHostToDevice),
                              "up pg zk");
          w4::gpu::cuda_check(cudaMemcpy(zv_d, pc.z_v.data(),
                                         pool_z * sizeof(uint32_t),
                                         cudaMemcpyHostToDevice),
                              "up pg zv");
        }
        w4::gpu::cuda_check(cudaMemcpy(q_d, q.data(),
                                       (size_t)c.H * c.D * sizeof(__half),
                                       cudaMemcpyHostToDevice),
                            "up pg q");
        w4::gpu::cuda_check(cudaMemcpy(bt_d, bt.data(),
                                       npages * sizeof(int),
                                       cudaMemcpyHostToDevice),
                            "up pg bt");

        const w4::gpu::DeviceKVW4 dpp{c.H,    c.S,    c.D, c.gs, cc.G,
                                      sym ? 1 : 0,    qk_d,  sk_d, zk_d,
                                      qv_d,   sv_d,   zv_d};
        w4::gpu::attention_w4_flash_paged(dpp, bt_d, c.page, npages, q_d,
                                          o_p, 0, ko_d, vo_d);
        w4::gpu::cuda_check(cudaDeviceSynchronize(), "pg sync");

        // Contiguous flash reference on the same logical cache. The
        // contiguous pool is a prefix view of the packed rows: the pool
        // holds the same logical content, but through the shuffled table,
        // so build a compact contiguous upload from the host cache.
        {
          const size_t nq = (size_t)c.H * c.S * wd;
          const size_t ns = (size_t)c.H * c.S * cc.G;
          const size_t nz = (size_t)c.H * c.S * w4::words(cc.G);
          uint32_t *cq_d = nullptr, *cv_d = nullptr, *cz_d = nullptr;
          __half *cs_d = nullptr, *cw_d = nullptr;
          w4::gpu::cuda_check(cudaMalloc(&cq_d, nq * sizeof(uint32_t)), "cq");
          w4::gpu::cuda_check(cudaMalloc(&cv_d, nq * sizeof(uint32_t)), "cv");
          w4::gpu::cuda_check(cudaMalloc(&cs_d, ns * sizeof(__half)), "cs");
          w4::gpu::cuda_check(cudaMalloc(&cw_d, ns * sizeof(__half)), "cw");
          if (!sym)
            w4::gpu::cuda_check(cudaMalloc(&cz_d, nz * sizeof(uint32_t)),
                                "cz");
          w4::gpu::cuda_check(cudaMemcpy(cq_d, cc.q_k.data(),
                                         nq * sizeof(uint32_t),
                                         cudaMemcpyHostToDevice),
                              "up cq");
          w4::gpu::cuda_check(cudaMemcpy(cv_d, cc.q_v.data(),
                                         nq * sizeof(uint32_t),
                                         cudaMemcpyHostToDevice),
                              "up cv");
          w4::gpu::cuda_check(cudaMemcpy(cs_d, cc.s_k.data(),
                                         ns * sizeof(__half),
                                         cudaMemcpyHostToDevice),
                              "up cs");
          w4::gpu::cuda_check(cudaMemcpy(cw_d, cc.s_v.data(),
                                         ns * sizeof(__half),
                                         cudaMemcpyHostToDevice),
                              "up cw");
          uint32_t* cz2 = nullptr;
          if (!sym) {
            w4::gpu::cuda_check(cudaMalloc(&cz2, nz * sizeof(uint32_t)),
                                "cz2");
            w4::gpu::cuda_check(cudaMemcpy(cz2, cc.z_v.data(),
                                           nz * sizeof(uint32_t),
                                           cudaMemcpyHostToDevice),
                                "up cz2");
            w4::gpu::cuda_check(cudaMemcpy(cz_d, cc.z_k.data(),
                                           nz * sizeof(uint32_t),
                                           cudaMemcpyHostToDevice),
                                "up cz");
          }
          const w4::gpu::DeviceKVW4 dpc{c.H,    c.S,   c.D, c.gs, cc.G,
                                        sym ? 1 : 0,   cq_d, cs_d, cz_d,
                                        cv_d,   cw_d,  cz2};
          w4::gpu::attention_w4_flash(dpc, q_d, o_c);
          w4::gpu::cuda_check(cudaDeviceSynchronize(), "pg contig sync");
          cudaFree(cq_d);
          cudaFree(cv_d);
          cudaFree(cz_d);
          cudaFree(cs_d);
          cudaFree(cw_d);
          cudaFree(cz2);
        }

        std::vector<__half> hp((size_t)c.H * c.D), hc((size_t)c.H * c.D);
        w4::gpu::cuda_check(cudaMemcpy(hp.data(), o_p,
                                       (size_t)c.H * c.D * sizeof(__half),
                                       cudaMemcpyDeviceToHost),
                            "down pg p");
        w4::gpu::cuda_check(cudaMemcpy(hc.data(), o_c,
                                       (size_t)c.H * c.D * sizeof(__half),
                                       cudaMemcpyDeviceToHost),
                            "down pg c");
        ++checks;
        if (std::memcmp(hp.data(), hc.data(),
                        (size_t)c.H * c.D * sizeof(__half)) != 0) {
          ++fails;
          std::printf(
              "FAIL paged-vs-contiguous bitwise %s H=%d S=%d D=%d page=%d\n",
              sym ? "sym" : "asym", c.H, c.S, c.D, c.page);
        }
        // Hooks: logical-order dequant must reproduce dequant_head bitwise.
        std::vector<__half> buf(ne);
        bool k_ok = true, v_ok = true;
        for (int h = 0; h < c.H; ++h) {
          const auto kref = w4::kv::dequant_head(cc, h, false);
          const auto vref = w4::kv::dequant_head(cc, h, true);
          w4::gpu::cuda_check(cudaMemcpy(buf.data(),
                                         ko_d + (size_t)h * c.S * c.D,
                                         (size_t)c.S * c.D * sizeof(__half),
                                         cudaMemcpyDeviceToHost),
                              "down pg ko");
          if (std::memcmp(buf.data(), kref.data(),
                          (size_t)c.S * c.D * sizeof(__half)) != 0)
            k_ok = false;
          w4::gpu::cuda_check(cudaMemcpy(buf.data(),
                                         vo_d + (size_t)h * c.S * c.D,
                                         (size_t)c.S * c.D * sizeof(__half),
                                         cudaMemcpyDeviceToHost),
                              "down pg vo");
          if (std::memcmp(buf.data(), vref.data(),
                          (size_t)c.S * c.D * sizeof(__half)) != 0)
            v_ok = false;
        }
        ++checks;
        if (!k_ok || !v_ok) {
          ++fails;
          std::printf("FAIL paged dequant bitwise %s H=%d S=%d D=%d\n",
                      sym ? "sym" : "asym", c.H, c.S, c.D);
        }

        cudaFree(qk_d);
        cudaFree(qv_d);
        cudaFree(zk_d);
        cudaFree(zv_d);
        cudaFree(sk_d);
        cudaFree(sv_d);
        cudaFree(q_d);
        cudaFree(o_p);
        cudaFree(o_c);
        cudaFree(ko_d);
        cudaFree(vo_d);
        cudaFree(bt_d);
      }
    }
  }

  // Prefix sharing: one physical pool holds sequence B (S2 positions) and
  // its own prefix A (S1 < S2) — A's block table is a prefix of B's. Both
  // decode correctly out of the single shared pool, bitwise vs their
  // contiguous references.
  {
    const int H = 2, D = 64, gs = 32, page = 16, S1 = 200, S2 = 520;
    for (const bool sym : {true, false}) {
      const auto k_src = w4::normal_weights(H * S2 * D, 0xBEEF);
      const auto v_src = w4::normal_weights(H * S2 * D, 0xBEE0);
      const auto ccB = w4::kv::quantize_cache(k_src, v_src, H, S2, D, gs, sym);
      const int npages = (S2 + page - 1) / page;  // 33
      std::vector<int> btB(npages);
      std::iota(btB.begin(), btB.end(), 0);
      std::mt19937_64 shuf(0xCAFE);
      std::shuffle(btB.begin(), btB.end(), shuf);
      const auto pc = w4::kv::paged_from_contiguous(ccB, page, btB);
      // A's view: the first ceil(S1/page) entries of B's table.
      const int na = (S1 + page - 1) / page;  // 13
      const std::vector<int> btA(btB.begin(), btB.begin() + na);
      // A's contiguous cache: per-head prefix slices (the source layout is
      // head-major, so a flat prefix slice would be the wrong rows).
      std::vector<double> kA((size_t)H * S1 * D), vA((size_t)H * S1 * D);
      for (int h = 0; h < H; ++h) {
        std::copy_n(k_src.begin() + (size_t)h * S2 * D, (size_t)S1 * D,
                    kA.begin() + (size_t)h * S1 * D);
        std::copy_n(v_src.begin() + (size_t)h * S2 * D, (size_t)S1 * D,
                    vA.begin() + (size_t)h * S1 * D);
      }
      const auto ccA = w4::kv::quantize_cache(kA, vA, H, S1, D, gs, sym);
      const auto q = w4::normal_activations(H * D, 0xFEED);

      const int wd = w4::words(D);
      const size_t pool_q = (size_t)H * npages * page * wd;
      const size_t pool_s = (size_t)H * npages * page * ccB.G;
      const size_t pool_z = (size_t)H * npages * page * w4::words(ccB.G);
      uint32_t *qk_d = nullptr, *qv_d = nullptr, *zk_d = nullptr,
               *zv_d = nullptr;
      __half *sk_d = nullptr, *sv_d = nullptr, *q_d = nullptr, *o_d = nullptr;
      int *btB_d = nullptr, *btA_d = nullptr;
      w4::gpu::cuda_check(cudaMalloc(&qk_d, pool_q * sizeof(uint32_t)), "pf qk");
      w4::gpu::cuda_check(cudaMalloc(&qv_d, pool_q * sizeof(uint32_t)), "pf qv");
      w4::gpu::cuda_check(cudaMalloc(&sk_d, pool_s * sizeof(__half)), "pf sk");
      w4::gpu::cuda_check(cudaMalloc(&sv_d, pool_s * sizeof(__half)), "pf sv");
      if (!sym) {
        w4::gpu::cuda_check(cudaMalloc(&zk_d, pool_z * sizeof(uint32_t)),
                            "pf zk");
        w4::gpu::cuda_check(cudaMalloc(&zv_d, pool_z * sizeof(uint32_t)),
                            "pf zv");
        w4::gpu::cuda_check(cudaMemcpy(zk_d, pc.z_k.data(),
                                       pool_z * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice),
                            "up pf zk");
        w4::gpu::cuda_check(cudaMemcpy(zv_d, pc.z_v.data(),
                                       pool_z * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice),
                            "up pf zv");
      }
      w4::gpu::cuda_check(
          cudaMalloc(&q_d, (size_t)H * D * sizeof(__half)), "pf q");
      w4::gpu::cuda_check(
          cudaMalloc(&o_d, (size_t)H * D * sizeof(__half)), "pf o");
      w4::gpu::cuda_check(cudaMalloc(&btB_d, npages * sizeof(int)), "pf btB");
      w4::gpu::cuda_check(cudaMalloc(&btA_d, na * sizeof(int)), "pf btA");
      w4::gpu::cuda_check(cudaMemcpy(qk_d, pc.q_k.data(),
                                     pool_q * sizeof(uint32_t),
                                     cudaMemcpyHostToDevice),
                          "up pf qk");
      w4::gpu::cuda_check(cudaMemcpy(qv_d, pc.q_v.data(),
                                     pool_q * sizeof(uint32_t),
                                     cudaMemcpyHostToDevice),
                          "up pf qv");
      w4::gpu::cuda_check(cudaMemcpy(sk_d, pc.s_k.data(),
                                     pool_s * sizeof(__half),
                                     cudaMemcpyHostToDevice),
                          "up pf sk");
      w4::gpu::cuda_check(cudaMemcpy(sv_d, pc.s_v.data(),
                                     pool_s * sizeof(__half),
                                     cudaMemcpyHostToDevice),
                          "up pf sv");
      w4::gpu::cuda_check(cudaMemcpy(q_d, q.data(),
                                     (size_t)H * D * sizeof(__half),
                                     cudaMemcpyHostToDevice),
                          "up pf q");
      w4::gpu::cuda_check(cudaMemcpy(btB_d, btB.data(),
                                     npages * sizeof(int),
                                     cudaMemcpyHostToDevice),
                          "up pf btB");
      w4::gpu::cuda_check(cudaMemcpy(btA_d, btA.data(), na * sizeof(int),
                                     cudaMemcpyHostToDevice),
                          "up pf btA");

      const w4::gpu::DeviceKVW4 pool{H,    S2,    D, gs, ccB.G,
                                     sym ? 1 : 0,  qk_d,  sk_d, zk_d,
                                     qv_d,  sv_d,  zv_d};
      // B decodes from the pool.
      w4::gpu::attention_w4_flash_paged(pool, btB_d, page, npages, q_d, o_d);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "pf B sync");
      std::vector<__half> oB((size_t)H * D);
      w4::gpu::cuda_check(cudaMemcpy(oB.data(), o_d,
                                     (size_t)H * D * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down pf B");
      // Contiguous B reference through the FP64 host reference.
      const auto refB = w4::kv::attention_reference(q, ccB);
      double numB = 0.0, denB = 0.0;
      for (size_t j = 0; j < oB.size(); ++j) {
        const double d = (double)__half2float(oB[j]) - refB[j];
        numB += d * d;
        denB += refB[j] * refB[j];
      }
      const double relB = std::sqrt(numB / (denB > 0.0 ? denB : 1.0));
      ++checks;
      if (relB > 1e-3) {
        ++fails;
        std::printf("FAIL prefix B oracle %s rel=%.3e\n", sym ? "sym" : "asym",
                    relB);
      }
      // A decodes from the same pool via the prefix table.
      w4::gpu::DeviceKVW4 poolA = pool;
      poolA.S = S1;
      w4::gpu::attention_w4_flash_paged(poolA, btA_d, page, npages, q_d, o_d);
      w4::gpu::cuda_check(cudaDeviceSynchronize(), "pf A sync");
      std::vector<__half> oA((size_t)H * D);
      w4::gpu::cuda_check(cudaMemcpy(oA.data(), o_d,
                                     (size_t)H * D * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          "down pf A");
      const auto refA = w4::kv::attention_reference(q, ccA);
      double numA = 0.0, denA = 0.0;
      for (size_t j = 0; j < oA.size(); ++j) {
        const double d = (double)__half2float(oA[j]) - refA[j];
        numA += d * d;
        denA += refA[j] * refA[j];
      }
      const double relA = std::sqrt(numA / (denA > 0.0 ? denA : 1.0));
      ++checks;
      if (relA > 1e-3) {
        ++fails;
        std::printf("FAIL prefix A oracle %s rel=%.3e\n", sym ? "sym" : "asym",
                    relA);
      }
      // The share is structural: the pool holds B's ceil(S2/page) pages and
      // A added none — its 200 positions are read out of B's prefix pages.

      cudaFree(qk_d);
      cudaFree(qv_d);
      cudaFree(zk_d);
      cudaFree(zv_d);
      cudaFree(sk_d);
      cudaFree(sv_d);
      cudaFree(q_d);
      cudaFree(o_d);
      cudaFree(btB_d);
      cudaFree(btA_d);
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
    // gridDim.y caps at 65535: with the 64-row m-tile the MMA launcher must
    // reject M > 4194240, and N near INT32_MAX must not overflow the
    // ceil-rounding add. Throw side only, dummy pointers never launched.
    expect(throws_invalid([&] { w4::gpu::gemm_mma(good, out, out, 4194241); }),
           "mma M past grid.y bound");
    {
      auto huge_n = good;
      huge_n.N = INT32_MAX;
      expect(throws_invalid([&] { w4::gpu::gemm_mma(huge_n, out, out, 2); }),
             "mma N overflow");
    }

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
    // MX launch bounds: same gridDim.y / ceil-add limits as the W4 entries.
    expect(throws_invalid(
               [&] { w4::gpu::gemm_fused_mx(good_mx, out, out, 262141); }),
           "mx fused M past grid.y bound");
    expect(throws_invalid(
               [&] { w4::gpu::gemm_mma_mx(good_mx, out, out, 4194241); }),
           "mx mma M past grid.y bound");
    {
      auto huge_mx = good_mx;
      huge_mx.N = INT32_MAX;
      expect(throws_invalid(
                 [&] { w4::gpu::gemm_fused_mx(huge_mx, out, out, 2); }),
             "mx fused N overflow");
      expect(throws_invalid(
                 [&] { w4::gpu::gemm_mma_mx(huge_mx, out, out, 2); }),
             "mx mma N overflow");
    }

    // Dispatcher entry contracts: malformed input must throw on both sides
    // of the threshold (M=2 routes SIMT, M=16 routes MMA), and the dispatch
    // inherits the MMA launch bound past the threshold.
    expect(throws_invalid([&] { w4::gpu::gemm(bad_gs0, out, out, 2); }),
           "dispatch gs=0 simt side");
    expect(throws_invalid([&] { w4::gpu::gemm(bad_gs0, out, out, 16); }),
           "dispatch gs=0 mma side");
    expect(throws_invalid([&] { w4::gpu::gemm(good, nullptr, out, 2); }),
           "dispatch x null");
    expect(throws_invalid([&] { w4::gpu::gemm(good, out, out, 0); }),
           "dispatch M=0");
    expect(throws_invalid([&] { w4::gpu::gemm(good, out, out, 4194241); }),
           "dispatch M past mma grid.y bound");
    expect(throws_invalid([&] { w4::gpu::gemm_mx(bad_mxb, out, out, 2); }),
           "dispatch mx B wrong simt side");
    expect(throws_invalid([&] { w4::gpu::gemm_mx(bad_mxb, out, out, 16); }),
           "dispatch mx B wrong mma side");
    expect(throws_invalid([&] { w4::gpu::gemm_mx(good_mx, out, out, 0); }),
           "dispatch mx M=0");
    expect(throws_invalid([&] { w4::gpu::gemm_mx(good_mx, out, out, 4194241); }),
           "dispatch mx M past mma grid.y bound");

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
               [&] { w4::gpu::attention_fp16(out, out, out, out, 1, 4, 513); }),
           "kv fp16 D bound");
    // Flash entry contracts: dims, the D shared bound, gs/G, nulls, and the
    // tile-grid S bound (65535 * 256). S > 4096 alone must NOT throw here —
    // that is the point of the flash path — so only the tile bound is
    // tested for rejection.
    expect(throws_invalid([&] { w4::gpu::attention_w4_flash(kv_s0, out, out); }),
           "kv flash S=0");
    expect(throws_invalid(
               [&] { w4::gpu::attention_w4_flash(kv_big_d, out, out); }),
           "kv flash D=513");
    expect(throws_invalid([&] { w4::gpu::attention_w4_flash(kv_gs, out, out); }),
           "kv flash gs=7");
    expect(throws_invalid([&] { w4::gpu::attention_w4_flash(kv_g, out, out); }),
           "kv flash G wrong");
    expect(throws_invalid([&] { w4::gpu::attention_w4_flash(kv_q, out, out); }),
           "kv flash q null");
    expect(throws_invalid([&] { w4::gpu::attention_w4_flash(kv_z, out, out); }),
           "kv flash asym zeros null");
    expect(throws_invalid(
               [&] { w4::gpu::attention_w4_flash(kv_sz, out, out); }),
           "kv flash sym carries zeros");
    expect(throws_invalid(
               [&] { w4::gpu::attention_w4_flash(good_kv, nullptr, out); }),
           "kv flash x null");
    expect(throws_invalid(
               [&] { w4::gpu::attention_w4_flash(good_kv, out, nullptr); }),
           "kv flash out null");
    {
      auto kv_past_tiles = good_kv;
      kv_past_tiles.S = 65535 * 256 + 1;
      expect(throws_invalid(
                 [&] { w4::gpu::attention_w4_flash(kv_past_tiles, out, out); }),
             "kv flash S past tile-grid bound");
    }
    expect(throws_invalid([&] {
             w4::gpu::attention_fp16_flash(out, out, out, out, 1, 4, 513);
           }),
           "kv fp16 flash D bound");
    expect(throws_invalid([&] {
             w4::gpu::attention_fp16_flash(out, out, out, out, 1,
                                           65535 * 256 + 1, 8);
           }),
           "kv fp16 flash S bound");
    expect(throws_invalid([&] {
             w4::gpu::attention_fp16_flash(nullptr, out, out, out, 1, 4, 8);
           }),
           "kv fp16 flash q null");
    // FP8 flash entry contracts: dims, D bounds (shared stage and word
    // alignment), gs, G, nulls, the tile-grid S bound, base alignment of
    // the word-walked caches, dequant_f8 rules.
    const w4::gpu::DeviceKVF8 good_f8{1, 4, 8, 8, 1, (const uint8_t*)1,
                                      (const __half*)1, (const uint8_t*)1,
                                      (const __half*)1};
    auto f8_s0 = good_f8; f8_s0.S = 0;
    auto f8_bigd = good_f8; f8_bigd.D = 513;
    auto f8_d2 = good_f8; f8_d2.D = 2; f8_d2.G = 1;  // D % 4 != 0
    auto f8_gs = good_f8; f8_gs.gs = 7;
    auto f8_g = good_f8; f8_g.G = 3;
    auto f8_q = good_f8; f8_q.q_k = nullptr;
    expect(throws_invalid(
               [&] { w4::gpu::attention_f8_flash(f8_s0, out, out); }),
           "f8 flash S=0");
    expect(throws_invalid(
               [&] { w4::gpu::attention_f8_flash(f8_bigd, out, out); }),
           "f8 flash D=513");
    expect(throws_invalid(
               [&] { w4::gpu::attention_f8_flash(f8_d2, out, out); }),
           "f8 flash D%4");
    expect(throws_invalid(
               [&] { w4::gpu::attention_f8_flash(f8_gs, out, out); }),
           "f8 flash gs=7");
    expect(throws_invalid(
               [&] { w4::gpu::attention_f8_flash(f8_g, out, out); }),
           "f8 flash G wrong");
    expect(throws_invalid(
               [&] { w4::gpu::attention_f8_flash(f8_q, out, out); }),
           "f8 flash q null");
    expect(throws_invalid(
               [&] { w4::gpu::attention_f8_flash(good_f8, nullptr, out); }),
           "f8 flash x null");
    {
      auto f8_big = good_f8;
      f8_big.S = 65535 * 256 + 1;
      expect(throws_invalid(
                 [&] { w4::gpu::attention_f8_flash(f8_big, out, out); }),
             "f8 flash S past tile-grid bound");
    }
    expect(throws_invalid(
               [&] { w4::gpu::dequant_f8(nullptr, (const __half*)1, out, 4, 8, 8); }),
           "f8 dequant q null");
    expect(throws_invalid(
               [&] { w4::gpu::dequant_f8((const uint8_t*)1, (const __half*)1, out, 4, 6, 8); }),
           "f8 dequant D%4");
    expect(throws_invalid(
               [&] { w4::gpu::dequant_f8((const uint8_t*)1, (const __half*)1, out, 4, 8, 7); }),
           "f8 dequant gs=7");
    // The word walk casts byte pointers to uint32: bases must be 4 B
    // aligned, rejected pre-launch (a misaligned base is an async device
    // fault otherwise).
    expect(throws_invalid(
               [&] { w4::gpu::dequant_f8((const uint8_t*)3, (const __half*)4, out, 4, 8, 8); }),
           "f8 dequant q misaligned");
    expect(throws_invalid(
               [&] { w4::gpu::attention_f8_flash(good_f8, out, out); }),
           "f8 flash cache base alignment");
    // Paged flash entry contracts.
    {
      const w4::gpu::DeviceKVW4 pk{1, 4, 8, 8, 1, 1, (const uint32_t*)1,
                                   (const __half*)1, nullptr,
                                   (const uint32_t*)1, (const __half*)1,
                                   nullptr};
      const int* btp = (const int*)1;
      expect(throws_invalid([&] {
               w4::gpu::attention_w4_flash_paged(pk, nullptr, 16, 1, out, out);
             }),
             "paged bt null");
      expect(throws_invalid([&] {
               w4::gpu::attention_w4_flash_paged(pk, btp, 0, 1, out, out);
             }),
             "paged page=0");
      expect(throws_invalid([&] {
               w4::gpu::attention_w4_flash_paged(pk, btp, 16, 0, out, out);
             }),
             "paged npages=0");
      {
        auto big_s = pk;
        big_s.S = 100;
        expect(throws_invalid([&] {
                 w4::gpu::attention_w4_flash_paged(big_s, btp, 16, 1, out,
                                                   out);
               }),
               "paged pool smaller than S");
      }
      {
        auto big_d = pk;
        big_d.D = 513;
        expect(throws_invalid([&] {
                 w4::gpu::attention_w4_flash_paged(big_d, btp, 16, 33, out,
                                                   out);
               }),
               "paged D=513");
      }
      {
        auto big_t = pk;
        big_t.S = 65535 * 256 + 1;
        expect(throws_invalid([&] {
                 w4::gpu::attention_w4_flash_paged(big_t, btp, 16,
                                                   65535 * 16 + 1, out, out);
               }),
               "paged S past tile-grid bound");
      }
      // npages*page passes the size_t geometry check but overflows the
      // 32-bit pool_rows the kernel receives (50000^2 = 2.5e9 > 2^31).
      expect(throws_invalid([&] {
               w4::gpu::attention_w4_flash_paged(pk, btp, 50000, 50000, out,
                                                 out);
             }),
             "paged pool rows overflow");
    }
    expect(throws_invalid(
               [&] { w4::gpu::attention_fp16(nullptr, out, out, out, 1, 4, 8); }),
           "kv fp16 q null");
  }

  std::printf("gpu: %d checks, %d failures, worst gemm rel L2 = %.3e\n",
              checks, fails, worst_rel);
  return fails ? 1 : 0;
}
