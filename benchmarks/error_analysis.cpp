// Error-analysis measurements behind ERROR_ANALYSIS.md. Host-only,
// deterministic seeds; prints the tables. Everything here is a value
// measurement, not a timing: machine state does not touch the numbers.
//
//   A. Quantizer rel L2 vs group size (W4 sym/asym, MXFP4), Gaussian.
//   B. Where the error lives: absolute and relative error binned by
//      |w|/amax within the element's group (W4 sym gs=128).
//   C. Outlier sensitivity: 0.1% of elements at 20x, gs 16 vs 128.
//   D. Distribution shape: Gaussian vs Laplace vs uniform, gs=128 sym.
//   E. End-to-end GEMM error vs the FP16 reference, K in {256, 1024,
//      4096}: the quantized W is promoted exactly to FP64, so the
//      measured gap is quantization only: no accumulation noise.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "w4/mx.hpp"
#include "w4/pack.hpp"

namespace {

double rel_l2(const std::vector<double>& a, const std::vector<__half>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = (double)__half2float(b[i]) - a[i];
    num += d * d;
    den += a[i] * a[i];
  }
  return std::sqrt(num / (den > 0.0 ? den : 1.0));
}

double max_abs(const std::vector<double>& a, const std::vector<__half>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs((double)__half2float(b[i]) - a[i]));
  return m;
}

std::vector<double> laplace_weights(int count, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::exponential_distribution<double> e(1.0);
  std::uniform_int_distribution<int> s(0, 1);
  std::vector<double> v(count);
  for (auto& x : v) x = s(rng) ? e(rng) : -e(rng);
  return v;
}

std::vector<double> uniform_weights(int count, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(count);
  for (auto& x : v) x = u(rng);
  return v;
}

}  // namespace

int main() {
  const int N = 512, K = 4096;
  const auto w = w4::normal_weights(N * K, 0xEA11);

  std::printf("A. rel L2 of dequant(quant(W)) vs W, Gaussian %dx%d\n", N, K);
  std::printf("%-14s %-6s %-10s %-12s %s\n", "format", "gs", "bits/elt",
              "rel L2", "max abs err");
  double bits = 0.0;
  for (const int gs : {8, 16, 32, 64, 128}) {
    for (const bool sym : {true, false}) {
      const auto p = w4::quantize(w, N, K, gs, sym);
      const auto wq = w4::dequant_all(p);
      bits = 4.0 + 16.0 / gs + (sym ? 0.0 : 4.0 / gs);
      std::printf("%-14s %-6d %-10.3f %-12.4e %.3e\n",
                  sym ? "w4 sym" : "w4 asym", gs, bits, rel_l2(w, wq),
                  max_abs(w, wq));
    }
  }
  {
    const auto p = w4::mx::quantize_mx(w, N, K);
    const auto wq = w4::mx::dequant_all_mx(p);
    std::printf("%-14s %-6s %-10.3f %-12.4e %.3e\n", "mxfp4", "32", 4.25,
                rel_l2(w, wq), max_abs(w, wq));
  }

  std::printf(
      "\nB. where the error lives (w4 sym gs=128): bins by |w|/amax_group\n");
  {
    const int gs = 128;
    const auto p = w4::quantize(w, N, K, gs, true);
    const auto wq = w4::dequant_all(p);
    double abs_sum[10] = {0}, rel_sum[10] = {0};
    int64_t cnt[10] = {0};
    for (int n = 0; n < N; ++n)
      for (int g = 0; g < p.G; ++g) {
        double amax = 0.0;
        for (int k = g * gs; k < (g + 1) * gs; ++k)
          amax = std::max(amax, std::fabs(w[(size_t)n * K + k]));
        for (int k = g * gs; k < (g + 1) * gs; ++k) {
          const double wv = w[(size_t)n * K + k];
          const double err =
              std::fabs((double)__half2float(wq[(size_t)n * K + k]) - wv);
          const double frac = amax > 0.0 ? std::fabs(wv) / amax : 1.0;
          const int bin = std::min(9, (int)(frac * 10.0));
          abs_sum[bin] += err;
          rel_sum[bin] += err / (std::fabs(wv) > 1e-12 ? std::fabs(wv) : 1e-12);
          ++cnt[bin];
        }
      }
    std::printf("%-14s %-14s %-14s %s\n", "|w|/amax bin", "count", "mean abs",
                "mean rel");
    for (int b = 0; b < 10; ++b)
      std::printf("%.1f-%.1f      %-14lld %-14.3e %.3e\n", b / 10.0,
                  (b + 1) / 10.0, (long long)cnt[b], abs_sum[b] / cnt[b],
                  rel_sum[b] / cnt[b]);
  }

  std::printf("\nC. outlier sensitivity: 0.1%% of elements at 20x magnitude\n");
  {
    auto wo = w;
    std::mt19937_64 rng(0x07A1);
    std::uniform_int_distribution<int> pick(0, N * K - 1);
    for (int i = 0; i < N * K / 1000; ++i) wo[pick(rng)] *= 20.0;
    for (const int gs : {16, 128}) {
      const auto p = w4::quantize(wo, N, K, gs, true);
      const auto wq = w4::dequant_all(p);
      std::printf("w4 sym gs=%-4d rel L2 %.4e (max abs %.3e)\n", gs,
                  rel_l2(wo, wq), max_abs(wo, wq));
    }
  }

  std::printf("\nD. distribution shape (w4 sym gs=128)\n");
  for (const auto& [name, src] :
       std::vector<std::pair<const char*, std::vector<double>>>{
           {"gaussian", w},
           {"laplace", laplace_weights(N * K, 0x1A9A)},
           {"uniform", uniform_weights(N * K, 0x0B1E)}}) {
    const auto p = w4::quantize(src, N, K, 128, true);
    const auto wq = w4::dequant_all(p);
    std::printf("%-10s rel L2 %.4e\n", name, rel_l2(src, wq));
  }

  std::printf(
      "\nE. end-to-end: rel L2 of quantized-W GEMM vs FP16-reference GEMM "
      "(both FP64-accumulated; the gap is quantization only)\n");
  {
    const int M = 4, Ne = 512;
    std::printf("%-8s %-10s %-12s %s\n", "K", "format", "vs fp16 ref",
                "fp16 ref vs fp64");
    for (const int Ke : {256, 1024, 4096}) {
      const auto we = w4::normal_weights(Ne * Ke, 0xE110 + Ke);
      const auto x = w4::normal_activations(M * Ke, 0xE220 + Ke);
      // FP16 reference: W rounded to FP16, no quantization.
      std::vector<__half> w16(Ne * Ke);
      for (size_t i = 0; i < we.size(); ++i)
        w16[i] = __float2half_rn((float)we[i]);
      const auto c16 = w4::gemm_oracle(x, M, Ke, w16, Ne);
      // FP64 source: same x, original W, FP64 accumulation.
      std::vector<double> csrc((size_t)M * Ne, 0.0);
      for (int m = 0; m < M; ++m)
        for (int nn = 0; nn < Ne; ++nn) {
          double acc = 0.0;
          for (int kk = 0; kk < Ke; ++kk)
            acc += (double)__half2float(x[(size_t)m * Ke + kk]) *
                   we[(size_t)nn * Ke + kk];
          csrc[(size_t)m * Ne + nn] = acc;
        }
      double numr = 0.0, denr = 0.0;
      for (size_t i = 0; i < c16.size(); ++i) {
        numr += (c16[i] - csrc[i]) * (c16[i] - csrc[i]);
        denr += csrc[i] * csrc[i];
      }
      const double fp16_only = std::sqrt(numr / (denr > 0.0 ? denr : 1.0));
      for (const auto& [name, wq] :
           std::vector<std::pair<const char*, std::vector<__half>>>{
               {"w4s gs=32",
                w4::dequant_all(w4::quantize(we, Ne, Ke, 32, true))},
               {"w4s gs=128",
                w4::dequant_all(w4::quantize(we, Ne, Ke, 128, true))},
               {"mxfp4",
                w4::mx::dequant_all_mx(w4::mx::quantize_mx(we, Ne, Ke))}}) {
        const auto cq = w4::gemm_oracle(x, M, Ke, wq, Ne);
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < cq.size(); ++i) {
          num += (cq[i] - c16[i]) * (cq[i] - c16[i]);
          den += c16[i] * c16[i];
        }
        std::printf("%-8d %-10s %-12.4e %.3e\n", Ke, name,
                    std::sqrt(num / (den > 0.0 ? den : 1.0)), fp16_only);
      }
    }
  }

  return 0;
}
