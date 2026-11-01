// Host-side gates: quantizer per-element bounds across the full group-size
// matrix (including ragged K and group_size % 32 != 0, the silent-wrong-
// output shape from the vLLM RDNA W4A16 issue class), plus a fixed tiny
// case pinning the FP64 GEMM oracle to a hand-computed answer.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "w4/kvcache.hpp"
#include "w4/mx.hpp"
#include "w4/pack.hpp"

struct Cfg {
  int M, N, K, gs;
};

int main() {
  const std::vector<Cfg> cfgs = {
      {1, 64, 64, 8},      {8, 128, 96, 16},  {16, 256, 100, 32},
      {4, 96, 72, 48},     {2, 32, 128, 64},  {3, 40, 256, 128},
      {5, 72, 40, 8},  // K not a multiple of gs, smallest group
  };
  int checks = 0, fails = 0;

  for (const bool sym : {true, false}) {
    for (size_t i = 0; i < cfgs.size(); ++i) {
      const Cfg c = cfgs[i];
      const uint64_t seed = 0x44A1ULL ^ (uint64_t)(i * 977 + (sym ? 0 : 1));
      const auto w = w4::normal_weights(c.N * c.K, seed);
      const w4::PackedW p = w4::quantize(w, c.N, c.K, c.gs, sym);
      const auto wd = w4::dequant_all(p);

      double worst = 0.0;
      for (int n = 0; n < c.N; ++n)
        for (int k = 0; k < c.K; ++k) {
          const int g = k / p.gs;
          const double sf = __half2float(p.scales[(size_t)n * p.G + g]);
          const int q = w4::get_q(p, n, k);
          const bool clamped = (q == 0 || q == 15);
          const double tol = (clamped ? 1.1 : 0.52) * sf + 1e-12;
          const double err =
              std::fabs((double)__half2float(wd[(size_t)n * c.K + k]) -
                        w[(size_t)n * c.K + k]);
          worst = std::max(worst, err / tol);
          ++checks;
          if (err > tol) {
            ++fails;
            std::printf(
                "FAIL %s N=%d K=%d gs=%d n=%d k=%d: |wd-w|=%.3e > %.3e\n",
                sym ? "sym" : "asym", c.N, c.K, c.gs, n, k, err, tol);
          }
        }
      std::printf("%s N=%d K=%d gs=%d: worst err/tol = %.3f\n",
                  sym ? "sym  " : "asym ", c.N, c.K, c.gs, worst);
    }
  }

  // Oracle pinned to a hand-computed result: x = [1, 2],
  // W = [[3, 4], [5, 6]] -> C = [11, 17].
  {
    const std::vector<__half> x = {__float2half_rn(1.0f),
                                   __float2half_rn(2.0f)};
    const std::vector<__half> w = {
        __float2half_rn(3.0f), __float2half_rn(4.0f),
        __float2half_rn(5.0f), __float2half_rn(6.0f)};
    const auto c = w4::gemm_oracle(x, 1, 2, w, 2);
    ++checks;
    if (c[0] != 11.0 || c[1] != 17.0) {
      ++fails;
      std::printf("FAIL oracle 2x2: got [%g, %g], want [11, 17]\n", c[0],
                  c[1]);
    }
  }

  // Degenerate inputs: all-zero group must not produce NaN scales.
  for (const bool sym : {true, false}) {
    const std::vector<double> w(64 * 64, 0.0);
    const auto p = w4::quantize(w, 64, 64, 32, sym);
    const auto wd = w4::dequant_all(p);
    for (const auto& v : wd) {
      ++checks;
      if (__half2float(v) != 0.0f) {
        ++fails;
        std::printf("FAIL zero input %s: nonzero dequant\n",
                    sym ? "sym" : "asym");
        break;
      }
    }
  }

  // Validation contracts: malformed input must throw, never quantize
  // silently. NaN is the dangerous one — std::min/max swallow it.
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
    const std::vector<double> ok(64 * 64, 0.5);
    expect(throws_invalid([&] { w4::quantize(ok, 0, 64, 8, true); }), "N=0");
    expect(throws_invalid([&] { w4::quantize(ok, 64, 64, 0, true); }), "gs=0");
    expect(throws_invalid([&] { w4::quantize(ok, 64, 64, 7, true); }), "gs=7");
    expect(throws_invalid([&] { w4::quantize(ok, 64, 64, 9, true); }), "gs=9");
    expect(throws_invalid(
               [&] { w4::quantize(std::vector<double>(10, 0.0), 64, 64, 8, true); }),
           "size mismatch");
    for (const bool sym : {true, false}) {
      auto bad = ok;
      bad[3] = std::nan("");
      expect(throws_invalid([&] { w4::quantize(bad, 64, 64, 8, sym); }),
             sym ? "NaN sym" : "NaN asym");
      bad[3] = 1e300;  // scale overflows FP16
      expect(throws_invalid([&] { w4::quantize(bad, 64, 64, 8, sym); }),
             sym ? "scale overflow sym" : "scale overflow asym");
      std::vector<double> tiny(64 * 64, 1e-12);  // scale underflows FP16
      expect(throws_invalid([&] { w4::quantize(tiny, 64, 64, 8, sym); }),
             sym ? "scale underflow sym" : "scale underflow asym");
    }
    // Malformed PackedW must not reach the reference dequant either.
    auto p = w4::quantize(ok, 64, 64, 8, true);
    auto bad_gs = p;
    bad_gs.gs = 7;
    expect(throws_invalid([&] { w4::dequant_all(bad_gs); }), "dequant gs=7");
    auto bad_sizes = p;
    bad_sizes.q.resize(3);
    expect(throws_invalid([&] { w4::dequant_all(bad_sizes); }), "dequant size");
    expect(throws_invalid([&] {
             w4::gemm_oracle(std::vector<__half>(3), 2, 2,
                             std::vector<__half>(4), 1);
           }),
           "oracle size mismatch");
  }

  // ---- MXFP4 (E2M1 + E8M0 block scale) host gates ----
  // Per-element error bounds against the exact quantization interval
  // (half the wider neighboring code spacing), the no-saturation
  // invariant, ragged blocks, degenerate blocks, a hand-computed pin, and
  // the throw contracts.
  {
    const std::vector<Cfg> mcfgs = {
        {1, 64, 64, 0},   {8, 128, 96, 0},  {16, 256, 100, 0},  // ragged K
        {4, 96, 72, 0},   {2, 32, 33, 0},   // block of 1 element
        {3, 40, 256, 0},  {5, 72, 40, 0},
    };
    for (size_t i = 0; i < mcfgs.size(); ++i) {
      const Cfg c = mcfgs[i];
      const uint64_t seed = 0x9E4FULL ^ (uint64_t)(i * 977);
      const auto w = w4::normal_weights(c.N * c.K, seed);
      const w4::mx::PackedWm p = w4::mx::quantize_mx(w, c.N, c.K);
      const auto wd = w4::mx::dequant_all_mx(p);

      double worst = 0.0;
      for (int n = 0; n < c.N; ++n)
        for (int k = 0; k < c.K; ++k) {
          const uint8_t se = p.scales[(size_t)n * p.B + k / 32];
          const double s = std::ldexp(1.0, (int)se - 127);
          const int q = w4::mx::get_q(p, n, k);
          const int idx = q & 7;
          // wider of the two neighboring code spacings, halved
          const double left =
              idx > 0 ? (double)(w4::mx::kE2M1[idx] - w4::mx::kE2M1[idx - 1])
                      : 0.5;
          const double right =
              idx < 7 ? (double)(w4::mx::kE2M1[idx + 1] - w4::mx::kE2M1[idx])
                      : left;
          const double tol = 0.5 * std::max(left, right) * s + 1e-12;
          const double err =
              std::fabs((double)__half2float(wd[(size_t)n * c.K + k]) -
                        w[(size_t)n * c.K + k]);
          worst = std::max(worst, err / tol);
          ++checks;
          if (err > tol) {
            ++fails;
            std::printf("FAIL mx N=%d K=%d n=%d k=%d: |wd-w|=%.3e > %.3e\n",
                        c.N, c.K, n, k, err, tol);
          }
        }
      // No-saturation invariant: no block's amax exceeds 6 * scale.
      for (int n = 0; n < c.N; ++n)
        for (int b = 0; b < p.B; ++b) {
          const int k0 = b * 32, k1 = std::min(c.K, k0 + 32);
          double amax = 0.0;
          for (int k = k0; k < k1; ++k)
            amax = std::max(amax, std::fabs(w[(size_t)n * c.K + k]));
          const double s =
              std::ldexp(1.0, (int)p.scales[(size_t)n * p.B + b] - 127);
          ++checks;
          if (amax > 6.0 * s * (1.0 + 1e-12)) {
            ++fails;
            std::printf("FAIL mx saturation N=%d K=%d n=%d b=%d\n", c.N,
                        c.K, n, b);
          }
        }
      std::printf("mx N=%d K=%d: worst err/tol = %.3f\n", c.N, c.K, worst);
    }

    // Hand pin: a block of all 3.0 -> amax 3, e = ceil(log2(0.5)) = -1,
    // scale byte 126, every code = 7 (magnitude 6.0), dequant exact 3.0.
    {
      const std::vector<double> w(64, 3.0);
      const auto p = w4::mx::quantize_mx(w, 2, 32);
      ++checks;
      if (p.scales[0] != 126 || p.scales[1] != 126) {
        ++fails;
        std::printf("FAIL mx pin scale: got %d, %d\n", (int)p.scales[0],
                    (int)p.scales[1]);
      }
      const auto wd = w4::mx::dequant_all_mx(p);
      ++checks;
      if (__half2float(wd[0]) != 3.0f || __half2float(wd[63]) != 3.0f) {
        ++fails;
        std::printf("FAIL mx pin value: got %g\n", (double)__half2float(wd[0]));
      }
      ++checks;
      if (w4::mx::get_q(p, 0, 0) != 7) {
        ++fails;
        std::printf("FAIL mx pin code: got %d\n", w4::mx::get_q(p, 0, 0));
      }
    }

    // Sign handling: an all-negative block must produce negative codes.
    {
      const std::vector<double> w(32, -2.0);
      const auto p = w4::mx::quantize_mx(w, 1, 32);
      const auto wd = w4::mx::dequant_all_mx(p);
      ++checks;
      if (__half2float(wd[0]) != -2.0f || (w4::mx::get_q(p, 0, 0) & 8) == 0) {
        ++fails;
        std::printf("FAIL mx sign: got %g code %d\n",
                    (double)__half2float(wd[0]), w4::mx::get_q(p, 0, 0));
      }
    }

    // Degenerate all-zero input: no NaN scales, all-zero codes.
    {
      const std::vector<double> w(64 * 64, 0.0);
      const auto p = w4::mx::quantize_mx(w, 64, 64);
      const auto wd = w4::mx::dequant_all_mx(p);
      bool bad = false;
      for (const auto& v : wd)
        if (__half2float(v) != 0.0f) bad = true;
      ++checks;
      if (bad) {
        ++fails;
        std::printf("FAIL mx zero input: nonzero dequant\n");
      }
    }

    // Throw contracts: NaN, E8M0 overflow, dims, size.
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
          std::printf("FAIL mx no-throw: %s\n", name);
        }
      };
      const std::vector<double> ok(64 * 64, 0.5);
      expect(throws_invalid([&] { w4::mx::quantize_mx(ok, 0, 64); }),
             "mx N=0");
      expect(throws_invalid(
                 [&] { w4::mx::quantize_mx(std::vector<double>(10), 64, 64); }),
             "mx size mismatch");
      auto bad = ok;
      bad[3] = std::nan("");
      expect(throws_invalid([&] { w4::mx::quantize_mx(bad, 64, 64); }),
             "mx NaN");
      bad[3] = 1e300;  // block scale would need e > 127
      expect(throws_invalid([&] { w4::mx::quantize_mx(bad, 64, 64); }),
             "mx scale overflow");
      auto p = w4::mx::quantize_mx(ok, 64, 64);
      auto bad_sizes = p;
      bad_sizes.q.resize(3);
      expect(throws_invalid([&] { w4::mx::dequant_all_mx(bad_sizes); }),
             "mx dequant size");
    }
  }

  // ---- KV-cache attention (R5) host gates ----
  // The cache is H slabs of PackedW (N=S positions, K=D head dim); the
  // gates cover the slab layout against independent per-head quantization,
  // the FP64 reference attention against a hand pin, softmax max
  // subtraction at overflow-class score magnitudes, and the throw
  // contracts.
  {
    // Layout: quantize_cache over H heads must reproduce, slab for slab,
    // what independent per-head quantize calls produce.
    for (const bool sym : {true, false}) {
      const int H = 3, S = 5, D = 24, gs = 8;
      const auto ksrc = w4::normal_weights(H * S * D, 0xC1U);
      const auto vsrc = w4::normal_weights(H * S * D, 0xC2U);
      const auto c = w4::kv::quantize_cache(ksrc, vsrc, H, S, D, 8, sym);
      for (int h = 0; h < H; ++h) {
        const auto ki = w4::dequant_all(w4::quantize(
            std::vector<double>(ksrc.begin() + (size_t)h * S * D,
                                ksrc.begin() + (size_t)(h + 1) * S * D),
            S, D, gs, sym));
        const auto kh = w4::kv::dequant_head(c, h, false);
        ++checks;
        if (std::memcmp(ki.data(), kh.data(), (size_t)S * D * sizeof(__half)) !=
            0) {
          ++fails;
          std::printf("FAIL kv layout %s h=%d\n", sym ? "sym" : "asym", h);
        }
      }
    }

    // Hand pin: k all zeros -> scores all zero -> softmax uniform -> output
    // is the mean of the two V rows, computed in the same accumulation
    // order, so the match is exact.
    {
      const int H = 1, S = 2, D = 8;
      const std::vector<double> k(H * S * D, 0.0);
      std::vector<double> v(H * S * D, 0.0);
      for (int d = 0; d < D; ++d) {
        v[d] = 1.0;
        v[D + d] = 3.0;
      }
      const auto c = w4::kv::quantize_cache(k, v, H, S, D, 8, true);
      const auto vdq = w4::kv::dequant_head(c, 0, true);
      const std::vector<__half> q(H * D, __float2half_rn(1.0f));
      const auto out = w4::kv::attention_reference(q, c);
      for (int d = 0; d < D; ++d) {
        const double want = 0.5 * (double)__half2float(vdq[d]) +
                            0.5 * (double)__half2float(vdq[D + d]);
        ++checks;
        if (out[d] != want) {
          ++fails;
          std::printf("FAIL kv pin d=%d: got %g want %g\n", d, out[d], want);
        }
      }
    }

    // Softmax stability: scores of magnitude ~3e4 overflow a naive exp;
    // max subtraction keeps the result finite and correct (uniform here).
    {
      const int H = 1, S = 2, D = 8;
      const std::vector<double> k(H * S * D, 100.0);
      const std::vector<double> v(H * S * D, 0.5);
      const auto c = w4::kv::quantize_cache(k, v, H, S, D, 8, true);
      const std::vector<__half> q(H * D, __float2half_rn(100.0f));
      const auto out = w4::kv::attention_reference(q, c);
      ++checks;
      if (!std::isfinite(out[0])) {
        ++fails;
        std::printf("FAIL kv softmax overflow: got %g\n", out[0]);
      }
      const auto vdq = w4::kv::dequant_head(c, 0, true);
      ++checks;
      if (out[0] != (double)__half2float(vdq[0])) {
        ++fails;
        std::printf("FAIL kv softmax value: got %g\n", out[0]);
      }
    }

    // Throw contracts.
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
          std::printf("FAIL kv no-throw: %s\n", name);
        }
      };
      const std::vector<double> ok(1 * 4 * 8, 0.5);
      expect(throws_invalid(
                 [&] { w4::kv::quantize_cache(ok, ok, 0, 4, 8, 8, true); }),
             "kv H=0");
      expect(throws_invalid(
                 [&] { w4::kv::quantize_cache(ok, ok, 1, 4, 8, 7, true); }),
             "kv gs=7");
      // gs=0 divided into G before the per-head quantizer could validate it
      // (SIGFPE at a host entry point, caught in the hardening pass).
      expect(throws_invalid(
                 [&] { w4::kv::quantize_cache(ok, ok, 1, 4, 8, 0, true); }),
             "kv gs=0");
      expect(throws_invalid([&] {
               w4::kv::quantize_cache(std::vector<double>(10), ok, 1, 4, 8, 8,
                                      true);
             }),
             "kv size mismatch");
      auto c = w4::kv::quantize_cache(ok, ok, 1, 4, 8, 8, true);
      auto bad = c;
      bad.G = 2;
      expect(throws_invalid([&] { w4::kv::dequant_head(bad, 0, false); }),
             "kv G inconsistent");
      expect(throws_invalid([&] { w4::kv::dequant_head(c, 1, false); }),
             "kv h out of range");
      const std::vector<__half> q(8, __float2half_rn(1.0f));
      expect(throws_invalid(
                 [&] { w4::kv::attention_reference(std::vector<__half>(3), c); }),
             "kv q size");
      (void)q;
    }
  }

  std::printf("oracle: %d checks, %d failures\n", checks, fails);
  return fails ? 1 : 0;
}
