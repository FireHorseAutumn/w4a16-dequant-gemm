// Host-side gates: quantizer per-element bounds across the full group-size
// matrix (including ragged K and group_size % 32 != 0, the silent-wrong-
// output shape from the vLLM RDNA W4A16 issue class), plus a fixed tiny
// case pinning the FP64 GEMM oracle to a hand-computed answer.

#include <cmath>
#include <cstdio>
#include <vector>

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

  std::printf("oracle: %d checks, %d failures\n", checks, fails);
  return fails ? 1 : 0;
}
