#pragma once

// Packed W4A16 weight format, host quantizer, and FP64 oracle.
//
// Logical W is N x K row-major. Groups partition K: group g covers
// k in [g*gs, min((g+1)*gs, K)). gs must be a multiple of 8 so one packed
// word never straddles a group boundary.
//   q:      8 nibbles per uint32; word w of row n holds k = 8w + j,
//           low nibble first (j = 0 is bits 3..0).
//   scales: __half, index n*G + g.
//   zeros:  asymmetric mode only, one nibble per (n, g), packed like q
//           along G instead of K.
//   symmetric:    w = scale * (q - 8)
//   asymmetric:   w = scale * (q - z)

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__)
#define W4_HD __host__ __device__
#else
#define W4_HD
#endif

namespace w4 {

W4_HD inline int words(int n) { return (int)(((int64_t)n + 7) / 8); }

// Every kernel walks K with 32-bit ints and ceil-adds (words, chunk counts,
// k0 = 8*w). K is bounded so none of that can wrap; the slack covers the
// 32-wide chunk ceil-add in the MMA kernels.
constexpr int W4_MAX_K = INT32_MAX - 31;

struct PackedW {
  int N = 0, K = 0, gs = 0, G = 0;
  bool symmetric = true;
  std::vector<uint32_t> q;      // N * words(K)
  std::vector<__half> scales;   // N * G
  std::vector<uint32_t> zeros;  // N * words(G), asymmetric only
};

inline int get_q(const PackedW& p, int n, int k) {
  return (int)((p.q[(size_t)n * words(p.K) + k / 8] >> (4 * (k % 8))) & 0xF);
}

inline int get_z(const PackedW& p, int n, int g) {
  return (int)((p.zeros[(size_t)n * words(p.G) + g / 8] >> (4 * (g % 8))) & 0xF);
}

inline int zero_point(const PackedW& p, int n, int g) {
  return p.symmetric ? 8 : get_z(p, n, g);
}

// The one dequantization step. Host reference and device kernel both call
// this so the two-pass pipeline and its reference share identical arithmetic
// (float multiply, round-to-nearest-even half conversion).
W4_HD inline __half dequant_one(__half s, int q, int z) {
  return __float2half_rn(__half2float(s) * (float)(q - z));
}

// Quantize FP64 source weights against the half-rounded scale actually
// stored, so dequantization is self-consistent with the packed artifact.
inline PackedW quantize(const std::vector<double>& w, int N, int K, int gs,
                        bool symmetric) {
  if (N < 1 || K < 1) throw std::invalid_argument("quantize: bad dims");
  if (K > W4_MAX_K) throw std::invalid_argument("quantize: K out of range");
  if (gs < 8 || gs % 8 != 0)
    throw std::invalid_argument("quantize: gs must be a multiple of 8");
  if (w.size() != (size_t)N * (size_t)K)
    throw std::invalid_argument("quantize: size");
  for (const double v : w)
    if (!std::isfinite(v))
      throw std::invalid_argument("quantize: non-finite input");
  PackedW p;
  p.N = N;
  p.K = K;
  p.gs = gs;
  p.G = (int)(((int64_t)K + gs - 1) / gs);
  p.symmetric = symmetric;
  p.q.assign((size_t)N * words(K), 0);
  p.scales.assign((size_t)N * p.G, __float2half_rn(0.0f));
  if (!symmetric) p.zeros.assign((size_t)N * words(p.G), 0);

  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < p.G; ++g) {
      const int k0 = g * gs, k1 = std::min(K, k0 + gs);
      double lo = w[(size_t)n * K + k0], hi = lo;
      for (int k = k0 + 1; k < k1; ++k) {
        const double v = w[(size_t)n * K + k];
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
      if (!symmetric) {
        // Range must bracket zero so the zero-point lands in [0, 15];
        // one-sided groups would otherwise clamp z and corrupt the group.
        lo = std::min(lo, 0.0);
        hi = std::max(hi, 0.0);
      }
      double s;
      if (symmetric) {
        const double amax = std::max(std::fabs(lo), std::fabs(hi));
        s = amax > 0.0 ? amax / 7.0 : 1.0;  // codes land in [1, 15]
      } else {
        s = hi > lo ? (hi - lo) / 15.0 : 1.0;
      }
      const __half sh = __float2half_rn((float)s);
      const float sf = __half2float(sh);
      // The scale must survive FP16 storage; overflow (inf) and underflow
      // to zero on a nonzero group are rejected, not quantized through.
      const bool nonzero = symmetric ? (lo != 0.0 || hi != 0.0) : (hi != lo);
      if (!std::isfinite(sf) || (sf == 0.0f && nonzero))
        throw std::invalid_argument("quantize: group scale outside FP16 range");
      int z = 8;
      if (!symmetric) {
        z = (int)std::lround(-lo / (double)sf);
        z = std::max(0, std::min(15, z));
      }
      p.scales[(size_t)n * p.G + g] = sh;
      for (int k = k0; k < k1; ++k) {
        int q = (int)std::lround(w[(size_t)n * K + k] / (double)sf) + z;
        q = std::max(0, std::min(15, q));
        p.q[(size_t)n * words(K) + k / 8] |= (uint32_t)q << (4 * (k % 8));
      }
      if (!symmetric)
        p.zeros[(size_t)n * words(p.G) + g / 8] |= (uint32_t)z
                                                   << (4 * (g % 8));
    }
  }
  return p;
}

// Structural invariants of the packed artifact. dequant_all and the tests
// route through this; a malformed PackedW is rejected, never read.
inline void validate_packed(const PackedW& p) {
  if (p.N < 1 || p.K < 1) throw std::invalid_argument("packed: bad dims");
  if (p.K > W4_MAX_K) throw std::invalid_argument("packed: K out of range");
  if (p.gs < 8 || p.gs % 8 != 0)
    throw std::invalid_argument("packed: gs must be a multiple of 8");
  if ((int64_t)p.G != ((int64_t)p.K + p.gs - 1) / p.gs)
    throw std::invalid_argument("packed: G inconsistent with K, gs");
  if (p.q.size() != (size_t)p.N * (size_t)words(p.K))
    throw std::invalid_argument("packed: q size");
  if (p.scales.size() != (size_t)p.N * (size_t)p.G)
    throw std::invalid_argument("packed: scales size");
  if (!p.symmetric && p.zeros.size() != (size_t)p.N * (size_t)words(p.G))
    throw std::invalid_argument("packed: zeros size");
  if (p.symmetric && !p.zeros.empty())
    throw std::invalid_argument("packed: symmetric carries no zeros");
}

inline std::vector<__half> dequant_all(const PackedW& p) {
  validate_packed(p);
  std::vector<__half> w((size_t)p.N * p.K);
  for (int n = 0; n < p.N; ++n)
    for (int k = 0; k < p.K; ++k) {
      const int g = k / p.gs;
      w[(size_t)n * p.K + k] =
          dequant_one(p.scales[(size_t)n * p.G + g], get_q(p, n, k),
                      zero_point(p, n, g));
    }
  return w;
}

// C[m,n] = sum_k x[m,k] * w[n,k]. FP64 accumulation over the exact FP16
// inputs the device pipeline sees, so the only expected device deviation is
// FP32 accumulation order plus the final half rounding.
inline std::vector<double> gemm_oracle(const std::vector<__half>& x, int M,
                                       int K, const std::vector<__half>& w,
                                       int N) {
  if (M < 1 || N < 1 || K < 1)
    throw std::invalid_argument("gemm_oracle: bad dims");
  if (x.size() != (size_t)M * (size_t)K ||
      w.size() != (size_t)N * (size_t)K)
    throw std::invalid_argument("gemm_oracle: size");
  std::vector<double> c((size_t)M * N, 0.0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k)
        acc += (double)__half2float(x[(size_t)m * K + k]) *
               (double)__half2float(w[(size_t)n * K + k]);
      c[(size_t)m * N + n] = acc;
    }
  return c;
}

inline std::vector<double> normal_weights(int count, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> d(0.0, 1.0);
  std::vector<double> v(count);
  for (auto& x : v) x = d(rng);
  return v;
}

inline std::vector<__half> normal_activations(int count, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> d(0.0, 1.0);
  std::vector<__half> v(count);
  for (auto& x : v) x = __float2half_rn((float)d(rng));
  return v;
}

}  // namespace w4
