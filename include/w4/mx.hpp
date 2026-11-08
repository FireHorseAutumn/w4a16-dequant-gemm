#pragma once

// MXFP4 (OCP microscaling) packed format, host quantizer, and reference
// dequant. This is NOT the W4 affine scheme: elements are E2M1 minifloats
// (1 sign, 2 exponent, 1 mantissa — a 16-entry value table, no zero-point)
// and the shared scale is E8M0 (8 exponent bits, no mantissa — a pure power
// of two) per block of 32 elements. Dequant is a table lookup times an
// exponent shift, so the product is exact in float and the only rounding is
// the final round-to-nearest half conversion.
//
//   q:      8 nibbles per uint32 along K, low nibble first (same packing as
//           the W4 format); code = sign<<3 | magnitude index.
//   scales: one uint8 E8M0 per (row, block), block = 32 consecutive k.
//           value = 2^(byte - 127); byte 255 is NaN and is never emitted.
//   dequant: w = 2^(e-127) * e2m1[q]
//
// Quantizer rule: e = ceil(log2(amax / 6)), clamped to [-127, 127], so
// amax / scale <= 6 by construction — no element ever saturates to the
// endstop code. The floor variant (full range use, saturating the block
// max) is the noted alternative; we take the no-saturation side.
// amax == 0 quantizes to scale 2^0 with all-zero codes. A block whose amax
// would need e > 127 throws; clamping at e = -127 is lossless enough
// (absolute error bounded by 0.25 * 2^-127).

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "pack.hpp"

namespace w4::mx {

constexpr int MX_BLOCK = 32;

// E2M1 magnitude table, index 0..7. Sign is code bit 3.
constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

struct PackedWm {
  int N = 0, K = 0, B = 0;  // B = ceil(K / 32) blocks per row
  std::vector<uint32_t> q;       // N * words(K)
  std::vector<uint8_t> scales;   // N * B, E8M0 bytes
};

// The one dequant step, host and device (same contract as dequant_one:
// identical arithmetic on both sides so the bitwise gate will apply).
// Semantically a LUT lookup times an exponent shift, but implemented as a
// branch-free arithmetic decode: a function-local table indexed by a
// runtime nibble lands in local memory on device, and the per-element
// loads cost the fused SIMT kernel ~7x. E2M1 decodes exactly: magnitude
// index i splits as exponent field ee = i>>1 and mantissa m = i&1;
// ee == 0 -> m * 2^-1, else (2 + m) * 2^(ee - 2). Every value is exactly
// representable in float, so host and device produce identical bits.
W4_HD inline __half dequant_one_mx(uint8_t se, int q) {
  const int i = q & 7;
  const float mag = (i >> 1) == 0 ? ldexpf((float)(i & 1), -1)
                                  : ldexpf((float)(2 + (i & 1)), (i >> 1) - 2);
  const float s = ldexpf(1.0f, (int)se - 127);
  const float v = (q & 8) ? -mag : mag;
  return __float2half_rn(s * v);
}

W4_HD inline int blocks(int K) {
  return (int)(((int64_t)K + MX_BLOCK - 1) / MX_BLOCK);
}

inline int get_q(const PackedWm& p, int n, int k) {
  return (int)((p.q[(size_t)n * words(p.K) + k / 8] >> (4 * (k % 8))) & 0xF);
}

inline PackedWm quantize_mx(const std::vector<double>& w, int N, int K) {
  if (N < 1 || K < 1) throw std::invalid_argument("quantize_mx: bad dims");
  if (K > W4_MAX_K) throw std::invalid_argument("quantize_mx: K out of range");
  if (w.size() != (size_t)N * (size_t)K)
    throw std::invalid_argument("quantize_mx: size");
  for (const double v : w)
    if (!std::isfinite(v))
      throw std::invalid_argument("quantize_mx: non-finite input");
  PackedWm p;
  p.N = N;
  p.K = K;
  p.B = blocks(K);
  p.q.assign((size_t)N * words(K), 0);
  p.scales.assign((size_t)N * p.B, 0);

  for (int n = 0; n < N; ++n) {
    for (int b = 0; b < p.B; ++b) {
      const int k0 = b * MX_BLOCK, k1 = std::min(K, k0 + MX_BLOCK);
      double amax = 0.0;
      for (int k = k0; k < k1; ++k)
        amax = std::max(amax, std::fabs(w[(size_t)n * K + k]));
      int e = 0;
      if (amax > 0.0) {
        e = (int)std::ceil(std::log2(amax / 6.0));
        // log2 at an exact power of two can land one low; the loop is the
        // cheap insurance that the no-saturation invariant really holds.
        while (amax > 6.0 * std::ldexp(1.0, e)) ++e;
        if (e > 127)
          throw std::invalid_argument(
              "quantize_mx: block scale outside E8M0 range");
        if (e < -127) e = -127;
      }
      const double s = std::ldexp(1.0, e);
      p.scales[(size_t)n * p.B + b] = (uint8_t)(e + 127);
      for (int k = k0; k < k1; ++k) {
        const double v = w[(size_t)n * K + k] / s;
        const double av = std::fabs(v);
        // nearest magnitude, ties toward the lower index
        int idx = 0;
        for (int i = 1; i < 8; ++i)
          if (std::fabs((double)kE2M1[i] - av) <
              std::fabs((double)kE2M1[idx] - av))
            idx = i;
        const int q = (std::signbit(v) ? 8 : 0) | idx;
        p.q[(size_t)n * words(K) + k / 8] |= (uint32_t)q << (4 * (k % 8));
      }
    }
  }
  return p;
}

inline void validate_packed_mx(const PackedWm& p) {
  if (p.N < 1 || p.K < 1) throw std::invalid_argument("packed mx: bad dims");
  if (p.K > W4_MAX_K)
    throw std::invalid_argument("packed mx: K out of range");
  if (p.B != blocks(p.K))
    throw std::invalid_argument("packed mx: B inconsistent with K");
  if (p.q.size() != (size_t)p.N * (size_t)words(p.K))
    throw std::invalid_argument("packed mx: q size");
  if (p.scales.size() != (size_t)p.N * (size_t)p.B)
    throw std::invalid_argument("packed mx: scales size");
}

inline std::vector<__half> dequant_all_mx(const PackedWm& p) {
  validate_packed_mx(p);
  std::vector<__half> w((size_t)p.N * p.K);
  for (int n = 0; n < p.N; ++n)
    for (int k = 0; k < p.K; ++k)
      w[(size_t)n * p.K + k] =
          dequant_one_mx(p.scales[(size_t)n * p.B + k / MX_BLOCK],
                         get_q(p, n, k));
  return w;
}

}  // namespace w4::mx
