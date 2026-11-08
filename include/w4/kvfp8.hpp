#pragma once

// FP8 (E4M3) KV cache, host side (R6): the deployed-incumbent format on
// the R5 cache structure. One E4M3 byte per element, one FP16 scale per
// (position, group) — the group machinery is the W4 one, so the gates and
// the layout disciplines carry over. E4M3 is signed: no zero-points, no
// asymmetry mode. Production FP8 KV is often per-tensor scale; per-group
// is the finer-grained variant, so accuracy figures here flatter FP8
// slightly. The byte story (1 B/element) is identical either way.
//
// Decode is branch-free and exact in float (the R4 LUT lesson: no
// function-local tables on device):
//   e == 0:  v = m * 2^-9            (subnormal)
//   else:    v = (8 + m) * 2^(e-10)  (1 + m/8) * 2^(e-7)
// NaN encodings (e=15, m=7) are never emitted by the quantizer
// (SATFINITE caps at ±448 = 0x7E) and decode as 480 if they appear —
// caches are validated artifacts, so this is documented, not handled.
//
// Quantizer rule: group scale s = amax / 448 (no saturation, same ethos
// as the MX rung), stored FP16, elements RNE via the CUDA conversion.
// Scale overflow/underflow-to-zero on a nonzero group throws, mirroring
// the R1 contract.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "pack.hpp"

namespace w4::kv8 {

struct KVCacheF8 {
  int H = 0, S = 0, D = 0, gs = 0, G = 0;
  std::vector<uint8_t> q_k, q_v;  // H slabs of S * D E4M3 bytes
  std::vector<__half> s_k, s_v;   // H slabs of S * G FP16 scales
};

W4_HD inline float decode_f8_e4m3(uint8_t b) {
  const int e = (b >> 3) & 0xF, m = b & 7;
  const float mag = (e == 0) ? ldexpf((float)m, -9)
                             : ldexpf((float)(8 + m), e - 10);
  return (b & 0x80) ? -mag : mag;
}

// The one dequant step, host and device (same contract as dequant_one:
// identical arithmetic on both sides so the bitwise gate applies).
W4_HD inline __half dequant_one_f8(__half s, uint8_t b) {
  return __float2half_rn(__half2float(s) * decode_f8_e4m3(b));
}

inline void validate_cache_f8(const KVCacheF8& c) {
  if (c.H < 1 || c.S < 1 || c.D < 1)
    throw std::invalid_argument("cache f8: bad dims");
  if (c.gs < 8 || c.gs % 8 != 0)
    throw std::invalid_argument("cache f8: gs must be a multiple of 8");
  if ((int64_t)c.G != ((int64_t)c.D + c.gs - 1) / c.gs)
    throw std::invalid_argument("cache f8: G inconsistent with D, gs");
  const size_t qb = (size_t)c.H * c.S * c.D;
  const size_t sw = (size_t)c.H * c.S * c.G;
  if (c.q_k.size() != qb || c.q_v.size() != qb)
    throw std::invalid_argument("cache f8: q size");
  if (c.s_k.size() != sw || c.s_v.size() != sw)
    throw std::invalid_argument("cache f8: scales size");
}

inline KVCacheF8 quantize_cache_f8(const std::vector<double>& k,
                                   const std::vector<double>& v, int H, int S,
                                   int D, int gs) {
  if (H < 1 || S < 1 || D < 1)
    throw std::invalid_argument("quantize_cache_f8: bad dims");
  if (gs < 8 || gs % 8 != 0)
    throw std::invalid_argument("quantize_cache_f8: gs must be a multiple of 8");
  if (k.size() != (size_t)H * S * D || v.size() != (size_t)H * S * D)
    throw std::invalid_argument("quantize_cache_f8: size");
  for (const double x : k)
    if (!std::isfinite(x))
      throw std::invalid_argument("quantize_cache_f8: non-finite input");
  for (const double x : v)
    if (!std::isfinite(x))
      throw std::invalid_argument("quantize_cache_f8: non-finite input");
  KVCacheF8 c;
  c.H = H;
  c.S = S;
  c.D = D;
  c.gs = gs;
  c.G = (int)(((int64_t)D + gs - 1) / gs);
  c.q_k.assign((size_t)H * S * D, 0);
  c.q_v.assign((size_t)H * S * D, 0);
  c.s_k.assign((size_t)H * S * c.G, __float2half_rn(0.0f));
  c.s_v.assign((size_t)H * S * c.G, __float2half_rn(0.0f));

  for (int side = 0; side < 2; ++side) {
    const std::vector<double>& src = side ? v : k;
    std::vector<uint8_t>& qc = side ? c.q_v : c.q_k;
    std::vector<__half>& sc = side ? c.s_v : c.s_k;
    for (int h = 0; h < H; ++h) {
      const double* base = src.data() + (size_t)h * S * D;
      const size_t off_q = (size_t)h * S * D;
      const size_t off_s = (size_t)h * S * c.G;
      for (int t = 0; t < S; ++t) {
        for (int g = 0; g < c.G; ++g) {
          const int d0 = g * gs, d1 = std::min(D, d0 + gs);
          double amax = 0.0;
          for (int d = d0; d < d1; ++d)
            amax = std::max(amax, std::fabs(base[(size_t)t * D + d]));
          const double s = amax > 0.0 ? amax / 448.0 : 1.0;
          const __half sh = __float2half_rn((float)s);
          const float sf = __half2float(sh);
          if (!std::isfinite(sf) || (sf == 0.0f && amax > 0.0))
            throw std::invalid_argument(
                "quantize_cache_f8: group scale outside FP16 range");
          sc[off_s + (size_t)t * c.G + g] = sh;
          for (int d = d0; d < d1; ++d) {
            const float x = (float)(base[(size_t)t * D + d] / (double)sf);
            qc[off_q + (size_t)t * D + d] =
                (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE,
                                               __NV_E4M3);
          }
        }
      }
    }
  }
  return c;
}

// Dequantize one head's slab through the shared step.
inline std::vector<__half> dequant_head_f8(const KVCacheF8& c, int h,
                                           bool vside) {
  validate_cache_f8(c);
  if (h < 0 || h >= c.H) throw std::invalid_argument("dequant_head_f8: h");
  const size_t off_q = (size_t)h * c.S * c.D;
  const size_t off_s = (size_t)h * c.S * c.G;
  const auto& qs = vside ? c.q_v : c.q_k;
  const auto& ss = vside ? c.s_v : c.s_k;
  std::vector<__half> w((size_t)c.S * c.D);
  for (int t = 0; t < c.S; ++t)
    for (int d = 0; d < c.D; ++d)
      w[(size_t)t * c.D + d] =
          dequant_one_f8(ss[off_s + (size_t)t * c.G + d / c.gs],
                         qs[off_q + (size_t)t * c.D + d]);
  return w;
}

// FP64 reference: decode attention over the dequantized (FP16-exact)
// cache. Same discipline as attention_reference in kvcache.hpp.
inline std::vector<double> attention_reference_f8(
    const std::vector<__half>& q, const KVCacheF8& c) {
  validate_cache_f8(c);
  if (q.size() != (size_t)c.H * c.D)
    throw std::invalid_argument("attention_reference_f8: q size");
  std::vector<double> out((size_t)c.H * c.D, 0.0);
  const double inv_sqrt_d = 1.0 / std::sqrt((double)c.D);
  std::vector<double> scores(c.S);
  for (int h = 0; h < c.H; ++h) {
    const auto kdq = dequant_head_f8(c, h, false);
    const auto vdq = dequant_head_f8(c, h, true);
    double smax = -INFINITY;
    for (int t = 0; t < c.S; ++t) {
      double acc = 0.0;
      for (int d = 0; d < c.D; ++d)
        acc += (double)__half2float(q[(size_t)h * c.D + d]) *
               (double)__half2float(kdq[(size_t)t * c.D + d]);
      scores[t] = acc * inv_sqrt_d;
      smax = std::max(smax, scores[t]);
    }
    double denom = 0.0;
    for (int t = 0; t < c.S; ++t) {
      scores[t] = std::exp(scores[t] - smax);
      denom += scores[t];
    }
    for (int t = 0; t < c.S; ++t) scores[t] /= denom;
    for (int d = 0; d < c.D; ++d) {
      double acc = 0.0;
      for (int t = 0; t < c.S; ++t)
        acc += scores[t] * (double)__half2float(vdq[(size_t)t * c.D + d]);
      out[(size_t)h * c.D + d] = acc;
    }
  }
  return out;
}

}  // namespace w4::kv8
