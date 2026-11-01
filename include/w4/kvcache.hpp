#pragma once

// KV-cache-quantized decode attention (R5), host side: cache format,
// quantizer, and FP64 reference. The cache is H independent head slabs of
// the existing PackedW format — positions are rows (N = S), head dim is the
// contracted axis (K = D), groups partition D — so the R1 quantizer, the
// zero-bracketing fix, and the whole gate matrix carry over unchanged.
// Contiguous layout, not paged: paging is a serving-systems concern and
// changes nothing about the dequant-at-load craft this rung exercises.
//
// Decode attention per head: one query vector q (D), scores
// s_t = dot(q, k_t) / sqrt(D) over the S cached positions, softmax with
// max subtraction, output o = sum_t p_t * v_t. The kernel side
// (kvattention.cuh) dequantizes K and V at load; FP16 cache tensors never
// materialize.

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "pack.hpp"

namespace w4::kv {

struct KVCacheW4 {
  int H = 0, S = 0, D = 0, gs = 0, G = 0;
  bool symmetric = true;
  std::vector<uint32_t> q_k, q_v;  // H slabs of S * words(D)
  std::vector<__half> s_k, s_v;    // H slabs of S * G
  std::vector<uint32_t> z_k, z_v;  // H slabs of S * words(G), asym only
};

inline void validate_cache(const KVCacheW4& c) {
  if (c.H < 1 || c.S < 1 || c.D < 1)
    throw std::invalid_argument("cache: bad dims");
  if (c.gs < 8 || c.gs % 8 != 0)
    throw std::invalid_argument("cache: gs must be a multiple of 8");
  if (c.G != (c.D + c.gs - 1) / c.gs)
    throw std::invalid_argument("cache: G inconsistent with D, gs");
  const size_t qw = (size_t)c.H * c.S * words(c.D);
  const size_t sw = (size_t)c.H * c.S * c.G;
  if (c.q_k.size() != qw || c.q_v.size() != qw)
    throw std::invalid_argument("cache: q size");
  if (c.s_k.size() != sw || c.s_v.size() != sw)
    throw std::invalid_argument("cache: scales size");
  if (!c.symmetric &&
      (c.z_k.size() != (size_t)c.H * c.S * words(c.G) ||
       c.z_v.size() != (size_t)c.H * c.S * words(c.G)))
    throw std::invalid_argument("cache: zeros size");
  if (c.symmetric && (!c.z_k.empty() || !c.z_v.empty()))
    throw std::invalid_argument("cache: symmetric carries no zeros");
}

// Quantize both sides of a cache, head by head. k and v are H*S*D,
// head-major, position-major. The per-head quantize() calls carry all of
// the R1 contracts (NaN, scale range, dims) — they are not re-checked here.
inline KVCacheW4 quantize_cache(const std::vector<double>& k,
                                const std::vector<double>& v, int H, int S,
                                int D, int gs, bool symmetric) {
  if (H < 1 || S < 1 || D < 1)
    throw std::invalid_argument("quantize_cache: bad dims");
  if (k.size() != (size_t)H * S * D || v.size() != (size_t)H * S * D)
    throw std::invalid_argument("quantize_cache: size");
  KVCacheW4 c;
  c.H = H;
  c.S = S;
  c.D = D;
  c.gs = gs;
  c.G = (D + gs - 1) / gs;
  c.symmetric = symmetric;
  const size_t qw = (size_t)H * S * words(D);
  const size_t sw = (size_t)H * S * c.G;
  c.q_k.assign(qw, 0);
  c.q_v.assign(qw, 0);
  c.s_k.assign(sw, __float2half_rn(0.0f));
  c.s_v.assign(sw, __float2half_rn(0.0f));
  if (!symmetric) {
    c.z_k.assign((size_t)H * S * words(c.G), 0);
    c.z_v.assign((size_t)H * S * words(c.G), 0);
  }
  for (int h = 0; h < H; ++h) {
    const double* kbase = k.data() + (size_t)h * S * D;
    const double* vbase = v.data() + (size_t)h * S * D;
    const PackedW pk =
        quantize(std::vector<double>(kbase, kbase + (size_t)S * D), S, D, gs,
                 symmetric);
    const PackedW pv =
        quantize(std::vector<double>(vbase, vbase + (size_t)S * D), S, D, gs,
                 symmetric);
    const size_t off_q = (size_t)h * S * words(D);
    const size_t off_s = (size_t)h * S * c.G;
    std::copy(pk.q.begin(), pk.q.end(), c.q_k.begin() + off_q);
    std::copy(pv.q.begin(), pv.q.end(), c.q_v.begin() + off_q);
    std::copy(pk.scales.begin(), pk.scales.end(), c.s_k.begin() + off_s);
    std::copy(pv.scales.begin(), pv.scales.end(), c.s_v.begin() + off_s);
    if (!symmetric) {
      const size_t off_z = (size_t)h * S * words(c.G);
      std::copy(pk.zeros.begin(), pk.zeros.end(), c.z_k.begin() + off_z);
      std::copy(pv.zeros.begin(), pv.zeros.end(), c.z_v.begin() + off_z);
    }
  }
  return c;
}

// Dequantize one head's slab through the shared host reference.
inline std::vector<__half> dequant_head(const KVCacheW4& c, int h,
                                        bool vside) {
  validate_cache(c);
  if (h < 0 || h >= c.H) throw std::invalid_argument("dequant_head: h");
  PackedW p;
  p.N = c.S;
  p.K = c.D;
  p.gs = c.gs;
  p.G = c.G;
  p.symmetric = c.symmetric;
  const size_t off_q = (size_t)h * c.S * words(c.D);
  const size_t off_s = (size_t)h * c.S * c.G;
  const size_t off_z = (size_t)h * c.S * words(c.G);
  const auto& qs = vside ? c.q_v : c.q_k;
  const auto& ss = vside ? c.s_v : c.s_k;
  p.q.assign(qs.begin() + off_q, qs.begin() + off_q + (size_t)c.S * words(c.D));
  p.scales.assign(ss.begin() + off_s,
                  ss.begin() + off_s + (size_t)c.S * c.G);
  if (!c.symmetric) {
    const auto& zs = vside ? c.z_v : c.z_k;
    p.zeros.assign(zs.begin() + off_z,
                   zs.begin() + off_z + (size_t)c.S * words(c.G));
  }
  return dequant_all(p);
}

// FP64 reference: decode attention over the dequantized (FP16-exact) cache.
// q is H*D half; output is H*D double. Max-subtracted softmax; the scores
// are FP16-input dots in double accumulation, same oracle discipline as
// gemm_oracle.
inline std::vector<double> attention_reference(const std::vector<__half>& q,
                                               const KVCacheW4& c) {
  validate_cache(c);
  if (q.size() != (size_t)c.H * c.D)
    throw std::invalid_argument("attention_reference: q size");
  std::vector<double> out((size_t)c.H * c.D, 0.0);
  const double inv_sqrt_d = 1.0 / std::sqrt((double)c.D);
  std::vector<double> scores(c.S);
  for (int h = 0; h < c.H; ++h) {
    const auto kdq = dequant_head(c, h, false);
    const auto vdq = dequant_head(c, h, true);
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

}  // namespace w4::kv
