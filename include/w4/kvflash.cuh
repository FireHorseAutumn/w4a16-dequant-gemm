#pragma once

// Split-S ("flash-decoding") decode attention (R5.5): the S <= 4096 bound
// of kvattention.cuh was the static scores array, not a law. Here the cache
// is processed in tiles of FLASH_TILE positions with an online softmax:
// phase 1 (one block per (head, tile)) computes the tile max, tile sum, and
// the tile's weighted V accumulation; phase 2 (one block per head) rescales
// and combines the partials. Nothing per-S lives in shared memory, so S is
// bounded only by the grid (65535 tiles) — and the grid is H x tiles, so
// the H=8 occupancy problem of the single-block-per-head kernel is gone
// too (8 x tiles blocks instead of 8).
//
// The dequant-at-load craft is unchanged: the W4 phase 1 reads packed words
// and calls dequant_one exactly like attention_w4, and the k_out/v_out
// hooks still reproduce dequant_head bitwise (every position is touched
// exactly once). The numerics differ from attention_w4 only in reduction
// order (tile max/sum, then a global rescale in FP32), so the cross-check
// tolerance is 2e-3, same as any two gated paths.
//
// Scratch: phase 1 writes per-tile partials (m, l, acc[D]) into a cached
// global buffer grown on demand; like x_scratch it is shared across calls,
// so calls on different streams must not be in flight concurrently.
//
// Bounds (hard throws at entry): D <= KV_MAX_D (the shared query stage),
// S <= 65535 * FLASH_TILE (grid.y). No other S limit.

#include <cuda_fp16.h>
#include <math_constants.h>

#include "kvattention.cuh"
#include "kvfp8.hpp"

namespace w4::gpu {

constexpr int FLASH_TILE = 256;

inline void validate_kv_flash(const DeviceKVW4& p) {
  if (p.H < 1 || p.S < 1 || p.D < 1)
    throw std::invalid_argument("kv flash: bad dims");
  if (p.D > KV_MAX_D)
    throw std::invalid_argument("kv flash: D exceeds the shared-memory bound");
  if (p.gs < 8 || p.gs % 8 != 0)
    throw std::invalid_argument("kv flash: gs must be a multiple of 8");
  if (p.G != (p.D + p.gs - 1) / p.gs)
    throw std::invalid_argument("kv flash: G inconsistent with D, gs");
  if (!p.q_k || !p.s_k || !p.q_v || !p.s_v)
    throw std::invalid_argument("kv flash: null pointer");
  if (!p.symmetric && (!p.z_k || !p.z_v))
    throw std::invalid_argument("kv flash: asymmetric without zeros");
  if (p.symmetric && (p.z_k || p.z_v))
    throw std::invalid_argument("kv flash: symmetric carries no zeros");
  if (p.S > 65535 * FLASH_TILE)
    throw std::invalid_argument("kv flash: S exceeds the tile-grid bound");
}

// Cached partials buffer, grown on demand; never shrunk. Shared across
// calls: same stream contract as x_scratch in fused.cuh.
inline float* kv_part_scratch(size_t nfloats) {
  static float* buf = nullptr;
  static size_t cap = 0;
  if (nfloats > cap) {
    if (buf) cudaFree(buf);
    cuda_check(cudaMalloc(&buf, nfloats * sizeof(float)), "kv part scratch");
    cap = nfloats;
  }
  return buf;
}

// W4 phase 1: per (head, tile), scores via warp-per-position packed-word
// walk (dequant at load), tile max/sum, weighted V accumulate.
__global__ void flash_w4_phase1(const DeviceKVW4 p,
                                const __half* __restrict__ q,
                                float* __restrict__ part,
                                __half* __restrict__ k_out,
                                __half* __restrict__ v_out) {
  __shared__ float q_sh[KV_MAX_D];
  __shared__ float sc[FLASH_TILE];
  __shared__ float acc[KV_MAX_D];
  __shared__ float red[8];
  const int h = blockIdx.x, tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int S = p.S, D = p.D;
  const int wd = words(D);
  const int t0 = tile * FLASH_TILE;
  const int t1 = min(S, t0 + FLASH_TILE);
  const float inv_sqrt_d = rsqrtf((float)D);

  for (int d = tid; d < D; d += 256) {
    q_sh[d] = __half2float(q[(size_t)h * D + d]);
    acc[d] = 0.0f;
  }
  __syncthreads();

  const size_t off_q = (size_t)h * S * wd;
  const size_t off_s = (size_t)h * S * p.G;
  const size_t off_z = (size_t)h * S * words(p.G);
  const size_t off_e = (size_t)h * S * D;

  // Scores for this tile (scaled), stored in shared.
  for (int t = t0 + wid; t < t1; t += 8) {
    float a = 0.0f;
    for (int w = lane; w < wd; w += 32) {
      const int k0 = 8 * w;
      const int g = k0 / p.gs;
      const __half s = p.s_k[off_s + (size_t)t * p.G + g];
      const int z =
          p.symmetric
              ? 8
              : (int)((p.z_k[off_z + (size_t)t * words(p.G) + g / 8] >>
                       (4 * (g % 8))) &
                      0xF);
      const uint32_t qw = p.q_k[off_q + (size_t)t * wd + w];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int k = k0 + j;
        if (k >= D) break;
        const __half wj = dequant_one(s, (int)((qw >> (4 * j)) & 0xF), z);
        if (k_out) k_out[off_e + (size_t)t * D + k] = wj;
        a += q_sh[k] * __half2float(wj);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      a += __shfl_down_sync(0xffffffffu, a, o);
    if (lane == 0) sc[t - t0] = a * inv_sqrt_d;
  }
  __syncthreads();

  // Tile max (block reduction through red[8]).
  {
    float m = -CUDART_INF_F;
    for (int t = wid * 32 + lane; t < t1 - t0; t += 256) m = fmaxf(m, sc[t]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    if (tid == 0) {
      float mm = red[0];
#pragma unroll
      for (int i = 1; i < 8; ++i) mm = fmaxf(mm, red[i]);
      red[0] = mm;
    }
    __syncthreads();
  }
  const float m = red[0];
  // Probabilities in place, tile sum.
  float l = 0.0f;
  for (int t = tid; t < t1 - t0; t += 256) {
    const float e = __expf(sc[t] - m);
    sc[t] = e;
    l += e;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    l += __shfl_down_sync(0xffffffffu, l, o);
  __syncthreads();
  if (lane == 0) red[wid] = l;
  __syncthreads();
  if (tid == 0) {
    float ss = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) ss += red[i];
    red[1] = ss;
  }
  __syncthreads();

  // Weighted V accumulate for this tile; each thread owns its d's.
  for (int d = tid; d < D; d += 256) {
    const int w = d / 8, j = d % 8, g = d / p.gs;
    float a = 0.0f;
    for (int t = t0; t < t1; ++t) {
      const __half s = p.s_v[off_s + (size_t)t * p.G + g];
      const int z =
          p.symmetric
              ? 8
              : (int)((p.z_v[off_z + (size_t)t * words(p.G) + g / 8] >>
                       (4 * (g % 8))) &
                      0xF);
      const uint32_t qw = p.q_v[off_q + (size_t)t * wd + w];
      const __half vj = dequant_one(s, (int)((qw >> (4 * j)) & 0xF), z);
      if (v_out) v_out[off_e + (size_t)t * D + d] = vj;
      a += sc[t - t0] * __half2float(vj);
    }
    acc[d] = a;
  }
  __syncthreads();

  // Partials: [m, l, acc[D]] per (head, tile).
  float* base = part + ((size_t)h * gridDim.y + tile) * (D + 2);
  if (tid == 0) {
    base[0] = m;
    base[1] = red[1];
  }
  for (int d = tid; d < D; d += 256) base[2 + d] = acc[d];
}

// FP16 phase 1: same tile structure over an already-FP16 cache.
__global__ void flash_fp16_phase1(const __half* __restrict__ q,
                                  const __half* __restrict__ k16,
                                  const __half* __restrict__ v16,
                                  float* __restrict__ part, const int S,
                                  const int D) {
  __shared__ float q_sh[KV_MAX_D];
  __shared__ float sc[FLASH_TILE];
  __shared__ float acc[KV_MAX_D];
  __shared__ float red[8];
  const int h = blockIdx.x, tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int t0 = tile * FLASH_TILE;
  const int t1 = min(S, t0 + FLASH_TILE);
  const float inv_sqrt_d = rsqrtf((float)D);

  for (int d = tid; d < D; d += 256) {
    q_sh[d] = __half2float(q[(size_t)h * D + d]);
    acc[d] = 0.0f;
  }
  __syncthreads();

  const __half* kh = k16 + (size_t)h * S * D;
  for (int t = t0 + wid; t < t1; t += 8) {
    float a = 0.0f;
    for (int d = lane; d < D; d += 32)
      a += q_sh[d] * __half2float(kh[(size_t)t * D + d]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      a += __shfl_down_sync(0xffffffffu, a, o);
    if (lane == 0) sc[t - t0] = a * inv_sqrt_d;
  }
  __syncthreads();

  {
    float m = -CUDART_INF_F;
    for (int t = wid * 32 + lane; t < t1 - t0; t += 256) m = fmaxf(m, sc[t]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    if (tid == 0) {
      float mm = red[0];
#pragma unroll
      for (int i = 1; i < 8; ++i) mm = fmaxf(mm, red[i]);
      red[0] = mm;
    }
    __syncthreads();
  }
  const float m = red[0];
  float l = 0.0f;
  for (int t = tid; t < t1 - t0; t += 256) {
    const float e = __expf(sc[t] - m);
    sc[t] = e;
    l += e;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    l += __shfl_down_sync(0xffffffffu, l, o);
  __syncthreads();
  if (lane == 0) red[wid] = l;
  __syncthreads();
  if (tid == 0) {
    float ss = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) ss += red[i];
    red[1] = ss;
  }
  __syncthreads();

  const __half* vh = v16 + (size_t)h * S * D;
  for (int d = tid; d < D; d += 256) {
    float a = 0.0f;
    for (int t = t0; t < t1; ++t)
      a += sc[t - t0] * __half2float(vh[(size_t)t * D + d]);
    acc[d] = a;
  }
  __syncthreads();

  float* base = part + ((size_t)h * gridDim.y + tile) * (D + 2);
  if (tid == 0) {
    base[0] = m;
    base[1] = red[1];
  }
  for (int d = tid; d < D; d += 256) base[2 + d] = acc[d];
}

// Phase 2 (shared by both cache types): per head, rescale and combine the
// tile partials. exp(m_t - M) is recomputed per (d, tile) — T is small
// against D at decode shapes, and the shared-staged variant is a measured
// question, not a guessed one.
__global__ void flash_phase2(const float* __restrict__ part,
                             __half* __restrict__ out, const int T,
                             const int D) {
  __shared__ float red[8];
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const size_t stride = (size_t)D + 2;
  const float* hb = part + (size_t)h * T * stride;

  // Global max over tile maxima.
  float m = -CUDART_INF_F;
  for (int t = tid; t < T; t += 256) m = fmaxf(m, hb[(size_t)t * stride]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
  if (lane == 0) red[wid] = m;
  __syncthreads();
  if (tid == 0) {
    float mm = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) mm = fmaxf(mm, red[i]);
    red[0] = mm;
  }
  __syncthreads();
  const float M = red[0];
  __syncthreads();

  // Denominator: sum of exp(m_t - M) * l_t.
  float den = 0.0f;
  for (int t = tid; t < T; t += 256)
    den += __expf(hb[(size_t)t * stride] - M) * hb[(size_t)t * stride + 1];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    den += __shfl_down_sync(0xffffffffu, den, o);
  if (lane == 0) red[wid] = den;
  __syncthreads();
  if (tid == 0) {
    float ss = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) ss += red[i];
    red[0] = ss;
  }
  __syncthreads();
  const float inv = 1.0f / red[0];

  for (int d = tid; d < D; d += 256) {
    float a = 0.0f;
    for (int t = 0; t < T; ++t) {
      const float* pb = hb + (size_t)t * stride;
      a += __expf(pb[0] - M) * pb[2 + d];
    }
    out[(size_t)h * D + d] = __float2half_rn(a * inv);
  }
}

inline void attention_w4_flash(const DeviceKVW4& p, const __half* q,
                               __half* out, cudaStream_t stream = 0,
                               __half* k_out = nullptr,
                               __half* v_out = nullptr) {
  validate_kv_flash(p);
  if (!q || !out) throw std::invalid_argument("kv flash w4: null pointer");
  const int T = (p.S + FLASH_TILE - 1) / FLASH_TILE;
  float* part = kv_part_scratch((size_t)p.H * T * (p.D + 2));
  const dim3 grid((unsigned)p.H, (unsigned)T);
  flash_w4_phase1<<<grid, 256, 0, stream>>>(p, q, part, k_out, v_out);
  cuda_check(cudaGetLastError(), "flash_w4_phase1");
  flash_phase2<<<p.H, 256, 0, stream>>>(part, out, T, p.D);
  cuda_check(cudaGetLastError(), "flash_phase2");
}

inline void attention_fp16_flash(const __half* q, const __half* k16,
                                 const __half* v16, __half* out, int H, int S,
                                 int D, cudaStream_t stream = 0) {
  if (H < 1 || S < 1 || D < 1)
    throw std::invalid_argument("kv flash fp16: bad dims");
  if (D > KV_MAX_D)
    throw std::invalid_argument(
        "kv flash fp16: D exceeds the shared-memory bound");
  if (S > 65535 * FLASH_TILE)
    throw std::invalid_argument("kv flash fp16: S exceeds the tile-grid bound");
  if (!q || !k16 || !v16 || !out)
    throw std::invalid_argument("kv flash fp16: null pointer");
  const int T = (S + FLASH_TILE - 1) / FLASH_TILE;
  float* part = kv_part_scratch((size_t)H * T * (D + 2));
  const dim3 grid((unsigned)H, (unsigned)T);
  flash_fp16_phase1<<<grid, 256, 0, stream>>>(q, k16, v16, part, S, D);
  cuda_check(cudaGetLastError(), "flash_fp16_phase1");
  flash_phase2<<<H, 256, 0, stream>>>(part, out, T, D);
  cuda_check(cudaGetLastError(), "flash_phase2");
}

// FP8 (E4M3) cache, flash path (R6). Same split-S structure as the W4
// flash kernel; the score walk reads one uint32 per lane (four E4M3
// bytes), gs % 8 == 0 keeps every word inside one group. No zero-points:
// E4M3 is signed. Hooks: k_out/v_out reproduce dequant_head_f8 bitwise.
struct DeviceKVF8 {
  int H, S, D, gs, G;
  const uint8_t* q_k;
  const __half* s_k;
  const uint8_t* q_v;
  const __half* s_v;
};

inline void validate_kv_f8(const DeviceKVF8& p) {
  if (p.H < 1 || p.S < 1 || p.D < 1)
    throw std::invalid_argument("kv f8 flash: bad dims");
  if (p.D > KV_MAX_D)
    throw std::invalid_argument("kv f8 flash: D exceeds the shared bound");
  if (p.gs < 8 || p.gs % 8 != 0)
    throw std::invalid_argument("kv f8 flash: gs must be a multiple of 8");
  if (p.G != (p.D + p.gs - 1) / p.gs)
    throw std::invalid_argument("kv f8 flash: G inconsistent with D, gs");
  if (!p.q_k || !p.s_k || !p.q_v || !p.s_v)
    throw std::invalid_argument("kv f8 flash: null pointer");
  if (p.S > 65535 * FLASH_TILE)
    throw std::invalid_argument("kv f8 flash: S exceeds the tile-grid bound");
}

// Naive per-head FP8 dequant (the two-pass denominator). One thread per
// 4-element word.
__global__ void dequant_kernel_f8(const uint8_t* __restrict__ q,
                                  const __half* __restrict__ s,
                                  __half* __restrict__ w, const int S,
                                  const int D, const int gs) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int wpr = (D + 3) / 4;
  if (idx >= S * wpr) return;
  const int t = idx / wpr;
  const int d0 = 4 * (idx % wpr);
  const int G = (D + gs - 1) / gs;
  const __half sc = s[(size_t)t * G + d0 / gs];
  const uint32_t qw = ((const uint32_t*)(q + (size_t)t * D))[idx % wpr];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (d0 + j >= D) break;
    w[(size_t)t * D + d0 + j] =
        kv8::dequant_one_f8(sc, (uint8_t)((qw >> (8 * j)) & 0xFF));
  }
}

inline void dequant_f8(const uint8_t* q, const __half* s, __half* w, int S,
                       int D, int gs, cudaStream_t stream = 0) {
  if (S < 1 || D < 1)
    throw std::invalid_argument("f8 dequant: bad dims");
  if (gs < 8 || gs % 8 != 0)
    throw std::invalid_argument("f8 dequant: gs must be a multiple of 8");
  if (!q || !s || !w) throw std::invalid_argument("f8 dequant: null pointer");
  // The kernel indexes with 32-bit ints; keep the whole launch inside that.
  const int64_t total = (int64_t)S * ((D + 3) / 4);
  if (total >= (int64_t{1} << 31))
    throw std::invalid_argument("f8 dequant: S*words4(D) exceeds 2^31");
  // 4-byte word loads require 4 B alignment of each row start: D % 4 or an
  // aligned base. cudaMalloc bases are 256 B-aligned; D % 4 != 0 would
  // misalign row starts, so reject it here.
  if (D % 4 != 0)
    throw std::invalid_argument("f8 dequant: D must be a multiple of 4");
  const int block = 256;
  dequant_kernel_f8<<<(unsigned)((total + block - 1) / block), block, 0,
                      stream>>>(q, s, w, S, D, gs);
  cuda_check(cudaGetLastError(), "dequant_kernel_f8");
}

__global__ void flash_f8_phase1(const DeviceKVF8 p,
                                const __half* __restrict__ q,
                                float* __restrict__ part,
                                __half* __restrict__ k_out,
                                __half* __restrict__ v_out) {
  __shared__ float q_sh[KV_MAX_D];
  __shared__ float sc[FLASH_TILE];
  __shared__ float acc[KV_MAX_D];
  __shared__ float red[8];
  const int h = blockIdx.x, tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int S = p.S, D = p.D;
  const int t0 = tile * FLASH_TILE;
  const int t1 = min(S, t0 + FLASH_TILE);
  const float inv_sqrt_d = rsqrtf((float)D);
  const int wd = (D + 3) / 4;  // 4 E4M3 bytes per word

  for (int d = tid; d < D; d += 256) {
    q_sh[d] = __half2float(q[(size_t)h * D + d]);
    acc[d] = 0.0f;
  }
  __syncthreads();

  const size_t off_q = (size_t)h * S * D;
  const size_t off_s = (size_t)h * S * p.G;

  for (int t = t0 + wid; t < t1; t += 8) {
    float a = 0.0f;
    for (int w = lane; w < wd; w += 32) {
      const int d0 = 4 * w;
      const int g = d0 / p.gs;
      const __half s = p.s_k[off_s + (size_t)t * p.G + g];
      const uint32_t qw =
          ((const uint32_t*)(p.q_k + off_q + (size_t)t * D))[w];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = d0 + j;
        if (k >= D) break;
        const __half wj =
            kv8::dequant_one_f8(s, (uint8_t)((qw >> (8 * j)) & 0xFF));
        if (k_out) k_out[off_q + (size_t)t * D + k] = wj;
        a += q_sh[k] * __half2float(wj);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      a += __shfl_down_sync(0xffffffffu, a, o);
    if (lane == 0) sc[t - t0] = a * inv_sqrt_d;
  }
  __syncthreads();

  {
    float m = -CUDART_INF_F;
    for (int t = wid * 32 + lane; t < t1 - t0; t += 256) m = fmaxf(m, sc[t]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    if (tid == 0) {
      float mm = red[0];
#pragma unroll
      for (int i = 1; i < 8; ++i) mm = fmaxf(mm, red[i]);
      red[0] = mm;
    }
    __syncthreads();
  }
  const float m = red[0];
  float l = 0.0f;
  for (int t = tid; t < t1 - t0; t += 256) {
    const float e = __expf(sc[t] - m);
    sc[t] = e;
    l += e;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    l += __shfl_down_sync(0xffffffffu, l, o);
  __syncthreads();
  if (lane == 0) red[wid] = l;
  __syncthreads();
  if (tid == 0) {
    float ss = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) ss += red[i];
    red[1] = ss;
  }
  __syncthreads();

  for (int d = tid; d < D; d += 256) {
    const int w = d / 4, j = d % 4, g = d / p.gs;
    float a = 0.0f;
    for (int t = t0; t < t1; ++t) {
      const __half s = p.s_v[off_s + (size_t)t * p.G + g];
      const uint32_t qw =
          ((const uint32_t*)(p.q_v + off_q + (size_t)t * D))[w];
      const __half vj =
          kv8::dequant_one_f8(s, (uint8_t)((qw >> (8 * j)) & 0xFF));
      if (v_out) v_out[off_q + (size_t)t * D + d] = vj;
      a += sc[t - t0] * __half2float(vj);
    }
    acc[d] = a;
  }
  __syncthreads();

  float* base = part + ((size_t)h * gridDim.y + tile) * (D + 2);
  if (tid == 0) {
    base[0] = m;
    base[1] = red[1];
  }
  for (int d = tid; d < D; d += 256) base[2 + d] = acc[d];
}

inline void attention_f8_flash(const DeviceKVF8& p, const __half* q,
                               __half* out, cudaStream_t stream = 0,
                               __half* k_out = nullptr,
                               __half* v_out = nullptr) {
  validate_kv_f8(p);
  if (!q || !out) throw std::invalid_argument("kv f8 flash: null io pointer");
  // The 4-byte word walk requires 4 B-aligned row starts.
  if (p.D % 4 != 0)
    throw std::invalid_argument("kv f8 flash: D must be a multiple of 4");
  const int T = (p.S + FLASH_TILE - 1) / FLASH_TILE;
  float* part = kv_part_scratch((size_t)p.H * T * (p.D + 2));
  const dim3 grid((unsigned)p.H, (unsigned)T);
  flash_f8_phase1<<<grid, 256, 0, stream>>>(p, q, part, k_out, v_out);
  cuda_check(cudaGetLastError(), "flash_f8_phase1");
  flash_phase2<<<p.H, 256, 0, stream>>>(part, out, T, p.D);
  cuda_check(cudaGetLastError(), "flash_phase2");
}

}  // namespace w4::gpu
