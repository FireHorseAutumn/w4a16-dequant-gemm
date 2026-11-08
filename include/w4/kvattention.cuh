#pragma once

// KV-cache-quantized decode attention (R5), device side. One decode token:
// per head, scores s_t = dot(q, k_t)/sqrt(D) over S cached positions,
// max-subtracted softmax, output o = sum_t p_t * v_t.
//
// attention_w4 dequantizes K and V at load via dequant_one — the FP16
// cache never materializes, so the K/V streams carry 0.5 bytes per element
// against the FP16 baseline's 2. attention_fp16 is the same kernel shape
// over an already-FP16 cache: the honest baseline, isolating the cache
// bandwidth win from the kernel structure.
//
// Mapping: one block per head, 256 threads. Scores: warp per position,
// lanes stride D (packed words of 8 on the W4 path) with a shuffle
// reduction; softmax does two block-wide reductions (max, then sum)
// through an 8-slot shared array; output: threads stride D, each summing
// over all positions with scores broadcast from shared. The query is
// staged in shared as float once — every position re-reads it.
//
// Bounds (hard throws at entry): S <= 4096 (scores live in a static shared
// array, 16 KB), D <= 512 (the shared query stage, 2 KB). Both are
// decode-realistic; larger S wants flash-style online softmax over tiles,
// which is a different kernel, not a bigger array.
//
// k_out / v_out (test hook): when non-null, attention_w4 also stores each
// dequantized cache element, reproducing dequant_all per head bitwise.

#include <cuda_fp16.h>
#include <math_constants.h>

#include "dequant.cuh"
#include "kvcache.hpp"

namespace w4::gpu {

constexpr int KV_MAX_S = 4096;
constexpr int KV_MAX_D = 512;

struct DeviceKVW4 {
  int H, S, D, gs, G;
  int symmetric;
  const uint32_t* q_k;
  const __half* s_k;
  const uint32_t* z_k;
  const uint32_t* q_v;
  const __half* s_v;
  const uint32_t* z_v;
};

inline void validate_kv(const DeviceKVW4& p) {
  if (p.H < 1 || p.S < 1 || p.D < 1)
    throw std::invalid_argument("kv: bad dims");
  if (p.S > KV_MAX_S)
    throw std::invalid_argument("kv: S exceeds the shared-memory bound");
  if (p.D > KV_MAX_D)
    throw std::invalid_argument("kv: D exceeds the shared-memory bound");
  if (p.gs < 8 || p.gs % 8 != 0)
    throw std::invalid_argument("kv: gs must be a multiple of 8");
  if ((int64_t)p.G != ((int64_t)p.D + p.gs - 1) / p.gs)
    throw std::invalid_argument("kv: G inconsistent with D, gs");
  if (!p.q_k || !p.s_k || !p.q_v || !p.s_v)
    throw std::invalid_argument("kv: null pointer");
  if (!p.symmetric && (!p.z_k || !p.z_v))
    throw std::invalid_argument("kv: asymmetric without zeros");
  if (p.symmetric && (p.z_k || p.z_v))
    throw std::invalid_argument("kv: symmetric carries no zeros");
}

// Block-wide max then sum over scores[S], in place. warp 0 lane 0 owns
// the combine; everyone re-reads the broadcast.
__device__ inline void block_softmax(float* scores, float* red, int S,
                                     int tid, float inv_sqrt_d) {
  const int lane = tid & 31, wid = tid >> 5;
  for (int t = tid; t < S; t += 256) scores[t] *= inv_sqrt_d;
  __syncthreads();
  float m = -CUDART_INF_F;
  for (int t = wid * 32 + lane; t < S; t += 256) m = fmaxf(m, scores[t]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
  if (lane == 0) red[wid] = m;
  __syncthreads();
  if (tid == 0) {
    float mm = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) mm = fmaxf(mm, red[i]);
    red[0] = mm;
  }
  __syncthreads();
  const float mx = red[0];
  float s = 0.0f;
  for (int t = wid * 32 + lane; t < S; t += 256)
    s += __expf(scores[t] - mx);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    s += __shfl_down_sync(0xffffffffu, s, off);
  __syncthreads();
  if (lane == 0) red[wid] = s;
  __syncthreads();
  if (tid == 0) {
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss += red[i];
    red[0] = ss;
  }
  __syncthreads();
  const float inv = 1.0f / red[0];
  for (int t = tid; t < S; t += 256)
    scores[t] = __expf(scores[t] - mx) * inv;
  __syncthreads();
}

__global__ void attention_fp16_kernel(const __half* __restrict__ q,
                                      const __half* __restrict__ k16,
                                      const __half* __restrict__ v16,
                                      __half* __restrict__ out, const int S,
                                      const int D) {
  __shared__ float scores[KV_MAX_S];
  __shared__ float red[8];
  __shared__ float q_sh[KV_MAX_D];
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const float inv_sqrt_d = rsqrtf((float)D);

  for (int d = tid; d < D; d += 256) q_sh[d] = __half2float(q[(size_t)h * D + d]);
  __syncthreads();

  const __half* kh = k16 + (size_t)h * S * D;
  for (int t = wid; t < S; t += 8) {
    float acc = 0.0f;
    for (int d = lane; d < D; d += 32)
      acc += q_sh[d] * __half2float(kh[(size_t)t * D + d]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) scores[t] = acc;
  }
  __syncthreads();

  block_softmax(scores, red, S, tid, inv_sqrt_d);

  const __half* vh = v16 + (size_t)h * S * D;
  for (int d = tid; d < D; d += 256) {
    float acc = 0.0f;
    for (int t = 0; t < S; ++t)
      acc += scores[t] * __half2float(vh[(size_t)t * D + d]);
    out[(size_t)h * D + d] = __float2half_rn(acc);
  }
}

__global__ void attention_w4_kernel(const DeviceKVW4 p,
                                    const __half* __restrict__ q,
                                    __half* __restrict__ out,
                                    __half* __restrict__ k_out,
                                    __half* __restrict__ v_out) {
  __shared__ float scores[KV_MAX_S];
  __shared__ float red[8];
  __shared__ float q_sh[KV_MAX_D];
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int S = p.S, D = p.D;
  const int wd = words(D);
  const float inv_sqrt_d = rsqrtf((float)D);

  for (int d = tid; d < D; d += 256) q_sh[d] = __half2float(q[(size_t)h * D + d]);
  __syncthreads();

  const size_t off_q = (size_t)h * S * wd;
  const size_t off_s = (size_t)h * S * p.G;
  const size_t off_z = (size_t)h * S * words(p.G);
  const size_t off_e = (size_t)h * S * D;

  for (int t = wid; t < S; t += 8) {
    float acc = 0.0f;
    for (int w = lane; w < wd; w += 32) {
      const int k0 = 8 * w;
      const int g = k0 / p.gs;
      const __half sc = p.s_k[off_s + (size_t)t * p.G + g];
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
        const __half wj = dequant_one(sc, (int)((qw >> (4 * j)) & 0xF), z);
        if (k_out) k_out[off_e + (size_t)t * D + k] = wj;
        acc += q_sh[k] * __half2float(wj);
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) scores[t] = acc;
  }
  __syncthreads();

  block_softmax(scores, red, S, tid, inv_sqrt_d);

  for (int d = tid; d < D; d += 256) {
    const int w = d / 8, j = d % 8, g = d / p.gs;
    float acc = 0.0f;
    for (int t = 0; t < S; ++t) {
      const __half sc = p.s_v[off_s + (size_t)t * p.G + g];
      const int z =
          p.symmetric
              ? 8
              : (int)((p.z_v[off_z + (size_t)t * words(p.G) + g / 8] >>
                       (4 * (g % 8))) &
                      0xF);
      const uint32_t qw = p.q_v[off_q + (size_t)t * wd + w];
      const __half vj = dequant_one(sc, (int)((qw >> (4 * j)) & 0xF), z);
      if (v_out) v_out[off_e + (size_t)t * D + d] = vj;
      acc += scores[t] * __half2float(vj);
    }
    out[(size_t)h * D + d] = __float2half_rn(acc);
  }
}

inline void attention_fp16(const __half* q, const __half* k16,
                           const __half* v16, __half* out, int H, int S,
                           int D, cudaStream_t stream = 0) {
  if (H < 1 || S < 1 || D < 1)
    throw std::invalid_argument("kv fp16: bad dims");
  if (S > KV_MAX_S)
    throw std::invalid_argument("kv fp16: S exceeds the shared-memory bound");
  if (D > KV_MAX_D)
    throw std::invalid_argument("kv fp16: D exceeds the shared-memory bound");
  if (!q || !k16 || !v16 || !out)
    throw std::invalid_argument("kv fp16: null pointer");
  attention_fp16_kernel<<<H, 256, 0, stream>>>(q, k16, v16, out, S, D);
  cuda_check(cudaGetLastError(), "attention_fp16_kernel");
}

inline void attention_w4(const DeviceKVW4& p, const __half* q, __half* out,
                         cudaStream_t stream = 0, __half* k_out = nullptr,
                         __half* v_out = nullptr) {
  validate_kv(p);
  if (!q || !out) throw std::invalid_argument("kv w4: null pointer");
  attention_w4_kernel<<<p.H, 256, 0, stream>>>(p, q, out, k_out, v_out);
  cuda_check(cudaGetLastError(), "attention_w4_kernel");
}

}  // namespace w4::gpu
