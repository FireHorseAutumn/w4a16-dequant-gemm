#pragma once

// MXFP4 GPU kernels (R4): naive dequant for the two-pass baseline, fused
// dequant+GEMM, and fused dequant+WMMA GEMM. The dequant step is
// dequant_one_mx (LUT + exponent shift, no zero-point); everything else —
// thread mapping, x staging, dispatch, store guards — follows the W4
// skeletons in fused.cuh / mma.cuh, which stay untouched.
//
// Scale layout: one E8M0 byte per (row, 32-element block). Words hold 8
// nibbles along K and 32 % 8 == 0, so a packed word never straddles a block
// boundary and the scale index of word w is w / 4.
//
// w_out (test hook): same contract as the W4 kernels — when non-null the
// fused/MMA kernels also store each dequantized weight, reproducing
// dequant_all_mx bitwise; racy-but-byte-identical across repeated m-tiles.

#include <cuda_fp16.h>
#include <mma.h>

#include "fused.cuh"
#include "mma.cuh"
#include "mx.hpp"

namespace w4::gpu {

struct DevicePackedWm {
  int N, K, B;  // B = mx::blocks(K)
  const uint32_t* q;
  const uint8_t* scales;
};

inline void validate_mx(const DevicePackedWm& p) {
  if (p.N < 1 || p.K < 1) throw std::invalid_argument("mx: bad dims");
  if (p.B != mx::blocks(p.K))
    throw std::invalid_argument("mx: B inconsistent with K");
  if (!p.q || !p.scales) throw std::invalid_argument("mx: null pointer");
}

// One thread per packed word; the naive materialization the fused kernels
// are measured against.
__global__ void dequant_kernel_mx(DevicePackedWm p, __half* w) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int wpr = words(p.K);
  if (idx >= p.N * wpr) return;
  const int n = idx / wpr;
  const int k0 = 8 * (idx % wpr);
  const uint8_t se = p.scales[(size_t)n * p.B + k0 / mx::MX_BLOCK];
  const uint32_t qw = p.q[idx];
  __half* out = w + (size_t)n * p.K + k0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (k0 + j >= p.K) break;
    out[j] = mx::dequant_one_mx(se, (int)((qw >> (4 * j)) & 0xF));
  }
}

inline void dequant_mx(const DevicePackedWm& p, __half* w,
                       cudaStream_t stream = 0) {
  validate_mx(p);
  if (!w) throw std::invalid_argument("mx dequant: null output");
  // The kernel indexes with 32-bit ints; keep the whole launch inside that.
  const int64_t total = (int64_t)p.N * words(p.K);
  if (total >= (int64_t{1} << 31))
    throw std::invalid_argument("mx dequant: N*words(K) exceeds 2^31");
  const int block = 256;
  dequant_kernel_mx<<<(unsigned)((total + block - 1) / block), block, 0,
                      stream>>>(p, w);
  cuda_check(cudaGetLastError(), "dequant_kernel_mx");
}

// SIMT fused path: identical mapping to fused_kernel (warp per R rows,
// lanes stride packed words along K, x pre-converted to FP32 once per
// call); only the scale fetch and dequant step differ.
template <int MTILE, int R>
__global__ void fused_kernel_mx(const DevicePackedWm p,
                                const float* __restrict__ xf,
                                __half* __restrict__ c, const int M,
                                __half* __restrict__ w_out) {
  const int lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
  if (n0 >= p.N) return;  // whole-warp exit: n0 is warp-uniform
  const int wpr = words(p.K);
  const int m0 = blockIdx.y * MTILE;
  const bool fast = (p.K % 8) == 0;

  float acc[R][MTILE];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int i = 0; i < MTILE; ++i) acc[r][i] = 0.0f;

  for (int w = lane; w < wpr; w += 32) {
    const int k0 = 8 * w;

    float xv[MTILE][8];
    if (fast) {
#pragma unroll
      for (int i = 0; i < MTILE; ++i) {
        const int m = m0 + i < M ? m0 + i : M - 1;
        const float4 xa =
            __ldg((const float4*)(xf + (size_t)m * p.K + k0));
        const float4 xb =
            __ldg((const float4*)(xf + (size_t)m * p.K + k0 + 4));
        xv[i][0] = xa.x;
        xv[i][1] = xa.y;
        xv[i][2] = xa.z;
        xv[i][3] = xa.w;
        xv[i][4] = xb.x;
        xv[i][5] = xb.y;
        xv[i][6] = xb.z;
        xv[i][7] = xb.w;
      }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = n0 + r;
      if (n >= p.N) continue;  // warp-uniform
      const uint8_t se = p.scales[(size_t)n * p.B + k0 / mx::MX_BLOCK];
      const uint32_t qw = p.q[(size_t)n * wpr + w];

      if (fast) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const __half wj = mx::dequant_one_mx(se, (int)((qw >> (4 * j)) & 0xF));
          if (w_out) w_out[(size_t)n * p.K + k0 + j] = wj;
          const float wf = __half2float(wj);
#pragma unroll
          for (int i = 0; i < MTILE; ++i)
            acc[r][i] = fmaf(wf, xv[i][j], acc[r][i]);
        }
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int k = k0 + j;
          if (k >= p.K) break;
          const __half wj = mx::dequant_one_mx(se, (int)((qw >> (4 * j)) & 0xF));
          if (w_out) w_out[(size_t)n * p.K + k] = wj;
          const float wf = __half2float(wj);
#pragma unroll
          for (int i = 0; i < MTILE; ++i) {
            const int m = m0 + i < M ? m0 + i : M - 1;
            acc[r][i] = fmaf(wf, __ldg(xf + (size_t)m * p.K + k), acc[r][i]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int n = n0 + r;
    if (n >= p.N) continue;  // warp-uniform
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
      for (int i = 0; i < MTILE; ++i)
        acc[r][i] += __shfl_down_sync(0xffffffffu, acc[r][i], off);
    if (lane == 0)
#pragma unroll
      for (int i = 0; i < MTILE; ++i)
        if (m0 + i < M) c[(size_t)(m0 + i) * p.N + n] = __float2half_rn(acc[r][i]);
  }
}

inline void gemm_fused_mx(const DevicePackedWm& p, const __half* x, __half* c,
                          int M, cudaStream_t stream = 0,
                          __half* w_out = nullptr) {
  validate_mx(p);
  if (!x || !c) throw std::invalid_argument("mx fused: null pointer");
  if (M < 1) throw std::invalid_argument("mx fused: bad M");
  const int mtile = M <= 1 ? 1 : (M <= 2 ? 2 : 4);
  const int rpw = M <= 4 ? 1 : 4;
  // Same launch bounds as gemm_fused: gridDim.y caps at 65535 (the
  // m-tiling bounds M); on gridDim.x only the ceil-rounding add can
  // overflow, N being an int.
  if (M > 65535 * mtile || p.N > INT32_MAX - 8 * rpw)
    throw std::invalid_argument("mx fused: launch grid out of range");

  float* xf = x_scratch((size_t)M * p.K);
  x_to_f32_kernel<<<(unsigned)(((size_t)M * p.K + 255) / 256), 256, 0,
                    stream>>>(x, xf, (size_t)M * p.K);
  cuda_check(cudaGetLastError(), "x_to_f32_kernel");

  const dim3 grid((unsigned)((p.N + 8 * rpw - 1) / (8 * rpw)),
                  (unsigned)((M + mtile - 1) / mtile));
  const int block = 256;
  if (mtile == 1) {
    fused_kernel_mx<1, 1><<<grid, block, 0, stream>>>(p, xf, c, M, w_out);
  } else if (mtile == 2) {
    fused_kernel_mx<2, 1><<<grid, block, 0, stream>>>(p, xf, c, M, w_out);
  } else if (rpw == 1) {
    fused_kernel_mx<4, 1><<<grid, block, 0, stream>>>(p, xf, c, M, w_out);
  } else {
    fused_kernel_mx<4, 4><<<grid, block, 0, stream>>>(p, xf, c, M, w_out);
  }
  cuda_check(cudaGetLastError(), "fused_kernel_mx");
}

// WMMA fused path: same 64x64 tile / 32-wide K chunk structure as
// mma_kernel, dequant step swapped for dequant_one_mx. A chunk is exactly
// one MX block per row, so each staged word reads one scale byte.
__global__ void mma_kernel_mx(const DevicePackedWm p,
                              const __half* __restrict__ x,
                              __half* __restrict__ c, const int M,
                              __half* __restrict__ w_out) {
  __shared__ __align__(32) __half ws[MMA_TILE_N][MMA_CHUNK_K];
  __shared__ __align__(32) __half xs[MMA_TILE_M][MMA_CHUNK_K];
  __shared__ __align__(32) float cs[8][16][16];  // per-warp store staging

  const int n0 = blockIdx.x * MMA_TILE_N;
  const int m0 = blockIdx.y * MMA_TILE_M;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int wpr = words(p.K);

  const int nf = wid / 2;
  const int mf = 2 * (wid % 2);
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[2];
  nvcuda::wmma::fill_fragment(acc[0], 0.0f);
  nvcuda::wmma::fill_fragment(acc[1], 0.0f);

  for (int kc = 0; kc < p.K; kc += MMA_CHUNK_K) {
    {
      const int nl = tid / 4;
      const int w = kc / 8 + (tid % 4);
      const int n = n0 + nl;
      const int k0 = 8 * w;
      if (n < p.N && w < wpr) {
        const uint8_t se = p.scales[(size_t)n * p.B + k0 / mx::MX_BLOCK];
        const uint32_t qw = p.q[(size_t)n * wpr + w];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int k = k0 + j;
          const bool in = k < p.K;
          const __half wj = in ? mx::dequant_one_mx(se, (int)((qw >> (4 * j)) & 0xF))
                               : __float2half_rn(0.0f);
          ws[nl][(tid % 4) * 8 + j] = wj;
          if (w_out && in) w_out[(size_t)n * p.K + k] = wj;
        }
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          ws[nl][(tid % 4) * 8 + j] = __float2half_rn(0.0f);
      }
    }
    for (int t = tid; t < MMA_TILE_M * MMA_CHUNK_K; t += 256) {
      const int ml = t / MMA_CHUNK_K, kl = t % MMA_CHUNK_K;
      const int m = m0 + ml, k = kc + kl;
      xs[ml][kl] = (m < M && k < p.K) ? x[(size_t)m * p.K + k]
                                      : __float2half_rn(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int kf = 0; kf < MMA_CHUNK_K / 16; ++kf) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half,
                             nvcuda::wmma::row_major>
          af[2];
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half,
                             nvcuda::wmma::col_major>
          bf;
      nvcuda::wmma::load_matrix_sync(af[0], &xs[mf * 16][kf * 16],
                                     MMA_CHUNK_K);
      nvcuda::wmma::load_matrix_sync(af[1], &xs[(mf + 1) * 16][kf * 16],
                                     MMA_CHUNK_K);
      nvcuda::wmma::load_matrix_sync(bf, &ws[nf * 16][kf * 16], MMA_CHUNK_K);
      nvcuda::wmma::mma_sync(acc[0], af[0], bf, acc[0]);
      nvcuda::wmma::mma_sync(acc[1], af[1], bf, acc[1]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int f = 0; f < 2; ++f) {
    nvcuda::wmma::store_matrix_sync(&cs[wid][0][0], acc[f], 16,
                                    nvcuda::wmma::mem_row_major);
    __syncwarp();
    const int mb = m0 + (mf + f) * 16;
    const int nb = n0 + nf * 16;
    for (int t = lane; t < 256; t += 32) {
      const int i = t / 16, j = t % 16;
      if (mb + i < M && nb + j < p.N)
        c[(size_t)(mb + i) * p.N + nb + j] = __float2half_rn(cs[wid][i][j]);
    }
    __syncwarp();
  }
}

inline void gemm_mma_mx(const DevicePackedWm& p, const __half* x, __half* c,
                        int M, cudaStream_t stream = 0,
                        __half* w_out = nullptr) {
  validate_mx(p);
  if (!x || !c) throw std::invalid_argument("mx mma: null pointer");
  if (M < 1) throw std::invalid_argument("mx mma: bad M");
  // Edge tiles are zero-padded in shared memory. Same launch bounds as
  // gemm_mma: gridDim.y caps at 65535 (the 64-row m-tiling bounds M); on
  // gridDim.x only the ceil-rounding add can overflow, N being an int.
  if (M > 65535 * MMA_TILE_M || p.N > INT32_MAX - MMA_TILE_N)
    throw std::invalid_argument("mx mma: launch grid out of range");

  const dim3 grid((unsigned)((p.N + MMA_TILE_N - 1) / MMA_TILE_N),
                  (unsigned)((M + MMA_TILE_M - 1) / MMA_TILE_M));
  mma_kernel_mx<<<grid, 256, 0, stream>>>(p, x, c, M, w_out);
  cuda_check(cudaGetLastError(), "mma_kernel_mx");
}

}  // namespace w4::gpu
