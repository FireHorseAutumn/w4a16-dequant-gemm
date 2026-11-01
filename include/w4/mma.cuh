#pragma once

// Fused dequant + WMMA GEMM (R3). Same contract as the SIMT fused kernel —
// packed weights dequantized via dequant_one, never materialized in DRAM —
// but the multiply runs on tensor cores. The SIMT R2 kernel hits its wall at
// M=64: N*M*K FP32 FMAs on CUDA cores, W re-read per m-tile, and x L2
// traffic. Fragment MMA removes all three at once: the M=64 tile is one
// m-tile (W read exactly once), the FMA floor drops to the tensor-core
// rate, and fragment loads reuse x and W from shared.
//
// Block tile: 64x64 outputs, K in chunks of 32. The 256 threads cooperatively
// dequant one packed word each (64 rows x 4 words = 2048 weights) into ws
// and stage the x tile into xs, both zero-padded past the N/M/K edges, so
// ragged K and partial tiles need no special path. 8 warps; warp w owns
// n-fragment w/2 and m-fragments {2*(w%2), 2*(w%2)+1}, two k-fragments per
// chunk. Outputs are stored through a per-warp shared tile so edge
// fragments never write out of bounds.
//
// The pipeline is single-buffered: staging and MMA alternate under two
// __syncthreads per chunk with no overlap, so the kernel is staging-latency
// bound, not bandwidth bound. Measured A/B (same session, unpinned): chunk
// 64 and N-tile 32 both lose to this configuration — wider chunks amortize
// syncs but double the x-staging footprint, and a narrower N tile halves W
// work per chunk while x staging stays fixed. The next lever is
// double-buffering (cp.async or register prefetch), not tile shape.
//
// x stays FP16 and is read from global through the xs stage (L2-resident at
// decode shapes) — the FP32 pre-pass of the SIMT kernel was only there to
// dodge per-FMA converts, which fragment loads do in hardware.
//
// w_out (test hook): the dequant step also stores each weight it produces,
// reproducing dequant_all bitwise — same racy-but-identical semantics as
// the SIMT kernel when m-tiles repeat (M > 64).

#include <cuda_fp16.h>
#include <mma.h>

#include "dequant.cuh"

namespace w4::gpu {

constexpr int MMA_TILE_M = 64;
constexpr int MMA_TILE_N = 64;
constexpr int MMA_CHUNK_K = 32;

__global__ void mma_kernel(const DevicePackedW p,
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

  const int nf = wid / 2;         // n-fragment 0..3
  const int mf = 2 * (wid % 2);   // m-fragments {mf, mf+1}
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[2];
  nvcuda::wmma::fill_fragment(acc[0], 0.0f);
  nvcuda::wmma::fill_fragment(acc[1], 0.0f);

  for (int kc = 0; kc < p.K; kc += MMA_CHUNK_K) {
    // W tile: one packed word per thread, 64 rows x 4 words.
    {
      const int nl = tid / 4;
      const int w = kc / 8 + (tid % 4);
      const int n = n0 + nl;
      const int k0 = 8 * w;
      if (n < p.N && w < wpr) {
        const int g = k0 / p.gs;
        const __half s = p.scales[(size_t)n * p.G + g];
        const int z =
            p.symmetric
                ? 8
                : (int)((p.zeros[(size_t)n * words(p.G) + g / 8] >>
                         (4 * (g % 8))) &
                        0xF);
        const uint32_t qw = p.q[(size_t)n * wpr + w];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int k = k0 + j;
          const bool in = k < p.K;
          const __half wj =
              in ? dequant_one(s, (int)((qw >> (4 * j)) & 0xF), z)
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
    // x tile: 64x32 halves, zero-padded past M and K.
    for (int t = tid; t < MMA_TILE_M * MMA_CHUNK_K; t += 256) {
      const int ml = t / MMA_CHUNK_K, kl = t % MMA_CHUNK_K;
      const int m = m0 + ml, k = kc + kl;
      xs[ml][kl] = (m < M && k < p.K) ? x[(size_t)m * p.K + k]
                                      : __float2half_rn(0.0f);
    }
    __syncthreads();

    // ws[n][k] read as a col_major B fragment: element (k, n) sits at
    // k + n*CHUNK_K, so the fragment base for (k0', n0') is &ws[n0'][k0']
    // with ldm CHUNK_K. xs is row-major A with ldm CHUNK_K.
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

inline void gemm_mma(const DevicePackedW& p, const __half* x, __half* c,
                     int M, cudaStream_t stream = 0,
                     __half* w_out = nullptr) {
  if (p.N < 1 || p.K < 1) throw std::invalid_argument("mma: bad dims");
  if (p.gs < 8 || p.gs % 8 != 0)
    throw std::invalid_argument("mma: gs must be a multiple of 8");
  if (p.G != (p.K + p.gs - 1) / p.gs)
    throw std::invalid_argument("mma: G inconsistent with K, gs");
  if (!p.q || !p.scales || !x || !c)
    throw std::invalid_argument("mma: null pointer");
  if (!p.symmetric && !p.zeros)
    throw std::invalid_argument("mma: asymmetric without zeros");
  if (p.symmetric && p.zeros)
    throw std::invalid_argument("mma: symmetric carries no zeros");
  if (M < 1) throw std::invalid_argument("mma: bad M");
  // Edge tiles are zero-padded in shared memory; all fragment addressing is
  // shared-internal 32-byte-aligned. gridDim.y caps at 65535, so the
  // 64-row m-tiling bounds M; gridDim.x is 2^31-1 and N is an int, so only
  // the ceil-rounding add could overflow there.
  if (M > 65535 * MMA_TILE_M || p.N > INT32_MAX - MMA_TILE_N)
    throw std::invalid_argument("mma: launch grid out of range");

  const dim3 grid((unsigned)((p.N + MMA_TILE_N - 1) / MMA_TILE_N),
                  (unsigned)((M + MMA_TILE_M - 1) / MMA_TILE_M));
  mma_kernel<<<grid, 256, 0, stream>>>(p, x, c, M, w_out);
  cuda_check(cudaGetLastError(), "mma_kernel");
}

}  // namespace w4::gpu
