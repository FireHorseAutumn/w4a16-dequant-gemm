#pragma once

// Fused dequant + WMMA GEMM (R3). Same contract as the SIMT fused kernel —
// packed weights dequantized via dequant_one, never materialize in DRAM —
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
// cp.async pipeline (R3.6): chunk staging rides cp.async straight into
// shared — the x tile lands without a register round trip, and the packed
// W words land in qs and pay a shared round trip through the dequant.
// Scales/zeros still ride registers, prefetched at issue time. Issue runs
// two chunks ahead of the MMA: xs is three slots deep (the issued slot, the
// dequanted slot, and the slot the MMA reads are all live), ws and qs two.
// One __syncthreads per chunk publishes the previous iteration's dequant
// and staged x and retires the previous MMA before its slots recycle; the
// next chunk's dequant sits between issue and MMA under no barrier of its
// own, so warps interleave dequant ALU with tensor-core work. qs needs no
// barrier at all — each word is read only by the thread that staged it, and
// cp.async.wait_group covers the issuing thread. cp.async wants a 4-byte
// aligned source, which x rows only guarantee at even K; odd-K or
// misaligned x falls back to the guarded scalar stores, same staged values.
// Numerics are unchanged: same dequant values, same fragment and
// accumulation order. Measured A/Bs: chunk 64 and N-tile 32 lost to
// 64x64x32 under the single-buffered loop (wider chunks double the
// x-staging footprint; a narrower N tile halves W work per chunk while x
// staging stays fixed) and the rejection predates pipelining; register
// prefetch (R3.5) measured flat at N=4096 and +4-6% at N=14336 unpinned;
// this cp.async restructure measured parity with R3.5 unpinned — three
// staging designs within noise at 25-45 GB/s effective, so the binding
// constraint is not the global-load path.
//
// x stays FP16 and is read from global through the xs stage (L2-resident at
// decode shapes) — the FP32 pre-pass of the SIMT kernel was only there to
// dodge per-FMA converts, which fragment loads do in hardware.
//
// w_out (test hook): the dequant step also stores each weight it produces,
// reproducing dequant_all bitwise — same racy-but-identical semantics as
// the SIMT kernel when m-tiles repeat (M > 64).

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
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
  __shared__ __align__(32) __half ws[2][MMA_TILE_N][MMA_CHUNK_K];
  __shared__ __align__(32) __half xs[3][MMA_TILE_M][MMA_CHUNK_K];
  __shared__ __align__(16) uint32_t qs[2][MMA_TILE_N][MMA_CHUNK_K / 8];
  __shared__ __align__(32) float cs[8][16][16];  // per-warp store staging

  const int n0 = blockIdx.x * MMA_TILE_N;
  const int m0 = blockIdx.y * MMA_TILE_M;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int wpr = words(p.K);
  const int nchunks = (p.K + MMA_CHUNK_K - 1) / MMA_CHUNK_K;

  const int nf = wid / 2;         // n-fragment 0..3
  const int mf = 2 * (wid % 2);   // m-fragments {mf, mf+1}
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[2];
  nvcuda::wmma::fill_fragment(acc[0], 0.0f);
  nvcuda::wmma::fill_fragment(acc[1], 0.0f);

  const int nl = tid / 4;  // W row this thread stages
  const int wq = tid % 4;  // its word lane within the chunk

  // cp.async sources must be 4-byte aligned; x rows are only when K is even
  // and the base is word-aligned.
  const bool x_async =
      (p.K % 2 == 0) && ((reinterpret_cast<uintptr_t>(x) & 3) == 0);

  // Issue chunk ch: this thread's packed word plus a quarter of the x tile
  // (four 4-byte pairs). Invalid W words skip the copy — the dequant never
  // reads them — and out-of-range x pairs take guarded scalar stores.
  auto stage_chunk = [&](int ch) {
    const int w = ch * (MMA_CHUNK_K / 8) + wq;
    const int n = n0 + nl;
    if (n < p.N && w < wpr)
      __pipeline_memcpy_async(&qs[ch & 1][nl][wq], &p.q[(size_t)n * wpr + w],
                              sizeof(uint32_t));
    const int kc = ch * MMA_CHUNK_K;
    const int slot = ch % 3;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = tid + 256 * i;
      const int r = t >> 4, k = kc + 2 * (t & 15);
      const int m = m0 + r;
      if (x_async && m < M && k + 1 < p.K) {
        __pipeline_memcpy_async(&xs[slot][r][2 * (t & 15)],
                                &x[(size_t)m * p.K + k], 2 * sizeof(__half));
      } else {
        xs[slot][r][2 * (t & 15)] =
            (m < M && k < p.K) ? x[(size_t)m * p.K + k]
                               : __float2half_rn(0.0f);
        xs[slot][r][2 * (t & 15) + 1] =
            (m < M && k + 1 < p.K) ? x[(size_t)m * p.K + k + 1]
                                   : __float2half_rn(0.0f);
      }
    }
    __pipeline_commit();
  };

  // Scale/zero for a chunk, riding registers from issue to dequant.
  struct ScaleZero {
    __half s;
    int z;
    bool valid;
  };
  auto load_sz = [&](int ch, ScaleZero& r) {
    const int w = ch * (MMA_CHUNK_K / 8) + wq;
    const int n = n0 + nl;
    r.valid = n < p.N && w < wpr;
    if (r.valid) {
      const int g = (8 * w) / p.gs;
      r.s = p.scales[(size_t)n * p.G + g];
      r.z = p.symmetric
                ? 8
                : (int)((p.zeros[(size_t)n * words(p.G) + g / 8] >>
                         (4 * (g % 8))) &
                        0xF);
    }
  };

  // Dequant chunk ch out of qs into ws[b]. Guards match the staged-data
  // validity exactly; the w_out hook reproduces dequant_all bitwise.
  auto dequant_chunk = [&](int ch, int b, const ScaleZero& r) {
    if (r.valid) {
      const uint32_t qw = qs[ch & 1][nl][wq];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int k = ch * MMA_CHUNK_K + wq * 8 + j;
        const bool in = k < p.K;
        const __half wj =
            in ? dequant_one(r.s, (int)((qw >> (4 * j)) & 0xF), r.z)
               : __float2half_rn(0.0f);
        ws[b][nl][wq * 8 + j] = wj;
        if (w_out && in) w_out[(size_t)(n0 + nl) * p.K + k] = wj;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        ws[b][nl][wq * 8 + j] = __float2half_rn(0.0f);
    }
  };

  // MMA on chunk ch. ws[b][n][k] read as a col_major B fragment: element
  // (k, n) sits at k + n*CHUNK_K, so the fragment base for (k0', n0') is
  // &ws[b][n0'][k0'] with ldm CHUNK_K. xs is row-major A, ldm CHUNK_K.
  auto mma_chunk = [&](int ch) {
    const int b = ch & 1, slot = ch % 3;
#pragma unroll
    for (int kf = 0; kf < MMA_CHUNK_K / 16; ++kf) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half,
                             nvcuda::wmma::row_major>
          af[2];
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half,
                             nvcuda::wmma::col_major>
          bf;
      nvcuda::wmma::load_matrix_sync(af[0], &xs[slot][mf * 16][kf * 16],
                                     MMA_CHUNK_K);
      nvcuda::wmma::load_matrix_sync(af[1], &xs[slot][(mf + 1) * 16][kf * 16],
                                     MMA_CHUNK_K);
      nvcuda::wmma::load_matrix_sync(bf, &ws[b][nf * 16][kf * 16],
                                     MMA_CHUNK_K);
      nvcuda::wmma::mma_sync(acc[0], af[0], bf, acc[0]);
      nvcuda::wmma::mma_sync(acc[1], af[1], bf, acc[1]);
    }
  };

  // Prologue: chunks 0 and 1 in flight, chunk 0 dequanted into ws[0].
  stage_chunk(0);
  if (nchunks > 1) stage_chunk(1);
  ScaleZero sz0{}, sz1{};
  load_sz(0, sz0);
  if (nchunks > 1) load_sz(1, sz1);
  __pipeline_wait_prior(nchunks > 1 ? 1 : 0);
  dequant_chunk(0, 0, sz0);

  for (int ch = 0; ch < nchunks; ++ch) {
    // Publishes the previous iteration's dequant and this chunk's staged x;
    // retires the previous MMA before its xs/ws slots recycle.
    __syncthreads();
    ScaleZero szn{};
    if (ch + 2 < nchunks) {
      stage_chunk(ch + 2);
      load_sz(ch + 2, szn);
    }
    if (ch + 1 < nchunks) {
      __pipeline_wait_prior(ch + 2 < nchunks ? 1 : 0);
      dequant_chunk(ch + 1, (ch + 1) & 1, sz1);
    }
    mma_chunk(ch);
    sz1 = szn;
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
