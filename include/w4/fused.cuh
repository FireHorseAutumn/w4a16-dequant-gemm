#pragma once

// Fused dequant+GEMM: packed weights are dequantized in registers via
// dequant_one and multiplied straight into FP32 accumulators; FP16 W never
// materializes in DRAM, so the 2*N*K FP16 round-trip of the two-pass
// baseline is gone.
//
// Mapping: one warp per group of R output rows, lanes stride packed words
// along K (consecutive lanes read consecutive words: one coalesced 128 B
// line per step). Each row's weights are dequanted once and reused for all
// MTILE rows of the m-tile; per-lane partial sums are shuffle-reduced per
// row at the end. gridDim.y tiles M in chunks of MTILE, so packed W is
// re-read ceil(M/MTILE) times.
//
// The other side of the tradeoff is x: every warp pass over a row re-reads
// the m-tile's x slice, and N * M * K * 4 bytes of L2 traffic is what binds
// the kernel once M grows (measured ~2.4 TB/s effective at M=64, the L2
// wall). R rows per warp keep the x float4s live in registers and reuse
// them across the R rows, dividing x traffic by R at the cost of
// R * MTILE accumulators. Shared-memory staging cannot do better: lanes
// stride 8 k's apart, and a lane stride of 8 floats is 4-way
// bank-conflicted under any padding.
//
// Dispatch (measured): M <= 4 -> MTILE = pow2-ceil(M), R = 1; M > 4 ->
// MTILE = 4, R = 4. MTILE 8+/R 1 loses on x traffic; MTILE 16 (94 regs)
// halves occupancy; MTILE 32 (148 regs) and 64 (255, spills) lose the W
// savings to occupancy and local memory.
//
// w_out (test hook): when non-null the kernel also stores each dequantized
// weight, reproducing dequant_all bitwise. With multiple m-tiles the same
// values are written to the same locations more than once... racy but
// byte-identical, so the memcmp gate stays valid.

#include <cuda_fp16.h>

#include "dequant.cuh"

namespace w4::gpu {

// x is converted to FP32 once per call: the SIMT inner loop would otherwise
// run __half2float on x per FMA, re-converting the same activations for
// every row of W (N * M * K converts at M=64: measured 2 TFLOPS, the
// converts bound it). The FP32 scratch is M*K*4 bytes, trivial at decode
// shapes.
__global__ void x_to_f32_kernel(const __half* __restrict__ x,
                                float* __restrict__ xf, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) xf[i] = __half2float(x[i]);
}

// Cached scratch, grown on demand; never shrunk. Bench warmup covers the
// one-time allocation, so timed reps never pay it. The buffer is shared
// across calls, so calls on different streams must not be in flight
// concurrently.
inline float* x_scratch(size_t n) {
  static float* buf = nullptr;
  static size_t cap = 0;
  if (n > cap) {
    float* nb = nullptr;
    cuda_check(cudaMalloc(&nb, n * sizeof(float)), "x scratch");
    if (buf) cudaFree(buf);
    buf = nb;
    cap = n;
  }
  return buf;
}

template <int MTILE, int R>
__global__ void fused_kernel(const DevicePackedW p,
                             const float* __restrict__ xf,
                             __half* __restrict__ c, const int M,
                             __half* __restrict__ w_out) {
  const int lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
  if (n0 >= p.N) return;  // whole-warp exit: n0 is warp-uniform
  const int wpr = words(p.K);
  const int m0 = blockIdx.y * MTILE;
  // K % 8 == 0 means every row of xf is 32 B-aligned and no packed word has
  // a ragged tail, so xf can be fetched as two float4 per (m, word).
  const bool fast = (p.K % 8) == 0;

  // Every i/r loop is fully unrolled: acc must stay register-resident, and
  // a rolled loop would index it with a runtime variable and spill it.
  float acc[R][MTILE];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int i = 0; i < MTILE; ++i) acc[r][i] = 0.0f;

  for (int w = lane; w < wpr; w += 32) {
    const int k0 = 8 * w;

    // Stage the m-tile's x slice for this word once; the R rows reuse it.
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
      const int g = k0 / p.gs;
      const __half s = p.scales[(size_t)n * p.G + g];
      const int z =
          p.symmetric
              ? 8
              : (int)((p.zeros[(size_t)n * words(p.G) + g / 8] >>
                       (4 * (g % 8))) &
                      0xF);
      const uint32_t qw = p.q[(size_t)n * wpr + w];

      if (fast) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const __half wj = dequant_one(s, (int)((qw >> (4 * j)) & 0xF), z);
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
          const __half wj = dequant_one(s, (int)((qw >> (4 * j)) & 0xF), z);
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

inline void gemm_fused(const DevicePackedW& p, const __half* x, __half* c,
                       int M, cudaStream_t stream = 0,
                       __half* w_out = nullptr) {
  if (p.N < 1 || p.K < 1) throw std::invalid_argument("fused: bad dims");
  if (p.K > W4_MAX_K) throw std::invalid_argument("fused: K out of range");
  if (p.gs < 8 || p.gs % 8 != 0)
    throw std::invalid_argument("fused: gs must be a multiple of 8");
  if ((int64_t)p.G != ((int64_t)p.K + p.gs - 1) / p.gs)
    throw std::invalid_argument("fused: G inconsistent with K, gs");
  if (!p.q || !p.scales || !x || !c)
    throw std::invalid_argument("fused: null pointer");
  if (!p.symmetric && !p.zeros)
    throw std::invalid_argument("fused: asymmetric without zeros");
  if (p.symmetric && p.zeros)
    throw std::invalid_argument("fused: symmetric carries no zeros");
  if (M < 1) throw std::invalid_argument("fused: bad M");
  const int mtile = M <= 1 ? 1 : (M <= 2 ? 2 : 4);
  const int rpw = M <= 4 ? 1 : 4;
  // gridDim.y caps at 65535, so the m-tiling bounds M; gridDim.x is 2^31-1
  // and N is an int, so only the ceil-rounding add could overflow there.
  // In-kernel address math is size_t throughout, so the 2^31 launch bound
  // of the dequant kernel does not apply here. The x pre-pass launches
  // ceil(M*K/256) blocks on gridDim.x, which bounds the product.
  if (M > 65535 * mtile || p.N > INT32_MAX - 8 * rpw)
    throw std::invalid_argument("fused: launch grid out of range");
  if ((int64_t)M * p.K > (int64_t)INT32_MAX * 256)
    throw std::invalid_argument("fused: x pre-pass grid out of range");

  float* xf = x_scratch((size_t)M * p.K);
  x_to_f32_kernel<<<(unsigned)(((size_t)M * p.K + 255) / 256), 256, 0,
                    stream>>>(x, xf, (size_t)M * p.K);
  cuda_check(cudaGetLastError(), "x_to_f32_kernel");

  const dim3 grid((unsigned)((p.N + 8 * rpw - 1) / (8 * rpw)),
                  (unsigned)((M + mtile - 1) / mtile));
  const int block = 256;
  if (mtile == 1) {
    fused_kernel<1, 1><<<grid, block, 0, stream>>>(p, xf, c, M, w_out);
  } else if (mtile == 2) {
    fused_kernel<2, 1><<<grid, block, 0, stream>>>(p, xf, c, M, w_out);
  } else if (rpw == 1) {
    fused_kernel<4, 1><<<grid, block, 0, stream>>>(p, xf, c, M, w_out);
  } else {
    fused_kernel<4, 4><<<grid, block, 0, stream>>>(p, xf, c, M, w_out);
  }
  cuda_check(cudaGetLastError(), "fused_kernel");
}

}  // namespace w4::gpu
