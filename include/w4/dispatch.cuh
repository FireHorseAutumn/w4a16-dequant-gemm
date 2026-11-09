#pragma once

// SIMT/MMA dispatcher (R8): one host-side branch picks the kernel per call.
// The crossover is measured, not argued: the pinned three-way run of record
// (2026-11-04, bench_fused): SIMT leads decode shapes through M=8 (2.29x /
// 2.87x at M=8 vs MMA's 1.35x / 1.92x, N=4096 / N=14336); MMA is within
// noise at M=16/N=4096 (1.38x vs SIMT's 1.37x) and leads from there
// (1.79-1.92x at N=14336, and it removes the SIMT M=64 loss outright,
// 0.44x -> 1.8x+). The MX paths reproduce the same picture on a different
// dequant arithmetic (SIMT 2.01x / 2.43x at M=8; MMA wins from M=16), so
// one threshold serves both. N is not dispatched on: the only split cell
// (W4, M=16, N=4096) is within noise, and every other M=16 cell says MMA.
//
// Dispatch changes which kernel runs, never what it computes: both callees
// share dequant_one and carry the same w_out hook, and the test gate pins
// the routing by memcmp against the expected callee's output.

#include "fused.cuh"
#include "mma.cuh"
#include "mxfused.cuh"

namespace w4::gpu {

// M below this goes SIMT; M at or above goes MMA.
constexpr int DISPATCH_MMA_MIN_M = 9;

inline void gemm(const DevicePackedW& p, const __half* x, __half* c, int M,
                 cudaStream_t stream = 0, __half* w_out = nullptr) {
  if (M < DISPATCH_MMA_MIN_M) {
    gemm_fused(p, x, c, M, stream, w_out);
  } else {
    gemm_mma(p, x, c, M, stream, w_out);
  }
}

inline void gemm_mx(const DevicePackedWm& p, const __half* x, __half* c,
                    int M, cudaStream_t stream = 0, __half* w_out = nullptr) {
  if (M < DISPATCH_MMA_MIN_M) {
    gemm_fused_mx(p, x, c, M, stream, w_out);
  } else {
    gemm_mma_mx(p, x, c, M, stream, w_out);
  }
}

}  // namespace w4::gpu
