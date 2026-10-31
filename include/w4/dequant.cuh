#pragma once

// Naive dequant kernel plus the cuBLAS two-pass baseline: materialize W in
// FP16 DRAM, then a plain FP16 GEMM with FP32 accumulation. This is the
// denominator every fused kernel is measured against, and the correctness
// reference the FP64 oracle gates.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "pack.hpp"

namespace w4::gpu {

inline void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(e));
}

inline void cublas_check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + ": cublas status " +
                             std::to_string((int)s));
}

struct DevicePackedW {
  int N, K, gs, G;
  int symmetric;  // bool across the host/device boundary
  const uint32_t* q;
  const __half* scales;
  const uint32_t* zeros;  // null when symmetric
};

// One thread per packed word; gs % 8 == 0 keeps every word inside one group.
__global__ void dequant_kernel(DevicePackedW p, __half* w) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int wpr = words(p.K);
  if (idx >= p.N * wpr) return;
  const int n = idx / wpr;
  const int k0 = 8 * (idx % wpr);
  const int g = k0 / p.gs;
  const __half s = p.scales[(size_t)n * p.G + g];
  const int z =
      p.symmetric
          ? 8
          : (int)((p.zeros[(size_t)n * words(p.G) + g / 8] >> (4 * (g % 8))) &
                  0xF);
  const uint32_t qw = p.q[idx];
  __half* out = w + (size_t)n * p.K + k0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (k0 + j >= p.K) break;
    out[j] = dequant_one(s, (int)((qw >> (4 * j)) & 0xF), z);
  }
}

inline void dequant(const DevicePackedW& p, __half* w, cudaStream_t stream = 0) {
  if (p.N < 1 || p.K < 1) throw std::invalid_argument("dequant: bad dims");
  if (p.gs < 8 || p.gs % 8 != 0)
    throw std::invalid_argument("dequant: gs must be a multiple of 8");
  if (p.G != (p.K + p.gs - 1) / p.gs)
    throw std::invalid_argument("dequant: G inconsistent with K, gs");
  if (!p.q || !p.scales || !w)
    throw std::invalid_argument("dequant: null pointer");
  if (!p.symmetric && !p.zeros)
    throw std::invalid_argument("dequant: asymmetric without zeros");
  if (p.symmetric && p.zeros)
    throw std::invalid_argument("dequant: symmetric carries no zeros");
  // The kernel indexes with 32-bit ints; keep the whole launch inside that.
  const int64_t total = (int64_t)p.N * words(p.K);
  if (total >= (int64_t{1} << 31))
    throw std::invalid_argument("dequant: N*words(K) exceeds 2^31");
  const int block = 256;
  dequant_kernel<<<(unsigned)((total + block - 1) / block), block, 0,
                   stream>>>(p, w);
  cuda_check(cudaGetLastError(), "dequant_kernel");
}

// Row-major X (M x K) and W (N x K) are column-major X^T (K x M) and
// W^T (K x N); OP_T(W) * OP_N(X) = W * X^T = C^T stored column-major
// N x M, i.e. C (M x N) row-major. FP32 accumulation, FP16 out.
inline void gemm_fp16(cublasHandle_t h, const __half* w, const __half* x,
                      __half* c, int M, int N, int K,
                      cudaStream_t stream = 0) {
  if (M < 1 || N < 1 || K < 1)
    throw std::invalid_argument("gemm_fp16: bad dims");
  if (!w || !x || !c) throw std::invalid_argument("gemm_fp16: null pointer");
  cublas_check(cublasSetStream(h, stream), "cublasSetStream");
  const float alpha = 1.0f, beta = 0.0f;
  cublas_check(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w,
                            CUDA_R_16F, K, x, CUDA_R_16F, K, &beta, c,
                            CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
}

inline cublasHandle_t make_cublas() {
  cublasHandle_t h = nullptr;
  cublas_check(cublasCreate(&h), "cublasCreate");
  return h;
}

}  // namespace w4::gpu
