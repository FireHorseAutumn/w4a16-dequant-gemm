# W4A16 Fused Dequant GEMM

Packed 4-bit weights dequantized in registers inside the GEMM kernel:
FP16/BF16 weights never materialize in DRAM. CUDA, WMMA tensor cores, FP16
activations, FP32 accumulation. Bandwidth-bound by construction; the
craft is bytes-in, MACs-out. cuBLAS has no fused-dequant kernel, so the
reference class is Marlin / AWQ / TensorRT-LLM.

## Status

- Rungs R1–R8 complete: SIMT and MMA GEMM, an M=9 dispatcher, MXFP4
  (E2M1 + E8M0) variants of all three, and KV-cache decode attention
  (single-block, split-S flash, FP8 E4M3 cache, paged with prefix
  sharing).
- CTest 2/2: 930 device + 212928 host checks, zero warnings.
  compute-sanitizer memcheck and racecheck clean.
- The group-walk classes vLLM's exllama kernels get wrong
  (vllm-project/vllm#60921) are gated here and pass.

## Headline numbers

Pinned (governor + platform profile), RTX 5060 Laptop, same-session
triad 320.6 GB/s. Ratios vs the two-pass baseline (dequant kernel +
cuBLAS):

| M | SIMT N=4096 | MMA N=4096 | SIMT N=14336 | MMA N=14336 |
|---|---|---|---|---|
| 1 | 7.33x | 1.41x | 8.25x | 1.88x |
| 8 | 2.29x | 1.35x | 2.87x | 1.92x |
| 16 | 1.37x | 1.35x | 1.64x | 1.90x |
| 64 | 0.44x | 1.40x | 0.45x | 1.79x |

The dispatcher picks SIMT below M=9, MMA from there. Decode attention:
split-S flash beats the single-block kernel everywhere (up to 10.4x)
and runs at triad at large S; W4 ties the FP16 cache from the L2 spill
up. The KV win is a 4x smaller cache, not wall time.

## Honest losses

- SIMT loses at M=64 (0.44x): structural FP32 FMA floor plus packed-W
  re-reads. The MMA path exists to remove exactly this, and does.
- The MMA staging lever is measured out across three designs
  (single-buffer, register prefetch, cp.async): all land at 34–46 GB/s
  effective. The kernel is instruction-stream-bound; no further MMA
  levers are chartered.
- FP8 KV loses on wall time at every size (0.19–0.41x). Its footing is
  accuracy: rel L2 2.57e-2 vs W4's 1.17e-1.
- Paged W4 costs 0.51–0.66x of contiguous at page=16; the tax is
  per-position div/mod, not page size, not placement.

## Build and test

```sh
cmake -B build
cmake --build build -j
ctest --test-dir build
```

Benchmarks: `build/bench_baseline`, `build/bench_fused`,
`build/bench_mx`, `build/bench_attention`. Raw reps + machine metadata
land in `data/`.

## Layout

- `include/w4/pack.hpp`: format, host quantizer, FP64 oracle. Host and
  device share `dequant_one`: reference and pipeline are the same
  arithmetic.
- `include/w4/dequant.cuh`, `fused.cuh`, `mma.cuh`, `dispatch.cuh`: the
  the W4 kernels and the M=9 dispatcher.
- `include/w4/mx.hpp`, `mxfused.cuh`: MXFP4 format and kernels.
- `include/w4/kvcache.hpp`, `kvattention.cuh`, `kvflash.cuh`,
  `kvfp8.hpp`, `kvpaged.hpp`: KV-cache decode attention.
- `tests/`: oracle and device gates. `benchmarks/`: timing harnesses
  and `error_analysis.cpp`.

## Deeper reading

- `docs/RUNGS.md`: the full engineering record: every rung, every
  pinned table, every measured rejection.
- `ERROR_ANALYSIS.md`: where the quantization error lives, group-size
  sensitivity, fixed-seed reproducible.

## Ceiling honesty

Matching Marlin-class SOTA is months of work and not the point. The
piece is: bottleneck identified, pipeline fused, measured honestly.
Bench-size correctness is covered by the bitwise dequant gate only; the
FP64 oracle is gated at small sizes, so no precision claim is made at
benchmark scale.
