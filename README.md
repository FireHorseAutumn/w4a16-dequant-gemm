# W4A16 Fused Dequant GEMM

**STATUS: R1 complete — packed format, host oracle, two-pass baseline,
gates, bench harness. R2 is the fused SIMT kernel.**

## The kernel

Fused weight-dequantization GEMM (W4A16 class): packed 4-bit weights with
per-group scales/zeros, dequantized in registers inside the same kernel that
does the MMA — FP16/BF16 weights never materialize in DRAM. Bandwidth-bound by
construction; the entire craft is bytes-in, MACs-out.

## Why this shape

- It is the kernel a digital-inference-numerics company exists to run:
  systolic MACs starved unless fed by minimal bytes.
- It exercises one skill list: bit packing, mixed-precision error discipline,
  fusion, bandwidth obsession.
- It is NOT "beat cuBLAS at GEMM" — cuBLAS has no fused-dequant kernel.
  Reference points are Marlin / AWQ / TensorRT-LLM class kernels.

## Packed format (what R1 fixed)

Logical W is N x K row-major. Groups partition K (`gs` a multiple of 8).
8 nibbles per uint32, low nibble first; scales are FP16 per (row, group);
asymmetric mode adds a packed 4-bit zero-point per (row, group). Asymmetric
group ranges are expanded to bracket zero before quantization — without that,
one-sided groups clamp the zero-point and corrupt the whole group (the R1
gate matrix caught this; bound failures were ~2x tolerance before the fix).
Symmetric dequant: `w = scale * (q - 8)`; asymmetric: `w = scale * (q - z)`.

## R1 contents (2026-10-29)

- `include/w4/pack.hpp` — format, host quantizer, host dequant reference,
  FP64 GEMM oracle. The single `dequant_one` step is shared by host and
  device so reference and pipeline are the same arithmetic.
- `include/w4/dequant.cuh` — naive dequant kernel (one thread per packed
  word) + cuBLAS FP16 GEMM with FP32 accumulation (the two-pass baseline and
  correctness reference).
- Gates (CTest 2/2):
  - quantizer per-element bounds over group sizes {8, 16, 32, 48, 64, 128}
    including ragged K and `group_size % 32 != 0` — the silent-wrong-output
    shape from the vLLM RDNA W4A16 issue class — plus degenerate all-zero
    groups and a hand-computed oracle pin;
  - validation contracts with expect-throw tests: malformed dims/group sizes,
    size mismatches, non-finite inputs (NaN is otherwise swallowed by
    min/max scans), and group scales outside the FP16 range (underflow to
    zero below |w|~4e-7, overflow above ~4.5e5) all throw, on both the host
    reference and the device entry points; the device launcher also rejects
    launches whose 32-bit kernel indexing would wrap;
  - device dequant bitwise-identical to the host reference on every config;
  - cuBLAS GEMM vs FP64 oracle at rel L2 <= 1e-3 (inputs are identical FP16
    on both sides; measured worst 2.2e-4 — the gate catches silent wrong
    output, not rounding).
- `benchmarks/bench_baseline.cu` — dequant and GEMM phases timed separately
  plus end-to-end wall, raw reps + device/driver/clock/seed metadata in
  `data/w4a16_baseline_bench.json`. Bandwidth figures are analytic bytes
  moved / time, labeled; the GEMM byte count is the analytic minimum.

## Pinned numbers (2026-10-30, RTX 5060 Laptop, governor-pinned idle machine)

Decode-class shapes, gs=128 symmetric; raw reps + metadata in
`data/w4a16_baseline_bench.json`. Same-session FP64 triad: 321.6 GB/s. The
dequant phase holds 147.9–151.2 GB/s across all eight shapes (~46–47% of
triad — the naive one-word-per-thread kernel is not bandwidth-optimal,
which is R2's opening measurement), rep spreads 0.5–1.8%; the cuBLAS phase
runs at 343–658 GB/s analytic-minimum, spreads 0.7–6.7%. The unpinned
2026-10-29 runs showed the same dequant rate in fast clock windows
(137–140 GB/s against a 256–259 GB/s triad) but intermittently degraded
~3.6x (to ~39 GB/s) as the laptop migrated clocks mid-case; under the
pinned governor that degradation is gone — it was clock migration, as
recorded at the time. The two-pass pays 2x N*K FP16 traffic (write W in
dequant, read W in the GEMM) that a fused kernel eliminates entirely.

## MVP scope

- Packed W4A16 GEMM, per-group scales, CUDA + WMMA/tensor cores.
- Benchmark vs the R1 two-pass baseline.
- Success = meaningful fraction of peak DRAM bandwidth **plus** a written
  error analysis vs FP16 reference (where the error lives, group-size
  sensitivity). The error analysis is the numerics-engineer tell.

## Stretch

- MXFP4/FP8 block-scaled variants; emulated formats on hardware without
  native FP4; KV-cache-quantized attention as a second kernel.

## Ceiling honesty

Matching Marlin-class SOTA is months of work and not the point. The portfolio
piece is: bottleneck identified, pipeline fused, measured honestly. Bench-size
correctness is covered by the bitwise dequant gate only; the FP64 oracle is
gated at small sizes, so no precision claim is made at benchmark scale.
