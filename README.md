# W4A16 Fused Dequant GEMM

**STATUS: R3 complete — WMMA/tensor-core fused kernel landed and gated;
it removes the SIMT M=64 loss (0.43x -> 1.68x vs baseline) and is flat
1.4-2.0x across M (staging-latency bound; double-buffering is the next
lever). SIMT fused stays the winner at M <= 16. Pinned rerun owed.**

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

## R3 contents (2026-10-31, same day — separate commit)

- `include/w4/mma.cuh` — fused dequant + WMMA GEMM (`gemm_mma`). Block
  tile 64x64, K in chunks of 32: the 256 threads dequant one packed word
  each into a shared FP16 tile (`dequant_one`, unchanged) and stage the x
  tile, both zero-padded past the N/M/K edges so ragged K and partial
  tiles need no special path; 8 warps each own two 16x16x16 WMMA fragment
  MMAs per k-fragment with FP32 accumulators; outputs store through a
  per-warp shared tile with guarded copies. W is read exactly once at
  M <= 64, the FMA floor drops to the tensor-core rate, and fragment loads
  take x reuse from shared — all three SIMT M=64 walls removed at once.
  The `w_out` hook keeps the bitwise dequant gate.
- Gates: the MMA path gets the full fused gate set (bitwise dequant,
  oracle 1e-3, cuBLAS cross-check 2e-3, entry throws). CTest 2/2, zero
  warnings. CUDA 13 note: `store_matrix_sync` has no
  float-accumulator-to-half-pointer overload, so fragments store to an
  FP32 shared tile and the guarded copy does the `__float2half_rn`.

## R3 numbers (2026-10-31, RTX 5060 Laptop, UNPINNED powersave governor)

Same session, same bench as R2 (`bench_fused` now three-way); triad
264.9 GB/s. Ratios vs the re-measured two-pass baseline:

| M | N=4096 | N=14336 |
|---|---|---|
| 1 | 1.45x | 1.95x |
| 8 | 1.38x | 2.00x |
| 16 | 1.38x | 1.78x |
| 64 | 1.42x | 1.68x |

Flat across M and far under triad (25-43 GB/s effective) — the signature
of a staging-latency-bound kernel: the chunk loop is single-buffered, two
`__syncthreads` per chunk, no overlap. The M=64 SIMT loss is gone
(0.43x -> 1.68x), which was the R3 charter; the margin headroom is
double-buffering (cp.async / register prefetch), recorded as the next
lever. Tile-shape A/Bs were run and rejected (chunk 64 doubles the
x-staging footprint; N-tile 32 halves W work per chunk while x staging
stays fixed). Best-of-dispatch vs baseline: SIMT M=1 7.5/7.7x, M=8
2.2/2.9x, M=16 1.4/1.6x; MMA M=64 1.4/1.7x. The pinned MEAS-2 rerun is
the authority on all of this.

## R2 contents (2026-10-31)

- `include/w4/fused.cuh` — fused dequant+GEMM (`gemm_fused`). One warp per
  group of R output rows, lanes stride packed words along K (coalesced
  128 B steps); each weight is dequantized once via `dequant_one` and FMA'd
  into R*MTILE FP32 accumulators; shuffle reduction per row at the end. x
  is converted to FP32 once per call (per-FMA `__half2float` re-conversion
  measured as the binding constraint without it). MTILE/R dispatch:
  M <= 4 -> (pow2(M), 1); M > 4 -> (4, 4) — 128 regs, no spills, 2
  blocks/SM. MTILE 16/32/64 were measured and rejected (occupancy,
  register spills, shared-staging bank conflicts). The same `dequant_one` call is the whole
  dequant path, so the bitwise gate below applies unchanged.
- Gates (CTest 2/2, zero warnings): everything R1 gated, plus fused
  dequant bitwise vs `dequant_all` through the `w_out` hook across the
  same matrix (including M > 16 configs for the m-tiling guards), fused
  GEMM vs the FP64 oracle at rel L2 <= 1e-3, fused vs cuBLAS output at
  rel L2 <= 2e-3, and the fused entry-point throw contracts.
- `benchmarks/bench_fused.cu` — fused vs two-pass re-measured in the same
  session (ratios immune to cross-session machine state), same shapes as
  the baseline bench, raw reps + metadata in `data/w4a16_fused_bench.json`.

## R2 numbers (2026-10-31, RTX 5060 Laptop, governor-pinned)

Same-session triad 320.1 GB/s; baseline re-measured in session. Raw reps +
metadata in `data/w4a16_fused_bench.json`. The pre-committed bar was
>= 1.5x e2e at decode shapes.

| M | N=4096 | N=14336 |
|---|---|---|
| 1 | 7.25x | 8.31x |
| 8 | 2.28x | 2.88x |
| 16 | 1.36x | 1.64x |
| 64 | 0.44x | 0.46x |

M=1 and M=8 clear the bar; M=16 clears it at N=14336 and misses at
N=4096; M=64 loses outright. The M=64 loss is structural for SIMT FP32:
N*M*K FMAs on CUDA cores plus packed-W re-reads from m-tiling and x L2
traffic (~2.4 TB/s measured effective — the kernel is L2-bound before it
is FMA-bound). Both are exactly what WMMA fragment tiling removes, which
is the R3 justification. An unpinned run earlier the same day showed the
same ordering at lower magnitude (M=1 5.8–7.3x, M=64 0.36–0.45x); the
pinned numbers above are the record.

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
