# W4A16 Fused Dequant GEMM

**STATUS: R7 complete — split-S flash decode attention (S past the old
single-block 4096 bound), FP8 (E4M3) KV cache, and paged W4 with prefix
sharing landed and gated on top of the R5 KV rung. Flash FP16 runs at
triad at large S and beats the single-block baseline everywhere (up to
10.4x at H=8); flash W4 ties it from the L2 spill up — the win is a 4x
smaller cache, not wall time, dequant at load being instruction-bound.
FP8 KV's footing is accuracy (rel L2 2.6e-2 vs W4's 1.2e-1), not speed.
R4 (MXFP4) and the R1–R3 GEMM rungs below. Governor-pinned numbers.**

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

## R4 contents (2026-10-31 — MXFP4)

- `include/w4/mx.hpp` — MXFP4 (E2M1 elements + E8M0 power-of-two block
  scale, 32-element blocks): packed format, host quantizer with a
  no-saturation scale rule, reference dequant. This is a different
  arithmetic from W4 (table lookup + exponent shift, no zero-point), so it
  gets its own header.
- Host gates live in `test_oracle.cpp`: per-element bounds (measured worst
  exactly saturates the nearest-rounding bound), the no-saturation
  invariant, ragged and single-element blocks, sign handling, degenerate
  all-zero blocks, a hand-computed pin, and the throw contracts.
- `include/w4/mxfused.cuh` — the kernel rung: naive dequant (the MX
  two-pass denominator), fused SIMT (`gemm_fused_mx`), and fused WMMA
  (`gemm_mma_mx`). The mapping, x staging, dispatch, and store guards
  follow the `fused.cuh` / `mma.cuh` skeletons unchanged; only the scale
  fetch (one E8M0 byte per 4 words) and the dequant step differ, so the
  R2/R3 performance conclusions carry over instead of being re-derived.
- The measured dequant lesson: `dequant_one_mx` is semantically a table
  lookup, but a function-local LUT indexed by a runtime nibble lands in
  local memory on device and cost the fused SIMT kernel ~7-8x end-to-end
  (M=1 went 0.71x -> 5.61x). The fix is a branch-free arithmetic decode of
  E2M1 (exponent field and mantissa split; every table value is exactly
  representable), which keeps host and device bitwise-identical.
- Gates: the full set on both MX paths — dequant bitwise vs
  `dequant_all_mx`, fused/MMA dequant bitwise through `w_out`, GEMM vs the
  FP64 oracle at rel L2 <= 1e-3, vs cuBLAS at 2e-3, entry-throw contracts
  (dims, B-vs-K consistency, nulls, M, the 2^31 naive-dequant launch
  bound). Partial 32-element blocks and ragged word tails are covered by
  the K in {72, 100, 104, 40} configs. CTest 2/2 (271 checks), zero
  warnings.
- `benchmarks/bench_mx.cu` — MX two-pass vs fused vs MMA, same session,
  same shapes as `bench_fused`; raw reps + metadata in
  `data/w4a16_mx_bench.json`. Each case also records the format-accuracy
  figure: rel L2 of dequant(quant(W)) vs the FP64 source, MXFP4 vs W4A16
  gs=128 symmetric, on identical data.

## R4 numbers (2026-11-01 pinned rerun, RTX 5060 Laptop, performance governor)

Same-session triad 321.4 GB/s; MX two-pass baseline re-measured in
session. SIMT-fused / MMA ratios vs baseline:

| M | SIMT N=4096 | SIMT N=14336 | MMA N=4096 | MMA N=14336 |
|---|---|---|---|---|
| 1 | 5.55x | 6.34x | 1.34x | 1.79x |
| 8 | 2.01x | 2.43x | 1.29x | 1.81x |
| 16 | 1.14x | 1.33x | 1.28x | 1.80x |
| 64 | 0.34x | 0.36x | 1.34x | 1.71x |

The W4 dispatch picture reproduces on a different dequant arithmetic:
SIMT wins decode shapes (M <= 16), MMA removes the M=64 loss. MX fused
SIMT runs under its W4 counterpart (5.55x vs 7.33x at M=1, N=4096) — the
scale-byte fetch per four words plus the decode ALU. Quantization
accuracy on Gaussian weights: MXFP4 rel L2 1.15e-1 vs W4A16 gs=128
symmetric 1.17e-1 — parity on this distribution; MX's finer blocks (32
vs 128 elements) do not buy accuracy here because the E8M0 scale is a
bare power of two where W4's is FP16. One distribution, one seed set;
not a general format ranking.

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

## R3 numbers (2026-11-01 pinned rerun, RTX 5060 Laptop, performance governor)

Same bench as R2 (`bench_fused` three-way), re-measured two-pass
baseline; triad 320.6 GB/s. MMA ratios vs baseline:

| M | N=4096 | N=14336 |
|---|---|---|
| 1 | 1.41x | 1.88x |
| 8 | 1.35x | 1.92x |
| 16 | 1.35x | 1.90x |
| 64 | 1.40x | 1.79x |

Flat across M and far under triad (35–45 GB/s effective) — the signature
of a staging-latency-bound kernel. The M=64 SIMT loss is gone
(0.44x -> 1.79x at N=14336), which was the R3 charter. Tile-shape A/Bs
were run and rejected (chunk 64 doubles the x-staging footprint; N-tile
32 halves W work per chunk while x staging stays fixed). The same run
re-measured the SIMT fused path — the R2 table above is this session —
reproducing the 2026-10-31 pinned numbers to a few hundredths.
Best-of-dispatch vs baseline: SIMT M <= 16, MMA M=64.

R3.5 (2026-11-03, separate commit): the chunk loop is now double-buffered
— the next chunk's packed word, scale/zero, and x slivers prefetch into
registers under the current chunk's MMA, ws/xs alternate two shared
buffers, one `__syncthreads` per chunk instead of two. Numerics
unchanged; CTest 2/2. The table above stays the single-buffered pinned
run of record: the replacement measured flat at N=4096 and +4–6% at
N=14336 in an unpinned same-session A/B (triad 264–265 GB/s, a
low-power-profile session), still 25–45 GB/s effective. Staging latency
still binds; cp.async into shared is the next lever. The pinned rerun
joins the next measurement pass.

## R5 contents (2026-10-31 — KV-cache decode attention)

- `include/w4/kvcache.hpp` (scaffold commit) — the cache as H independent
  slabs of the existing PackedW format: positions are rows, head dim is
  the contracted axis, groups partition D. The R1 quantizer and all its
  contracts carry over unchanged. Plus the FP64 reference attention
  (double-accumulated scores over the FP16-exact dequantized cache,
  max-subtracted softmax) and host gates: slab layout memcmp'd against
  independent per-head quantization, a hand pin (zero keys -> uniform
  softmax -> exact mean of the V rows), softmax stability at score
  magnitudes ~3e4, throw contracts.
- `include/w4/kvattention.cuh` (kernel commit) — `attention_fp16` (the
  FP16-cache baseline, same kernel shape, so the comparison isolates
  cache bytes) and `attention_w4` (dequant at load: K in the score phase,
  V in the output phase, `dequant_one` unchanged). One block per head;
  warp-per-position scores with shuffle reduction; softmax as two
  block-wide reductions (max, then sum); threads stride D for the output
  sum with probabilities broadcast from shared. The query is staged in
  shared once. Static shared bounds with hard entry throws: S <= 4096
  (scores, 16 KB) and D <= 512 (query stage, 2 KB) — past that wants
  flash-style online softmax over tiles, which is a different kernel.
- Gates: `k_out`/`v_out` hooks reproduce `dequant_head` bitwise per head;
  output vs the FP64 reference at rel L2 <= 1e-3; vs the device two-pass
  (per-head dequant + FP16 attention) at 2e-3. Configs cover S=1, ragged
  words (D=96), both sign modes, and S=4096 exactly on the bound. CTest
  2/2, 350 checks, zero warnings.
- `benchmarks/bench_attention.cu` — single-block FP16/W4 fused vs
  two-pass, plus the split-S flash paths (FP16, W4, FP8, paged W4), same
  session, MHA (H=32) and GQA (H=8) head counts with S past the
  single-block bound; raw reps + metadata in `data/w4a16_kv_bench.json`.

## R5 numbers (2026-11-01 pinned rerun, RTX 5060 Laptop, performance governor)

Same-session triad 319.7 GB/s. W4 fused vs the two denominators:

| H | S | vs two-pass | vs FP16 fused |
|---|---|---|---|
| 32 | 512 | 4.34x | 0.86x |
| 32 | 2048 | 2.48x | 0.59x |
| 32 | 4096 | 2.52x | 1.12x |
| 8 | 512 | 1.84x | 0.90x |
| 8 | 2048 | 1.44x | 0.90x |
| 8 | 4096 | 1.41x | 0.89x |

Read this honestly. The fusion win over the two-pass is real and
everywhere — the dequant round-trip to DRAM is eliminated as in the GEMM
rungs. But against the FP16-cache fused baseline the quantized kernel
only wins at the largest cache, and loses or washes below it. The output
phase is instruction-bound (per-element dequant with 8-way redundant word
touches — 12–44 GB/s effective, nowhere near triad), not byte-bound; the
4x byte savings only pay once the FP16 cache spills out of L2 (64 MB at
H=32/S=4096, where the FP16 path degrades 220 -> 118 GB/s while the 16 MB
packed cache stays resident). The win that exists is cache-residency
extension — the quantization argument that actually holds for long-context
decode. The named lever for the small-cache loss is a chunk-staged
shared-memory V pass (dequant a position tile once, not per output
element); recorded, not done.

(2026-11-02: the flash-rungs rerun re-measured these paths in session —
four of six cells reproduce this table exactly; the two L2-boundary cells
moved with the FP16 cache's residency regime: 32/2048 0.59x -> 1.11x
(219.7 -> 117.9 GB/s on the 33.5 MB cache) and 8/4096 0.89x -> 0.77x,
with the 8/4096 two-pass ratio moving 1.41x -> 0.91x on the same flip.
The conclusion — W4 wins once the FP16 cache spills L2 — is unchanged.
This table keeps its 2026-11-01 values; the JSON now holds the 11-02 run,
and the 11-01 values live in git history.)

## R5.5 + R6 + R7 contents (2026-11-01)

- `include/w4/kvflash.cuh` — split-S ("flash-decoding") decode attention.
  The S <= 4096 bound of `kvattention.cuh` was the static scores array,
  not a law: phase 1 (one block per (head, 256-position tile)) computes
  the tile max, tile sum, and weighted V accumulation into a cached
  partials scratch; phase 2 (one block per head) rescales and combines.
  Nothing per-S lives in shared memory, so S is bounded only by the tile
  grid (65535 tiles) — and since the grid is H x tiles, the H=8 occupancy
  hole of one-block-per-head is gone with it. Three cache variants:
  `attention_w4_flash` (the R5 dequant-at-load walk, unchanged),
  `attention_fp16_flash` (same structure over an FP16 cache),
  `attention_f8_flash` (R6 below). The `k_out`/`v_out` hooks still
  reproduce the per-head dequant bitwise — every position is touched
  exactly once.
- `include/w4/kvfp8.hpp` + `dequant_f8`/`flash_f8_phase1` — FP8 (E4M3) KV
  cache: one byte per element, one FP16 scale per (position, group) on
  the W4 group machinery. Quantizer s = amax/448, no saturation (the MX
  rung's ethos); branch-free exact decode (the R4 LUT lesson: no
  function-local tables on device). NaN encodings are never emitted by
  the quantizer and decode as +/-480 if they appear — documented, not
  handled.
- `include/w4/kvpaged.hpp` + `attention_w4_flash_paged` — paged W4: a
  per-head pool of npages x page positions plus a block table. Paging
  changes where a position lives, never what it contains; tiles partition
  the logical range, hooks write in logical order and stay bitwise, and
  the prefix-sharing exhibit decodes two sequences (one a prefix of the
  other) out of a single pool.
- Gates: hooks bitwise vs `dequant_head`/`dequant_head_f8` at every
  config, including S past the old bound; output vs the FP64 references
  at rel L2 <= 1e-3; flash vs single-block at 2e-3; paged bitwise vs
  contiguous flash on the same logical cache; entry-throw contracts
  (dims, D shared bound, gs/G, nulls, zeros discipline, the tile-grid S
  bound, FP8 word/base alignment, paged pool geometry and its 2^31 row
  bound). CTest 2/2 = 493 device + 212922 host checks, zero warnings.

## R5.5–R7 numbers (2026-11-02 pinned rerun, RTX 5060 Laptop, performance governor + platform profile)

Same-session triad 342.0 GB/s (the 2026-11-01 runs sat at ~320 — quote
the triad per session). Ten decode-class cases, raw reps + metadata in
`data/w4a16_kv_bench.json` (this run superseded the R5-era JSON).

Split-S flash16 vs the single-block FP16 kernel: wins everywhere,
1.5x–10.4x on the shared cases (2.6x at H=32/S=4096, 10.4x at H=8/S=4096
— the occupancy fix is the whole story at H=8), and runs at 290–350 GB/s
effective at large S — at triad.

flashw4 vs flash16 (same kernel structure, so the comparison isolates
cache bytes):

| H | S=512 | S=2048 | S=4096 | S=8192 | S=32768 |
|---|---|---|---|---|---|
| 32 | 0.74x | 0.58x | 1.01x | 1.00x | 1.02x |
| 8 | 0.83x | 0.74x | 0.63x | 0.97x | 0.99x |

Same direction as R5, sharper: the W4 cache wins nothing in wall time —
parity from the L2 spill up, losses below it. At H=32/S=32768 flashw4
moves a quarter of flash16's bytes and still ties (92 vs 350 GB/s
effective): dequant at load is instruction-bound, not byte-bound, at
every size measured. The win that exists is cache-residency extension —
a 4x smaller cache — and past S=4096 the flash path is the only path.

flashf8 vs flash16: 0.19–0.41x at every size — the E4M3 decode is
pricier and one uint32 covers only four elements (twice the score-walk
word iterations of W4); even at 134 MB, deep in DRAM, it runs 62 GB/s
effective. FP8 KV's footing on this data is accuracy: rel L2 2.57e-2 vs
W4's 1.17e-1 — 4.6x closer for 2x the bytes. W4 wins bytes, FP8 wins
accuracy, neither wins wall time against FP16 at decode shapes; that
trade is now measured, not asserted.

Paged vs contiguous flashw4 at page=16: 0.51–0.66x — the block-table
indirection costs a third to a half at small pages; the page-size sweep
is the recorded lever.

## R2 contents (2026-10-31)

- `include/w4/fused.cuh` — fused dequant+GEMM (`gemm_fused`). One warp per
  group of R output rows, lanes stride packed words along K (coalesced
  128 B steps); each weight is dequantized once via `dequant_one` and FMA'd
  into R*MTILE FP32 accumulators; shuffle reduction per row at the end. x
  is converted to FP32 once per call (per-FMA `__half2float` re-conversion
  measured as the binding constraint without it). MTILE/R dispatch:
  M <= 4 -> (pow2(M), 1); M > 4 -> (4, 4) — 128 regs, no spills, 2
  blocks/SM. MTILE 16/32/64 were measured and rejected (94 regs halves
  occupancy; 148 regs drops to 1 block/SM; 255 spills). The same
  `dequant_one` call is the whole dequant path, so the bitwise gate below
  applies unchanged.
- Gates (CTest 2/2, zero warnings): everything R1 gated, plus fused
  dequant bitwise vs `dequant_all` through the `w_out` hook across the
  same matrix (including M > 16 configs for the m-tiling guards), fused
  GEMM vs the FP64 oracle at rel L2 <= 1e-3, fused vs cuBLAS output at
  rel L2 <= 2e-3, and the fused entry-point throw contracts.
- `benchmarks/bench_fused.cu` — fused vs two-pass re-measured in the same
  session (ratios immune to cross-session machine state), same shapes as
  the baseline bench, raw reps + metadata in `data/w4a16_fused_bench.json`.

## R2 numbers (2026-11-01, RTX 5060 Laptop, governor-pinned)

Same-session triad 320.6 GB/s; baseline re-measured in session. Raw reps +
metadata in `data/w4a16_fused_bench.json`. The pre-committed bar was
>= 1.5x e2e at decode shapes.

| M | N=4096 | N=14336 |
|---|---|---|
| 1 | 7.33x | 8.25x |
| 8 | 2.29x | 2.87x |
| 16 | 1.37x | 1.64x |
| 64 | 0.44x | 0.45x |

M=1 and M=8 clear the bar; M=16 clears it at N=14336 and misses at
N=4096; M=64 loses outright. The M=64 loss is structural for SIMT FP32:
N*M*K FMAs on CUDA cores plus packed-W re-reads from m-tiling and x L2
traffic (~2.4 TB/s measured effective — the kernel is L2-bound before it
is FMA-bound). Both are exactly what WMMA fragment tiling removes, which
is the R3 justification. An unpinned run on 2026-10-31 showed the same
ordering at lower magnitude (M=1 5.8–7.3x, M=64 0.36–0.45x), and a pinned
run that same night matched the table above to a few hundredths
(7.25/8.31x at M=1); the numbers here are the record.

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

- Emulated formats beyond the two landed (MXFP4 GEMM, E4M3 KV); the
  chunk-staged V pass named in the R5 numbers; cp.async staging for the
  MMA pipeline and the SIMT/MMA dispatcher; the paged page-size sweep;
  the equal-bytes experiment the flash path now reaches — W4 at 4x the
  context length for the FP16 cache's bytes, at the same latency class.

## Ceiling honesty

Matching Marlin-class SOTA is months of work and not the point. The portfolio
piece is: bottleneck identified, pipeline fused, measured honestly. Bench-size
correctness is covered by the bitwise dequant gate only; the FP64 oracle is
gated at small sizes, so no precision claim is made at benchmark scale.
