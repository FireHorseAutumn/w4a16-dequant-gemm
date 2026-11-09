# Rung record

The full engineering record for the W4A16 fused dequant GEMM: every
rung, every gate, every pinned table, every measured rejection. The
README carries the headline; this file carries the evidence. All
numbers are governor-pinned runs on an RTX 5060 Laptop (8151 MiB) with
the same-session FP64 triad quoted per run, unless marked unpinned.
Error numerics live in `../ERROR_ANALYSIS.md`.

## Packed format (R1)

- Logical W is N x K row-major. Groups partition K; `gs` is a multiple
  of 8.
- 8 nibbles per uint32, low nibble first. FP16 scales per (row, group).
  Asymmetric mode adds a packed 4-bit zero-point per (row, group).
- Asymmetric group ranges are expanded to bracket zero before
  quantization. Without it, one-sided groups clamp the zero-point and
  corrupt the whole group. The R1 gate matrix caught this; bound
  failures were ~2x tolerance before the fix.
- Symmetric dequant: `w = scale * (q - 8)`. Asymmetric:
  `w = scale * (q - z)`.
- `dequant_one` is shared by host and device, so the reference and the
  pipeline are the same arithmetic.

## R1: format, oracle, two-pass baseline

Contents:

- `include/w4/pack.hpp`: format, host quantizer, host dequant
  reference, FP64 GEMM oracle.
- `include/w4/dequant.cuh`: naive dequant (one thread per packed word)
  plus the cuBLAS FP16-GEMM/FP32-accum two-pass baseline, which is also
  the correctness reference.

Gates (CTest 2/2):

- Quantizer per-element bounds over gs {8, 16, 32, 48, 64, 128},
  including ragged K and `gs % 32 != 0` (the silent-wrong-output shape
  from the vLLM RDNA W4A16 issue class), degenerate all-zero groups,
  and a hand-computed pin.
- Throw contracts: malformed dims/group sizes, size mismatches,
  non-finite inputs (NaN is otherwise swallowed by min/max scans),
  group scales outside the FP16 range (underflow below |w| ~4e-7,
  overflow above ~4.5e5). All throw on host and device entries. The
  device launcher also rejects launches whose 32-bit indexing would
  wrap.
- Device dequant bitwise-identical to the host reference on every
  config.
- cuBLAS GEMM vs the FP64 oracle at rel L2 <= 1e-3; measured worst
  2.2e-4 on identical FP16 inputs. The gate catches silent wrong
  output, not rounding.

Baseline numbers (triad 321.6 GB/s; `data/w4a16_baseline_bench.json`):

- Dequant phase: 147.9–151.2 GB/s across all eight shapes, ~46–47% of
  triad. One word per thread is not bandwidth-optimal; that gap is R2's
  opening measurement. Rep spreads 0.5–1.8%.
- cuBLAS phase: 343–658 GB/s analytic-minimum bytes; spreads 0.7–6.7%.
- Unpinned runs showed the same dequant rate in fast clock windows
  (137–140 GB/s against a 256–259 GB/s triad) but intermittent ~3.6x
  degradation to ~39 GB/s under mid-case clock migration. Pinned, the
  degradation is gone: it was clock migration.
- The two-pass pays 2·N·K FP16 traffic (write W in dequant, read W in
  the GEMM). The fused kernel never pays it.

## R2: fused SIMT dequant+GEMM

Contents: `include/w4/fused.cuh` (`gemm_fused`).

- One warp per group of R output rows; lanes stride packed words along
  K in coalesced 128 B steps. Each weight dequants once via
  `dequant_one` into R·MTILE FP32 accumulators; shuffle reduction per
  row at the end.
- x is converted to FP32 once per call; per-FMA `__half2float`
  re-conversion measured as the binding constraint without it.
- Dispatch: M <= 4 -> (pow2-ceil(M), 1); M > 4 -> (4, 4). 128 regs, no
  spills, 2 blocks/SM.
- MTILE 16/32/64 measured and rejected: 94 regs halves occupancy, 148
  regs drops to 1 block/SM, 255 spills.

Gates: everything R1 gated, plus fused dequant bitwise via the `w_out`
hook across the full matrix (including M > 16 m-tiling configs), fused
vs oracle at rel L2 <= 1e-3, fused vs cuBLAS at 2e-3, entry throws.
CTest 2/2, zero warnings.

Numbers (triad 320.6 GB/s; baseline re-measured in session;
pre-committed bar >= 1.5x e2e at decode shapes):

| M | N=4096 | N=14336 |
|---|---|---|
| 1 | 7.33x | 8.25x |
| 8 | 2.29x | 2.87x |
| 16 | 1.37x | 1.64x |
| 64 | 0.44x | 0.45x |

- M=1 and M=8 clear the bar; M=16 clears it at N=14336 and misses at
  N=4096; M=64 loses outright.
- The M=64 loss is structural for SIMT FP32: N·M·K FMAs on CUDA cores
  plus packed-W re-reads from m-tiling and x L2 traffic (~2.4 TB/s
  measured effective; L2-bound before FMA-bound). Both are exactly what
  WMMA fragment tiling removes: the R3 charter.
- An unpinned run showed the same ordering at lower magnitude (M=1
  5.8–7.3x, M=64 0.36–0.45x); a same-night pinned run matched the table
  to a few hundredths (7.25/8.31x at M=1).

## R3: fused WMMA GEMM

Contents: `include/w4/mma.cuh` (`gemm_mma`).

- Block tile 64x64, K in chunks of 32. 256 threads dequant one packed
  word each into a shared FP16 tile (`dequant_one`, unchanged) and
  stage the x tile, both zero-padded past the N/M/K edges: ragged K and
  partial tiles need no special path.
- 8 warps each own two 16x16x16 WMMA fragment MMAs per k-fragment, FP32
  accumulators; stores go through a per-warp FP32 shared tile with
  guarded copies. (CUDA 13 has no float-acc -> half-pointer
  `store_matrix_sync` overload; the guarded copy does the
  `__float2half_rn`.)
- W is read exactly once at M <= 64, the FMA floor drops to the
  tensor-core rate, and fragment loads take x reuse from shared: all
  three SIMT M=64 walls removed at once. The `w_out` bitwise hook is
  intact.

Gates: the full fused gate set on the MMA path. CTest 2/2, zero
warnings.

Numbers (triad 320.6 GB/s; same bench, baseline re-measured):

| M | N=4096 | N=14336 |
|---|---|---|
| 1 | 1.41x | 1.88x |
| 8 | 1.35x | 1.92x |
| 16 | 1.35x | 1.90x |
| 64 | 1.40x | 1.79x |

- Flat across M and far under triad (35–45 GB/s effective): the
  signature of a staging-latency-bound kernel.
- The M=64 SIMT loss is gone (0.44x -> 1.79x at N=14336). That was the
  charter.
- Tile-shape A/Bs run and rejected: chunk 64 doubles the x-staging
  footprint; N-tile 32 halves W work per chunk while x staging stays
  fixed.
- The same run re-measured the SIMT path and reproduced the R2 pinned
  table to a few hundredths.

## R3.5: register-prefetch double buffer

- The next chunk's packed word, scale/zero, and x slivers prefetch into
  registers under the current chunk's MMA; ws/xs alternate two shared
  buffers; one `__syncthreads` per chunk instead of two. Numerics
  unchanged; CTest 2/2.
- Unpinned same-session A/B (triad 264–265 GB/s, a low-power-profile
  session): flat at N=4096, +4–6% at N=14336.
- Pinned A/B (triad 341–342 GB/s): 1.35–1.44x at N=4096 and
  1.82–1.88x at N=14336, the single-buffered table within a few
  hundredths. The +4–6% was a low-power-session artifact. Register
  prefetch is pinned parity with single-buffering, still 35–46 GB/s
  effective.

## R3.6: cp.async staging

- The x tile rides cp.async straight into shared, no register round
  trip; packed W words land in a small shared stage and pay the round
  trip through the dequant; scales/zeros still ride registers.
- Issue runs two chunks ahead of the MMA (xs three slots deep, ws and
  the W stage two); one `__syncthreads` per chunk; the next chunk's
  dequant sits between issue and MMA with no barrier of its own.
- Occupancy 2 -> 3 blocks/SM (96 -> 57 registers, 25.6 -> 31.7 KB
  shared). Numerics unchanged; CTest 2/2.
- The landing-day A/B was unpinned and unresolvable (triads 259–265
  GB/s, clock migration up to 4x inside single cases); on fast-window
  minima cp.async read as parity, 0.92–1.13x per case.
- A hybrid (W keeps the register ride, only x on cp.async) measured the
  same parity and was rejected: dropping the register ride was the
  point of the rung.
- Pinned A/B (governor + platform profile, two interleaved runs per
  binary, triads 341.1–342.3 GB/s, run-to-run medians within 0.5%):
  parity at N=4096 for M >= 8, +2% at M=64/N=4096, −3.5% at
  M=1/N=4096, −1.4 to −4.2% at N=14336 for M <= 16.
- Verdict: the staging restructure is measured out as the lever across
  all three designs. Single-buffer, register prefetch, and cp.async
  land within a few percent of each other at 34–46 GB/s effective; the
  binding constraint is something none of them touched. The named
  re-opens: the per-chunk barrier and the chunk size itself (the R3
  chunk-64 rejection predates pipelining).
- Caveat, recorded plainly: the sampler logged transient SW-power-cap
  flags in 14 of 360 samples, all at 28–32 W between-case transitions;
  every run's triad sat in the 341–342 band and MMA rep spreads stayed
  <= 1.02 apart from one 1.11 on the 0.25 ms M=1/N=4096 case.
- `data/w4a16_fused_bench.json` holds the R3.6-era tip run; the R3
  table stays the single-buffered run of record (its JSON is in git
  history at `1d65ece`).

## R4: MXFP4

Contents: `include/w4/mx.hpp` (format, host quantizer, reference
dequant) and `include/w4/mxfused.cuh` (naive dequant, `gemm_fused_mx`,
`gemm_mma_mx`).

- MXFP4: E2M1 elements plus an E8M0 power-of-two block scale over
  32-element blocks. Table lookup plus exponent shift, no zero-point:
  different arithmetic from W4, so its own header.
- The mapping, x staging, dispatch, and store guards follow the
  `fused.cuh` / `mma.cuh` skeletons unchanged; only the scale fetch
  (one E8M0 byte per 4 words) and the dequant step differ, so the R2/R3
  performance conclusions carry over instead of being re-derived.
- Measured dequant lesson: `dequant_one_mx` is semantically a table
  lookup, but a function-local LUT indexed by a runtime nibble lands in
  local memory on device and cost the fused SIMT kernel ~7–8x
  end-to-end (M=1 went 0.71x -> 5.61x after the fix). The fix is a
  branch-free arithmetic decode of E2M1; every table value is exactly
  representable, so host and device stay bitwise-identical.

Gates: full set on both MX paths (bitwise dequant vs `dequant_all_mx`,
fused/MMA bitwise through `w_out`, oracle 1e-3, cuBLAS 2e-3, entry
throws). Partial 32-element blocks and ragged word tails covered by
K in {72, 100, 104, 40}. CTest 2/2 (271 checks at landing), zero
warnings.

Numbers (triad 321.4 GB/s; MX two-pass re-measured in session):

| M | SIMT N=4096 | SIMT N=14336 | MMA N=4096 | MMA N=14336 |
|---|---|---|---|---|
| 1 | 5.55x | 6.34x | 1.34x | 1.79x |
| 8 | 2.01x | 2.43x | 1.29x | 1.81x |
| 16 | 1.14x | 1.33x | 1.28x | 1.80x |
| 64 | 0.34x | 0.36x | 1.34x | 1.71x |

- The W4 dispatch picture reproduces on a different dequant arithmetic:
  SIMT wins decode shapes, MMA removes the M=64 loss.
- MX fused SIMT runs under its W4 counterpart (5.55x vs 7.33x at M=1,
  N=4096): the scale-byte fetch per four words plus the decode ALU.
- Quantization accuracy on Gaussian weights: MXFP4 rel L2 1.15e-1 vs
  W4A16 gs=128 symmetric 1.17e-1. Parity on this distribution; MX's
  finer blocks (32 vs 128 elements) do not buy accuracy because the
  E8M0 scale is a bare power of two where W4's is FP16. One
  distribution, one seed set; not a general format ranking.

## R5: KV-cache decode attention (single-block)

Contents: `include/w4/kvcache.hpp` (the cache as H independent slabs of
the PackedW format: positions are rows, head dim is the contracted
axis, groups partition D) and `include/w4/kvattention.cuh`
(`attention_fp16` baseline, `attention_w4` with dequant at load).

- The R1 quantizer and its contracts carry over unchanged.
- One block per head; warp-per-position scores with shuffle reduction;
  softmax as two block-wide reductions (max, then sum); threads stride
  D for the output sum with probabilities broadcast from shared; query
  staged in shared once.
- Static shared bounds with hard entry throws: S <= 4096 (scores,
  16 KB) and D <= 512 (query stage, 2 KB). Past that wants flash-style
  online softmax over tiles: a different kernel (R5.5).

Gates: `k_out`/`v_out` hooks reproduce `dequant_head` bitwise per head;
output vs the FP64 reference at rel L2 <= 1e-3; vs the device two-pass
at 2e-3. Configs cover S=1, ragged words (D=96), both sign modes, and
S=4096 exactly on the bound. CTest 2/2, 350 checks at landing, zero
warnings.

Numbers (triad 319.7 GB/s; `data/w4a16_kv_bench.json`):

| H | S | vs two-pass | vs FP16 fused |
|---|---|---|---|
| 32 | 512 | 4.34x | 0.86x |
| 32 | 2048 | 2.48x | 0.59x |
| 32 | 4096 | 2.52x | 1.12x |
| 8 | 512 | 1.84x | 0.90x |
| 8 | 2048 | 1.44x | 0.90x |
| 8 | 4096 | 1.41x | 0.89x |

Read honestly:

- The fusion win over the two-pass is real and everywhere: the dequant
  round trip to DRAM is eliminated, as in the GEMM rungs.
- Against the FP16-cache fused baseline, W4 wins only at the largest
  cache and loses or washes below it. The output phase is
  instruction-bound (per-element dequant with 8-way redundant word
  touches; 12–44 GB/s effective), not byte-bound. The 4x byte savings
  pay only once the FP16 cache spills L2 (64 MB at H=32/S=4096, where
  FP16 degrades 220 -> 118 GB/s while the 16 MB packed cache stays
  resident).
- The win that exists is cache-residency extension: the quantization
  argument that actually holds for long-context decode.
- Named lever for the small-cache loss: a chunk-staged shared-memory V
  pass (dequant a position tile once, not per output element).
  Recorded, not done.
- A later flash-rungs rerun re-measured these paths in session: four of
  six cells reproduce exactly; the two L2-boundary cells moved with the
  FP16 cache's residency regime (32/2048 0.59x -> 1.11x on 219.7 ->
  117.9 GB/s on the 33.5 MB cache; 8/4096 0.89x -> 0.77x; the 8/4096
  two-pass ratio moved 1.41x -> 0.91x on the same flip). The conclusion is unchanged: W4 wins
  once the FP16 cache spills L2. The table above keeps its original
  values; the JSON holds the later run; the earlier values live in git
  history.

## R5.5 + R6 + R7: split-S flash, FP8 cache, paged W4

Contents:

- `include/w4/kvflash.cuh`: split-S (flash-decoding) decode attention.
  The S <= 4096 bound was the static scores array, not a law. Phase 1
  (one block per (head, 256-position tile)) computes tile max, tile
  sum, and weighted V into a cached partials scratch; phase 2 (one
  block per head) rescales and combines. Nothing per-S lives in shared:
  S is bounded only by the tile grid (65535 tiles). The grid is
  H x tiles, so the H=8 occupancy hole of one-block-per-head is gone
  with it. Variants: `attention_w4_flash`, `attention_fp16_flash`,
  `attention_f8_flash`.
- `include/w4/kvfp8.hpp` plus `dequant_f8` / `flash_f8_phase1`: FP8
  E4M3 KV cache, one byte per element, one FP16 scale per (position,
  group) on the W4 group machinery. Quantizer s = amax/448, no
  saturation; branch-free exact decode (the R4 LUT lesson: no
  function-local tables on device). NaN encodings are never emitted by
  the quantizer and decode as +/-480 if they appear: documented, not
  handled.
- `include/w4/kvpaged.hpp` plus `attention_w4_flash_paged`: paged W4, a
  per-head pool of npages x page positions plus a block table. Paging
  changes where a position lives, never what it contains; tiles
  partition the logical range, hooks write in logical order and stay
  bitwise; the prefix-sharing exhibit decodes two sequences (one a
  prefix of the other) out of a single pool.

Gates: hooks bitwise vs `dequant_head` / `dequant_head_f8` at every
config including S past the old bound; output vs FP64 references at
1e-3; flash vs single-block at 2e-3; paged bitwise vs contiguous flash
on the same logical cache; entry-throw contracts (dims, D shared bound,
gs/G, nulls, zeros discipline, tile-grid S bound, FP8 word/base
alignment, paged pool geometry and its 2^31 row bound). CTest 2/2 =
493 device + 212922 host checks at landing, zero warnings.

Numbers (triad 342.0 GB/s, governor + platform profile; earlier runs
sat at ~320, so quote the triad per session; ten decode-class cases,
`data/w4a16_kv_bench.json`):

- Split-S flash16 vs the single-block FP16 kernel: wins everywhere,
  1.5x–10.4x on the shared cases (2.6x at H=32/S=4096, 10.4x at
  H=8/S=4096; the occupancy fix is the whole story at H=8), and runs at
  290–350 GB/s effective at large S. At triad.

flashw4 vs flash16 (same kernel structure, so the comparison isolates
cache bytes):

| H | S=512 | S=2048 | S=4096 | S=8192 | S=32768 |
|---|---|---|---|---|---|
| 32 | 0.74x | 0.58x | 1.01x | 1.00x | 1.02x |
| 8 | 0.83x | 0.74x | 0.63x | 0.97x | 0.99x |

- The W4 cache wins nothing in wall time: parity from the L2 spill up,
  losses below it. At H=32/S=32768 flashw4 moves a quarter of flash16's
  bytes and still ties (92 vs 350 GB/s effective): dequant at load is
  instruction-bound, not byte-bound, at every size measured.
- flashf8 vs flash16: 0.19–0.41x at every size. The E4M3 decode is
  pricier and one uint32 covers only four elements (twice the
  score-walk word iterations of W4); even at 134 MB, deep in DRAM, it
  runs 62 GB/s effective. FP8 KV's footing on this data is accuracy:
  rel L2 2.57e-2 vs W4's 1.17e-1, 4.6x closer for 2x the bytes. W4
  wins bytes, FP8 wins accuracy, neither wins wall time against FP16 at
  decode shapes. That trade is now measured, not asserted.
- Paged vs contiguous flashw4 at page=16: 0.51–0.66x. The block-table
  indirection costs a third to a half at small pages.
- The page-size sweep (page in {16, 32, 64, 128, 256}, shuffled and
  in-order block tables, six (H, S) cells up to H=32/S=32768, bitwise
  clean): the ratio is FLAT at 0.60–0.64x, and shuffled placement costs
  nothing against in-order. The paged tax is not per-page table lookups
  and not placement; it is per-position address arithmetic, the runtime
  `pos / page` and `pos % page` on every position of both walks, which
  a power-of-two page size cannot remove because `page` is a runtime
  value. The renamed lever: a power-of-two fast path (shift/mask) or
  page-aligned tiling. A kernel change, recorded not done.

## R8: SIMT/MMA dispatcher

Contents: `include/w4/dispatch.cuh` (`gemm`, `gemm_mx`): one host-side
branch on M per call. M < 9 -> SIMT fused; M >= 9 -> MMA.

- The crossover is the pinned run of record, not an argument: SIMT
  leads through M=8 (2.29x / 2.87x vs MMA's 1.35x / 1.92x), MMA is
  within noise at M=16/N=4096 (1.38x vs SIMT's 1.37x) and leads
  everywhere from there, and it is the only path that does not lose at
  M=64.
- The MX pinned table reproduces the same picture on a different
  dequant arithmetic (SIMT 2.01x / 2.43x at M=8; MMA wins from M=16),
  so one threshold serves both.
- N is not dispatched on: the only split cell (W4, M=16, N=4096) is a
  wash.
- Dispatch changes which kernel runs, never what it computes: both
  callees share `dequant_one` and carry the same `w_out` hook.

Gates: dispatch output memcmp'd against the expected callee's output on
the full config matrix (routing pinned bitwise), `w_out` bitwise
through the dispatch, entry contracts thrown on both sides of the
threshold. CTest 2/2 = 568 device + 212922 host checks at landing, zero
warnings.

The two named MMA re-opens, exercised and rejected on measurement
(variants bitwise-gated against the tip, 9 reps, triad 341.1 GB/s):

- Chunk 64 loses 26–29% at every shape: the 48 KB static-shared limit
  forces two-slot buffers and issue-1-ahead; occupancy 3 -> 2 blocks/SM.
  The R3 rejection stands post-pipelining.
- One barrier per two chunks loses 1–7%: four-slot buffers, 2
  blocks/SM. The halved barrier count does not pay for the lost
  occupancy.
- Verdict: the MMA kernel is not staging-bound, not barrier-bound, not
  chunk-size-bound. What remains is the instruction stream itself. No
  further MMA levers are chartered.

## Correctness audit

Deep line-by-line audit of code and tests (branch `correctness-pass`):

- The group-walk classes vLLM's exllama kernels get wrong
  (vllm-project/vllm#60921: gs in {16, 48, 96, 160, 192} with
  K % gs == 0, N % 128 != 0 at 2 <= M <= 8, single-group K with
  K % 32 != 0, gs > K) are gated and pass. This kernel computes each
  packed word's group as `k0 / gs`, so the failing classes are a gate
  matrix here, not a fix.
- Three latent contract/UB bugs fixed, each red-demonstrated then
  pinned: K bounded at every entry (`W4_MAX_K`, 32-bit ceil-adds could
  wrap), the fused x pre-pass grid guarded (`ceil(M*K/256)` on
  gridDim.x), the MXFP4 subnormal-amax floor taken before log2.
- compute-sanitizer memcheck and racecheck clean.
- Current suite: CTest 2/2 = 930 device + 212928 host checks, zero
  warnings.

## Open levers

- The chunk-staged shared-memory V pass for the small-cache KV loss.
- The paged power-of-two fast path.
- The equal-bytes experiment the flash path now reaches: W4 at 4x the
  context length for the FP16 cache's bytes, at the same latency class.
- Emulated formats beyond the two landed (MXFP4 GEMM, E4M3 KV).

## Ceiling honesty

Matching Marlin-class SOTA is months of work and not the point. The
portfolio piece is: bottleneck identified, pipeline fused, measured
honestly. Bench-size correctness is covered by the bitwise dequant gate
only; the FP64 oracle is gated at small sizes, so no precision claim is
made at benchmark scale.
