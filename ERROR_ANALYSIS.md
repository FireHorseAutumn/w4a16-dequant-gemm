# Error analysis — W4A16 (and MXFP4) vs the FP16 reference

All numbers below are produced by `benchmarks/error_analysis.cpp` (host,
fixed seeds) and are exactly reproducible: `cmake --build build --target
error_analysis && ./build/error_analysis`. These are value measurements;
machine state does not move them.

## The two error sources, separated

The end-to-end error has exactly two components, and they are three
orders of magnitude apart:

- **Quantization** of W to 4-bit: rel L2 ~1e-1 (tables below).
- **FP16 storage of W and x, plus accumulation-order noise**: rel L2
  2.0e-4 (FP16-rounded W vs the FP64 source, FP64 accumulation, K up to
  4096). The device kernels add nothing beyond this: the suite gates
  fused output against cuBLAS on the *same* dequantized W at 2e-3, and
  the measured worst is ~3e-4.

So the error question for this pipeline is the quantization question.
Everything below is about it.

## Group size: the error vs the bits (Gaussian weights, 512x4096)

| format | gs | bits/elt | rel L2 | max abs err |
|---|---|---|---|---|
| w4 sym | 8 | 6.000 | 7.18e-2 | 3.35e-1 |
| w4 sym | 16 | 5.000 | 8.53e-2 | 3.47e-1 |
| w4 sym | 32 | 4.500 | 9.71e-2 | 3.47e-1 |
| w4 sym | 64 | 4.250 | 1.08e-1 | 3.47e-1 |
| w4 sym | 128 | 4.125 | 1.17e-1 | 3.47e-1 |
| mxfp4 | 32 | 4.250 | 1.15e-1 | 9.03e-1 |

Two reads. First: error grows slowly with group size — 16x fewer scale
bytes costs 1.6x the error. The grid is set by the group's max, and a
Gaussian group's max grows like sqrt(2 log gs), which is all the growth
there is. Second: MXFP4 sits at parity with W4 sym at matched bits
(4.25), as the R4 bench found — but note the max abs err column: E8M0's
power-of-two scale overshoots the block max by up to 2x (no-saturation
rule), so MX's worst single-element error is 2.6x W4's while its
average matches. MX buys tail behavior with worst-case behavior.

Asymmetric mode (zero-point) buys back 15–20% of the error at every
group size (5.71e-2 vs 7.18e-2 at gs=8, 1.01e-1 vs 1.17e-1 at gs=128)
for 0.03–0.5 extra bits/elt. On a zero-mean distribution this is pure
grid placement: 16 codes spanning [lo, hi] beat 15 codes spanning
[-amax, amax] whenever the group is not symmetric.

## Where the error lives

Within a group, the max element sets the scale; the error budget is
spent by the largest element and paid by the smallest. W4 sym gs=128,
elements binned by |w|/amax of their group:

| \|w\|/amax bin | share of elements | mean abs err | mean rel err |
|---|---|---|---|
| 0.0–0.1 | 22.2% | 1.20e-1 | 9.1e-1 |
| 0.1–0.2 | 20.4% | 7.2e-2 | 1.7e-1 |
| 0.2–0.3 | 17.3% | 1.1e-1 | 1.6e-1 |
| 0.4–0.5 | 9.9% | 7.7e-2 | 6.0e-2 |
| 0.7–0.8 | 2.6% | 8.6e-2 | 4.2e-2 |
| 0.9–1.0 | 1.6% | 5.9e-2 | 2.4e-2 |

Absolute error is roughly uniform across the group (~0.1, half a ULP of
the coarse grid); relative error is not. The bottom decile — a fifth of
all weights — carries ~91% relative error. This is why W4 works at all:
the elements that carry the dot product (the large ones) are the ones
the grid resolves, and the error on the rest is bounded, uniform, and
uncorrelated with x.

## Outliers are the group-size argument

Inject 0.1% of elements at 20x magnitude:

| gs | rel L2 (clean 8.53e-2 / 1.17e-1) |
|---|---|
| 16 | 9.28e-2 (+9%) |
| 128 | 1.98e-1 (+69%) |

Same outliers, same total count. At gs=16 an outlier inflates the scale
for 16 elements; at gs=128, for 128. Group size is the outlier
exposure, and this — not the average case — is what group size is
actually for. Weight distributions with outlier structure (post-training
LLM weights have it) buy their error back with smaller groups, not with
finer codes.

## Distribution shape is a tails question

W4 sym gs=128: uniform 7.06e-2, Gaussian 1.17e-1, Laplace 1.60e-1.
Heavier tail, worse error, because the group max separates further from
the bulk while the grid still serves the max. Nothing in the format
cares about the body of the distribution; everything rides on the tail.

## End-to-end: what reaches the output

Quantized-W GEMM vs the FP16-reference GEMM, both FP64-accumulated so
the gap is quantization only (the dequantized values are FP16 by
construction and promote exactly):

| K | w4 sym gs=32 | w4 sym gs=128 | mxfp4 | FP16 ref vs FP64 |
|---|---|---|---|---|
| 256 | 9.39e-2 | 1.15e-1 | 1.10e-1 | 2.03e-4 |
| 1024 | 9.60e-2 | 1.16e-1 | 1.15e-1 | 2.08e-4 |
| 4096 | 9.85e-2 | 1.23e-1 | 1.17e-1 | 2.09e-4 |

The output error tracks the per-element error and is essentially flat in
K — 16x more accumulation moves it 5%, not 4x. The quantization error
is independent per element and partially cancels in the dot product;
there is no growth mechanism. And the whole pipeline sits ~600x above
the FP16-storage error, which never moves.

## What this means

For a serving stack: W4A16's error is a model-quality budget, not a
numerics risk. The kernel contributes nothing measurable beyond the
quantization itself — the device output is the dequantized-W GEMM to
within accumulation noise, gated. The levers that matter are group size
(outlier exposure), sign mode (grid placement), and format choice at
the tails (W4 vs MX), in that order. All three are quantization-time
decisions; none of them touch the kernel.

*Method note: per-element bounds, degenerate groups, and the
zero-bracketing invariant are gated in `tests/test_oracle.cpp`; the
tables above measure the error the gates permit, not correctness of the
artifact.*
