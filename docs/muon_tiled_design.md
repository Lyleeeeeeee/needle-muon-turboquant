# Reusing tiled intermediates in Muon Newton-Schulz iterations

The user approved the second optimization: reduce repeated layout conversions,
measure the actual speedup, and retain the implementation/results for the report.
Baseline revision: `e94a928cbecc78db6aaaaa047363b2f7140a007a`, including the earlier
Compact indexing optimization.

## Local native path

Use one native CPU function for the Newton-Schulz recurrence after the existing
Python/Needle float32 Frobenius normalization. Pack normalized X once, maintain
X, its transpose, Gram, squared Gram and product in the existing tiled format,
and unpack the final X once. X changes each iteration, so refresh its tiled
transpose once per iteration; do not claim that all data rearrangement vanishes.

Reuse existing MatmulTiled/AlignedDot, ScalarMul and EwiseAdd helpers in the
same operation/accumulation order. Reuse scratch buffers within a call without
retaining gradients or state across calls. Separate scalar multiplication and
addition passes preserve reference float32 rounding; do not introduce fused
multiply-add expressions for the NS polynomial.

Muon keeps normalization, momentum, NS coefficients/iteration count, final
transpose/aspect scaling and auxiliary AdamW unchanged. Add `use_tiled_ns=True`
with automatic native-CPU dispatch only for positive, tile-aligned dimensions.
Keep `use_tiled_ns=False` as the exact old reference path; other backends and
unaligned shapes fall back automatically. No generic NDArray/autodiff layout
change, global cache or stale-input reuse is introduced.

Alternatives were global matrix-layout caching or a new packed NDArray type.
Those require broader mutation/autodiff semantics. The local operator handles
the known NS recurrence without changing unrelated model operations.

## Correctness and counters

Before implementation, add failing tests for the native entry point, shape/size
validation, source preservation, exact reference float32 outputs, multi-step
optimizer parity, independent SVD-polynomial agreement, zero/missing gradients,
backend/shape dispatch and profiling counts. Test square, wide and tall matrices
whose normalized orientation is tile-aligned, plus unsupported fallbacks.

Keep logical matrix work explicit: three products per NS iteration, five NS
iterations, three hidden matrices =45 matrix products per hybrid update. The
reference path performs150 NS full-matrix layout passes plus7 other native
compactions for this model. The proposed native path has21 NS layout passes
(pack+five tiled transposes+unpack per matrix) plus the same7 other compactions.
These counts are not all equal-sized transfers. Track written elements/bytes
and distinguish measured generic Compact calls from derived internal passes.
Update profiler accounting so native NS time is not mislabeled as Python time.

## Bounded comparison

Compare reference and packed switches using the same final CPU binary, current
Compact implementation, Python/model/optimizer sources and fixed inputs. Five
alternating sequential process pairs measure aligned transforms, uninstrumented
fixed-gradient hybrid updates, an unchanged AdamW control, and128-step deep
Fashion-MNIST training at seed8/batch64 with the existing rates. Warmup and input
hashes are matched. Verify parameter/loss/feature hashes exactly before claiming
that the optimization preserves the reference trajectory. If exact parity is
not achieved, investigate rather than silently relaxing the primary comparison.

Report means/medians/paired speedups, timing variability and conversion counts;
five timing repetitions are not independent model-quality seeds. Retain source
and binary hashes, the reference switch, tests, raw reports and a scientific
plot. No new220-second retrieval-quality or general Muon superiority claim.
Historical reports remain tied to their recorded revisions.
