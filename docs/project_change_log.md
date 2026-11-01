# Project implementation change log

This file preserves implementation decisions and evidence to revisit while
writing the final project report. Repository documentation is in English.

## 2026-11-01: CPU Compact coordinate-carry indexing

**Reason.** The earlier Muon audit measured 157 native compactions per hybrid
optimizer update, taking about 6.66 ms in that diagnostic run. The inherited
HW3 `Compact` implementation decoded every element's coordinates using integer
division/remainder in every dimension. The surrounding tile/untile conversion
pipeline is the course-provided Python implementation.

**Change.** Replace only `Compact`'s coordinate decoding with an incrementing
coordinate vector and a signed 64-bit source index. For rank two, use direct
row/column loops to avoid coordinate bookkeeping. For higher ranks, advance the
last dimension and propagate carries at boundaries; stop after the final copy.
The change removes repeated division/remainder from the copy loop. The generic
path adds one small
O(rank) coordinate vector per call, whose allocation is included in timings;
the rank-two path does not allocate this vector.

**Scope.** The amount/order of copied data, number of Compact calls, matrix
product kernels, Python tile/untile pipeline, Muon/AdamW formulas, optimizer
routing and quantizers are unchanged. This is not fused Newton-Schulz or a new
optimizer. Both AdamW and Muon receive the improved shared CPU backend.

**Baseline.** Commit `efe063d3b67153ced96c8435a381866ee0077605` contains the
old kernel and all preceding equal-time results. Original homework archives
remain unchanged. Historical source-hash validators belong to their recorded
revision; do not reinterpret their timings as the optimized backend's results.

**Verification.** The full suite passes 105 checks. New direct-native tests
compare exact copied bits with independent NumPy strided views on 109 layouts,
including scalar/empty, singleton axes, nonzero offsets, signed and zero strides,
transpose, tile and untile. Source-buffer preservation and special floating
values (signed zero, infinities, NaN payloads) are checked. Old-vs-old control
measurement gives approximately 1.008x paired-median speedup on the predefined
128x128 tiled case, correctly failing a 1.2x speedup requirement before the edit.

**Measurement design.** See [the accepted design](compact_optimization_design.md)
and [the separate before/after report](../reports/compact_optimization/summary.md).
The comparison rebuilds both native versions with the same compiler/flags,
alternates five process-isolated old/new pairs, and records hashes and all raw
times. Fixed-gradient updates and 128-update Fashion-MNIST loops check exact
parameter/loss/embedding parity. These short measurements establish execution
speed and semantics, not a new convergence/retrieval-quality result.

**Measured refinement.** The first generic-carry candidate improved four-dimensional
packing but slowed two-dimensional transpose/broadcast/slicing. Its complete
source and five paired raw trials are preserved under
`reports/compact_optimization/generic_carry/`. The final implementation adds the
rank-two nested-loop path to address that measured regression. Do not describe
the initial generic carry as universally faster.

**Report-ready measured outcome.** Across the seven measured layouts, final native
Compact paired-median speedups are 2.23x to 8.15x. Instrumented native compaction
within a Muon update decreases from 6.69 to 2.67 ms, while the 45 native matrix
products remain about 7.8 ms. Uninstrumented Muon optimizer updates improve
1.263x. In 128-update fixed-seed training, the median full update decreases
from 25.82 to 19.97 ms for Muon (about 23% less time) and from 10.37 to 8.50 ms
for AdamW (about 18% less time). All measured parameter/loss/feature hashes are
identical. The measurements show execution-speed improvement; equal-time
retrieval quality has not been rerun with this backend. Five blocks are timing
repetitions, not five independent model-quality seeds.

## 2026-11-01: Reuse tiled Muon Newton-Schulz intermediates

**Reason.** The first Compact optimization made each layout conversion cheaper,
but left repeated pack/matmul/unpack sequences in every NS iteration. The user
approved reducing these repeated conversions and preserving the evidence.

**Change.** Add a local native CPU NS path in `src/muon_cpu.h`. Normalize with the
same Needle operations as before, pack X once, and keep all NS intermediate
matrices tiled. Refresh X's tiled transpose each iteration because X changes;
unpack the final result once. Reuse the existing MatmulTiled/AlignedDot,
ScalarMul and EwiseAdd operations and their separate float32 rounding stages.
Five scratch buffers are reused within a call, not cached across updates.

**Dispatch and reference.** Muon uses the native path by default for supported
tile-aligned CPU matrices. `use_tiled_ns=False` preserves the reference path;
NumPy/CUDA and unsupported dimensions automatically fall back. Momentum,
normalization, NS coefficients/count, aspect scaling, weight decay and auxiliary
AdamW are unchanged. The generic NDArray/autodiff/matmul interfaces are unchanged.

**Evidence and scope.** The full suite passes 126 checks, including direct
native/reference bit parity, independent float64 singular-value oracles, source
preservation, shape errors and fallback routing. Profiling counts logical matrix
products and internal derived layout passes separately, so work inside the new
native call is not mislabeled as Python time. See the
[accepted design](muon_tiled_design.md) and
[separate second-stage measurements](../reports/muon_tiled/summary.md).
The comparison uses the same final CPU binary for both switches, with the
previous Compact optimization already enabled. Baseline revision is
`e94a928cbecc78db6aaaaa047363b2f7140a007a`; this measures the second stage's
incremental effect, not an unpaired comparison with older timing results.

**Report-ready measured outcome.** Five paired timing repetitions show a Muon
hybrid optimizer median of 14.38 to 10.96 ms/update (paired-median speedup 1.351x).
The 128-step training median decreases from 19.56 to 15.94 ms/update (paired
speedup 1.239x; approximately 19% less time), while AdamW stays approximately
unchanged. All final parameters, 128 losses and 128 validation embeddings are
bit-identical between paths and reproduce the prior Compact-stage outputs.

**Layout work.** Each hybrid update still performs 45 logical matrix products.
Generic Compact calls decrease from 157 to 7, but the native path also performs
21 internal passes (three packs, fifteen tiled transposes and three unpacks).
Total logical layout passes decrease from 157 to 28. Counted/derived float32
layout writes decrease from 8.077312 to 1.441792 decimal MB; these are not physical
DRAM traffic, retained-state memory or peak process memory measurements.

**Interpretation.** The native path also reduces Python dispatch and reuses
within-call scratch buffers; the measured gain is not an isolated estimate of
layout reuse alone. These are execution-speed measurements at fixed work,
not evidence that Muon now beats AdamW at equal time. The 220-second retrieval
comparison has not been rerun. Five timing repetitions use one training seed,
not five independent model-quality seeds.
