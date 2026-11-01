# Coordinate-carry optimization for native CPU Compact

The user approved replacing per-element division/remainder coordinate decoding
with the previously reviewed coordinate-carry loop, and requested a durable
record for the final project report. This document records that accepted scope.

## Change and invariants

Modify only `Compact` in `src/ndarray_backend_cpu.cc`, plus explicit standard
headers needed by the implementation. Maintain one coordinate vector and signed
64-bit source index. Copy the current element, increment the last coordinate,
and propagate a carry only at dimension boundaries. Stop after the last copy.
Support valid nonzero-offset, permuted, sliced, broadcast/zero-stride, signed-
stride, scalar, singleton-axis and zero-element layouts. Preserve copied bits,
including signed zero and NaN payloads; do not change the source buffer.

The alternative is to specialize fixed ranks or fuse matrix layout conversion.
Those can reduce further work but increase scope. The approved generic carry
loop directly addresses repeated division/remainder without changing the number
of copies, matrix-product algorithm, Muon formulas or optimizer routing.

Baseline: commit `efe063d3b67153ced96c8435a381866ee0077605`. The old HW3 code
and historical measurement protocols remain intact in Git. Existing reports
describe that old backend and should be reproduced at their recorded revision,
not silently relabeled as measurements of the optimized backend.

## Verification before and after editing

Use independent NumPy strided views as byte-level references for direct native
Compact calls, including layouts used in Muon's transpose/tile/untile path and
random valid layouts up to rank six. Run these checks on the old kernel first.
Add tests of paired benchmark aggregation and rejection of missing/mismatched
pairs before implementing the new measurement helper.

Because correct outputs must remain unchanged, the new performance requirement
is a separate measured acceptance check, not an artificial output difference.
Before modifying Compact, compare two builds of the old code and observe that
the predefined 1.2x target on the representative 128x128 tiled layout is unmet.
Do not put timing assertions in the regular unit-test suite.

## Bounded measurements

The initial generic-carry trial was measured and retained under
`reports/compact_optimization/generic_carry/`, including its complete C++ source.
It improved four-dimensional packing but regressed two-dimensional transpose,
broadcast and sliced copies. Therefore add a direct two-dimensional nested-loop
path inside Compact, without allocating a coordinate vector. Keep generic carry
for other ranks. This evidence-driven adjustment remains a single-function
indexing optimization; retain all initial results rather than hiding regressions.
Before editing that path, the measured transpose/broadcast non-regression target
(at least 0.95x baseline speed) fails. Rerun all semantic and paired speed checks.

Compile the old C++ source from the baseline Git revision and the new current
source with the existing build helper, compiler and flags. Use the same Python,
model and optimizer sources for both variants; isolate extensions in separate
processes. Record source/binary hashes, compiler details and thread limits.

Run five alternating old/new paired blocks sequentially. Measure representative
transpose, tile, untile, sliced-offset and broadcast layouts; preallocate output
so direct native Compact timing excludes allocation. Then measure original
AdamW and Muon+auxiliary AdamW fixed-gradient updates, with identical inputs and
warmups. Finally compare short 128-update deep Fashion-MNIST training loops for
both optimizers. Compare complete parameter/loss/feature hashes across versions.
This fixed-step run checks semantics and execution speed, not convergence or
retrieval quality. Report both optimizers' speedups, not only Muon's.

Retain raw paired records and a concise English change log/report with the old
code, new mechanism, measured gains, byte-identical results and limitations.
Do not run or claim a new 220-second optimizer-quality experiment. Keep binary
snapshots, model/data artifacts and logs under ignored `results/` directories.
