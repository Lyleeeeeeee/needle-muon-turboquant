# Shared stride-aware CPU matrix multiplication

The user approved trying the shared matrix multiplication path after the two
earlier CPU optimizations. Baseline: `ac6dca9b34b60d968558e0978667b69a33b0949e`.
Both AdamW and Muon must benefit from improvements in model forward/backward
operations. Keep Muon's already optimized native NS path enabled in both arms.

## Implementation

For positive, tile-aligned native CPU products, pack each original two-dimensional
view directly into the existing 8x8 tiled format in C++, including its strides
and offset. Transposed operands therefore avoid a separate full-matrix compact
copy. Call unchanged MatmulTiled/AlignedDot, then unpack directly to an ordinary
contiguous output. Allocate three scratch arrays per call; retain no cache.
Check dimensions, buffer sizes, aliasing and signed view bounds at the native
boundary. Preserve ordinary NDArray/Tensor/autograd layout semantics.

Keep the original Python path for other devices, unsupported dimensions and
older native extensions. `NEEDLE_CPU_MATMUL=reference` selects the original path
for reproducible same-binary ablations; the default is `strided`. This switch
affects public NDArray multiplication, not Muon's internal native NS products.

Alternatives are a global packed Tensor layout (much broader autograd/mutation
changes) or improving generic Compact again (does not eliminate redundant
transposed-input copies). The bounded native operator isolates the shared work
without changing arithmetic or the model.

## Verification and measurement

First add failing tests for native view handling, bounds, exact old-path parity,
source preservation, dispatch/fallback and Tensor backward correctness. Include
contiguous, transposed, offset, sliced, broadcast and negative-stride views.
Use independent NumPy mathematical references as well as exact native reference
comparisons. Instrument the new native entry explicitly in optimizer profiling.

Measure five alternating old/new process pairs on the same final binary and
fixed inputs: representative forward/backward products and 128-update deep
Fashion-MNIST training for both AdamW and Muon. Keep data, seeds, rates, batches,
NS optimization and output fingerprints matched. Exclude loading/model setup
and final feature export from training timing, include the complete update loop.
Preserve raw reports, source/binary hashes, variability and unsuccessful cases.
Do not interpret timing repetitions as independent quality seeds or extrapolate
short-loop speedups to a new equal-time retrieval-quality result.
