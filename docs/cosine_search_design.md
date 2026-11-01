# Native Needle cosine retrieval

The user approved integrating the diagnosed scorer into Needle and repeating
the comparison. Add `store.search(..., metric='cosine')`, while retaining
`metric='inner_product'` as the backward-compatible default.

## Runtime

Normalize query rows before any TurboQuant rotation. Normalize each decoded
gallery block in its search basis, then perform the existing Needle matrix
product and deterministic top-k merge. TurboQuant keeps its rotated-basis
search, avoiding inverse rotation of every block. Orthogonality makes this
equivalent to original-space cosine up to float32 rounding.

Normalization uses Needle NDArray operations and the existing native CPU
kernels, not NumPy norm calculations. First divide by the row maximum absolute
value, then divide by the L2 norm of the rescaled row. Replace zero divisors
with one so zero rows remain zero and have zero cosine scores. This avoids
squaring huge or tiny float32 values. Work is blockwise; do not retain a dense
normalized gallery or add persistent per-row norm caches.

NumPy remains in the existing input validation/top-k/quantizer setup and
reference backend as previously documented. This change specifically migrates
the new scorer's normalization into Needle; it does not remove every NumPy
dependency from the project.

## Verification

Test both native CPU and the reference device against independently normalized
decoded vectors for FP32, uniform and TurboQuant stores. Include non-unit
queries, zero rows, tails across block sizes, deterministic ties, extreme
float32 magnitudes, explicit raw-IP compatibility and invalid score modes.
Add a CLI integration check that the selected metric reaches isolated workers.

## Retained experiment

Use the frozen 60,000-example encoder, unchanged saved features, a 60,000-image
gallery and 1,000 common queries. Evaluate FP32/uniform2/uniform4/Turbo2/Turbo4
under both raw IP and cosine for rotations7/8/9, one warm-up plus three timings.
Alternate the scorer order across rotations: IP/cosine for7, cosine/IP for8,
IP/cosine for9. Run workers sequentially. No retraining or hyperparameter tuning.

Ground truth is the FP32 store using the selected score mode. Unit-vector raw
IP and cosine are mathematically equivalent, but float32 near-ties may differ;
record the definition and compare quality with the independent diagnosis.
Report time for the complete 1,000-query batch. Include query and block
normalization in the timed region. Persistent store bytes stay unchanged;
existing logical decoded/score counters exclude normalization/matmul/top-k
workspace and must not be described as exact peak memory.

## Completed experiment

The [native results](../reports/native_cosine_summary.md) and raw records confirm
this implementation. All 52 project checks passed. FP32 raw-IP/cosine neighbor
sets agree on every query in the retained input, old raw-IP quality reproduces
exactly, and native cosine matches the independent diagnosis within 0.04
percentage points. No model retraining or persistent storage changes occurred.
