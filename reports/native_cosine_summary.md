# Native Needle cosine search: measured comparison

Decoded-vector cosine is now part of the Needle retrieval API. Both query and gallery-block normalization execute through Needle NDArray CPU kernels. The earlier NumPy float64 diagnosis remains an independent reference, not the implementation used for these measurements.

## API and behavior

```python
result = store.search(queries, k=10, block_size=256, metric='cosine')
```

The compatible default remains `metric='inner_product'`. FP32, affine uniform and TurboQuant stores share the cosine path. Stable max-absolute-value rescaling avoids norm overflow/underflow, zero rows score zero, and queries normalize before TurboQuant rotation. Gallery processing remains blockwise, without a dense decoded-gallery or persistent norm cache. TurboQuant keeps the rotated-basis search.

The new normalization uses Needle operations; existing input validation, quantizer setup and top-k merging retain their documented NumPy/SciPy dependencies. No model training, index encoding or storage-format changes were made.

## Matched native comparison

The frozen full-training encoder, 60,000-image gallery, 1,000 common queries, 64-dimensional features, top-10 retrieval, blocks of 256 and rotations 7/8/9 match the diagnosis. Both score modes were rerun in fresh sequential workers, with one warm-up and three recorded samples per worker. The scorer order alternated across rotations as specified in the design. FP32 ground truth uses the selected score metric.

| Method | Raw-IP recall@10 | Native cosine recall@10 | Native cosine same-label precision |
|---|---:|---:|---:|
| fp32 | 100.00% | 100.00% +/- 0.00% | 97.25% |
| uniform_2 | 15.67% | 44.95% +/- 0.00% | 96.88% |
| uniform_4 | 56.41% | 86.78% +/- 0.00% | 97.23% |
| turbo_2 | 14.20% | 50.20% +/- 0.55% | 97.14% |
| turbo_4 | 36.50% | 81.64% +/- 0.06% | 97.25% |

Recall is the mean overlap of exact FP32 top-10 image IDs. Label precision asks whether retrieved images have the query digit label. Rotation variation refers to one fixed trained encoder, not independent training runs.

## Query time with normalization included

Times are for the **complete batch of 1,000 queries**, not individual-query latency. Values are medians of nine retained samples. Cosine includes query preparation, query/block normalization, rotation, decode, Needle matrix products and top-k. Model feature extraction and index construction are excluded.

| Method | Raw-IP batch seconds | Cosine batch seconds | Observed mode time change |
|---|---:|---:|---:|
| fp32 | 3.3695 | 3.3768 | +0.22% |
| uniform_2 | 3.3706 | 3.3750 | +0.13% |
| uniform_4 | 3.3407 | 3.3389 | -0.05% |
| turbo_2 | 3.3222 | 3.4058 | +2.52% |
| turbo_4 | 3.3428 | 3.4180 | +2.25% |

These are observed whole-search mode differences, including changed ranking inputs and measurement noise, rather than isolated normalization-kernel timings. This work establishes the quality improvement in the actual Needle API; it does not establish compressed arithmetic acceleration.

## Persistent storage

| Method | Store bytes | Compression versus FP32 |
|---|---:|---:|
| fp32 | 15,360,000 | 1.000x |
| uniform_2 | 1,440,000 | 10.667x |
| uniform_4 | 2,400,000 | 6.400x |
| turbo_2 | 1,216,400 | 12.627x |
| turbo_4 | 2,176,448 | 7.057x |

Persistent bytes are identical across score modes. Norm scaling uses temporary query/block buffers. Logical decoded-block and score-array counters exclude normalization, matmul and top-k workspace. Whole-worker peak RSS in raw records includes loading, construction and quality checks and is not exact query-only memory.

## Verification

All 52 project checks passed, including native/reference devices, three store types, zero rows, extreme finite magnitudes, ties, block-size independence, raw-IP compatibility and CLI worker propagation. Every old full-gallery raw-IP quality metric was reproduced exactly. Encoding/reconstruction MSE and store bytes agree across modes.

For the original normalized gallery, native FP32 raw-IP versus cosine top-10 overlap is 100.0000%. Unit-vector IP and cosine are mathematically equivalent; float32 near-ties can differ. The reference-comparison table reports the actual float32 differences rather than forcing exact agreement.

| Method | Independent float64 cosine recall | Native cosine recall | Difference in percentage points |
|---|---:|---:|---:|
| uniform_2 | 44.99% | 44.95% | -0.0400 |
| uniform_4 | 86.78% | 86.78% | +0.0000 |
| turbo_2 | 50.20% | 50.20% | +0.0000 |
| turbo_4 | 81.64% | 81.64% | +0.0000 |

## Interpretation and scope

The native implementation confirms the diagnostic direction: TurboQuant is stronger at two bits under the common cosine scorer, while affine uniform remains stronger at four bits on these nonnegative ReLU features. Neither bit width establishes a universal ranking of quantizers. There is one trained encoder seed, an exploratory MNIST evaluation and no QJL correction; the planned Fashion-MNIST and Muon studies remain separate work.

## Reproduce

```bash
python -m pytest -q
python - <<'PY'
import os
import subprocess
import sys
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
conditions = [(7, 'inner_product'), (7, 'cosine'), (8, 'cosine'), (8, 'inner_product'), (9, 'inner_product'), (9, 'cosine')]
for seed, metric in conditions:
    subprocess.run([sys.executable, 'apps/benchmark_quantization.py', '--features', 'results/scale/features_train60000_gallery60000.npz', '--device', 'cpu', '--metric', metric, '--seed', str(seed), '--repeats', '3', '--output', f'reports/native_{metric}_full_seed{seed}.json'], check=True, env=env)
PY
```

Generate the frozen input archive using the [scale-study commands](../docs/quantization_scale_reproduce.md). See the [design](../docs/cosine_search_design.md), [validation](native_cosine_validation.json) and [machine-readable summary](native_cosine_summary.json). Raw worker records follow.

- [native_inner_product_full_seed7.json](native_inner_product_full_seed7.json)
- [native_inner_product_full_seed8.json](native_inner_product_full_seed8.json)
- [native_inner_product_full_seed9.json](native_inner_product_full_seed9.json)
- [native_cosine_full_seed7.json](native_cosine_full_seed7.json)
- [native_cosine_full_seed8.json](native_cosine_full_seed8.json)
- [native_cosine_full_seed9.json](native_cosine_full_seed9.json)
