# Quantization scale study: MNIST

Both approved checks are complete: larger retrieval evaluation with frozen weights, and larger-data encoder training.

Follow-up: the [independent quality diagnosis](quantization_diagnosis.md) finds
that reconstructed-norm variation affects the raw-IP scorer used below. A fair
decoded-cosine comparison greatly improves both methods and reverses the
two-bit ordering. These original measurements are preserved for their stated
scoring rule; they do not establish general task unsuitability.

## Protocol and verification

The same Needle CPU 784-128-64-10 ReLU model was trained with AdamW (learning rate 0.003, batch 64, seed 7) for three epochs on either 4,096 or 60,000 images. Both models use exactly the same nested 4,096/60,000-image galleries and the same 1,000 test-split query images. Features are L2 normalized. Top-10 FP32 neighbors are computed separately for each model and gallery.

All 40 project checks passed. The original small-model features were reproduced bit-for-bit; all five original 256-query quality results were also reproduced exactly for each of the three rotation seeds. Checkpoint reload and nested export preserve feature values. Three rotations (7/8/9) and three retained timings per method/rotation were run sequentially after one warm-up. No quantizer tuning was performed.

The common-query classification accuracy improved from **93.0% to 96.8%** with larger-data training. These are the same 1,000 images; the old 95.3125% figure used only the original 256-image subset. Training-log validation selections differ across training sizes because the legacy RNG sequence is preserved; use the common exported-query accuracy for this cross-model comparison.

## Neighbor recovery

Recall@10 measures how many of the same image IDs in the FP32 top 10 remain after compression. Values are mean +/- sample standard deviation across rotations. Uniform quality is deterministic for fixed features.

### 4-bit comparison

| Training images | Gallery images | Uniform recall@10 | TurboQuant-MSE recall@10 |
|---:|---:|---:|---:|
| 4,096 | 4,096 | 70.08% +/- 0.00% | 49.56% +/- 1.75% |
| 4,096 | 60,000 | 44.89% +/- 0.00% | 22.87% +/- 1.37% |
| 60,000 | 4,096 | 79.66% +/- 0.00% | 64.83% +/- 1.12% |
| 60,000 | 60,000 | 56.41% +/- 0.00% | 36.50% +/- 0.48% |

### 2-bit comparison

| Training images | Gallery images | Uniform recall@10 | TurboQuant-MSE recall@10 |
|---:|---:|---:|---:|
| 4,096 | 4,096 | 28.55% +/- 0.00% | 24.71% +/- 0.63% |
| 4,096 | 60,000 | 8.41% +/- 0.00% | 6.33% +/- 0.36% |
| 60,000 | 4,096 | 40.34% +/- 0.00% | 38.55% +/- 0.31% |
| 60,000 | 60,000 | 15.67% +/- 0.00% | 14.20% +/- 0.51% |

## Reconstruction error

Relative reconstruction MSE is total squared error divided by original feature
energy; lower is better. Means across the three rotations are shown below.
Uniform also has lower reconstruction error in these learned-feature runs.

| Training images | Gallery images | Uniform-2 | TurboQuant-2 | Uniform-4 | TurboQuant-4 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 4,096 | 0.049554 | 0.114496 | 0.001970 | 0.009419 |
| 4,096 | 60,000 | 0.050053 | 0.114608 | 0.001988 | 0.009365 |
| 60,000 | 4,096 | 0.039194 | 0.113808 | 0.001585 | 0.009812 |
| 60,000 | 60,000 | 0.039480 | 0.113627 | 0.001592 | 0.009724 |

## Same-class retrieval at 4 bits

Label precision@10 measures whether retrieved images have the same digit label as the query. It is a different objective from recovering the exact FP32 neighbors.

| Training images | Gallery images | FP32 label precision | Uniform-4 label precision | TurboQuant-4 label precision |
|---:|---:|---:|---:|---:|
| 4,096 | 4,096 | 92.69% +/- 0.00% | 92.76% +/- 0.00% | 92.00% +/- 0.27% |
| 4,096 | 60,000 | 93.68% +/- 0.00% | 93.45% +/- 0.00% | 92.83% +/- 0.35% |
| 60,000 | 4,096 | 96.40% +/- 0.00% | 96.34% +/- 0.00% | 96.30% +/- 0.06% |
| 60,000 | 60,000 | 97.25% +/- 0.00% | 97.20% +/- 0.00% | 97.09% +/- 0.08% |

## Query-count sensitivity at 4 bits

Each comparison below keeps the encoder and gallery unchanged and evaluates the nested 256-query prefix versus all 1,000 queries.

| Training | Gallery | Uniform: 256 queries | Uniform: 1,000 queries | TurboQuant: 256 queries | TurboQuant: 1,000 queries |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 4,096 | 69.92% | 70.08% | 48.61% | 49.56% |
| 4,096 | 60,000 | 45.20% | 44.89% | 22.60% | 22.87% |
| 60,000 | 4,096 | 78.75% | 79.66% | 64.45% | 64.83% |
| 60,000 | 60,000 | 55.00% | 56.41% | 35.60% | 36.50% |

## Storage and timing

Persistent numerical feature-store bytes include the current metadata formats. A 60,000-image FP32 gallery uses 15,360,000 bytes. Uniform-4 uses 2,400,000 bytes (6.40x compression); TurboQuant-4 uses 2,176,448 bytes (about 7.06x). TurboQuant amortizes its shared rotation and stores one norm per row, while this uniform implementation stores two floats per row. This difference is specific to these metadata layouts; both store four index bits per coordinate.

These are not whole-process memory reductions. Whole-worker RSS, actual NPZ sizes and logical block counters remain separately available in the raw reports.

Batch timing below is for **all 1,000 queries**, not one query. Entries are the median of nine retained samples, in milliseconds, excluding feature extraction and index construction.

| Training | Gallery | FP32 | Uniform-2 | Uniform-4 | TurboQuant-2 | TurboQuant-4 |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 4,096 | 225.96 | 225.96 | 225.44 | 227.04 | 227.88 |
| 4,096 | 60,000 | 3292.66 | 3324.24 | 3326.82 | 3319.60 | 3331.20 |
| 60,000 | 4,096 | 225.58 | 226.29 | 226.88 | 228.09 | 228.30 |
| 60,000 | 60,000 | 3317.06 | 3341.91 | 3322.18 | 3301.09 | 3299.19 |

## Interpretation and limits

Increasing query coverage does not remove the uniform-baseline advantage. Larger-data training improves encoder classification and both quantizers' neighbor recovery. Larger galleries make exact-neighbor recovery harder. TurboQuant-MSE still trails uniform at four bits in every tested model/gallery configuration; increased data alone does not resolve that gap in this study. Numerical storage compression is established, while an end-to-end speedup including image encoding is not.

This does not establish why the gap occurs or prove a general ranking of the methods. Nonnegative ReLU features and the MSE-versus-inner-product objective difference remain hypotheses requiring targeted ablations. The QJL-corrected inner-product variant is not implemented.

The larger training run has 2,814 optimizer updates versus 192 for the small run, so this is not a fixed-compute causal data-only comparison. There is one encoder training seed; rotation standard deviations do not quantify training-run uncertainty. MNIST test images have been used for exploratory evaluation, so this is not an untouched final test or the planned Fashion-MNIST study. Runtime NumPy/SciPy dependencies are unchanged.

## Reproduce and inspect

- [Reproduction commands](../docs/quantization_scale_reproduce.md)
- [Design and controls](../docs/quantization_scale_design.md)
- [Frozen-feature manifest](scale_feature_manifest.json)
- [Machine-readable aggregate](quantization_scale_summary.json)

- Training 4,096, gallery 4,096: [scale_train4096_gallery4096_seed7.json](scale_train4096_gallery4096_seed7.json), [scale_train4096_gallery4096_seed8.json](scale_train4096_gallery4096_seed8.json), [scale_train4096_gallery4096_seed9.json](scale_train4096_gallery4096_seed9.json)
- Training 4,096, gallery 60,000: [scale_train4096_gallery60000_seed7.json](scale_train4096_gallery60000_seed7.json), [scale_train4096_gallery60000_seed8.json](scale_train4096_gallery60000_seed8.json), [scale_train4096_gallery60000_seed9.json](scale_train4096_gallery60000_seed9.json)
- Training 60,000, gallery 4,096: [scale_train60000_gallery4096_seed7.json](scale_train60000_gallery4096_seed7.json), [scale_train60000_gallery4096_seed8.json](scale_train60000_gallery4096_seed8.json), [scale_train60000_gallery4096_seed9.json](scale_train60000_gallery4096_seed9.json)
- Training 60,000, gallery 60,000: [scale_train60000_gallery60000_seed7.json](scale_train60000_gallery60000_seed7.json), [scale_train60000_gallery60000_seed8.json](scale_train60000_gallery60000_seed8.json), [scale_train60000_gallery60000_seed9.json](scale_train60000_gallery60000_seed9.json)
