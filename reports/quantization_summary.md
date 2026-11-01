# Quantization component: preliminary results

Follow-up: the [completed scale study](quantization_scale_summary.md) expands to
60,000 training/gallery images and 1,000 common queries. The original records
below are preserved. Larger-data training improves retrieval, but the uniform
baseline advantage remains in the tested configurations.

The later [quality diagnosis](quantization_diagnosis.md) narrows that conclusion:
the original raw-inner-product scorer is sensitive to reconstructed norms;
decoded-cosine scoring makes TurboQuant competitive at two bits. Existing
results below retain their original scoring definition.

## What is implemented

TurboQuant-MSE now has exact-distribution codebooks, Needle rotations, native CPU 2/4-bit packing, block decoding/search, and NPZ persistence. FP32 and packed affine-uniform baselines use the same retrieval pipeline. All 37 current checks pass locally.

## Learned MNIST features

One shared encoder was trained on 4,096 MNIST images for 3 epochs using AdamW, seed 7. The gallery has 4,096 rows of 64 features; 256 held-out smoke-run images provide queries. The encoder reached 95.3125% classification accuracy on those images. Features are L2 normalized before retrieval. This is a component check, not the formal Fashion-MNIST test set.

The same saved features were reused for rotation seeds 7, 8 and 9; there was no retraining for quantization variants. Each condition used a fresh worker, one query warm-up and seven recorded timing samples. Quality entries show mean plus/minus sample standard deviation across the three rotation runs; FP32 and uniform quality are deterministic for this fixed gallery. Timing is the median of all 21 samples.

| Method | Numeric buffer KiB | Compression | Neighbor recall@10 | Label precision@10 | Median query ms |
|---|---:|---:|---:|---:|---:|
| fp32 | 1024.000 | 1.000x | 100.00% +/- 0.00% | 94.06% +/- 0.00% | 56.471 |
| uniform_2 | 96.000 | 10.667x | 27.34% +/- 0.00% | 91.99% +/- 0.00% | 56.948 |
| uniform_4 | 160.000 | 6.400x | 69.92% +/- 0.00% | 94.30% +/- 0.00% | 57.195 |
| turbo_2 | 96.016 | 10.665x | 24.64% +/- 1.41% | 90.82% +/- 0.69% | 56.753 |
| turbo_4 | 160.062 | 6.398x | 48.61% +/- 2.42% | 93.18% +/- 0.40% | 57.141 |

Neighbor recall compares the identities of retrieved rows with this encoder's FP32 top-10 neighbors. Label precision asks whether retrieved images share the query's class. A method can change the exact neighbors while preserving most class-level matches; these metrics are not interchangeable.

At 4 bits, TurboQuant uses about 160 KiB including metadata, versus 1,024 KiB for FP32. Ordinary uniform quantization currently has better reconstruction and neighbor recall on these features at essentially the same storage. TurboQuant's label precision remains near the FP32 baseline in this small sample, but no accuracy-neutrality or statistical-equivalence claim is made. Query times are in the same range; no end-to-end acceleration has been established.

## Synthetic unit-sphere check

[The synthetic run](quantization_synthetic.md) uses a 4,096 x 64 gallery, 128 queries and seed 7. TurboQuant-2 has relative reconstruction MSE about 0.114 and recall@10 about 55.5%, compared with uniform-2 MSE about 0.202 and recall about 51.6%. At 4 bits, uniform has slightly better recall (88.0% versus 85.9%). The TurboQuant distortion is consistent with the expected low-bit MSE scale, but neither bit width establishes universal superiority.

## Interpretation and limits

A plausible explanation for the learned-feature result is that these representations are nonnegative after ReLU, while per-row uniform min/scale can adapt to that range. TurboQuant uses a generic rotated sphere-coordinate codebook. The current experiment does not isolate this cause; it would require controlled feature/rotation ablations.

Buffer bytes count the packed indices and all retained norm/min/scale/rotation/codebook arrays once. They exclude Python objects, source images, labels and queries. Actual NPZ bytes and whole-worker peak RSS are retained in each JSON. Whole-worker RSS includes dense input loading, construction and reconstruction checks, so it must not be interpreted as query-only peak memory or confused with feature-store compression.

Next evaluation steps: larger Fashion-MNIST galleries with a separate validation/test protocol, feature-distribution ablations, bit-width/block-size sensitivity, and the planned Muon comparison. Existing negative findings should remain visible alongside any later improvements.

## Reproduce

```bash
python tools/build_cpu.py
python -m pytest -q
python apps/benchmark_quantization.py --gallery-size 4096 --queries 128 --dim 64 --repeats 7 --output reports/quantization_synthetic.json
python apps/train_baseline.py --data-dir /path/to/mnist --device cpu --train-limit 4096 --eval-limit 256 --epochs 3 --feature-output results/mnist_features.npz --output reports/encoder_mnist_quantization.json
python apps/benchmark_quantization.py --features results/mnist_features.npz --seed 7 --repeats 7 --output reports/quantization_mnist_encoder.json
python apps/benchmark_quantization.py --features results/mnist_features.npz --seed 8 --repeats 7 --output reports/quantization_mnist_encoder_seed8.json
python apps/benchmark_quantization.py --features results/mnist_features.npz --seed 9 --repeats 7 --output reports/quantization_mnist_encoder_seed9.json
```
