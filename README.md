# Needle: Muon + TurboQuant

A private 10-714 project workspace built from the completed HW1 autodiff, HW2 layers/optimizers/data utilities, and HW3 NDArray backends.

The two project workstreams are Muon training optimization and TurboQuant-MSE feature storage. Quantization is the current implementation phase; the optimizer comparison follows the [implementation plan](docs/implementation_plan.md).

## Setup

```bash
python -m pip install -r requirements-dev.txt
python tools/build_cpu.py
python -m pytest -q
```

The build helper compiles the CPU extension for the active Python interpreter on macOS or Linux. It uses pybind11 headers, optionally reusing headers from an installed PyTorch distribution; model training and retrieval use Needle. Native CPU is the primary quantization backend. The `cpu_numpy()` device provides a reference path. CUDA source is retained, but the new quantization kernels are CPU-only.

## Quantization

`needle.quantization` implements Algorithm 1 (TurboQuant-MSE): exact sphere-distribution Lloyd-Max codebooks, shared random orthogonal rotation, real packed 2/4-bit indices, row norms, block decoding, and inner-product top-k search. FP32 and per-row affine uniform quantization provide comparison baselines. Packing and both quantized decoders have native CPU kernels.

```python
import numpy as np
from needle.quantization import TurboQuantMSE

gallery = np.random.default_rng(7).normal(size=(2048, 64)).astype('float32')
quantizer = TurboQuantMSE(dim=64, bits=4, seed=7)
store = quantizer.encode(gallery)
result = store.search(gallery[:8], k=10, block_size=256)
print(store.storage_bytes, result.indices)
store.save('gallery.npz')
```

Input norms are restored within float32 range; zero vectors are supported. Search uses inner products. Normalize gallery and query rows before encoding/search when cosine similarity is intended. Search rotates queries once and decodes gallery blocks in the rotated basis, avoiding an inverse rotation per gallery block.

## Run comparisons

```bash
python apps/benchmark_quantization.py --output results/synthetic_quantization.json
```

This compares FP32, uniform 2/4-bit, and TurboQuant-MSE 2/4-bit in isolated workers. Reports include numerical buffer bytes, actual archive bytes, setup/build costs, query latency, reconstruction error, recall, decoded-block sizes, and whole-worker peak RSS. Read the [methodology](docs/quantization.md) before interpreting memory or timing numbers. [Current measurements](reports/quantization_summary.md) include both improvements and negative findings.

The [scale-study design](docs/quantization_scale_design.md) crosses training-set
and gallery sizes using frozen checkpoints and identical query images. The
completed [scale-study results](reports/quantization_scale_summary.md) cover both
training sizes, both gallery sizes, and three rotations on 1,000 common queries.
Larger-data training helps both methods; uniform still has better exact-neighbor
recall in these MNIST experiments. The
trainer accepts `--checkpoint-output results/encoder.npz`; independently expand
retrieval without retraining with:

```bash
python apps/export_features.py --checkpoint results/encoder.npz --gallery-size 60000 --queries 1000 --output results/expanded_features.npz
```

The exporter preserves the original 4,096-image gallery and 256-image query
selections as prefixes by default. Match `--reference-train`,
`--reference-queries`, and `--seed` to other original smoke-run settings.

For an encoder-feature integration check, provide MNIST's four original `.gz` files in one directory:

```bash
python apps/train_baseline.py --data-dir /path/to/mnist --device cpu --train-limit 1024 --eval-limit 256 --epochs 3 --feature-output results/mnist_features.npz --output results/encoder.json
python apps/benchmark_quantization.py --features results/mnist_features.npz --output results/encoder_quantization.json
```

Within the original course workspace, `--data-dir` defaults to the adjacent `hw0/data`. The shared model is `784 -> 128 -> 64 -> 10`; `model.features(x)` exposes the 64-dimensional gallery representation. This command is an MNIST integration check, not the final Fashion-MNIST study. The original baseline record is in [reports](reports/).

## Verification and scope

Tests cover exact-distribution centroids, independent numerical references, rotation orthogonality, packed layout and padding, native/reference parity, zero/constant vectors, metadata accounting, serialization, stable blockwise retrieval, gradients, optimizers, and small-model training. They are project checks rather than the complete homework grading suite.

The current quantizer reproduces the MSE variant. QJL residual correction, LLM KV-cache experiments, GPU quantization kernels, and the Muon comparison are later work. Comparative findings depend on dataset, bit width, metadata amortization, and implementation overhead; no acceleration or synergy is assumed.

## Provenance

Original source archives remain local in `sources/`. [The manifest](sources/manifest.json) records their sources and SHA-256 hashes. HW2 already includes the completed HW1 implementation. Integration adapts backend selection, initialization, reductions, stable log-sum-exp, and gradient device placement to HW3.

The original HW2 `Adam` is preserved. A separate, tested `AdamW` provides decoupled weight decay for project comparisons. Original homework folders remain separate. Keep this repository private because it contains course-framework solutions.
