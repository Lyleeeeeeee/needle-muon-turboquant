# Needle: Muon + TurboQuant

A private 10-714 project workspace built from the completed HW1 autodiff, HW2 layers/optimizers/data utilities, and HW3 NDArray backends.

The two project workstreams are Muon training optimization and TurboQuant-MSE
feature storage. Both are implemented in the integrated Needle framework.
Controlled experiments and the remaining formal study follow the
[implementation plan](docs/implementation_plan.md).

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
result = store.search(gallery[:8], k=10, block_size=256, metric='cosine')
print(store.storage_bytes, result.indices)
store.save('gallery.npz')
```

Input norms are restored within float32 range; zero vectors are supported.
Search defaults to raw inner products for compatibility. Select
`metric='cosine'` to normalize queries and reconstructed gallery blocks using
Needle NDArray operations before scoring. Zero rows have zero scores. Query
normalization precedes rotation; gallery normalization stays blockwise without
caching a dense decoded gallery. TurboQuant searches in the rotated basis,
avoiding an inverse rotation per gallery block.

## Run comparisons

```bash
python apps/benchmark_quantization.py --metric cosine --output results/synthetic_quantization.json
```

This compares FP32, uniform 2/4-bit, and TurboQuant-MSE 2/4-bit in isolated workers. Reports include numerical buffer bytes, actual archive bytes, setup/build costs, query latency, reconstruction error, recall, decoded-block sizes, and whole-worker peak RSS. Read the [methodology](docs/quantization.md) before interpreting memory or timing numbers. [Current measurements](reports/quantization_summary.md) include both improvements and negative findings.

The [scale-study design](docs/quantization_scale_design.md) crosses training-set
and gallery sizes using frozen checkpoints and identical query images. The
completed [scale-study results](reports/quantization_scale_summary.md) cover both
training sizes, both gallery sizes, and three rotations on 1,000 common queries.
Larger-data training helps both methods; uniform still has better exact-neighbor
recall under the original raw-inner-product scorer. The subsequent
[quality diagnosis](reports/quantization_diagnosis.md) finds a substantial
reconstructed-norm scoring effect and coordinate dependence. With the same
decoded-cosine scorer, TurboQuant leads at two bits and uniform leads at four
bits on the full learned gallery. Cosine is now implemented in Needle and
available with `search(..., metric='cosine')` or benchmark `--metric cosine`.
The production default remains raw inner product. The
trainer accepts `--checkpoint-output results/encoder.npz`; independently expand
retrieval without retraining with:

```bash
python apps/export_features.py --checkpoint results/encoder.npz --gallery-size 60000 --queries 1000 --output results/expanded_features.npz
```

The exporter preserves the original 4,096-image gallery and 256-image query
selections as prefixes by default. Match `--reference-train`,
`--reference-queries`, and `--seed` to other original smoke-run settings.

The [native cosine comparison](reports/native_cosine_summary.md) repeats both
score modes on the full frozen gallery, including normalization in query
timing. It confirms the two-bit TurboQuant advantage and four-bit uniform
advantage in the actual Needle API, with unchanged persistent storage.

For an encoder-feature integration check, provide MNIST's four original `.gz` files in one directory:

```bash
python apps/train_baseline.py --data-dir /path/to/mnist --device cpu --train-limit 1024 --eval-limit 256 --epochs 3 --feature-output results/mnist_features.npz --output results/encoder.json
python apps/benchmark_quantization.py --features results/mnist_features.npz --output results/encoder_quantization.json
```

Within the original course workspace, `--data-dir` defaults to the adjacent `hw0/data`. The shared model is `784 -> 128 -> 64 -> 10`; `model.features(x)` exposes the 64-dimensional gallery representation. This command is an MNIST integration check, not the final Fashion-MNIST study. The original baseline record is in [reports](reports/).

## Muon training

`needle.optim.Muon` implements the pinned author's momentum/Newton-Schulz
update in Needle float32. It expects hidden Linear matrices in Needle's
`(input, output)` layout and adapts shape scaling accordingly.
`MuonWithAuxAdamW` combines disjoint Muon and auxiliary AdamW groups. The shared
encoder uses Muon only for `hidden.weight`, with AdamW for other parameters.
The [paired training pilot](reports/muon_pilot_summary.md) reports three seeds,
ten-epoch validation curves, update cost and state bytes. In the predefined CPU
setting, Muon does not establish a quality or speed advantage over AdamW.

```bash
python apps/train_baseline.py --device cpu --optimizer muon --train-limit 55000 --validation-size 5000 --eval-limit 5000 --epochs 10 --seed 7 --best-checkpoint-output results/muon_best.npz --output results/muon_training.json
```

Select `--optimizer adamw` for the baseline. Holdout selection uses a fixed split
seed; initialization and sample-order hashes enable paired comparisons. The
trainer records per-epoch loss/accuracy, total training-loop and optimizer time,
numeric optimizer-state bytes, and the minimum-validation-loss checkpoint.
See the [Muon design](docs/muon_design.md) for algorithm pinning, float32/layout
adaptations and pilot limitations. The historical three-epoch AdamW training
and fixed-feature quantizer comparisons remain separate records.
`train_baseline.py --feature-output` exports the final epoch's features. Use
`export_features.py --checkpoint <best-checkpoint>` for features from a selected
validation checkpoint.

## Learning-rate and dataset comparison

The [predeclared study](docs/lr_dataset_study_design.md) gives AdamW and Muon
three paired learning-rate configurations each on MNIST and Fashion-MNIST.
Seed7 selects rates by validation loss; fresh seeds8/9/10 assess the selected
configurations. The [full report](reports/lr_dataset_study/summary.md) includes
all candidates, validation curves, CPU costs and the final Fashion test.

```bash
python experiments/run_lr_dataset_study.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/summarize_lr_dataset_study.py
```

Use `--dataset Fashion-MNIST` with `train_baseline.py`; its default directory is
`data/fashion-mnist/`. Both datasets use the original IDX loader and native
Needle training. Records include dataset identity and file SHA256 hashes.
Download the official Fashion training files before the runner. The summarizer
locks all choices before requesting the official test files; see the study
report for the staged evaluation protocol. Checkpoints and downloaded data
remain local ignored artifacts.
For a fresh full reproduction, archive the published `reports/lr_dataset_study/`
directory outside that output path first; its matching checkpoints are local.
The runner then creates a new set of records and checkpoints instead of silently
mixing old records with missing model files.
This study evaluates classification. The joint retrieval study below uses its
fixed selected models to evaluate embeddings and compression separately.

## Joint retrieval from tuned encoders

The [joint study](reports/joint_retrieval/summary.md) connects each tuned model to
its own 64D gallery/query embeddings and FP32, uniform 2/4-bit and TurboQuant-MSE 2/4-bit
stores. All six models share 55,000 gallery images and 1,000 query image IDs. The
fixed validation holdout is excluded from the gallery. Three rotation trials
are averaged within each training seed before aggregating three training seeds.

```bash
python experiments/run_joint_retrieval.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/summarize_joint_retrieval.py
```

`export_features.py --dataset Fashion-MNIST --validation-size 5000 --split-seed 17`
supports dataset-aware provenance and holdout exclusion; legacy MNIST selections
are preserved. `benchmark_quantization.py --methods ...` selects method workers
without changing their algorithms. The report separates category precision,
own-model FP32-neighbor recall, signed compression loss, persistent bytes and
whole-query-batch time. Classification test data was already viewed; this is a
fixed-model retrieval extension with no new selection or untouched-test claim.
Only gallery vectors are compressed; queries remain float32.
Local models from the learning-rate study are required. See the report for
artifact checks and fresh-reproduction instructions.

## Deeper encoder comparison

`train_baseline.py --extra-hidden-layers 2` inserts two 128-to-128 Linear+ReLU
layers, giving 784-128-128-128-64-10 while keeping 64D embeddings. Muon covers all
three hidden weight matrices: 40,960/142,410 parameters (28.76%), versus 7.49%
in the default shallow model. Input/classifier weights and all biases retain
auxiliary AdamW. The default model and old version1 checkpoints stay compatible;
version2 records and restores every added layer.

```bash
python experiments/run_depth_study.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/summarize_depth_study.py
```

The [predefined depth study](docs/depth_study_design.md) repeats the same
three-candidate tuning budget and three repeat seeds on Fashion-MNIST, followed
by identical gallery/query images and quantizers. The [depth report](reports/depth_study/summary.md)
compares both encoders and both architectures, with training cost and compression
loss separate from absolute retrieval quality. This comparison changes depth,
parameter count and compute; it cannot isolate Muon coverage as a causal effect.

## Equal-time training and optimizer profiling

The [equal-time protocol](docs/time_budget_design.md) gives each existing tuned
deep encoder 220 seconds, with four common-time validation checkpoints and three
paired training seeds. The [report](reports/time_budget_study/summary.md) compares
retrieval after validation-based selection, and separates optimizer update costs
from other training work. A fixed-gradient microbenchmark diagnoses Muon's
Newton-Schulz products, native matrix kernels and layout compaction separately.

```bash
python experiments/run_time_budget_study.py --phase all
python experiments/profile_optimizer.py
python experiments/summarize_time_budget_study.py
```

Intermediate validation/checkpoint costs count toward the training budget;
setup and terminal evaluation/export do not. Previously selected learning rates
are retained, so this is not a new equal-time hyperparameter search. Historical
epoch-budget records and all optimizer/quantizer formulas remain unchanged.

## Verification and scope

Tests cover exact-distribution centroids, independent numerical references, rotation orthogonality, packed layout and padding, native/reference parity, zero/constant vectors, metadata accounting, serialization, stable blockwise retrieval, gradients, optimizers, and small-model training. They are project checks rather than the complete homework grading suite.

The current quantizer reproduces the MSE variant. QJL residual correction, LLM
KV-cache experiments and GPU quantization kernels remain later work.
Comparative findings depend on dataset, bit
width, metadata amortization and implementation overhead; no acceleration or
synergy is assumed.

## Provenance

Original source archives remain local in `sources/`. [The manifest](sources/manifest.json) records their sources and SHA-256 hashes. HW2 already includes the completed HW1 implementation. Integration adapts backend selection, initialization, reductions, stable log-sum-exp, and gradient device placement to HW3.

The original HW2 `Adam` is preserved. A separate, tested `AdamW` provides decoupled weight decay for project comparisons. Original homework folders remain separate. Keep this repository private because it contains course-framework solutions.
