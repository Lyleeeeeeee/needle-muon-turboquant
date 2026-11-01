# Implementation plan

## Integrated foundation

Completed HW1 autodiff and operations, HW2 layers/data utilities/SGD/Adam, and HW3 NDArray have been integrated. A native CPU build and small-model training run pass locally. CUDA remains unverified. A separate AdamW implementation has independent checks for decoupled decay and missing-gradient step counts.

`apps/models.py` defines the shared encoder; `apps/train_baseline.py` trains it and optionally exports gallery/query features. Preserve the source archives and provenance manifest.

## Workstream A: Muon training optimization

The implementation and controlled MNIST pilot follow
[the pinned design](muon_design.md). Needle float32 Muon, auxiliary AdamW,
layout-aware scaling, holdout splits and independent numerical tests are now
implemented. The checklist below also defines the broader formal study.
The completed [paired pilot](../reports/muon_pilot_summary.md) compares three
training seeds, validation-selected checkpoints and the earlier training budget.
The [learning-rate and dataset study](../reports/lr_dataset_study/summary.md)
adds three candidate configurations per method/dataset, separate repeat seeds,
and a final Fashion-MNIST test with all choices frozen before test access.
Its small MLP and bounded grid remain distinct from large-model paper results.

1. Pin an author implementation and record momentum, Nesterov behavior, Newton-Schulz coefficients/steps, and shape scaling. Start with Needle float32.
2. Add Muon to `needle.optim`. Initially assign `model.hidden.weight` to Muon; input/output layers and biases use auxiliary AdamW. Check that parameter groups are disjoint and exhaustive.
3. Validate zero gradients, rectangular matrices, optimizer state, and agreement with an independent NumPy iteration. A five-step approximation need not exactly equal the SVD polar factor.
4. Compare loss, validation accuracy, time to a target quality, per-step latency, and optimizer-state bytes against AdamW.

Completion means correct updates and a controlled comparison on the same model; benefits are measured rather than assumed.

References: [author explanation](https://kellerjordan.github.io/posts/muon/), [author implementation](https://github.com/KellerJordan/Muon).

## Workstream B: TurboQuant-MSE feature storage

The first implementation provides:

- Exact sphere-coordinate Lloyd-Max codebooks, cached per dimension and bit width.
- A shared Haar orthogonal rotation from Gaussian QR with sign correction.
- Real uint8-backed 2/4-bit storage with per-row float32 norms.
- Native CPU packing and block decoders, plus a NumPy reference path.
- Blockwise inner-product retrieval in the rotated basis.
- FP32 and packed uniform baselines, safe NPZ serialization, and isolated-worker comparisons.

See [quantization methodology](quantization.md) for math, contracts, and memory/timing definitions. The MSE variant is distinct from the unbiased-inner-product QJL extension.

Remaining validation work includes larger Fashion-MNIST feature galleries, multiple rotation seeds, sensitivity to bit width/block size/gallery size, and more complete process-memory profiling. These are experiments on the implemented component rather than assumptions about paper-level results.

Reference: [TurboQuant Algorithm 1](https://arxiv.org/html/2504.19874v1).

## Shared study and interactions

1. Fix Fashion-MNIST training/validation/test splits, architecture, feature dimension, data order, and hardware. Keep final test queries separate from hyperparameter selection.
2. Give AdamW and Muon the same tuning budget, select using validation data, and retain multiple seeds.
3. Generate one feature gallery per trained model. Reuse it for FP32, uniform 2/4-bit, and TurboQuant-MSE 2/4-bit comparisons; compression variants do not require retraining.
4. Report model quality separately from each model's compression-induced loss relative to its own FP32 neighbors. Consider matching base quality before attributing differences to compression tolerance.
5. Keep independent results for each workstream, then test interactions. A valid negative result requires correct code and controlled measurements.

The current MNIST run and synthetic-vector comparisons are integration checks. They do not establish superiority on the final task, paper-scale speedups, or optimizer/quantizer synergy.
