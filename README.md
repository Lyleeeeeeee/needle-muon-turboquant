# Muon Training and TurboQuant Retrieval in Needle

A 10-714 deep learning systems project built on the completed HW1–HW3 Needle
framework. Each AdamW/Muon encoder generates its own retrieval embeddings;
FP32, uniform quantization and TurboQuant-MSE then store and search those same
embeddings. Training, autodiff, optimizer updates, decoding and scoring run in
Needle/native C++. NumPy/SciPy support data handling, calibration, rotation setup,
checkpoint files and analysis. PyTorch is not a training/inference dependency.

## Fresh consolidated results

Every training run, embedding archive, gallery comparison and timing below was
regenerated on the final native CPU implementation. Three paired training seeds
(8/9/10) use the same initialization, train/validation split and data-order prefixes.
The learning rates were locked from the earlier validation search before this
rerun. The table includes all five gallery formats, all training seeds and all
three TurboQuant rotations, with both positive and negative outcomes.

This is the only results table. [CSV](report/results.csv) exports the same rows;
[the evidence JSON](report/results.json) contains raw samples, selection rules,
model/data/source/native hashes, paired differences and artifact-validation proof.

<!-- RESULTS:START -->
| Study | Optimizer | Gallery format | P@10 (%) | Δ vs FP32 (pp) | Own-FP32 R@10 (%) | Gallery MB | Compression | Training cost | Updates | Optimizer state MB | Query s / 1k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 epochs / final CPU | AdamW | FP32 | 86.11 ± 0.21 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 43.566 ± 0.521 s | 8,600 | 1.1393 | 1.984 ± 0.182 |
| 10 epochs / final CPU | AdamW | Uniform 2-bit | 85.45 ± 0.17 | -0.66 ± 0.38 | 25.26 ± 1.44 | 1.3200 | 10.67× | 43.566 ± 0.521 s | 8,600 | 1.1393 | 1.895 ± 0.067 |
| 10 epochs / final CPU | AdamW | Uniform 4-bit | 85.95 ± 0.31 | -0.16 ± 0.13 | 74.42 ± 1.21 | 2.2000 | 6.40× | 43.566 ± 0.521 s | 8,600 | 1.1393 | 1.864 ± 0.060 |
| 10 epochs / final CPU | AdamW | TurboQuant 2-bit | 85.60 ± 0.19 | -0.51 ± 0.02 | 34.95 ± 1.51 | 1.1164 | 12.61× | 43.566 ± 0.521 s | 8,600 | 1.1393 | 1.875 ± 0.024 |
| 10 epochs / final CPU | AdamW | TurboQuant 4-bit | 86.01 ± 0.30 | -0.10 ± 0.12 | 68.46 ± 1.00 | 1.9964 | 7.05× | 43.566 ± 0.521 s | 8,600 | 1.1393 | 1.888 ± 0.009 |
| 10 epochs / final CPU | Muon + aux AdamW | FP32 | 86.25 ± 0.41 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 88.760 ± 2.179 s | 8,600 | 0.9754 | 1.874 ± 0.036 |
| 10 epochs / final CPU | Muon + aux AdamW | Uniform 2-bit | 85.90 ± 0.45 | -0.35 ± 0.07 | 31.70 ± 0.73 | 1.3200 | 10.67× | 88.760 ± 2.179 s | 8,600 | 0.9754 | 1.957 ± 0.136 |
| 10 epochs / final CPU | Muon + aux AdamW | Uniform 4-bit | 86.33 ± 0.35 | +0.08 ± 0.09 | 76.94 ± 0.64 | 2.2000 | 6.40× | 88.760 ± 2.179 s | 8,600 | 0.9754 | 1.904 ± 0.054 |
| 10 epochs / final CPU | Muon + aux AdamW | TurboQuant 2-bit | 85.98 ± 0.33 | -0.27 ± 0.11 | 43.93 ± 0.77 | 1.1164 | 12.61× | 88.760 ± 2.179 s | 8,600 | 0.9754 | 1.950 ± 0.050 |
| 10 epochs / final CPU | Muon + aux AdamW | TurboQuant 4-bit | 86.18 ± 0.50 | -0.07 ± 0.11 | 74.99 ± 0.49 | 1.9964 | 7.05× | 88.760 ± 2.179 s | 8,600 | 0.9754 | 1.949 ± 0.078 |
| 220s / final CPU | AdamW | FP32 | 86.11 ± 0.21 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 220.002 ± 0.001 s | 44,069 ± 303 | 1.1393 | 1.865 ± 0.057 |
| 220s / final CPU | AdamW | Uniform 2-bit | 85.73 ± 0.35 | -0.38 ± 0.20 | 27.08 ± 1.97 | 1.3200 | 10.67× | 220.002 ± 0.001 s | 44,069 ± 303 | 1.1393 | 1.861 ± 0.053 |
| 220s / final CPU | AdamW | Uniform 4-bit | 86.18 ± 0.20 | +0.07 ± 0.05 | 75.68 ± 0.87 | 2.2000 | 6.40× | 220.002 ± 0.001 s | 44,069 ± 303 | 1.1393 | 1.887 ± 0.048 |
| 220s / final CPU | AdamW | TurboQuant 2-bit | 86.00 ± 0.27 | -0.11 ± 0.15 | 36.44 ± 0.91 | 1.1164 | 12.61× | 220.002 ± 0.001 s | 44,069 ± 303 | 1.1393 | 1.914 ± 0.042 |
| 220s / final CPU | AdamW | TurboQuant 4-bit | 86.11 ± 0.23 | +0.00 ± 0.09 | 70.00 ± 0.97 | 1.9964 | 7.05× | 220.002 ± 0.001 s | 44,069 ± 303 | 1.1393 | 1.900 ± 0.029 |
| 220s / final CPU | Muon + aux AdamW | FP32 | 86.47 ± 0.89 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 220.004 ± 0.004 s | 21,108 ± 523 | 0.9754 | 1.966 ± 0.045 |
| 220s / final CPU | Muon + aux AdamW | Uniform 2-bit | 86.08 ± 0.56 | -0.39 ± 0.33 | 32.00 ± 1.22 | 1.3200 | 10.67× | 220.004 ± 0.004 s | 21,108 ± 523 | 0.9754 | 2.000 ± 0.126 |
| 220s / final CPU | Muon + aux AdamW | Uniform 4-bit | 86.46 ± 0.89 | -0.02 ± 0.08 | 77.02 ± 1.99 | 2.2000 | 6.40× | 220.004 ± 0.004 s | 21,108 ± 523 | 0.9754 | 1.948 ± 0.091 |
| 220s / final CPU | Muon + aux AdamW | TurboQuant 2-bit | 86.30 ± 0.75 | -0.17 ± 0.14 | 44.95 ± 0.51 | 1.1164 | 12.61× | 220.004 ± 0.004 s | 21,108 ± 523 | 0.9754 | 1.948 ± 0.076 |
| 220s / final CPU | Muon + aux AdamW | TurboQuant 4-bit | 86.47 ± 0.77 | -0.01 ± 0.12 | 75.74 ± 0.28 | 1.9964 | 7.05× | 220.004 ± 0.004 s | 21,108 ± 523 | 0.9754 | 1.952 ± 0.058 |
| 10 epochs / terminal check | AdamW | FP32 | 86.10 ± 0.21 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 43.566 ± 0.521 s | 8,600 | 1.1393 | 1.901 ± 0.042 |
| 10 epochs / terminal check | Muon + aux AdamW | FP32 | 85.97 ± 0.46 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 88.760 ± 2.179 s | 8,600 | 0.9754 | 1.893 ± 0.072 |
| 220s / terminal check | AdamW | FP32 | 88.52 ± 0.08 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 220.002 ± 0.001 s | 44,069 ± 303 | 1.1393 | 1.885 ± 0.044 |
| 220s / terminal check | Muon + aux AdamW | FP32 | 87.59 ± 0.71 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 220.004 ± 0.004 s | 21,108 ± 523 | 0.9754 | 1.905 ± 0.040 |
| 128 updates / original CPU | AdamW | — | — | — | — | — | — | 7.290 ms/step | 128 | — | — |
| 128 updates / original CPU | Muon + aux AdamW | — | — | — | — | — | — | 18.197 ms/step | 128 | — | — |
| 128 updates / final CPU | AdamW | — | — | — | — | — | — | 5.082 ms/step (1.435×) | 128 | — | — |
| 128 updates / final CPU | Muon + aux AdamW | — | — | — | — | — | — | 10.341 ms/step (1.749×) | 128 | — | — |
<!-- RESULTS:END -->

Measured takeaways:

- TurboQuant 2-bit reduces the gallery from 14.08 to 1.1164 MB (12.61×), with
  mean category-precision changes of −0.11 to −0.51 percentage points across
  the four encoder/budget conditions. It has higher mean category precision
  and own-FP32 neighbor recall than uniform 2-bit in all four conditions.
  Four-bit uniform preserves exact neighbors better than TurboQuant 4-bit;
  TurboQuant 4-bit uses less gallery storage.
- Muon uses 14.38% less retained numeric optimizer state. Under the fixed
  validation-loss checkpoint rule its FP32 precision is higher by 0.14 points
  at ten epochs and 0.37 points at equal time, winning two of three seeds in
  each comparison. The terminal 220-second check favors AdamW by 0.93 points,
  with Muon winning zero of three seeds. This CPU study has not established a
  robust equal-time retrieval advantage for Muon.
- The three systems changes accelerate complete fixed-work updates by 1.435×
  for AdamW and 1.749× for Muon, preserving exact outputs. Muon's optimizer
  computation remains more expensive per update in this small CPU model.

The first 20 rows are the full optimizer × gallery-format comparison, under equal
training exposure (10 epochs) and equal wall time (220 seconds). Each trained
encoder supplies all five gallery stores; repeated training costs in those rows
do not represent five separate training runs. The next four FP32 rows check the
terminal models. The last four rows isolate the original-to-final CPU systems
improvement with fixed work and exact output parity.

P@10 is the fraction of the ten returned images sharing the query category.
Δ is its signed change against the same encoder's FP32 store, in percentage
points. R@10 is overlap with that encoder's exact FP32 neighbors. Category
precision, neighbor fidelity and classifier accuracy are separate measurements.
Means ± sample SD use three training seeds. TurboQuant rotations 7/8/9 are
averaged within each training seed first; they are not nine independent models.

Gallery MB uses 1,000,000 bytes and includes packed codes and persistent quantizer
metadata. Compression is relative to the FP32 gallery. Images, labels, queries
and encoder weights are outside this storage count. Optimizer-state MB counts
retained numerical buffers after training, excluding parameters, gradients,
Python objects and temporary Newton–Schulz workspace; it is not peak process
memory. State memory and gallery storage belong to different pipeline stages.

Equal-epoch training cost includes the complete training loops and excludes data
loading, validation and checkpoint I/O. Equal-time cost includes intermediate
validation/checkpoint overhead; initial setup and final evaluation/export are
outside both methods' budgets. Updates are indivisible, so small recorded
one-step overruns are allowed. Updates reports the actual count, not an estimate
from elapsed time. Query cost is for 1,000 queries with one warmup and three
measured searches, including query normalization/rotation, block decoding,
native scoring and top-k; it excludes embedding generation and index construction.

The primary checkpoint rule was fixed before this rerun: minimum classification
validation loss, sampled at epoch boundaries or the four common times
55/110/165/220 seconds. Terminal FP32 checks were also fixed in advance. They
show whether that rule changes the retrieval comparison; model selection never
uses the reported test-query retrieval results. Historical test images were
already explored during the project, so this is an exploratory benchmark.

The systems rows report the median of five alternating sequential process pairs,
each measuring 128 complete updates at seed 8/batch 64. Parenthetical speedup is
the median within-pair ratio. These are timing repetitions, not training-quality
seeds. Python/model/optimizer formulas and compiler flags are shared; native
capabilities select the original or optimized paths. Parameters, losses and
features must match bit-for-bit. This implementation gain is distinct from the
AdamW-versus-Muon quality/cost comparison in the primary rows.

## Model, methods and controls

Fashion-MNIST has ten clothing categories. Split seed 17 fixes 55,000 training
and 5,000 validation images. The gallery contains the training pool and excludes
validation. Queries are 1,000 common official-test images. The plain ReLU MLP is
784–128–128–128–64–10 with a classification objective and 64-dimensional retrieval
features. All parameters train. Batch size is 64 and weight decay is 0.01.

AdamW uses first/second gradient moments with decoupled weight decay. Muon uses
momentum and five Newton–Schulz orthogonalization iterations on three hidden
weight matrices: 40,960 of 142,410 parameters. Input/classifier weights and all
biases use auxiliary AdamW. A three-candidate validation-loss search at seed 7
previously locked AdamW/auxiliary LR 0.001 and Muon LR 0.007. That complete
selection record remains in the evidence; this rerun compares those locked
configurations rather than newly optimized equal-time hyperparameters.

Uniform quantization stores each row's affine scale/offset and real packed
2/4-bit indices. TurboQuant-MSE applies a shared random orthogonal rotation,
uses sphere-distribution Lloyd–Max codebooks, and stores packed indices and row
norms. It implements Algorithm 1, without QJL residual correction. Only gallery
vectors are compressed; queries remain FP32. All methods use cosine similarity
after normalizing reconstructed vectors. Search decodes blocks of 256 without
retaining a dense reconstructed gallery.

The final CPU implementation has three systems changes:

- Compact uses direct rank-two indexing and coordinate carry for other ranks.
- Muon retains tiled intermediates across NS iterations, packs/unpacks once per
  hidden matrix, and refreshes the changing tiled transpose once per iteration.
- Shared matrix multiplication packs directly from strided/transpose views in
  C++ and unpacks directly to ordinary output, benefiting both optimizers.

MatmulTiled/AlignedDot arithmetic and accumulation order, separate float32
stages and optimizer formulas are preserved. Scratch buffers are temporary
within each call. `NEEDLE_CPU_MATMUL=reference` selects the original public
wrapper before import; `use_tiled_ns=False` selects reference NS. The total
systems ablation separately compiles the original Compact backend from its
pinned Git revision, with the same current Python code and C++17/-O2 build helper.

Algorithm references: [Muon author implementation](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py)
and [TurboQuant Algorithm 1](https://arxiv.org/html/2504.19874v1).

## Setup and reproduction

```bash
python -m pip install -r requirements-dev.txt
python tools/build_cpu.py
python -m pytest -q
python tools/build_report.py --check --verify-history
```

Put Fashion-MNIST's four standard train/t10k image/label IDX gzip files in
`data/fashion-mnist/`. The evaluated backend is native CPU float32; CPU NumPy
is a numerical reference. The build helper can reuse installed PyTorch pybind11
headers as a build fallback without importing PyTorch for model execution.
Data, model checkpoints, feature archives and generated runs stay ignored.

```bash
# Fast training -> own-model embeddings -> gallery comparison integration check.
python experiments/run_report.py --smoke --checkpoint-check

# Rebuild/benchmark first, then keep that native binary fixed for both studies.
python experiments/benchmark_system.py --blocks 5 --output results/reproduction/system.json
python experiments/run_report.py --study epochs --checkpoint best --checkpoint-check --resume --seeds 8 9 10 --output-dir results/reproduction/epochs
python experiments/run_report.py --study timed --checkpoint best --checkpoint-check --resume --seeds 8 9 10 --output-dir results/reproduction/timed

# Optional fixed-gradient optimizer profile.
python experiments/profile_optimizer.py --output results/profile.json

# Regenerate this one README table/CSV from the committed frozen evidence.
python tools/build_report.py
python tools/build_report.py --check --verify-history
```

New runs use fresh output directories. `--resume` reuses completed stages only
when frozen protocols, source/data/native hashes and checkpoint/feature hashes
match. New artifacts do not overwrite the published evidence. The original
backend ablation and origin check require the pinned Git history.

On the original measurement workspace, before changing its measured binary or
artifacts, `python tools/build_report.py --check --verify-history --verify-artifacts`
also reloads every selected and terminal checkpoint, reproduces full validation,
checks all common image IDs/labels, reproduces gallery/query feature samples
and compares retained metrics with original worker output. Its proof is recorded
in the evidence. The maintained tree contains the framework/kernels, six app
modules, four experiment modules, two tools, relevant tests and this single
report dataset/table. Earlier exploratory work remains in Git history.
