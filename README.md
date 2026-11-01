# Muon Training and TurboQuant Retrieval in Needle

A 10-714 deep learning systems project using the completed HW1–HW3 Needle
framework. The project trains image encoders with AdamW or Muon plus auxiliary
AdamW, then stores their retrieval embeddings as FP32, uniform quantization, or
TurboQuant-MSE. Training, autodiff, decoding and scoring use Needle/native C++.
NumPy/SciPy support data handling, calibration, random rotations and analysis;
PyTorch is not a training or inference dependency.

## Consolidated results

This is the only results table. [The CSV](report/results.csv) is an export of
this same table; [the evidence JSON](report/results.json) retains the required
samples, protocols, source/checkpoint hashes and historical provenance.

<!-- RESULTS:START -->
| Study | Optimizer | Gallery format | P@10 (%) | Own-FP32 R@10 (%) | Gallery MB | Compression | Training cost | Query s / 1k |
|---|---|---|---|---|---|---|---|---|
| 10 epochs / original CPU | AdamW | FP32 | 86.11 ± 0.21 | 100.00 ± 0.00 | 14.0800 | 1.00× | 87.762 ± 0.339 s | 3.036 ± 0.019 |
| 10 epochs / original CPU | AdamW | Uniform 2-bit | 85.45 ± 0.17 | 25.26 ± 1.44 | 1.3200 | 10.67× | 87.762 ± 0.339 s | 3.024 ± 0.016 |
| 10 epochs / original CPU | AdamW | Uniform 4-bit | 85.95 ± 0.31 | 74.42 ± 1.21 | 2.2000 | 6.40× | 87.762 ± 0.339 s | 3.029 ± 0.015 |
| 10 epochs / original CPU | AdamW | TurboQuant 2-bit | 85.60 ± 0.19 | 34.95 ± 1.51 | 1.1164 | 12.61× | 87.762 ± 0.339 s | 3.067 ± 0.057 |
| 10 epochs / original CPU | AdamW | TurboQuant 4-bit | 86.01 ± 0.30 | 68.46 ± 1.00 | 1.9964 | 7.05× | 87.762 ± 0.339 s | 3.052 ± 0.034 |
| 10 epochs / original CPU | Muon + aux AdamW | FP32 | 86.25 ± 0.41 | 100.00 ± 0.00 | 14.0800 | 1.00× | 220.025 ± 0.158 s | 3.065 ± 0.084 |
| 10 epochs / original CPU | Muon + aux AdamW | Uniform 2-bit | 85.90 ± 0.45 | 31.70 ± 0.73 | 1.3200 | 10.67× | 220.025 ± 0.158 s | 3.032 ± 0.068 |
| 10 epochs / original CPU | Muon + aux AdamW | Uniform 4-bit | 86.33 ± 0.35 | 76.94 ± 0.64 | 2.2000 | 6.40× | 220.025 ± 0.158 s | 3.053 ± 0.056 |
| 10 epochs / original CPU | Muon + aux AdamW | TurboQuant 2-bit | 85.98 ± 0.33 | 43.93 ± 0.77 | 1.1164 | 12.61× | 220.025 ± 0.158 s | 3.070 ± 0.012 |
| 10 epochs / original CPU | Muon + aux AdamW | TurboQuant 4-bit | 86.18 ± 0.50 | 74.99 ± 0.49 | 1.9964 | 7.05× | 220.025 ± 0.158 s | 3.061 ± 0.021 |
| 220s terminal / original CPU (posthoc) | AdamW | FP32 | 87.57 ± 0.64 | 100.00 ± 0.00 | 14.0800 | 1.00× | 220.003 ± 0.002 s | 3.124 ± 0.023 |
| 220s terminal / original CPU (posthoc) | Muon + aux AdamW | FP32 | 86.07 ± 0.44 | 100.00 ± 0.00 | 14.0800 | 1.00× | 220.015 ± 0.006 s | 3.131 ± 0.056 |
| 128 updates / original CPU | AdamW | — | — | — | — | — | 10.362 ms/step | — |
| 128 updates / original CPU | Muon + aux AdamW | — | — | — | — | — | 25.840 ms/step | — |
| 128 updates / final CPU | AdamW | — | — | — | — | — | 7.310 ms/step (1.430×) | — |
| 128 updates / final CPU | Muon + aux AdamW | — | — | — | — | — | 14.778 ms/step (1.750×) | — |
<!-- RESULTS:END -->

P@10 is the fraction of the ten returned images sharing the query category.
R@10 is overlap with each encoder's own exact FP32 neighbors. These measure
category usefulness and neighbor fidelity respectively; neither is classifier
accuracy. Gallery size is decimal MB, including persistent quantizer metadata;
compression is relative to FP32. Images, model weights, labels and FP32 queries
are outside this storage count. Query time is for a batch of 1,000 queries,
including normalization, rotation, decoding, native scoring and top-k.

The first ten rows use ten epochs on the original backend and validation-loss
checkpoint selection. Quality and cost are mean ± sample SD over training
seeds 8/9/10. TurboQuant rotations 7/8/9 are averaged within training seed first;
they are not nine independent trained models. The two historical equal-time
rows use terminal 220-second models as a posthoc sensitivity check. The original
validation-loss checkpoint comparison slightly favored Muon; terminal selection
reversed that outcome. Both results remain in the evidence file.

The last four rows are a new, direct original-versus-final backend comparison:
128 complete updates, batch 64, seed 8, five alternating process pairs. They
report median milliseconds per update and paired-median speedup. The complete
loop includes batch/Tensor preparation, forward/backward, loss reads and optimizer
updates. Setup and feature export are outside timing. These are timing repetitions,
not model-quality seeds. The old and new paths must produce bit-identical
parameters, losses and features. These short timings are not a rerun of the
220-second retrieval comparison; historical quality rows retain their original
backend and measurement revisions.

The evidence supports gallery compression with small category-precision changes.
Two-bit TurboQuant preserves exact neighbors better than uniform quantization
on these encoders; four-bit uniform can preserve neighbors better than TurboQuant.
Muon has not shown a robust equal-time retrieval advantage in this CPU setting.
A small retrieval gain at equal epochs is not evidence of faster training.

## Method and implementation

Fashion-MNIST has ten clothing categories. A fixed split (seed 17) uses 55,000
training and 5,000 validation images. The gallery excludes validation; queries
are 1,000 common official-test images already explored during the project.
The encoder is a plain ReLU MLP, 784–128–128–128–64–10, with 64-dimensional
retrieval features and a classification training objective. All parameters train.

AdamW uses first/second gradient moments with decoupled weight decay. Muon uses
momentum followed by five Newton–Schulz orthogonalization iterations on three
hidden weight matrices (40,960 of 142,410 parameters). Input/classifier weights
and biases use auxiliary AdamW. A three-candidate learning-rate search at seed 7
selected AdamW/auxiliary LR 0.001 and Muon LR 0.007 by minimum validation loss,
before repeat seeds. Batch size is 64 and weight decay is 0.01. All candidates
and the selection rule are retained in the evidence JSON.

Uniform quantization stores each row's affine scale/offset and packed 2/4-bit
coordinate indices. TurboQuant-MSE applies a shared random orthogonal rotation,
uses sphere-distribution Lloyd–Max codebooks, and stores packed indices plus row
norms. This implements the MSE variant, without QJL residual correction. Only the
gallery is compressed. All methods use cosine scores after normalizing reconstructed
vectors, avoiding a ranking bias from reconstruction-length changes. Search
decodes blocks of 256 rather than a permanent dense gallery. The implementation
demonstrates storage reduction; it does not establish faster queries.

The final CPU implementation includes three systems changes:

- Compact uses direct rank-two indexing and coordinate carry for other ranks.
- Muon retains tiled NS intermediates across iterations, with one changing tiled
  transpose per iteration. Pack once and unpack once per hidden matrix.
- Shared matrix multiplication packs directly from strided/transpose views in
  C++ and unpacks directly to ordinary output, benefiting both optimizers.

MatmulTiled/AlignedDot arithmetic, accumulation order, optimizer formulas and
float32 stages are preserved. Scratch storage is temporary within each call;
there is no persistent packed cache. `NEEDLE_CPU_MATMUL=reference` before import
selects the original public wrapper. `use_tiled_ns=False` selects reference NS.
The total backend benchmark separately builds the original Compact kernel from
its pinned Git revision, keeping current Python code and compiler flags fixed.

Algorithm references: [Muon author implementation](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py)
and [TurboQuant Algorithm 1](https://arxiv.org/html/2504.19874v1).

## Setup and reproduction

```bash
python -m pip install -r requirements-dev.txt
python tools/build_cpu.py
python -m pytest -q
python tools/build_report.py --check
```

The build uses C++17 and -O2. The helper can reuse installed PyTorch pybind11
headers as a build fallback; it does not import PyTorch into model execution.
Native CPU float32 is the evaluated backend; CPU NumPy provides a numerical
reference. Put the four standard Fashion-MNIST IDX gzip files in
`data/fashion-mnist/` (train/t10k images and labels). Data, checkpoints, feature
archives and generated runs are ignored by Git.

```bash
# Fast end-to-end check of training -> own-model features -> all gallery stores.
python experiments/run_report.py --smoke

# Rerun the locked equal-epoch configurations on the current backend.
python experiments/run_report.py --study epochs --seeds 8 9 10

# Current-backend equal-time terminal models; this is a new experiment.
python experiments/run_report.py --study timed --checkpoint terminal --seeds 8 9 10

# Direct total systems ablation and an optional fixed-gradient cost audit.
python experiments/benchmark_system.py --blocks 5 --output results/system.json
python experiments/profile_optimizer.py --output results/profile.json

# Verify the published table, CSV and historical evidence references.
python tools/build_report.py --check --verify-history
```

New runs write to `results/` and do not overwrite published evidence. Historical
quality numbers require the measurement revisions recorded in `report/results.json`;
current-backend equal-time quality still needs measurement. All previous exploratory
runs, designs and plots are available in Git history at commit
`c19dcebfb416fe68e7c26024724de9fe79129213`, rather than in the current report tree.

The maintained tree contains the Needle framework/native kernels, six application
modules for training/export/retrieval, a unified experiment runner, one total
systems benchmark, profiling/aggregation helpers, correctness tests, and the
single consolidated report dataset/table. Tests cover independent numerical
oracles, exact reference parity, optimizer updates, strided layouts, checkpoint
round trips, quantization/scoring, data separation and report completeness.
