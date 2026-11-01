# Muon Training and TurboQuant Retrieval in Needle

A 10-714 deep learning systems project built on the completed HW1–HW3 Needle
framework. AdamW and Muon train image encoders; each encoder's own embeddings
then feed FP32, uniform and TurboQuant-MSE galleries. Training, autodiff,
optimizer updates, decoding and scoring run in Needle/native C++. NumPy/SciPy
support data handling, quantizer setup, checkpoint files and analysis. PyTorch
is not a training or inference dependency.

## Consolidated results

The integrated compression comparison uses six existing **validation-tuned batch-512 models**:
AdamW and Muon at seeds 8/9/10, each trained for 100 epochs (10,800 updates).
Every model contributes all five gallery formats. The integration stage loads
the fixed final checkpoint, exports its features and measures retrieval; it
performs no additional training and does not choose the best retrieval snapshot.
A shared-min metadata follow-up strengthens the ordinary uniform baseline on
those same six models. A separately labelled CIFAR-100 patch-model pilot adds
one paired seed at 20 epochs; its numbers are validation results and have no
between-training-seed error bars. Dataset and architecture both change there.
A follow-up on that fixed CIFAR configuration uses three fresh paired seeds
and a 600-second training budget per model to measure predefined quality goals.

This is the only results table. [CSV](report/results.csv) exports the same rows.
[Evidence JSON](report/results.json) retains every current raw trial, provenance
hash, paired difference and earlier experiment, including negative outcomes.
Superseded batch-64 quality studies remain validated in that evidence and Git
history rather than appearing as competing primary comparisons.

<!-- RESULTS:START -->
| Study | Optimizer | Gallery format | P@10 (%) | Δ vs FP32 (pp) | Own-FP32 R@10 (%) | Gallery MB | Compression | Training cost | Updates | Optimizer state MB | Query s / 1k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Batch 512 / 100 epochs / tuned terminal | AdamW | FP32 | 88.29 ± 0.67 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 341.974 ± 0.804 s | 10,800 | 1.1393 | 2.825 ± 0.019 |
| Batch 512 / 100 epochs / tuned terminal | AdamW | Uniform 2-bit | 88.11 ± 0.73 | -0.19 ± 0.14 | 39.53 ± 0.92 | 1.3200 | 10.67× | 341.974 ± 0.804 s | 10,800 | 1.1393 | 2.827 ± 0.021 |
| Batch 512 / 100 epochs / tuned terminal | AdamW | Uniform 4-bit | 88.31 ± 0.64 | +0.02 ± 0.07 | 82.64 ± 1.27 | 2.2000 | 6.40× | 341.974 ± 0.804 s | 10,800 | 1.1393 | 2.825 ± 0.022 |
| Batch 512 / 100 epochs / tuned terminal | AdamW | TurboQuant 2-bit | 88.05 ± 0.64 | -0.24 ± 0.06 | 45.61 ± 0.94 | 1.1164 | 12.61× | 341.974 ± 0.804 s | 10,800 | 1.1393 | 2.808 ± 0.015 |
| Batch 512 / 100 epochs / tuned terminal | AdamW | TurboQuant 4-bit | 88.25 ± 0.72 | -0.04 ± 0.05 | 78.07 ± 0.93 | 1.9964 | 7.05× | 341.974 ± 0.804 s | 10,800 | 1.1393 | 2.808 ± 0.004 |
| Batch 512 / 100 epochs / tuned terminal | Muon + aux AdamW | FP32 | 87.14 ± 0.56 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 426.649 ± 1.829 s | 10,800 | 0.9754 | 2.821 ± 0.021 |
| Batch 512 / 100 epochs / tuned terminal | Muon + aux AdamW | Uniform 2-bit | 87.02 ± 0.37 | -0.12 ± 0.19 | 52.98 ± 2.65 | 1.3200 | 10.67× | 426.649 ± 1.829 s | 10,800 | 0.9754 | 2.816 ± 0.011 |
| Batch 512 / 100 epochs / tuned terminal | Muon + aux AdamW | Uniform 4-bit | 87.16 ± 0.54 | +0.02 ± 0.10 | 87.83 ± 0.66 | 2.2000 | 6.40× | 426.649 ± 1.829 s | 10,800 | 0.9754 | 2.825 ± 0.007 |
| Batch 512 / 100 epochs / tuned terminal | Muon + aux AdamW | TurboQuant 2-bit | 86.73 ± 0.46 | -0.41 ± 0.11 | 49.83 ± 1.15 | 1.1164 | 12.61× | 426.649 ± 1.829 s | 10,800 | 0.9754 | 2.838 ± 0.023 |
| Batch 512 / 100 epochs / tuned terminal | Muon + aux AdamW | TurboQuant 4-bit | 87.09 ± 0.46 | -0.05 ± 0.10 | 82.52 ± 0.59 | 1.9964 | 7.05× | 426.649 ± 1.829 s | 10,800 | 0.9754 | 2.836 ± 0.014 |
| Fashion / shared-min follow-up / same terminal models | AdamW | Uniform shared-min 2-bit | 88.11 ± 0.73 | -0.19 ± 0.14 | 39.53 ± 0.92 | 1.1000 | 12.80× | 341.974 ± 0.804 s | 10,800 | 1.1393 | 2.800 ± 0.003 |
| Fashion / shared-min follow-up / same terminal models | AdamW | Uniform shared-min 4-bit | 88.31 ± 0.64 | +0.02 ± 0.07 | 82.64 ± 1.27 | 1.9800 | 7.11× | 341.974 ± 0.804 s | 10,800 | 1.1393 | 2.788 ± 0.042 |
| Fashion / shared-min follow-up / same terminal models | Muon + aux AdamW | Uniform shared-min 2-bit | 87.02 ± 0.37 | -0.12 ± 0.19 | 52.98 ± 2.65 | 1.1000 | 12.80× | 426.649 ± 1.829 s | 10,800 | 0.9754 | 2.817 ± 0.010 |
| Fashion / shared-min follow-up / same terminal models | Muon + aux AdamW | Uniform shared-min 4-bit | 87.16 ± 0.54 | +0.02 ± 0.10 | 87.83 ± 0.66 | 1.9800 | 7.11× | 426.649 ± 1.829 s | 10,800 | 0.9754 | 2.810 ± 0.005 |
| CIFAR-100 pilot / 20 epochs / seed 8 | AdamW | FP32 | 11.98 | +0.00 | 100.00 | 2.5600 | 1.00× | 137.055 s | 800 | 0.4383 | 0.534 |
| CIFAR-100 pilot / 20 epochs / seed 8 | AdamW | Uniform 2-bit | 11.28 | -0.70 | 57.28 | 0.2400 | 10.67× | 137.055 s | 800 | 0.4383 | 0.514 |
| CIFAR-100 pilot / 20 epochs / seed 8 | AdamW | Uniform 4-bit | 12.21 | +0.23 | 90.25 | 0.4000 | 6.40× | 137.055 s | 800 | 0.4383 | 0.515 |
| CIFAR-100 pilot / 20 epochs / seed 8 | AdamW | Uniform compact 2-bit | 11.28 | -0.70 | 57.28 | 0.2400 | 10.67× | 137.055 s | 800 | 0.4383 | 0.521 |
| CIFAR-100 pilot / 20 epochs / seed 8 | AdamW | Uniform compact 4-bit | 12.21 | +0.23 | 90.25 | 0.4000 | 6.40× | 137.055 s | 800 | 0.4383 | 0.512 |
| CIFAR-100 pilot / 20 epochs / seed 8 | AdamW | Uniform symmetric 2-bit | 11.01 | -0.97 | 53.36 | 0.2000 | 12.80× | 137.055 s | 800 | 0.4383 | 0.524 |
| CIFAR-100 pilot / 20 epochs / seed 8 | AdamW | Uniform symmetric 4-bit | 11.81 | -0.17 | 88.73 | 0.3600 | 7.11× | 137.055 s | 800 | 0.4383 | 0.513 |
| CIFAR-100 pilot / 20 epochs / seed 8 | AdamW | TurboQuant 2-bit | 11.52 | -0.46 | 67.30 | 0.2164 | 11.83× | 137.055 s | 800 | 0.4383 | 0.515 |
| CIFAR-100 pilot / 20 epochs / seed 8 | AdamW | TurboQuant 4-bit | 11.96 | -0.02 | 89.75 | 0.3764 | 6.80× | 137.055 s | 800 | 0.4383 | 0.516 |
| CIFAR-100 pilot / 20 epochs / seed 8 | Muon + aux AdamW | FP32 | 12.86 | +0.00 | 100.00 | 2.5600 | 1.00× | 142.934 s | 800 | 0.2990 | 0.512 |
| CIFAR-100 pilot / 20 epochs / seed 8 | Muon + aux AdamW | Uniform 2-bit | 12.16 | -0.70 | 59.12 | 0.2400 | 10.67× | 142.934 s | 800 | 0.2990 | 0.515 |
| CIFAR-100 pilot / 20 epochs / seed 8 | Muon + aux AdamW | Uniform 4-bit | 12.71 | -0.15 | 90.67 | 0.4000 | 6.40× | 142.934 s | 800 | 0.2990 | 0.515 |
| CIFAR-100 pilot / 20 epochs / seed 8 | Muon + aux AdamW | Uniform compact 2-bit | 12.16 | -0.70 | 59.12 | 0.2400 | 10.67× | 142.934 s | 800 | 0.2990 | 0.515 |
| CIFAR-100 pilot / 20 epochs / seed 8 | Muon + aux AdamW | Uniform compact 4-bit | 12.71 | -0.15 | 90.67 | 0.4000 | 6.40× | 142.934 s | 800 | 0.2990 | 0.515 |
| CIFAR-100 pilot / 20 epochs / seed 8 | Muon + aux AdamW | Uniform symmetric 2-bit | 11.69 | -1.17 | 56.65 | 0.2000 | 12.80× | 142.934 s | 800 | 0.2990 | 0.516 |
| CIFAR-100 pilot / 20 epochs / seed 8 | Muon + aux AdamW | Uniform symmetric 4-bit | 12.53 | -0.33 | 89.44 | 0.3600 | 7.11× | 142.934 s | 800 | 0.2990 | 0.515 |
| CIFAR-100 pilot / 20 epochs / seed 8 | Muon + aux AdamW | TurboQuant 2-bit | 12.31 | -0.55 | 68.04 | 0.2164 | 11.83× | 142.934 s | 800 | 0.2990 | 0.512 |
| CIFAR-100 pilot / 20 epochs / seed 8 | Muon + aux AdamW | TurboQuant 4-bit | 12.81 | -0.05 | 90.48 | 0.3764 | 6.80× | 142.934 s | 800 | 0.2990 | 0.508 |
| CIFAR / 600s budget / First P@10 ≥ 10% (3/3 reached) | AdamW | FP32 | 10.12 ± 0.03 | +0.00 ± 0.00 | 100.00 ± 0.00 | 2.5600 | 1.00× | 52.773 ± 9.488 s | 433 ± 76 | 0.4383 | — |
| CIFAR / 600s budget / First P@10 ≥ 10% (3/3 reached) | Muon + aux AdamW | FP32 | 10.24 ± 0.29 | +0.00 ± 0.00 | 100.00 ± 0.00 | 2.5600 | 1.00× | 25.666 ± 1.326 s | 207 ± 12 | 0.2990 | — |
| CIFAR / 600s budget / First P@10 ≥ 15% (0/3 reached) | AdamW | FP32 | — | — | — | 2.5600 | 1.00× | — | — | 0.4383 | — |
| CIFAR / 600s budget / First P@10 ≥ 15% (0/3 reached) | Muon + aux AdamW | FP32 | — | — | — | 2.5600 | 1.00× | — | — | 0.2990 | — |
| CIFAR / 600s budget / First P@10 ≥ 20% (0/3 reached) | AdamW | FP32 | — | — | — | 2.5600 | 1.00× | — | — | 0.4383 | — |
| CIFAR / 600s budget / First P@10 ≥ 20% (0/3 reached) | Muon + aux AdamW | FP32 | — | — | — | 2.5600 | 1.00× | — | — | 0.2990 | — |
| CIFAR / 600s budget / First P@10 ≥ 30% (0/3 reached) | AdamW | FP32 | — | — | — | 2.5600 | 1.00× | — | — | 0.4383 | — |
| CIFAR / 600s budget / First P@10 ≥ 30% (0/3 reached) | Muon + aux AdamW | FP32 | — | — | — | 2.5600 | 1.00× | — | — | 0.2990 | — |
| 128 updates / original CPU | AdamW | — | — | — | — | — | — | 7.290 ms/step | 128 | — | — |
| 128 updates / original CPU | Muon + aux AdamW | — | — | — | — | — | — | 18.197 ms/step | 128 | — | — |
| 128 updates / final CPU | AdamW | — | — | — | — | — | — | 5.082 ms/step (1.435×) | 128 | — | — |
| 128 updates / final CPU | Muon + aux AdamW | — | — | — | — | — | — | 10.341 ms/step (1.749×) | 128 | — | — |
| Batch 64 / 32,768 images | AdamW | — | — | — | — | — | — | 113.550 ± 1.742 µs/image | 512 | 1.1393 | — |
| Batch 64 / 32,768 images | Muon + aux AdamW | — | — | — | — | — | — | 231.198 ± 1.032 µs/image | 512 | 0.9754 | — |
| Batch 256 / 32,768 images | AdamW | — | — | — | — | — | — | 69.215 ± 0.401 µs/image | 128 | 1.1393 | — |
| Batch 256 / 32,768 images | Muon + aux AdamW | — | — | — | — | — | — | 98.658 ± 0.455 µs/image | 128 | 0.9754 | — |
| Batch 512 / 32,768 images | AdamW | — | — | — | — | — | — | 62.721 ± 0.341 µs/image | 64 | 1.1393 | — |
| Batch 512 / 32,768 images | Muon + aux AdamW | — | — | — | — | — | — | 77.458 ± 0.406 µs/image | 64 | 0.9754 | — |
| Batch 512 / First P@10 ≥ 80% (3/3 reached) | AdamW | FP32 | 80.77 ± 0.37 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 3.429 ± 0.027 s | 108 | 1.1393 | — |
| Batch 512 / First P@10 ≥ 80% (3/3 reached) | Muon + aux AdamW | FP32 | 80.95 ± 0.27 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 4.389 ± 0.150 s | 108 | 0.9754 | — |
| Batch 512 / First P@10 ≥ 85% (3/3 reached) | AdamW | FP32 | 85.21 ± 0.20 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 25.162 ± 6.638 s | 792 ± 204 | 1.1393 | — |
| Batch 512 / First P@10 ≥ 85% (3/3 reached) | Muon + aux AdamW | FP32 | 85.19 ± 0.19 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 25.100 ± 2.441 s | 630 ± 62 | 0.9754 | — |
| Batch 512 / First P@10 ≥ 87% (3/3 reached) | AdamW | FP32 | 87.21 ± 0.16 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 93.711 ± 34.271 s | 2,952 ± 1,087 | 1.1393 | — |
| Batch 512 / First P@10 ≥ 87% (3/3 reached) | Muon + aux AdamW | FP32 | 87.10 ± 0.10 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 173.610 ± 72.104 s | 4,392 ± 1,811 | 0.9754 | — |
| Batch 512 / First P@10 ≥ 88% (3/3 reached) | AdamW | FP32 | 88.20 ± 0.18 | +0.00 ± 0.00 | 100.00 ± 0.00 | 14.0800 | 1.00× | 191.401 ± 65.940 s | 6,048 ± 2,083 | 1.1393 | — |
| Batch 512 / First P@10 ≥ 88% (0/3 reached) | Muon + aux AdamW | FP32 | — | — | — | 14.0800 | 1.00× | — | — | 0.9754 | — |
<!-- RESULTS:END -->


Measured takeaways for the labelled studies:

- **Compression is effective for both encoders.** TurboQuant 2-bit reduces
  the gallery from 14.08 to 1.1164 MB (12.61×), with mean P@10 losses of
  0.24 percentage points for AdamW and 0.41 for Muon. TurboQuant 4-bit uses
  1.9964 MB (7.05×) with losses of 0.04/0.05 points.
- **A stronger uniform format changes the Fashion storage comparison.**
  Sharing the repeated zero minimum reduces uniform 2-bit to 1.100005 MB
  (12.80×) and uniform 4-bit to 1.980005 MB (7.11×), including a one-byte mode
  flag. Packed codes, decoded values and retrieval results stay identical to
  their affine counterparts. Those P@10 losses remain 0.19/0.12 points at two
  bits. TurboQuant therefore has no gallery-byte advantage over this stronger
  baseline on these ReLU features, and its mean category precision is lower.
  Neighbor-fidelity comparisons differ: TurboQuant 2-bit has higher R@10 for
  the AdamW model, while four-bit uniform has higher R@10 for both encoders.
- **Fashion models show no overall Muon training advantage.** At the fixed
  100-epoch endpoint, AdamW costs 341.97 ± 0.80 s with 88.29 ± 0.67% P@10;
  Muon costs 426.65 ± 1.83 s with 87.14 ± 0.56%. All three paired terminal
  comparisons favor AdamW in both time and precision. Muon retains 14.38% less
  numeric optimizer state, which does not establish lower peak training memory.
- **Fashion step efficiency and time efficiency differ.** At the earlier 85% target,
  Muon averages 630 rather than 792 updates (20.5% fewer), while times are
  essentially tied at 25.10/25.16 s. Muon wins only one of three paired primary
  timing comparisons. Higher targets favor AdamW. Requiring two consecutive
  85% observations gives a modest secondary Muon mean advantage (27.31 versus
  29.71 s; two of three wins), without changing the higher-target conclusion.
- **Storage savings do not establish query acceleration.** Current mean query
  costs are approximately 2.81–2.84 s per 1,000 queries across formats, with
  no clear consistent speed advantage. The separate CPU implementation changes
  speed up complete fixed-work updates by 1.435× for AdamW and 1.749× for Muon
  versus each optimizer's original implementation, with bit-identical outputs.



- **CIFAR pilot: promising, not a confirmed optimizer win.** With the same
  800 updates, AdamW takes 137.05 s and reaches 11.98% P@10; Muon takes
  142.93 s and reaches 12.86%. Both start at 4.94%. Muon uses 4.3% more
  training time for a 0.88-point higher terminal result in this one seed.
  That single-seed endpoint alone cannot establish matched-quality timing;
  the fresh-seed follow-up supplies that comparison.
- **CIFAR fresh seeds: Muon reaches 10% with about half the training time.**
  AdamW needs 433 ± 76 updates and 52.77 ± 9.49 s; Muon needs 207 ± 12
  and 25.67 ± 1.33 s. All three paired timing comparisons favor Muon,
  including the two-observation confirmation check. Neither optimizer reaches
  15%, 20% or 30% in any of the three 600-second runs. This establishes an
  early-target benefit for the fixed small CPU configuration and validation cohort.
- **CIFAR pilot: TurboQuant 2-bit improves on affine uniform in both models.**
  AdamW P@10 is 11.52% versus 11.28%; Muon is 12.31% versus 12.16%.
  TurboQuant uses 216,400 bytes versus 240,000 and has higher own-FP32 R@10.
  The stronger symmetric baseline uses only 200,001 bytes, with lower P@10
  of 11.01%/11.69%; that comparison remains a storage/quality trade-off.
  Four-bit results are mixed. These observations average three quantizer
  rotations on each of two trained models, not independent training repeats.


## Reading the results

The first ten rows are the current optimizer × gallery-format comparison.
Training time and optimizer state repeat across formats because a single model
supplies all five stores; they are not five separate training runs. The next four
rows strengthen the uniform metadata format on those same Fashion checkpoints.
The following 18 rows are the separate one-seed CIFAR pilot, including symmetric
and compact uniform controls. Eight fresh-seed CIFAR rows report the four
predeclared goals, including full-budget non-attainment. Four CPU implementation rows, six batch-throughput
rows and eight validation-tuned Fashion target-crossing rows follow. These supporting studies answer different
questions and must not be combined as if they used the same checkpoint.

- **P@10:** fraction of the ten returned images sharing the query's category.
- **Δ:** signed P@10 change against that same encoder's FP32 store, in percentage
  points. This isolates compression's effect from encoder quality.
- **Own-FP32 R@10:** overlap with that encoder's exact FP32 neighbors. Preserving
  identical neighbors is a separate objective from finding same-category images.
- **Gallery MB:** actual persistent numerical buffers, including packed codes,
  norms/scales/offsets and shared rotation/codebook metadata, using decimal MB.
  Images, labels, queries and model weights are outside this count.
- **Optimizer-state MB:** retained numerical optimizer buffers only. Parameters,
  gradients, Python objects and temporary Newton–Schulz workspace are excluded;
  this is not peak training memory and belongs to a different pipeline stage.

Fashion quality means ± sample SD use three training seeds. Single-seed CIFAR
values display no SD; repeat uncertainty is not measured there. TurboQuant rotations 7/8/9
are averaged **within** each model first, not counted as nine trained models.
Query cost uses each worker's median of three measurements after one warmup,
then the same within-model/between-model aggregation. Its scope includes query
normalization/rotation, block decoding, reconstructed-vector normalization,
native scoring and top-k for 1,000 queries. Feature generation and index building
are excluded and recorded separately in raw artifacts. Compressed storage does
not automatically imply faster search.

The current training cost is the source model's complete measured training loop
to the 100-epoch endpoint, including all optimizer work; setup, checkpoint I/O
and offline evaluation are excluded. It is **not** the time to first reach 85%.
The fixed-target rows use the original trajectory's own saved observations.
No checkpoint-saving or offline-evaluation cost is disguised as training speed.

## Protocol and controls

Fashion-MNIST supplies ten clothing categories. Split seed 17 fixes 55,000
training and 5,000 validation images. All encoders share each seed's initial
parameters, training image order and retrieval image IDs. The gallery contains
all 55,000 training images and excludes validation. Queries are 1,000 common
historically explored official-test images, so this is an exploratory benchmark,
not an untouched final test set. The ReLU MLP is 784–128–128–128–64–10, trained
with classification loss, with 64-dimensional features and weight decay 0.01.

The learning-rate search uses seed 7 and **training-holdout retrieval queries**,
which are disjoint from the gallery. Each optimizer receives three candidates
and 15 epochs per candidate. AdamW rates are 0.001/0.003/0.01; Muon main/auxiliary
rates are (0.007, 0.001), (0.021, 0.003), (0.07, 0.01). Selection maximizes the
normalized trapezoidal area of validation P@10 versus updates at epochs
0/1/2/3/5/8/10/15, breaking ties by terminal precision then candidate ID.
The resulting **AdamW 0.003 and Muon 0.007 / auxiliary AdamW 0.001** are sealed
before any final-seed evaluation. Tuning consumes 155.27/190.59 training-loop
seconds for AdamW/Muon, plus offline validation evaluation, separately from final
training costs. All candidates are retained.

Both selected configurations then train from scratch at seeds 8/9/10 for the
same predefined 100-epoch budget: 10,800 updates and 5.5 million image exposures.
The last minibatch has 216 images. Original trajectory snapshots occur every
54 updates through 1,080, then every 216 through 10,800, including initialization.
For the gallery integration, every model uses its fixed step-10,800 checkpoint.
This is an additional analysis of the existing completed trajectories, not a
new independent model experiment or a pre-registered result.

The integration verifies the sealed parent run, selected learning rates,
training-record/checkpoint hashes, own-model feature archives, common image IDs,
source files, dataset and native binary. Recomputed terminal FP32 P@10 must equal
the original trajectory observation exactly. All 54 optimizer/seed/format/rotation
trials run in isolated **sequential** CPU workers with thread limits of one;
no other retained benchmark or heavy test runs concurrently. Each compressed
format is compared to its own model's FP32 neighbors. Export wall time,
quantizer setup/build time, all query timing samples, numerical buffer sizes,
relative reconstruction MSE and process RSS remain in raw evidence. Process RSS
is a whole-worker high-water mark, not query-only peak memory.

Targets 80/85/87/88% use the first **observed** qualifying snapshot. Their true
first crossing is unresolved between observations and a crossing can later
revert. Evidence also includes a two-consecutive-observation check, charging
through the confirming second snapshot. Failed/unreached seeds are retained;
incomplete target samples do not receive success-only means. Training is
sequential, and trajectory quality evaluation runs after training. Those
monitoring times are not query-performance benchmarks.

The batch diagnostic uses five isolated timing repetitions per configuration at
seed 8. Each warms on the same first 2,048 ordered images, then measures the same
next 32,768 images: 512/128/64 updates at batch 64/256/512. Its total includes
batch/Tensor preparation, loss reads, forward, backward and optimizer work.
Initialization, warmup and output hashing are excluded. These are throughput
repetitions, not independent quality trials. Repeated same-configuration output
hashes match; different batch sizes need not produce the same parameters.

The original-to-final systems study uses five alternating sequential process
pairs, each measuring 128 updates at seed 8/batch 64. Displayed step costs are
medians; speedup is the median within-pair ratio. The two implementations use
identical Python/model/optimizer formulas and compiler flags. Parameters,
losses and features match bit-for-bit. Implementation gains are distinct from
claims about Muon outperforming AdamW.


## CIFAR-100 feasibility pilot and stronger controls

The additional CIFAR-100 study changes **both dataset and architecture**. It is
not an isolated test of dataset difficulty, and its single paired training seed
is not a replication of the three-seed Fashion benchmark. Its role is to test
whether a more image-oriented model can learn useful features within a small CPU
budget. The official test set is downloaded but is not used for training,
selection or the reported pilot results.

A deterministic stratified split of the official training set supplies 10,000
training images (100 per fine class) and 2,000 validation images (20 per class).
The gallery uses those 10,000 training images; queries use ten validation images
per class, for 1,000 total. Normalization statistics come only from the selected
training images. Training uses paired deterministic horizontal flips. Fine-label
P@10 is the primary retrieval measurement; classification validation loss and
accuracy provide a separate learning check.

The encoder is a small [MLP-Mixer-inspired adaptation](https://arxiv.org/abs/2105.01601):
16 nonoverlapping 8×8 RGB patches, width 64, two residual mixing blocks, token-MLP
width 32 and channel-MLP width 128. It uses existing Needle Linear, LayerNorm,
ReLU, transpose and reduction operators. No convolution operator or GPU backend
is added. Final LayerNorm features are averaged over patches, producing signed
64-dimensional embeddings without an extra final ReLU. This is a from-scratch
CPU adaptation, not the paper's pretrained large-model result. Muon handles
the eight internal mixing matrices: 34,816 of 54,788 parameters (63.55%). Patch
projection, classifier, normalization parameters and biases use auxiliary AdamW.

At seed 7, both optimizers receive three configurations and five epochs each.
AdamW rates are 0.0003/0.001/0.003; Muon main/auxiliary rates are
(0.0021, 0.0003), (0.007, 0.001), (0.021, 0.003). The validation P@10 area over
updates at epochs 0/1/3/5 selects the rates, breaking ties by terminal precision
then candidate ID. All six candidates remain in evidence. This selects AdamW
0.003 and Muon main 0.021/auxiliary 0.003. The selection is sealed before fresh
seed-8 training. There is one final paired seed, with no claimed between-model
standard deviation.

A small synthetic timing probe, after warmup, determines the final epoch budget
without using quality: 20 epochs if the slower optimizer's median step is at
most 0.65 s, ten if at most 1.3 s, otherwise five. The measured medians are
0.178/0.182 s, selecting 20 epochs: 800 updates and 200,000 image presentations
per final model at batch 256. Formal training seconds include batch preparation,
augmentation, forward, backward and all optimizer work; setup, checkpoint I/O
and offline validation/retrieval evaluation are excluded. Initial, intermediate
and terminal checkpoints are retained. Both optimizers use the same starts and
augmented data order.

Two stronger uniform storage controls avoid attributing baseline metadata costs
to a TurboQuant algorithmic advantage:

- **Shared-min affine uniform** keeps the original codes/scales. If all row
  minima are identical, it stores one float32 minimum instead of a full column;
  otherwise it retains the column. A one-byte mode tag is included. Required
  decoder metadata arrays are generated temporarily, inside measured search.
  The six existing Fashion feature galleries have zero minimum in every row.
  This format is tested for bit-identical decoded vectors and search results.
- **Symmetric uniform** stores one scale per row plus the one-byte mode tag.
  Its minimum is derived from the scale, and all 2/4-bit codes are really packed.
  It is a separate quantization grid and does not promise affine-equivalent
  reconstructions. This control is useful for the signed CIFAR embeddings.

Gallery comparisons retain raw timing samples and the actual serialized packed
stores. All formats use the same model's vectors, cosine normalization and exact
FP32 neighbors, with three TurboQuant rotations averaged within each trained
model. Isolated query workers run sequentially only after all retained training
and required tests finish. Learning-rate selection and reported pilot quality
both use the holdout; pilot results are exploratory validation measurements,
not untouched-test generalization estimates. Existing Fashion results are
preserved rather than replaced by the new task. Dataset source and download
checksum: [official CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html), binary
archive MD5 `03b5dce01913d631647c71ecec9e9cb8`.


## Predeclared CIFAR training-time milestones

The follow-up measures time to the same retrieval quality using three fresh
paired seeds **9/10/11**, distinct from the explored pilot seed 8. Model,
10,000/2,000 split, 1,000 query images, batch 256, augmentation, FP32 gallery
and selected rates are fixed to the pilot configuration. No further tuning is
performed. The official test set remains unused. This tests repeat variation
on the existing validation cohort rather than untouched-test generalization.

Targets are fixed before the new runs at **10%, 15%, 20% and 30% P@10**.
Ten percent corresponds to one same-category result among ten, averaged over
queries; 20% and 30% correspond to two and three. Fifteen percent is an
intermediate milestone beyond the pilot's approximately 12–13% terminal level.
All targets remain in the report, including those neither optimizer reaches.

Every model receives **600 seconds of pure training updates**, including batch
preparation, augmentation, forward, backward and all optimizer/Newton–Schulz
work. Initial data/model setup is excluded. Checkpoint I/O and offline quality
evaluation are also excluded, with their costs recorded separately. The final
indivisible update may slightly exceed the
budget; its full time is charged. All six models run sequentially in isolated
single-thread CPU processes, without quality-based early stopping or extensions.

Snapshots include initialization, every 20 updates through update 400, then
every 40 updates, and the actual budget endpoint even when off-grid. This is
much denser than the pilot's five-epoch intervals. Three offline quality workers
run only after all retained training timings finish. Their monitoring costs are
not reported as query-performance benchmarks.

The primary result is each target's first **observed** qualifying snapshot,
with the preceding observation retained as a bracket. The exact crossing update
is unresolved between observations, and a brief crossing can later revert.
A predefined two-consecutive-observation sensitivity charges time through the
second confirming checkpoint. Per-seed statuses, observations, actual budget
usage and paired time/step ratios remain in the evidence. A target receives a
three-seed mean only when all three runs reach it; otherwise its reach count is
shown, without averaging only successful runs. Numerical failures retain their
real observation prefix and are distinguished from full-budget non-attainment.
Targets and resource caps are not changed after observing the new curves.

All six runs completed their full budgets with finite parameters, producing
794 checked snapshots. At 10% P@10, the three first-observed AdamW times are
48.97/63.57/45.77 s, versus Muon's 27.20/24.83/24.98 s at paired seeds 9/10/11.
The observed time gaps exceed the adjacent checkpoint intervals in every pair.
Mean paired Muon/AdamW time is 0.497.
Requiring two consecutive observations gives 56.96 ± 10.64 s for AdamW and
28.11 ± 1.26 s for Muon, again three paired Muon wins. The 15/20/30% goals
remain 0/3 reached for both methods, with no success-only time averages.

At the fixed 600-second endpoint, P@10 is 11.88 ± 0.64% for AdamW and
12.51 ± 0.37% for Muon; the paired Muon differences are +0.42/+1.01/+0.46
percentage points. These terminal values are a descriptive secondary outcome.
Mean complete-update costs are 123.08/125.30 ms for AdamW/Muon, so Muon still
costs slightly more per update. Its smaller step count to 10% outweighs that
overhead here. The optimizer phase accounts for 2.04%/3.64% of full training
time; all of it is charged. Retained numeric state is 438,304/299,040 bytes,
a 31.77% Muon reduction, excluding parameters, gradients and temporary workspace.

## Needle implementation

AdamW maintains first/second gradient moments and uses decoupled weight decay.
Muon applies momentum and five Newton–Schulz iterations to three hidden weight
matrices: 40,960 of 142,410 parameters. Input/classifier weights and biases use
auxiliary AdamW. These are the Fashion encoder groups. The separate CIFAR patch encoder uses
63.55% Muon-eligible parameters. The studies do not establish which optimizer
is best for larger models or other hardware.

Uniform quantization stores real packed 2/4-bit indices with per-row affine
scale/offset. TurboQuant-MSE applies a shared random orthogonal rotation and
sphere-distribution Lloyd–Max codebooks, then stores packed indices and row
norms. It implements Algorithm 1 without QJL residual correction. Only gallery
vectors are compressed; queries remain FP32. All formats use cosine scoring
with reconstructed vectors normalized again. Search decodes blocks of 256,
without retaining a full dense reconstructed gallery. The two methods are
combined as training and storage stages; a positive optimizer/compression
interaction is measured rather than assumed.

Three CPU systems changes preserve optimizer formulas and float32 accumulation:

- Compact uses direct rank-two indexing and coordinate carry for other ranks.
- Muon retains tiled intermediates across Newton–Schulz iterations, packs/unpacks
  once per hidden matrix and refreshes the tiled transpose once per iteration.
- Shared matrix multiplication packs directly from strided/transpose views in
  C++ and unpacks to ordinary output, benefiting both optimizers.

Scratch buffers are temporary within a call. `NEEDLE_CPU_MATMUL=reference` selects
the original public wrapper before import; `use_tiled_ns=False` selects reference
Newton–Schulz. The systems ablation separately compiles the pinned original
Compact backend using the same current Python code and C++17/-O2 build helper.

References: [Muon author implementation](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py)
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
is a numerical reference. The build helper may reuse installed PyTorch pybind11
headers without using PyTorch for model execution. Data, models, exported
features and generated runs stay ignored.

```bash
# Validation-only tuning, sealed selection, then six paired batch-512 trajectories.
python experiments/tune_batch512.py --output-dir results/reproduction/tuned_batch512

# Reuse those fixed terminal models; export features and compare all galleries.
python experiments/evaluate_tuned.py --tuned-run results/reproduction/tuned_batch512/run.json --output-dir results/reproduction/integrated

# Small CIFAR-100 patch-model pilot (requires the official training binary).
python experiments/cifar_pilot.py --train-file data/cifar-100/train.bin --output-dir results/reproduction/cifar_pilot

# Reuse each final pilot model's embeddings; includes both stronger uniform controls.
python experiments/compare_galleries.py --features results/reproduction/cifar_pilot/final/adamw_features.npz --output results/reproduction/cifar_galleries/adamw.json
python experiments/compare_galleries.py --features results/reproduction/cifar_pilot/final/muon_features.npz --output results/reproduction/cifar_galleries/muon.json

# Metadata-equivalent Fashion control; repeat for each existing encoder/seed archive.
python experiments/compare_galleries.py --features results/reproduction/integrated/adamw_seed8_features.npz --methods fp32 uniform_2 uniform_4 compact_2 compact_4 --output results/reproduction/shared_min/adamw_seed8.json

# Three fresh paired seeds, 600s training/model, then denser offline quality.
python experiments/cifar_time_to_quality.py --output-dir results/reproduction/cifar_time_to_quality

# Supporting implementation and batch-cost studies, run separately from queries.
python experiments/benchmark_system.py --blocks 5 --output results/reproduction/system.json
python experiments/benchmark_batch.py --output-dir results/reproduction/batch_cost

# Rebuild the one README/CSV table from committed evidence without rerunning models.
python tools/build_report.py
python tools/build_report.py --check --verify-history
```

Use a fresh output directory for each experiment. `--resume` reuses only stages
whose frozen protocols, source/data/native and checkpoint/feature hashes match.
Training and integration can run separately. The synthetic-data integration test
exercises both optimizers, all stores, checked resume and corruption rejection.
New measurements do not silently replace committed evidence.

The original-workspace integration is attached with:

```bash
python tools/build_report.py --integrated-run results/tuned_integration_20261101/run.json
```

Its parent must match the committed tuned study. Current and historical studies
remain sealed in the same evidence JSON. Earlier batch-64 experiments are
reproducible with `run_report.py` and `time_to_quality.py`; their protocols and
selection sensitivities remain in the evidence. The original artifact audit is
available through `tools/build_report.py --check --verify-history --verify-artifacts`
on the measurement workspace. Current integration and pilot audit proofs record original worker reports,
checkpoint identity, independently reconstructed data orders, augmented-image
streams, exported-feature samples, packed gallery reloads, retained neighbor
arrays, true buffer bytes and FP32 retrieval parity.

The curated repository contains Needle/kernels, runnable app and experiment
modules, build/report tools, relevant tests and this single report dataset/table. Earlier
exploratory documents and experiments remain in Git history. Keep the repository
private because it includes Needle homework implementations.
