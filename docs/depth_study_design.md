# Depth-only encoder comparison

The user requests a slightly deeper model so more parameters use Muon, followed
by the current learning-rate and joint retrieval/compression workflow. Keep
Fashion-MNIST fixed for this comparison. It is already clothing images, not
handwritten digits. Harder-dataset discussion remains a possible later study.

## Predefined architecture and training

Add two128-to128 Linear+ReLU layers between the input projection and the64D
embedding layer. Old:784-128-64-10. New:784-128-128-128-64-10. Keep width128,
embedding64, classifier10 and all other training/data choices fixed. This is a
plain deeper MLP, with no residual links, normalization or dropout changes.

Muon updates the two new128x128 weight matrices and the existing128x64 embedding
matrix. All input/classifier weights and all biases use auxiliary AdamW. The
deep model has142,410 parameters;40,960 (28.76%) use Muon, versus8,192/109,386
(7.49%) before. Parameter groups must be unique, disjoint and exhaustive.
Muon still follows the previously pinned float32/Linear-layout implementation.

Use the same55,000 training/5,000 validation split17, batch64, ten epochs, decay0.01
and three predefined configurations per method. Low/medium/high AdamW rates are
0.001/0.003/0.010; Muon matrix rates0.007/0.020/0.060 paired with auxiliary
AdamW0.001/0.003/0.010. Select by seed7 minimum validation loss, tie by candidate
name. Lock choices, then newly train all seeds8/9/10; do not choose a best seed.
Each repeat checkpoint uses minimum validation loss. Do not extend depth/grid
or budget to make either method win after inspecting outcomes. If a selected
candidate is a grid endpoint, disclose the remaining tuning uncertainty.

Compare paired optimizers within depth first. Shallow versus deep results are
a predefined architecture comparison, not matched parameter counts/compute.
Neither means alone establish general optimizer superiority. Do not attribute
any difference uniquely to greater Muon coverage: depth also changes the model.

## Serialization and compatibility

Default extra depth remains zero. Preserve the original shallow construction
order/RNG and six-array version1 checkpoints. Version2 records the extra depth
and every added Linear weight/bias; validate shapes/finiteness and restore the
full architecture. Existing export/checkpoint users must still load old models.
Validate complete grouping, nonzero gradients through every added layer,
checkpoint roundtrip, malformed depth/shape rejection and trainer CLI before
retained runs. Reproduce a saved shallow training/export before and after changes.

## Joint retrieval protocol

Reuse the existing six-model joint study without quantizer/scorer changes:
same55,000 gallery image IDs excluding validation, same1,000 official-test query
IDs,64D embeddings, native cosine top10, blocks256. Each deep encoder generates
both its own gallery and query vectors; only gallery vectors are compressed.
Stores: FP32, uniform2/4 bits, TurboQuant-MSE2/4 bits; Turbo rotations7/8/9.
FP32/uniform once per model; three query timings after warmup in sequential
isolated CPU workers. Average rotations within training seed before aggregating
three training seeds. Verify exact image IDs/labels against the shallow study.

Report primary category precision@10, change from own FP32, paired optimizer and
depth effects, own-FP32-neighbor recall, storage and query time. Report training
loss/accuracy/timing/numeric optimizer state separately. Reload all six selected
checkpoints and reproduce validation metrics. Official Fashion test images were
already viewed; do not claim an untouched test or select using retrieval results.
Feature geometry is descriptive, not a causal explanation. More matrices can
increase Newton-Schulz CPU cost; do not assume better accuracy or faster training.

Publish English source/raw results/plots and the frozen protocol to the existing
private repository; keep data/checkpoints/features local and ignored. No paid
compute, GPU/backend work or changes to the original homework archives.
