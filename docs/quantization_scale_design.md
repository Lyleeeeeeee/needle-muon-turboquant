# Quantization scale experiment

The user approved two checks: expand retrieval with a fixed encoder, then train
the same encoder architecture on more images. The study changes data scale,
without changing either quantization algorithm.

## Controlled comparisons

- Encoder: existing 784-128-64-10 ReLU model, native Needle CPU, AdamW, learning
  rate 0.003, batch size 64, training seed 7, three epochs.
- Training sizes: 4,096 and 60,000 MNIST training examples.
- Galleries: the same nested 4,096- and 60,000-image selections for both models.
- Queries: the same 1,000 images from the MNIST test split for every condition.
  The first 256 preserve the original smoke-run query selection.
- Quantizers: FP32, affine uniform 2/4 bit, TurboQuant-MSE 2/4 bit. Each model and
  gallery uses shared saved features across methods and rotation seeds 7/8/9.
- Search: normalized features, inner product, top 10, blocks of 256, one warm-up
  and three retained timings per method/rotation. Workers run sequentially.

## Required support

Save encoder weights without changing the existing training sampling or update
order. Independently export features from that checkpoint, preserving the
original gallery/query selections as prefixes and appending extra examples.
Both models use the same export selections. Checkpoint round-trip and feature
identity tests must pass before retained experiments.

Reproduce the original small encoder and compare its exported features with the
existing archive. Preserve all prior records. Record checkpoint hashes, selected
image IDs, dimensions, classification quality, reconstruction error, retrieval
quality, numeric storage and timing scopes.

## Interpretation

The crossed comparisons separate gallery scale from encoder training scale.
Increasing query count improves coverage, but it is not itself an intervention
expected to improve retrieval. Nested queries permit direct comparison of the
original 256-query subset with the complete 1,000-query set.

Three fixed epochs give the larger training run more optimizer updates and
compute, so this is a practical larger-data training comparison, not a causal
data-only experiment under a fixed update budget. One training seed and three
rotation seeds do not establish training-run uncertainty. MNIST test images have
already been used for exploratory evaluation; these results are not an untouched
final test or the planned Fashion-MNIST study. NumPy/SciPy dependencies remain as
documented in the requirements audit; this study does not migrate them.

No parameter tuning or method selection is based on the new retrieval results.
Report negative findings alongside positive ones, and do not infer whole-process
memory savings or speedups from numeric-buffer compression alone.
