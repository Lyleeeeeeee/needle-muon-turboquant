# Joint optimizer, embedding and quantization study

The user approved connecting the two implemented components: export embeddings
from the tuned AdamW and Muon encoders, then compare uncompressed, uniform and
TurboQuant-MSE retrieval. Use the most recent Fashion-MNIST target; preserve all
prior MNIST and classification results. No model retraining or new selection.

## Fixed inputs and controls

Use all six minimum-validation-loss checkpoints from the completed learning-rate
study: AdamW and Muon+auxiliary AdamW, training seeds8/9/10. Both selected the low
candidate (AdamW/auxiliary0.001, Muon matrix0.007). Verify checkpoint hashes against
the original immutable training records; do not choose the best seed.

Export one gallery and query archive per encoder using native Needle CPU.
Gallery: all55,000 training-pool images, excluding the fixed5,000-image validation
holdout from split17. Queries:1,000 common official-test image IDs, selection7.
Use the same image IDs/labels in every archive. Keep legacy default MNIST export
IDs/values unchanged. Record dataset/file/checkpoint/archive and image-ID hashes.
Both gallery and query embeddings must come from that archive's encoder;
64-dimensional spaces from different encoders must never be mixed.

## Retrieval and measurements

Use cosine scoring through the existing native API, top10, blocks256, normalized
inputs and decoded-vector renormalization. Five stores: FP32, uniform2/4 bits,
TurboQuant-MSE2/4 bits. No quantizer/scorer changes, no QJL. Run Turbo rotation
seeds7/8/9. FP32 and uniform do not depend on the rotation seed, so measure them
only with rotation7. This gives18 parent comparisons and54 isolated method workers.
Alternate encoder order across training seeds and reverse rotation order for
seed9. Each worker warms up once, then measures three whole1,000-query timings.
Retained workers run sequentially with BLAS thread limits1.

Primary task metric: same-label precision@10 (fraction of retrieved images with
the query's clothing category). Also report exact-neighbor recall@10 relative
to each encoder's own FP32 search, compression-induced precision change relative
to that same baseline, reconstruction MSE, persistent store bytes and query time.
Classifying a query correctly is a separate exported diagnostic, not retrieval.

Average Turbo rotations within a training seed first, then report mean/sample SD
over the three training seeds. Rotation trials are not additional training seeds.
For each compressed method, report paired differences between encoders and the
interaction: (Muon compressed precision - Muon FP32 precision) minus (AdamW
compressed precision - AdamW FP32 precision). Positive means a smaller drop or
larger gain for Muon in this metric; it does not by itself establish useful synergy.
Keep signed changes; compression can accidentally improve category precision.

## Scope and validation

The Fashion official test was already viewed for classification. This is a
fixed-model retrieval extension, not a new untouched test or model-selection
experiment. Gallery images were used for training. The encoder was trained with
a classification loss, not a dedicated metric-learning/retrieval objective.
Do not infer general optimizer or quantizer superiority from this task.

Test dataset-aware holdout exclusion/export provenance, backward compatibility,
selective benchmark workers and hierarchical aggregation before implementation.
Pass the full suite before retained timings. Verify common IDs, every checkpoint
and source hash, archive dimensions, baseline reproducibility across rotations,
and own-FP32 comparisons. Produce English raw/aggregate reports and a scientific
plot; keep data/checkpoints/feature binaries ignored and publish to the existing
private repository.
