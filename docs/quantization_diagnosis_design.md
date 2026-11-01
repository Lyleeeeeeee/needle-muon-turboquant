# Quantization quality diagnosis

The user requests an evidence-based distinction between implementation defects
and the suitability of TurboQuant-MSE for the current retrieval experiment.
Existing algorithms, search defaults and benchmark reports remain unchanged.

## Numerical audit

Compare the code against Algorithm 1 of TurboQuant v1, including sphere density,
Lloyd-Max centroids, Haar orthogonal rotation, nearest-centroid assignment,
packing, inverse rotation and norm restoration. Independently integrate the
scalar density to verify 2/4-bit centroids and expected distortion. Reconstruct
learned features using NumPy float64 and explicit nearest-centroid distances;
compare with native CPU packed decoding and retrieval. Independent references
are diagnostic tools, not replacement Needle implementations or speed claims.

## Controls

Use the frozen 60,000-example encoder. Start with the existing 4,096-image gallery
and original 256 queries; confirm the key scoring result on all 60,000 gallery
images and 1,000 queries. Reuse rotations 7/8/9 and both 2/4-bit widths.

1. Measure feature zeros, signs, row ranges and decoded norm distributions.
2. Apply a fixed random coordinate sign flip or orthogonal rotation to both
   original gallery and queries. These preserve exact inner products. Compare
   how uniform and TurboQuant respond to changing coordinate structure while
   keeping the original retrieval geometry. Record numerical invariance error.
3. Keep decoded vectors fixed and rank by raw inner product, decoded cosine
   (normalizing reconstructed gallery vectors), or squared Euclidean distance.
   The three agree for exact unit vectors; reconstructed norms can differ.
   Changing the scorer is a diagnostic experiment, not a silent default change.
4. Include recall@10, label precision@10 and the paper's hit1@10 metric. The
   paper's nearest-neighbor section uses 1@k, unlike the earlier overlap@10.

## Interpretation limits

Matching an independent reference and theoretical distortion gives evidence
against specific implementation errors; it does not prove the absence of every
possible bug. Controlled coordinate changes can establish dependence on the
coordinate representation, without isolating every aspect of ReLU sparsity.
Scorer improvements establish that evaluation choices contribute to the gap,
without proving that the paper's unimplemented QJL variant would be superior.
Never generalize one model/dataset to all image retrieval.

ReLU features may give per-row affine uniform quantization unusually favorable
structure. TurboQuant's worst-case guarantee is not a promise to outperform
every adaptive quantizer on every structured dataset. The paper's comparison
uses different embedding dimensions, data and baselines (PQ/RabitQ), which must
be distinguished from this 64-dimensional per-row-uniform experiment.
