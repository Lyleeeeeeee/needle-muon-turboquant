# Why TurboQuant-MSE trails uniform in the original experiment

Runtime follow-up: decoded-cosine scoring is now available in Needle through
`search(..., metric='cosine')`. The [native comparison](native_cosine_summary.md)
includes quality and timed normalization costs. The independent-reference
diagnosis below is preserved as the evidence that motivated that change.

The evidence points to a substantial scoring-choice effect and a favorable
coordinate representation for affine uniform quantization. The audit did not
find an error in the core quantizer on the examined inputs. The original
raw-inner-product measurements remain valid for that documented scorer, but
they do not establish that TurboQuant is generally unsuitable for retrieval.

## Independent implementation audit

The implementation follows [Algorithm 1 of TurboQuant](https://arxiv.org/html/2504.19874v1):
orthogonal Gaussian-QR rotation, sphere-distribution Lloyd-Max centroids,
nearest-centroid indices, inverse rotation and norm restoration. Row-vector
right multiplication is the transpose convention of the paper's column form.

An independent implementation used numerical quadrature of the density and
conditional moments, rather than the production Beta CDF and analytic moments.
Native packed results were then checked against explicit nearest-centroid
distances and a float64 reconstruction on 4,096 learned 64D vectors, seed 7.

| Check | 2 bits | 4 bits |
|---|---:|---:|
| Maximum centroid difference from independent quadrature | 6.42e-9 | 1.50e-8 |
| Incorrect packed indices versus explicit nearest-centroid reference | 0 / 262,144 | 0 / 262,144 |
| Maximum native decode error using identical indices | 1.95e-7 | 2.06e-7 |
| Index changes using float64 rather than native rotation | 0 | 0 |
| Independently integrated expected unit-vector MSE | 0.114526 | 0.009132 |

The rotation's maximum orthogonality error was 2.87e-8. The separate random-sphere
control gave mean MSE 0.114647 / 0.009107 at 2/4 bits, consistent with these
expectations. Native raw-IP search and float64-reference top-10 sets agreed for
all checked 16-query subsets in the identity-frame conditions. These checks
target specific possible defects; they do not prove every input is bug-free.

## Finding 1: reconstructed length was influencing the ranking

The source feature vectors are normalized before compression. Quantization
changes their lengths. The existing search correctly computes its documented
raw inner product, so a reconstructed vector with a larger norm can obtain an
unwanted score advantage when the intended task is cosine-style similarity.

We kept every compressed code and decoded vector fixed, and evaluated three
scorers: raw inner product, decoded cosine, and squared Euclidean distance.
Cosine normalizes reconstructed gallery rows for scoring. Distance accounts
for their squared norms. All agree on exact unit vectors; compressed vectors
are no longer exactly unit length. Both quantizers receive the same scorer.

The table uses the frozen 60,000-example encoder, the full 60,000-image gallery
and 1,000 common queries. TurboQuant entries are means of rotations 7/8/9.
Recall@10 is the mean overlap with the exact FP32 top-10 image IDs.

| Bits | Quantizer | Original raw-IP recall | Decoded-cosine recall | Squared-L2 recall |
|---:|---|---:|---:|---:|
| 2 | Uniform | 15.67% | 44.99% | 44.24% |
| 2 | TurboQuant-MSE | 14.20% | **50.20%** | 50.27% |
| 4 | Uniform | 56.41% | **86.78%** | 86.74% |
| 4 | TurboQuant-MSE | 36.50% | 81.64% | 81.60% |

At four bits the gap falls from 19.91 to 5.14 percentage points. At two bits the
ordering reverses: TurboQuant leads by 5.21 points under cosine. Thus the
earlier broad interpretation of the raw-IP gap was incomplete. The four-bit
uniform advantage is still present under the fair cosine comparison and must
remain visible.

This is a quality-only independent-reference experiment. Its scoring arithmetic
is NumPy float64, and the tested quantization/decode kernels are native Needle
CPU. At the time of this diagnosis the cosine/distance scorers were not part of
the production Needle search API. Cosine was subsequently integrated as an
explicit option; the records here remain independent-reference measurements.
This page makes no new packed-kernel, runtime-speed or peak-memory claim, and
earlier records and the default raw-IP scoring rule are unchanged.

## Finding 2: the current coordinate representation favors uniform

In the full learned gallery, **65.24% of coordinates are exactly zero and none
are negative**. This is the encoder's final ReLU representation. Every row's
minimum is zero, and affine uniform quantization places one endpoint exactly
at zero. It therefore reconstructs all those zeros exactly. TurboQuant spreads
coordinates through rotation before scalar quantization, so it does not retain
this particular coordinate-wise zero advantage.

To test representation dependence without changing the retrieval problem, we
transformed both gallery and queries by either fixed coordinate sign flips or
a shared orthogonal matrix. Exact inner products and neighbors are preserved.
The sign-flip control has zero measured similarity error; the orthogonal control
has maximum sampled error 2.00e-8. FP32 top-10 overlap was 100% in both controls.

These controls use 4,096 gallery images and the original 256 queries from the
same full-trained encoder. Entries below all use decoded-cosine scoring.

| Coordinate representation | Uniform-2 recall | TurboQuant-2 recall | Uniform-4 recall | TurboQuant-4 recall |
|---|---:|---:|---:|---:|
| Original nonnegative ReLU features | 67.19% | 67.63% | 92.38% | 89.18% |
| Fixed coordinate sign flips | 22.81% | 67.40% | 82.73% | 89.32% |
| Fixed orthogonal change of basis | 54.06% | 67.63% | 89.30% | 88.70% |

The sign flip retains the exact same zero fraction and retrieval geometry while
changing the signs of some nonzero coordinates. Zero is no longer generally an
affine-quantizer endpoint. At four bits, uniform reconstruction MSE increases
from 0.001585 to 0.014942; its mean absolute error on original zeros changes
from exactly zero to 0.013423. TurboQuant's mean MSE stays near its expected
scale (0.009812 before, 0.008715 after). Under this control, TurboQuant beats
uniform even at four bits.

This establishes dependence on the coordinate representation and supports the
favorable nonnegative/zero-aligned explanation. It does not isolate sparsity
rate alone: the sign flip preserves sparsity but changes signs, ranges and zero
alignment. A single fixed sign mask/basis is also not a universal robustness
evaluation. Rotation alone is reversible and does not lose information; the
representation changes the error introduced by subsequent scalar quantization.

The paper's guarantee averages over random rotations and does not require the
dataset itself to be Gaussian. The observation here is not a violation of a
distribution assumption: a dataset-adaptive coordinate quantizer can exploit
structure that a generic worst-case construction does not exploit.

## Metric and paper-scope distinctions

The paper's near-neighbor section reports hit1@k: whether the single exact best
neighbor appears in the approximate top k. Our earlier recall@10 measures the
overlap of all ten IDs. Under decoded cosine on the full gallery:

| Bits | Quantizer | Overlap recall@10 | Paper-style hit1@10 | Same-label precision@10 |
|---:|---|---:|---:|---:|
| 2 | Uniform | 44.99% | 74.10% | 96.88% |
| 2 | TurboQuant-MSE | 50.20% | 82.80% | 97.14% |
| 4 | Uniform | 86.78% | 100.00% | 97.23% |
| 4 | TurboQuant-MSE | 81.64% | 99.77% | 97.25% |

These metrics answer different questions. Same-label results do not establish
exact-neighbor equivalence, and a change of metric must not hide the remaining
four-bit recall gap.

The current method is the paper's MSE variant, not its QJL-corrected unbiased
inner-product variant. This diagnosis does not measure a QJL variant or prove
it would win. The paper's retrieval study also uses different embedding
dimensions (200/1536/3072) and PQ/RabitQ baselines rather than this 64D per-row
affine-uniform baseline. It does not promise superiority in every such setup.

## Practical consequence and remaining work

Core packing/codebook/rotation checks provide evidence against an implementation
defect being the main cause. The tested scoring choice demonstrably amplifies
the gap, while coordinate structure demonstrably changes the comparison.
TurboQuant is competitive at two bits in the corrected quality experiment;
uniform remains better at four bits on the original ReLU features.

The subsequent native cosine implementation exposes scorer choice explicitly,
retains raw IP, and measures the added work in the
[native comparison](native_cosine_summary.md). Independent-reference quality
improvements alone are not a systems performance contribution. More encoder
seeds, coordinate transforms and the planned Fashion-MNIST evaluation remain
necessary before broad claims.

## Reproduce

```bash
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/diagnose_quantization.py
```

The default input is the frozen full-training feature archive generated by the
[scale-study reproduction commands](../docs/quantization_scale_reproduce.md).
The script writes [all diagnostic records](quantization_diagnosis.json), including
individual rotations, norm variation, zero errors and all three quality metrics.
All 43 project checks passed, including the independent scoring/tie/quadrature
checks. The default quantizer and retrieval implementations were not modified.
