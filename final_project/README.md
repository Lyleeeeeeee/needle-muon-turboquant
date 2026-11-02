# Muon Training and TurboQuant Retrieval in Needle

This folder is the standalone project entry point. It contains the native
Needle implementation, the CIFAR-100 experiment, two result tables, compact
supporting evidence and correctness tests. Earlier exploratory experiments
remain in the parent repository for provenance.

## Contributions

1. **A complete native Needle training-to-retrieval pipeline.** We integrated
   the completed HW1–HW3 framework, trained a small patch-based image encoder
   with AdamW and hybrid Muon, and compressed each model's own image features.
   Model execution, autodiff, optimizer matrix operations, decoding and scoring
   use Needle/native C++. NumPy/SciPy handle data, quantizer setup, top-k merging,
   files and analysis. PyTorch is not a training or inference dependency.
2. **Two CPU data-movement improvements.** We optimized Compact's index
   calculation, and reduced repeated matrix packing: tiled intermediates stay
   packed throughout Muon's Newton–Schulz loop, while shared matrix multiplication
   packs directly from strided views for both optimizers. The float32 optimizer
   formulas and accumulation order are preserved.
3. **Native packed vector compression and stronger baselines.** We implemented
   TurboQuant-MSE, affine uniform, metadata-compact uniform and symmetric uniform
   at two and four bits. Actual byte buffers are measured, and cosine scoring
   normalizes decoded vectors again. Queries remain FP32.
4. **Controlled paired measurements.** Three paired training seeds share
   initialization, data order, augmentation, model and validation images. All
   Muon work is charged to training time. All six final models contribute every
   gallery format, including three rotation seeds for TurboQuant.

Muon, AdamW and TurboQuant are credited reproductions/adaptations. Our contribution
is their Needle implementation, CPU systems work, integration and evaluation.

## Data, model and task

We use the official [CIFAR-100 binary dataset](https://cave.cs.toronto.edu/kriz/cifar.html):
32×32 RGB images and 100 fine-grained categories. From its 50,000 training images,
a fixed stratified split takes 100 images per class for training (10,000 total)
and 20 per class for validation (2,000). Split seed is 17. The official test set
is unused. Normalization statistics come only from the training subset; training
uses horizontal flips with probability 0.5.

The gallery is the 10,000 training images. Queries are 10 held-out images per
class, giving 1,000 queries with no gallery/query overlap. The encoder is trained
with 100-class cross-entropy; retrieval uses its 64-dimensional feature vector.
Our main retrieval metric is

$$
P@10 = \frac{1}{10Q}\sum_{q=1}^{Q}\sum_{j\in\operatorname{Top10}(q)}
\mathbf{1}[y_j=y_q].
$$

This is the fraction of returned images sharing the query's category, rather
than classifier accuracy. Own-FP32 neighbor recall is a separate diagnostic.

The small MLP-Mixer-inspired encoder divides an image into sixteen 8×8 RGB
patches, projects each 192-value patch to width 64, applies two residual
token/channel MLP blocks, then LayerNorm and average pooling. Token MLP width
is 32; channel MLP width is 128. It uses ReLU and existing matrix operators,
with no convolution operator. Signed pooled features feed a 100-class head.
See [PatchMixer and checkpoints](apps/cifar_model.py),
[binary reader and patch layout](experiments/cifar_data.py), and
[experiment configuration](config.json).

## Results: training efficiency

Both tables follow the same six training runs: AdamW and Muon at paired seeds
9/10/11, batch 256, each receiving 600 seconds of pure training updates.
Learning rates were selected earlier using equal three-candidate, five-epoch
validation searches at seed 7, then locked: AdamW 0.003; Muon 0.021 with auxiliary
AdamW 0.003. Weight decay is 0.01. The configuration records that selection.

The training table measures the first **observed** target crossing. Snapshots
are saved at initialization, every 20 updates through 400, then every 40, plus
the actual terminal update. Data-batch preparation, augmentation, forward,
backward and the full optimizer/Newton–Schulz update are timed. Initial setup,
checkpoint I/O and offline evaluation are excluded. The last indivisible update
can slightly exceed 600 seconds; its full time is charged. Training workers run
sequentially before any offline quality evaluation.

<!-- TRAINING:START -->
| Target P@10 | AdamW updates | AdamW training s | Muon updates | Muon training s |
|---|---|---|---|---|
| 8% | 220 ± 35 | 26.92 ± 4.29 | 127 ± 23 | 15.73 ± 2.92 |
| 10% | 433 ± 76 | 52.77 ± 9.49 | 207 ± 12 | 25.67 ± 1.33 |
| 12% | 1040 ± 144 | 127.89 ± 18.15 | 373 ± 12 | 46.04 ± 1.46 |
| 14% | 0/3 reached | Not observed within 600 s | 0/3 reached | Not observed within 600 s |
<!-- TRAINING:END -->

Values summarize all three seeds; a mean crossing time is shown only if all
three reach the target. Non-attainment remains explicit. First crossing can
later revert; two-consecutive-observation outcomes and full lightweight curves
are retained in [evidence](report/evidence.json).

The displayed 8/12/14% goals were added after inspecting the saved curves and
are **post hoc supplementary analyses**. The original predeclared goals were
10/15/20/30%: both methods reach 10% in 3/3 runs; neither reaches 15/20/30% in
any run. Budgets were not extended and rates were not retuned after results.
These measurements use the same validation cohort as tuning, and characterize
this small native CPU setup rather than untouched-test generalization.

## Results: compression of the final models

This table uses the **actual 600-second final checkpoint** from each of those
six trajectories. Each final model's own features feed FP32 and all quantizers.
It does not reuse the older one-seed, 20-epoch quantization pilot.

<!-- QUANTIZATION:START -->
| Gallery format | AdamW P@10 (%) | Muon P@10 (%) | Gallery MB | Compression |
|---|---|---|---|---|
| FP32 | 11.88 ± 0.64 | 12.51 ± 0.37 | 2.5600 | 1.00× |
| Affine uniform 2-bit | 11.24 ± 0.48 | 11.62 ± 0.34 | 0.2400 | 10.67× |
| Symmetric uniform 2-bit | 11.14 ± 0.56 | 11.39 ± 0.25 | 0.2000 | 12.80× |
| TurboQuant-MSE 2-bit | 11.31 ± 0.53 | 11.94 ± 0.42 | 0.2164 | 11.83× |
| Affine uniform 4-bit | 11.90 ± 0.56 | 12.42 ± 0.44 | 0.4000 | 6.40× |
| Symmetric uniform 4-bit | 11.82 ± 0.59 | 12.47 ± 0.29 | 0.3600 | 7.11× |
| TurboQuant-MSE 4-bit | 11.87 ± 0.59 | 12.44 ± 0.38 | 0.3764 | 6.80× |
<!-- QUANTIZATION:END -->

P@10 is averaged over three training seeds, with sample standard deviation.
TurboQuant rotations 7/8/9 are averaged within each model before averaging
models. Gallery MB counts persistent numerical buffers, including packed codes
and metadata, using decimal MB. Images, labels, model weights, queries and
temporary decoding buffers are excluded. Compact-affine controls have identical
decoded values and neighbors to affine uniform, so duplicate rows are omitted;
their trials remain in the evidence. The symmetric control remains visible.
Smaller storage alone does not establish faster queries.

At two bits, TurboQuant retains 11.31%/11.94% mean P@10 for AdamW/Muon, using
216,400 bytes (11.83× smaller than FP32). Affine uniform uses 240,000 bytes and
retains 11.24%/11.62%; symmetric uniform uses only 200,001 bytes and retains
11.14%/11.39%. TurboQuant thus trades some storage for higher mean quality
against the symmetric control. Four-bit outcomes are mixed. Mean TurboQuant
2-bit losses against each encoder's own FP32 are almost identical (about 0.57
percentage points), so these data do not establish a positive optimizer/compression
interaction. They show the two implementations working together in one pipeline.

## AdamW and Muon: equations and implementation

For a parameter matrix or vector $W$, gradient $g_t$, learning rate $\eta$ and
weight decay $\lambda$, AdamW computes

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t},\quad
W_{t+1}=(1-\eta\lambda)W_t-
\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

We use $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.
The comparison baseline is **AdamW**, with decoupled decay, rather than the
homework Adam's coupled L2 regularization. Code: `AdamW.step` in
[optim.py](python/needle/optim.py).

Muon keeps a momentum buffer and forms its Nesterov update

$$
B_t=\mu B_{t-1}+(1-\mu)g_t,\qquad
G_t=(1-\mu)g_t+\mu B_t,\qquad \mu=0.95.
$$

Orient $G_t$ so rows do not exceed columns, and initialize
$X_0=G_t/(\|G_t\|_F+10^{-7})$. Five Newton–Schulz iterations approximate
an orthogonalized update:

$$
A_k=X_kX_k^T,\qquad
X_{k+1}=3.4445X_k+(-4.7750A_k+2.0315A_k^2)X_k.
$$

Transpose back when needed. For Needle's weight layout `(input, output)`, scale
the result by $s=\sqrt{\max(1,\mathrm{output}/\mathrm{input})}$ and update
$W_{t+1}=(1-\eta\lambda)W_t-\eta sX_5$. The finite polynomial approximation
is not an exact SVD or exact orthogonal projection.

Code: `Muon._transform`, `Muon.step` and `MuonWithAuxAdamW` in
[optim.py](python/needle/optim.py); `optimizer_for` and `TrainingState` in
[workflow.py](experiments/workflow.py). The eight hidden token/channel matrices
use Muon: 34,816 of 54,788 parameters (63.55%). Input projection, classifier,
normalization and biases use auxiliary AdamW. **Every parameter is trained.**
All arithmetic here is float32; the author reference uses bfloat16 in its NS core.

## Quantizers: equations and implementation

One bit stores a binary digit. A $b$-bit code selects one of $L=2^b$ levels:
four levels at two bits, sixteen at four bits. Four 2-bit codes, or two 4-bit
codes, share one uint8 byte. See native `PackCodes`, `QuantizePack` and decoders
in [quantization_cpu.h](src/quantization_cpu.h).

**Affine uniform.** For row $x_i$, let $a_i=\min_jx_{ij}$ and
$s_i=(\max_jx_{ij}-a_i)/(L-1)$. Then

$$
c_{ij}=\operatorname{clip}\left(\operatorname{round}
\frac{x_{ij}-a_i}{s_i},0,L-1\right),\qquad
\hat x_{ij}=a_i+s_ic_{ij}.
$$

Zero-scale rows are handled explicitly. Codes, float32 minima and float32
scales are retained. Uniform encoding computes metadata and integer indices
with NumPy; code packing and decoding use native C++. Code: `UniformQuantizer` and `UniformVectorStore` in
[quantization.py](python/needle/quantization.py).

**Compact affine and symmetric uniform.** Compact affine shares a row minimum
when every minimum is identical; otherwise it retains the original minima.
Its decoded values are unchanged. Symmetric uniform sets
$r_i=\max_j|x_{ij}|$, $s_i=2r_i/(L-1)$ and $a_i=-s_i(L-1)/2$.
The same rounding/decoding equation applies, but minima are derived from scales
rather than stored per row. A one-byte format flag is counted. Code:
`CompactUniformQuantizer.encode` and `CompactUniformStore` in
[uniform_compact.py](experiments/uniform_compact.py).

**TurboQuant-MSE.** For a row vector $x$, store its norm $r=\|x\|_2$, and rotate
its unit direction with one shared random orthogonal matrix $R$:

$$
u=x/r,\qquad z=uR,\qquad
c_j=\arg\min_\ell|z_j-\alpha_\ell|,\qquad
\hat x=r(\alpha_{c_1},\ldots,\alpha_{c_d})R^T.
$$

Zero rows reconstruct as zero. Rotation is constructed by Gaussian QR with
sign correction. The nonuniform codebook is optimized by Lloyd–Max for the
coordinate density of a random point on the unit sphere,

$$
f_d(z)=\frac{\Gamma(d/2)}{\sqrt\pi\,\Gamma((d-1)/2)}
(1-z^2)^{(d-3)/2},\quad -1\le z\le1.
$$

Cell boundaries are adjacent-centroid midpoints. Each centroid is updated to
$\alpha_\ell=\int_{I_\ell}zf_d(z)dz/\int_{I_\ell}f_d(z)dz$ using exact Beta
cell masses and first moments. Code: `sphere_codebook`, `TurboQuantMSE.encode`
and `PackedVectorStore` in [quantization.py](python/needle/quantization.py).
We reproduce the MSE variant, Algorithm 1; QJL residual correction and LLM KV
cache inference are outside this project.

For $N$ vectors of dimension $d$, persistent bytes are $4Nd$ for FP32,
$N\lceil db/8\rceil+8N$ for affine, and
$N\lceil db/8\rceil+4N+1$ for symmetric uniform. TurboQuant uses
$N\lceil db/8\rceil+4N+4d^2+4L$, including shared rotation and codebook.

**Scoring after decoding.** Every format uses cosine similarity

$$
\operatorname{score}(q,\hat x)=
\frac{q^T\hat x}{\|q\|_2\|\hat x\|_2}.
$$

Decoded vectors are normalized again because quantization changes their norms.
Zero rows score zero. Blocks of 256 vectors are decoded and discarded; no full
decoded gallery is retained. TurboQuant can rotate the query and score in the
rotated space. Stable row IDs break ties. Code: `_VectorStore.search` and
`_unit_tensor_rows` in [quantization.py](python/needle/quantization.py).

## The two CPU improvements

**1. Compact indexing.** The original loop recomputed multi-dimensional
coordinates using division/remainder for every output element. We added direct
rank-two indexing and a coordinate-carry traversal for other ranks, preserving
the exact source offsets. Code: `Compact` in
[ndarray_backend_cpu.cc](src/ndarray_backend_cpu.cc).

**2. Reduce repeated packing.** Originally, each NS matrix product returned to
ordinary layout and repacked inputs for the next product. `NewtonSchulzTiled`
keeps $X$, Gram matrices, polynomial intermediates and products tiled throughout
all five iterations, packs/unpacks once, and refreshes the tiled transpose once
per iteration. Code: [muon_cpu.h](src/muon_cpu.h). Separately, shared `PackView`
reads strided/transposed inputs directly into tiles and `Strided` unpacks the
output once, benefiting forward/backward and both optimizers. Code:
[matmul_cpu.h](src/matmul_cpu.h) and the `NDArray.__matmul__` dispatch in
[ndarray.py](python/needle/backend_ndarray/ndarray.py).

Reference switches are `NEEDLE_CPU_MATMUL=reference` before import and
`use_tiled_ns=False` for Muon. Buffers are temporary within each call.
Earlier five-pair fixed-work checks on a separate Fashion encoder measured
complete-update median speedups of 1.435× for AdamW and 1.749× for Muon, with
bit-identical outputs. These are combined CPU implementation speedups on that
workload, not CIFAR optimizer-vs-optimizer speedups or isolated gains from each
individual change. Compact supporting measurements are retained in the evidence.

## Build, verify and reproduce

From this folder, using Python with NumPy, SciPy and a C++17 compiler:

```bash
python -m pip install -r requirements-dev.txt
python tools/build_cpu.py
python -m pytest -q
python tools/make_tables.py --check
```

Download and extract the official CIFAR-100 binary archive into
`data/cifar-100/`, giving `data/cifar-100/train.bin`. Official archive MD5 is
`03b5dce01913d631647c71ecec9e9cb8`. The expected training-file SHA256 is in
`config.json`. No data, checkpoints, feature matrices or compiled binaries are
committed. The CPU build can reuse installed pybind11 headers; model execution
does not require PyTorch.

```bash
# Six sequential 600s trainings, then offline evaluation/feature export.
python experiments/workflow.py --train-file data/cifar-100/train.bin --output-dir runs/cifar

# All nine gallery controls; three rotations for TurboQuant. Repeat for all six models.
python experiments/compare_galleries.py --features runs/cifar/seed9/final/adamw_features.npz --output runs/quant/adamw_seed9.json

# Regenerate the two committed tables from compact archived evidence.
python tools/make_tables.py
```

`--stage train` and `--stage evaluate --resume` separate acquisition and offline
evaluation; `--resume` checks frozen configuration/source/checkpoint identities.
Use a fresh output directory for a new run. Saved curves and per-method trial
JSONs allow analysis without rerunning training. Query benchmarks must run
separately from CPU builds, tests and other training/benchmarks.

The compact evidence records original run hashes, full lightweight training
curves, all quantization trials and original/packaged source manifests. Trailing
whitespace is cleaned in copied Python files; AST comparison excluding docstrings
verifies that this formatting leaves executable code unchanged. The tables
were measured with the original pinned runners before packaging; this folder
consolidates their CIFAR-only dependencies and preserves algorithm code.
Historical source and full archived evidence are linked by commit in the
evidence rather than copied wholesale into this folder.

## References and credit

- HW1–HW3 Needle foundation: recovered completed homework implementations plus
  the course backend; project additions are identified above.
- [AdamW: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
- [Muon author implementation, pinned reference](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py).
- [TurboQuant-MSE Algorithm 1](https://arxiv.org/html/2504.19874v1).
- [MLP-Mixer](https://arxiv.org/abs/2105.01601), inspiration for our small ReLU patch encoder.
