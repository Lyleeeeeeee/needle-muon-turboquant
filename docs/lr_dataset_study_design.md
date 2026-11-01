# Learning-rate and dataset study

The user approved the proposed three-candidate learning-rate comparison for
AdamW and Muon, followed by Fashion-MNIST. Preserve the existing MNIST pilot.
This file fixes choices before the new outcomes are inspected.

## Controls and selection

Use the existing 784-128-64-10 encoder, native float32 CPU, batch 64, ten epochs,
weight decay 0.01, and the fixed split 17 of 55,000 training/5,000 validation
examples. Keep Muon on hidden.weight; auxiliary AdamW handles all other weights
and biases. No architecture, optimizer formula, schedule, or quantizer changes.

Each method has exactly three candidate configurations:

| Candidate | AdamW learning rate | Muon matrix rate | Muon auxiliary AdamW rate |
|---|---:|---:|---:|
| low | 0.001 | 0.007 | 0.001 |
| medium | 0.003 | 0.020 | 0.003 |
| high | 0.010 | 0.060 | 0.010 |

The Muon group has two rates. These are three predefined *paired* configurations,
not an independent two-dimensional search. Moving both rates prevents 92.5% of
parameters from being frozen at one auxiliary rate during tuning. Search budgets
match by configurations/updates, not elapsed time. Decay/momentum are not tuned.

On each dataset separately, use seed 7 to choose the candidate with minimum
validation loss across its ten checkpoints. Break exact ties by candidate name.
Freeze selected rates, then run fresh paired seeds 8/9/10, excluded from tuning.
For each repeat select its minimum-validation-loss checkpoint. Report every
candidate, all per-epoch values, repeat mean/sample standard deviation, and paired
differences. Do not retune after viewing repeat/test outcomes or extend the grid
to make one method win. A boundary winner means the search is limited.

Run MNIST search and repeats first, then Fashion-MNIST search and repeats:24
complete ten-epoch trainings. Alternate optimizer order between candidate/seed
pairs; retain timings sequentially in one process with BLAS thread limits 1.
Verify initialization, data/split/order hashes across paired runs. Checkpoint
reload must reproduce recorded validation loss/accuracy.

## Data and final evaluation

Use the original local MNIST files and checksum-verified official
[Fashion-MNIST IDX files](https://github.com/zalandoresearch/fashion-mnist#Get-the-Data).
The original HW loader is format-compatible; extend trainer identification so
Fashion is never recorded as MNIST. Default MNIST behavior stays unchanged.
Record file SHA256 hashes. Download Fashion training files first; acquire and
read its official test files only after all Fashion choices/checkpoints are fixed.
Evaluate all six selected Fashion repeat checkpoints once on the same 10,000 test
images, with no test-based selection. The old MNIST test has previously been
viewed, so MNIST repeat results remain validation evidence, not an untouched test.

Numeric optimizer-state size excludes transient workspace; CPU timing is not a
GPU speed claim. Classification results are distinct from retrieval label
precision and compressed-neighbor fidelity. This task does not rerun quantization.

## Deliverables and verification

Test dataset identification/export provenance, loss-based deterministic candidate
selection, and rejection of mismatched paired records before implementation.
Pass the complete suite before retained timings. Preserve data/checkpoints as
ignored artifacts; publish source, raw JSON, frozen selections, English summary
and scientific plots to the existing private repository.
