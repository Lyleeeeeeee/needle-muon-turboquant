# Muon: native Needle pilot

Implement the optimizer workstream already specified in the project plan. The
user requests moving to Muon and checking whether the previous three-epoch
training budget was too short. Prior quantization records remain unchanged.

## Algorithm and adaptations

Pin [the author source](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py).
Use EMA momentum 0.95, the Nesterov blend of the gradient and updated momentum,
five quintic Newton-Schulz steps with coefficients (3.4445,-4.7750,2.0315), and
Frobenius normalization with epsilon 1e-7. The iteration approximates a polar
transformation; it is not exact SVD orthogonalization.

Implement in float32 Needle operations, without PyTorch or NumPy optimizer
math. The author uses bfloat16; this is a disclosed precision adaptation.
Needle Linear stores weights as (in,out), transposed from PyTorch (out,in), so
apply the author's shape scaling in logical output/input order:
sqrt(max(1,out/in)). Missing gradients are skipped without decay/state changes,
consistently with existing Needle optimizers; this differs from the author's
forced-zero-gradient single-device behavior.

Muon updates only model.hidden.weight. Input/output weights and all biases use
the existing AdamW with identical auxiliary settings to the baseline. Parameter
groups must be disjoint and exhaustive. Keep states detached from autograd.
Count retained numerical optimizer-state buffers separately from parameters,
gradients, Python dictionaries and transient Newton-Schulz workspace.

## Training and evaluation

Preserve the old trainer's defaults and exact RNG sequence. Add explicit
optimizer selection and an optional train-split validation holdout. The pilot
uses 55,000 training / 5,000 validation images, fixed split seed 17, training seeds 7/8/9,
batch size 64, ten epochs, the existing 784-128-64-10 model, and native CPU. AdamW and
auxiliary AdamW use learning rate 0.003 and decay 0.01. Muon uses learning rate 0.02,
momentum 0.95, five NS steps and decay 0.01. These are predefined pilot settings,
not equal-budget tuned evidence that one optimizer is universally better.

Record initialization, split and training-order hashes to verify paired runs.
Record every epoch's training/validation loss, validation accuracy, total
training-loop time, optimizer-update time, update count and state bytes.
Compare epoch3 versus10 and time to validation thresholds 95/97/98 percent.
Save final weights and the minimum-validation-loss checkpoint. The official
test set does not select hyperparameters/checkpoints in this pilot; it was
previously viewed in exploratory work and is not an untouched final test.

Export common seed 7 retrieval features from selected checkpoints for downstream
inspection; do not mix different encoders into the existing fixed-feature
quantizer comparison. The formal Fashion-MNIST and joint study remain separate.

## Verification

Before implementation, add failing tests for rectangular and square updates,
Nesterov on/off, native/reference devices, zero/absent gradients, detached state,
layout scaling, numeric state bytes, invalid groups and CLI integration. Use an
independent float64 SVD/singular-value-polynomial oracle, not a copy of the
matrix iteration. Verify legacy sampling and feature reproduction before
retained runs. Run timing-sensitive comparisons sequentially.
