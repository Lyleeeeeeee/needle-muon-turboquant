# Equal wall-clock training and Muon cost diagnosis

The user approved an equal-training-time AdamW/Muon retrieval comparison and
profiling of the current Muon implementation. This is a bounded follow-up to the
depth study, not a new optimizer, dataset, batch-size or kernel search.

## Frozen comparison

Use Fashion-MNIST, the same 55,000/5,000 split 17, batch 64 and architecture
784-128-128-128-64-10. All parameters train; hidden matrices use Muon and all
other parameters use auxiliary AdamW. Retain previously selected constant rates:
AdamW/auxiliary 0.001, Muon 0.007, weight decay 0.01, NS 5, momentum 0.95. No new
hyperparameter search: this compares the existing tuned configurations under a
new budget, not each optimizer's fully retuned equal-time optimum.

Each fresh run receives 220 seconds, seeds 8/9/10, with alternating optimizer
order. The wall-clock timer includes batch preparation, forward/backward,
updates, epoch permutation generation, intermediate validation and checkpoint
writing. It excludes imports, data loading, initial validation, final evaluation
and final serialization/export. Stop between updates; record the actual elapsed
time and overrun, bounded by the last synchronous update. No simultaneous
retained training, retrieval, tests or profiling jobs.

Offer both optimizers four checkpoint opportunities near 55/110/165/220 seconds.
Intermediate evaluations are inside the budget. The final snapshot is taken
after stopping; its evaluation/serialization is outside the timer for both
optimizers. Select minimum held-out validation loss, with earlier candidate on
ties. Save the terminal model as well. Record every evaluation, cumulative
updates/images, epoch-order hashes, paired initialization/split hashes, numeric
optimizer state and phase timings. Verify shared order prefixes, rather than
requiring identical complete order hashes when update counts differ.

Use a separate timed trainer, preserving all historical measured sources.
Before retained runs, test deadline stopping/overrun, charged validation cost,
checkpoint selection, invalid budgets and a synthetic IDX native-CPU roundtrip.
Reload all six selected checkpoints and reproduce full-validation metrics.

## Retrieval and analysis

Reuse the existing six-model pipeline unchanged: identical 55,000 gallery and
1,000 official-test query IDs, own-model 64D embeddings, native cosine top10,
block 256, FP32/uniform 2/4/TurboQuant-MSE 2/4, Turbo rotations 7/8/9. One warmup plus
three query timings per isolated worker. Average rotations within training seed
before paired three-seed means/sample deviations. Report same-category precision
as primary, own-FP32 neighbor recall separately, actual store bytes and query
time. These are previously explored test images, not an untouched final test.
Do not choose checkpoints or learning rates using retrieval outcomes.

## Slowdown diagnosis

First use uninstrumented retained train-loop/update timings. Then separately
profile original optimizers with fixed deterministic gradients and the current
parameter shapes. Warm up state, measure repeated blocks, alternate method
order. Attribute hybrid time to Muon and auxiliary AdamW, and instrument native
NDArray matrix products/compaction plus C++ dispatch with restored wrappers.
Record instrumentation overhead; do not substitute instrumented times for the
primary training comparison. Check NS performs 45 extra matrix products/update.
Use cProfile as a separate descriptive call-cost audit. Do not change formulas,
NS iteration count, parameter groups, data, or native kernels to improve results.

Write English raw records, provenance-checked summary and verified scientific
plots, then synchronize the existing private repository. Keep dataset/model/
feature/profile binary artifacts local and ignored. Conclusions apply to this
Needle float32 CPU implementation and these settings.

## Added terminal-model diagnostic

After inspecting the first selected-model retrieval pair, also evaluate all six
terminal220s models with FP32 retrieval. This is an explicitly posthoc diagnostic
of checkpoint-selection sensitivity, not a changed selection rule. Its frozen
protocol records which selected-model results already existed. Use all three
seeds, identical image IDs/scorer and no retrieval-based model selection. No new
training, tuning or compressed terminal-model search is added.
