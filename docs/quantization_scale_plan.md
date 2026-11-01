# Quantization Scale Implementation Plan

> Execute inline, task by task, using the existing planning/TDD workflow. The
> user has already authorized both scale experiments; do not pause for another
> execution-mode selection.

**Goal:** Separate gallery-size effects from larger-data encoder training.

**Architecture:** Preserve the existing trainer and add weight export. A separate
feature exporter loads frozen weights and uses common nested image selections.
Existing quantization workers consume the exported features unchanged.

**Tech Stack:** Python, Needle native CPU, NumPy NPZ I/O, pytest.

**Spec:** `docs/quantization_scale_design.md`.

## Global constraints

- English repository content; preserve old result files.
- Existing 784-128-64-10 model, AdamW 0.003, batch 64, three epochs, seed 7.
- Training sizes 4096/60000, galleries 4096/60000, 1000 common queries.
- Five existing methods, rotation seeds 7/8/9, block 256, three timings.
- Sequential retained benchmarks; no paid compute or algorithm changes.

## Task 1: Reproducible frozen feature export

Files: add `apps/encoder_checkpoint.py`, `apps/export_features.py`, and
`tests/test_scale_features.py`; modify `apps/train_baseline.py`.

Interfaces: `save_encoder(model, path)` stores six named float32 parameter arrays;
`load_encoder(path, device)` reconstructs the same architecture and parameters.
`retrieval_indices(train_count, query_count, reference_train, reference_queries,
gallery_size, query_size, seed)` returns common nested integer image IDs.

- [x] Write the checkpoint/export CLI test before production changes. Train 128
  images, save weights, independently export 256 gallery/96 query images, and
  require exact equality with the original 128 gallery/64 query feature prefixes.
- [x] Add an index-selection test for exact legacy selection, nesting, uniqueness
  and deterministic IDs. Run `python -m pytest tests/test_scale_features.py -q`
  and confirm missing checkpoint CLI/module failures.
- [x] Add named, pickle-free weight save/load, checking matrix/bias dimensions,
  float32 types and finite values before assigning to Needle parameters.
- [x] Add `--checkpoint-output` after training without changing any RNG calls.
  In the exporter, reconstruct the legacy selections with `default_rng(seed)`,
  append unused IDs, extract features in batches and record hashes/IDs/accuracy.
- [x] Run the targeted tests, then `python -m pytest -q` (40 passed).

## Task 2: Controlled experiment execution

Files: new `reports/scale_*` records and ignored `results/scale/` weight/features.

- [x] Reproduce the 4096-example run with the original CLI settings and save
  weights. Require `np.array_equal` with `results/mnist_features.npz`.
- [x] Train the 60000-example model using the same architecture/hyperparameters.
- [x] Export both models on the identical 60000-row gallery and 1000 queries;
  create the nested 4096-row gallery views from those saved matrices.
- [x] Run `apps/benchmark_quantization.py --features <archive> --seed <7|8|9>
  --repeats 3 --output <report>` for all four model/gallery conditions.
- [x] Use the exact first 256 queries for a quality-only subset comparison with
  all 1000; do not compare timing across differently sized query batches.

## Task 3: Report and publish

- [x] Record neighbor recall, label precision, reconstruction MSE, persistent
  numeric bytes, batch timing and uncertainty across rotations. Preserve limits
  from the spec, including extra updates and one training seed.
- [x] Update README/summary links, inspect the diff and publish only project files
  to the existing private repository. Verify remote visibility and changed files.
