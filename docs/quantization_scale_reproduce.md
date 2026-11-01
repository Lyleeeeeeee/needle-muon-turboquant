# Reproduce the MNIST scale study

Run from the repository root after installing dependencies and building the CPU
backend. These commands expect MNIST's four original gzip files in `../hw0/data`.
Pass a different `--data-dir` to both apps when needed. Training and quantization
use native Needle CPU; checkpoint/feature I/O and evaluation bookkeeping use
NumPy as documented in the implementation audit.

## Train and freeze both encoders

```bash
for count in 4096 60000; do
  python apps/train_baseline.py --device cpu --train-limit "$count" --eval-limit 256 --epochs 3 --seed 7 --checkpoint-output "results/scale/encoder_train${count}.npz" --feature-output "results/scale/legacy_features_train${count}.npz" --output "reports/scale_encoder_train${count}.json"
  python apps/export_features.py --checkpoint "results/scale/encoder_train${count}.npz" --gallery-size 60000 --queries 1000 --output "results/scale/features_train${count}_gallery60000.npz"
done
```

Both exporters use the original 4,096-training-image and 256-query selections as
prefixes. This selection reference is deliberately the same for both models;
it does not change to 60,000 for the larger training run. Checkpoint SHA-256,
query classification accuracy and selection settings are written beside each
feature archive. Integer image IDs are stored inside the archive.

Create the corresponding nested small galleries without retraining or changing
the query images:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

folder = Path('results/scale')
for count in (4096, 60000):
    with np.load(folder / f'features_train{count}_gallery60000.npz', allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    for key in ('gallery', 'gallery_labels', 'gallery_ids'):
        arrays[key] = arrays[key][:4096]
    np.savez(folder / f'features_train{count}_gallery4096.npz', **arrays)
PY
```

## Run the complete comparison sequentially

```bash
for count in 4096 60000; do
  for gallery in 4096 60000; do
    for seed in 7 8 9; do
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python apps/benchmark_quantization.py --features "results/scale/features_train${count}_gallery${gallery}.npz" --seed "$seed" --repeats 3 --output "reports/scale_train${count}_gallery${gallery}_seed${seed}.json"
    done
  done
done
```

Each command evaluates all five methods in fresh sequential workers, with one
warm-up and three timed searches. Quality on the original 256-query prefix and
all 1,000 queries is recorded in each method's `query_prefix_quality` field.
Only compare timing for the same query count and gallery size.

Three rotations measure rotation sensitivity for one fixed trained encoder.
They are not three independent model-training runs. The three-epoch larger-data
run also has more optimizer updates (2,814 versus 192); the design and report
retain this limitation.
