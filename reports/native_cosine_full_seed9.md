# Preliminary quantization comparison

Source: MNIST_frozen_encoder_6b3e0e6111e9. Gallery: (60000, 64); queries: (1000, 64); seed: 9; k: 10; metric: cosine.

| Method | Buffer bytes | Compression | Relative MSE | Recall@k | Median query ms |
|---|---:|---:|---:|---:|---:|
| fp32 | 15360000 | 1.00x | 0.000000 | 1.0000 | 3444.742 |
| uniform_2 | 1440000 | 10.67x | 0.039480 | 0.4495 | 3374.985 |
| uniform_4 | 2400000 | 6.40x | 0.001592 | 0.8678 | 3327.799 |
| turbo_2 | 1216400 | 12.63x | 0.109981 | 0.5053 | 3327.922 |
| turbo_4 | 2176448 | 7.06x | 0.008569 | 0.8169 | 3347.546 |

Single-seed component check, not a full paper reproduction or optimizer comparison.
Storage includes shared rotation/codebook and per-row metadata. Query timing includes normalization for cosine; excludes feature generation and build.
Process RSS is a whole-worker high-water mark, not exact query-only peak memory. Decoded-block counters exclude normalization/matmul/top-k workspace.
