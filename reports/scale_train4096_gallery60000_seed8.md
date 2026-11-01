# Preliminary quantization comparison

Source: MNIST_frozen_encoder_f6d296bf6177. Gallery: (60000, 64); queries: (1000, 64); seed: 8; k: 10.

| Method | Buffer bytes | Compression | Relative MSE | Recall@k | Median query ms |
|---|---:|---:|---:|---:|---:|
| fp32 | 15360000 | 1.00x | 0.000000 | 1.0000 | 3292.661 |
| uniform_2 | 1440000 | 10.67x | 0.050053 | 0.0841 | 3302.936 |
| uniform_4 | 2400000 | 6.40x | 0.001988 | 0.4489 | 3298.419 |
| turbo_2 | 1216400 | 12.63x | 0.124057 | 0.0671 | 3293.000 |
| turbo_4 | 2176448 | 7.06x | 0.009927 | 0.2131 | 3325.301 |

Single-seed component check, not a full paper reproduction or optimizer comparison.
Storage includes shared rotation/codebook and per-row metadata. Query timing excludes feature generation and build.
Process RSS is a whole-worker high-water mark, not exact query-only peak memory. Decoded-block counters exclude matmul/top-k workspace.
