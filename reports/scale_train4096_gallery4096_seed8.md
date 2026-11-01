# Preliminary quantization comparison

Source: MNIST_frozen_encoder_f6d296bf6177. Gallery: (4096, 64); queries: (1000, 64); seed: 8; k: 10.

| Method | Buffer bytes | Compression | Relative MSE | Recall@k | Median query ms |
|---|---:|---:|---:|---:|---:|
| fp32 | 1048576 | 1.00x | 0.000000 | 1.0000 | 226.156 |
| uniform_2 | 98304 | 10.67x | 0.049554 | 0.2855 | 227.200 |
| uniform_4 | 163840 | 6.40x | 0.001970 | 0.7008 | 225.438 |
| turbo_2 | 98320 | 10.66x | 0.124205 | 0.2525 | 226.704 |
| turbo_4 | 163904 | 6.40x | 0.010064 | 0.4758 | 227.882 |

Single-seed component check, not a full paper reproduction or optimizer comparison.
Storage includes shared rotation/codebook and per-row metadata. Query timing excludes feature generation and build.
Process RSS is a whole-worker high-water mark, not exact query-only peak memory. Decoded-block counters exclude matmul/top-k workspace.
