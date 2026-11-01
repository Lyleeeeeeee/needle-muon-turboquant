# Preliminary quantization comparison

Source: MNIST_encoder. Gallery: (4096, 64); queries: (256, 64); seed: 9; k: 10.

| Method | Buffer bytes | Compression | Relative MSE | Recall@k | Median query ms |
|---|---:|---:|---:|---:|---:|
| fp32 | 1048576 | 1.00x | 0.000000 | 1.0000 | 57.041 |
| uniform_2 | 98304 | 10.67x | 0.049554 | 0.2734 | 56.942 |
| uniform_4 | 163840 | 6.40x | 0.001970 | 0.6992 | 57.159 |
| turbo_2 | 98320 | 10.66x | 0.112308 | 0.2324 | 56.753 |
| turbo_4 | 163904 | 6.40x | 0.009245 | 0.4984 | 57.171 |

Single-seed component check, not a full paper reproduction or optimizer comparison.
Storage includes shared rotation/codebook and per-row metadata. Query timing excludes feature generation and build.
Process RSS is a whole-worker high-water mark, not exact query-only peak memory. Decoded-block counters exclude matmul/top-k workspace.
