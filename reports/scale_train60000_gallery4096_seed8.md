# Preliminary quantization comparison

Source: MNIST_frozen_encoder_6b3e0e6111e9. Gallery: (4096, 64); queries: (1000, 64); seed: 8; k: 10.

| Method | Buffer bytes | Compression | Relative MSE | Recall@k | Median query ms |
|---|---:|---:|---:|---:|---:|
| fp32 | 1048576 | 1.00x | 0.000000 | 1.0000 | 228.722 |
| uniform_2 | 98304 | 10.67x | 0.039194 | 0.4034 | 225.967 |
| uniform_4 | 163840 | 6.40x | 0.001585 | 0.7966 | 223.888 |
| turbo_2 | 98320 | 10.66x | 0.122533 | 0.3857 | 230.956 |
| turbo_4 | 163904 | 6.40x | 0.012315 | 0.6393 | 228.299 |

Single-seed component check, not a full paper reproduction or optimizer comparison.
Storage includes shared rotation/codebook and per-row metadata. Query timing excludes feature generation and build.
Process RSS is a whole-worker high-water mark, not exact query-only peak memory. Decoded-block counters exclude matmul/top-k workspace.
