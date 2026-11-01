# Preliminary quantization comparison

Source: synthetic_sphere. Gallery: (4096, 64); queries: (128, 64); seed: 7; k: 10.

| Method | Buffer bytes | Compression | Relative MSE | Recall@k | Median query ms |
|---|---:|---:|---:|---:|---:|
| fp32 | 1048576 | 1.00x | 0.000000 | 1.0000 | 29.087 |
| uniform_2 | 98304 | 10.67x | 0.201737 | 0.5164 | 29.456 |
| uniform_4 | 163840 | 6.40x | 0.008018 | 0.8805 | 29.375 |
| turbo_2 | 98320 | 10.66x | 0.114434 | 0.5547 | 31.249 |
| turbo_4 | 163904 | 6.40x | 0.009193 | 0.8586 | 29.950 |

Single-seed component check, not a full paper reproduction or optimizer comparison.
Storage includes shared rotation/codebook and per-row metadata. Query timing excludes feature generation and build.
Process RSS is a whole-worker high-water mark, not exact query-only peak memory. Decoded-block counters exclude matmul/top-k workspace.
