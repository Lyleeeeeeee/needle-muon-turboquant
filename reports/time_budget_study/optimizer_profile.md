# Optimizer cost audit

Fixed deterministic gradients, current deep parameter shapes, ten warmup updates, 5 alternating blocks of 100 steps. No forward/backward computation is included. These are separate diagnostic runs after primary training/search timings; the terminal diagnostic ran separately afterwards.

| Method | Uninstrumented median, ms/update | Instrumented, ms/update | Instrumentation change, % |
|---|---:|---:|---:|
| adamw | 3.147 | 3.203 | +1.78 |
| muon | 18.643 | 18.888 | +1.32 |

### Hybrid update phases

| Phase | ms/update |
|---|---:|
| muon | 16.585 |
| auxiliary_adamw | 2.302 |
| ns_transform | 16.332 |

The NS transform is inside the Muon phase; do not add these nested times.

### Matrix products and native layout copies

| Operation | Calls/update | ms/update |
|---|---:|---:|
| NDArray matmul, including packing and dispatch | 45 | 15.207 |
| Native compact | 157 | 6.656 |
| Native matmul_tiled | 45 | 7.796 |

All native kernels total 16.472 ms/update; remaining Python dispatch/allocation/timer overhead is approximately 2.416 ms/update. The remaining term is not an isolated allocator measurement. NDArray matmul is inclusive of native matrix kernels and their layout copies; do not sum nested rows.

Three hidden matrices × five NS iterations × three products  = 45 extra matrix products per update. The native tiled path repacks both inputs and unpacks its output for every product. The generic Compact kernel computes strided indices per element, so layout work can be substantial even for small matrices.

The NS matrix products alone add approximately 138.94 million multiply/add FLOPs per update. The five dense layers' forward and two backward products at batch 64 are approximately 54.51 million FLOPs; this estimate excludes elementwise/loss work and all AdamW operations. Muon overhead is therefore material for this small batch. The author's [runtime analysis](https://kellerjordan.github.io/posts/muon/#runtime-analysis) also makes overhead depend on matrix size, NS iterations and batch size; its large-token-batch GPU examples do not predict this float32 CPU implementation.

The profiler preserves original update outputs bit-for-bit and restores every wrapped method/native function. Independent optimizer numerical-reference tests remain in the full suite. No formulas, NS counts or kernels were changed. The cProfile tables in JSON are a separate call-cost audit with additional overhead.
