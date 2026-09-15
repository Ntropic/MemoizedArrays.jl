# Benchmarks

Julia 1.12.7, AMD Ryzen AI 7 350; local measurements on 2026-09-14.
Medians in **ns/request**; lower is better. Four-thread columns report elapsed time divided by
all requests, not individual request latency. Short timings fluctuate; use these as indicative costs.

Workloads:

- **Read** → return an existing matrix reference into a preallocated output.
- **4-thread read** → the same workload, shared across four threads.
- **Generate** → cache a precomputed `4×4 Matrix{Float64}` on a miss, with capacity available.
- **Generate + evict** → the same miss after filling capacity 512. `—` means no eviction policy.

The generator returns the same matrix, allocated once before timing. Generation measures lookup,
claiming, bookkeeping, and publication; matrix allocation and filling are excluded. The direct-array
row measures assigning that same reference. Reported allocations belong to cache/task machinery.

Sparse coordinates are `(2mod(i,32), 2div(i,32))`, `i=1:512`: a small region with gaps of one grid
position. Eviction requests shift the first coordinate by 128. This replaces the previous widely
scattered workload. All sparse maps declare the same full signed coordinate domain.

**Array storage**

| Cache | Read | 4-thread read | Generate | Generate + evict |
|---|---:|---:|---:|---:|
| Direct array | 3.92 | 3.4 | 3.89 | — |
| JITArray | 9.38 | 5.72 | 101.82 | — |
| ExactLRUArray | 111.13 | 391.59 | 413.61 | 304.11 |
| SampledLRUArray | 14.7 | 37.21 | 401.02 | 874.05 |

**Dictionary storage**

| Cache | Read | 4-thread read | Generate | Generate + evict |
|---|---:|---:|---:|---:|
| JITArray | 26.91 | 49.47 | 384.88 | — |
| ExactLRUArray | 131.29 | 487.1 | 538.88 | 697.15 |
| SampledLRUArray | 28.54 | 103.16 | 600.57 | 1447.05 |

**MSB tree — block (4, 8)**

`block_shape=(4, 8)`.

| Cache | Read | 4-thread read | Generate | Generate + evict |
|---|---:|---:|---:|---:|
| JITArray | 69.54 | 92.83 | 288.08 | — |
| ExactLRUArray | 166.16 | 626.64 | 531.4 | 878.6 |
| SampledLRUArray | 70.25 | 117.52 | 623.47 | 1898.66 |

**MSB tree — block (8, 4)**

`block_shape=(8, 4)`.

| Cache | Read | 4-thread read | Generate | Generate + evict |
|---|---:|---:|---:|---:|
| JITArray | 64.45 | 26.7 | 168.86 | — |
| ExactLRUArray | 157.12 | 532.95 | 353.88 | 703.24 |
| SampledLRUArray | 72.74 | 70.28 | 385.85 | 1097.14 |

**LSB tree — block (4, 8)**

`block_shape=(4, 8)`.

| Cache | Read | 4-thread read | Generate | Generate + evict |
|---|---:|---:|---:|---:|
| JITArray | 46.4 | 34.61 | 437.29 | — |
| ExactLRUArray | 175.25 | 577.42 | 668.37 | 1110.08 |
| SampledLRUArray | 50.57 | 75.09 | 902.85 | 2255.58 |

**LSB tree — block (8, 4)**

`block_shape=(8, 4)`.

| Cache | Read | 4-thread read | Generate | Generate + evict |
|---|---:|---:|---:|---:|
| JITArray | 20.07 | 16.28 | 131.58 | — |
| ExactLRUArray | 120.01 | 506.82 | 357.69 | 365.01 |
| SampledLRUArray | 29.12 | 52.42 | 337.43 | 724.0 |

MSB means most significant bit; LSB means least significant bit. `block_shape` sets both the
child-reference block and cached-entry block dimensions—not the dimensions of the generated matrix.
See [storage](api/storage.md) for details.

**Reproduce**

From the workspace root:

```sh
julia --startup-file=no --compiled-modules=existing --project=JITArrays.jl --threads=4 JITArrays.jl/benchmark.jl --sparse
```

The script prints Markdown tables with medians, p10–p90 spreads, and allocations. It uses 15 interleaved
rounds; compilation, cache construction, prefilling, and validation are outside timing. Read samples make
32,768 requests across 512 entries; generation samples use 8 independent caches × 512 missing keys (4,096 requests). No CPU frequency or affinity
settings are changed. Waiting is reported separately per four-caller batch and includes the precomputed-value return,
notification, and scheduling after releasing the blocked generator.
