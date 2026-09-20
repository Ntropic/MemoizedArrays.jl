# Benchmarks

Current implementation, Julia 1.13.0, AMD Ryzen AI 7 350, 2026-09-20.
Tables report median time per request for the workloads below.

## Run

Run all maps and cache policies, followed by the matrix-operation benchmarks:

```sh
julia --startup-file=no --compiled-modules=existing --project=. benchmark.jl
```

Select maps with `--maps=array,dict,radixtree,subtree` and cache policies with
`--caches=jit,exact,sieve,batch_reinsert,batch_age`. For example, `--maps=array --caches=jit,exact`
runs only those cache/map combinations, followed by matrix operations.

## Cache workload

Fifteen rotated rounds, eight containers per sample, and 512 distinct resident keys.
The generator returns an existing 4×4 matrix. Each timed request consumes its first
element in a scalar checksum; no output array is written.

- **Read:** retrieve an existing value from a populated cache, with 32,768 requests per container.
- **Initial generation / insertion:** populate an initially empty container with 512 values.
  This includes entry allocation and any map growth or tree-node creation.
- **Generation / insertion (central leaf):** prefill one central-leaf value and 448 scattered
  values, then generate the remaining 63 central-leaf values. The leaf already exists and
  capacity is sufficient: no branch/leaf creation or eviction is included. Verify outside
  timing that all target values were missing and the existing central leaf remains attached.
  Average eight fresh batches of eight containers per round: 4,032 timed insertions, comparable
  to the 4,096 in initial population. Report the median of these averages over the 15 rotated rounds.
- **Generate + evict / replace:** prefill 512 values, then request 512 new keys.
  Bounded caches choose eviction victims; raw maps replace specified existing keys.
- **Generate + evict / replace (central leaf):** fill the cache with 32 central-leaf
  entries first, then 480 entries outside that leaf. Request the other 32 central-leaf
  coordinates, moving x by 4. Every request generates; raw maps delete the
  corresponding old central entry. Verify outside timing that the original central
  entries were evicted, the new ones are present, the outside entries remain, and the
  same central leaf is retained. Average 16 fresh batches of eight containers per round
  (4,096 replacements). Victim-selection scans remain included in timing. `:batch_reinsert` and
  `:batch_age` use `batch_size=32` in this column: the first request evicts the 32 old values,
  and the remaining requests refill those freed entries. This keeps every victim central.

Construction, compilation, prefill, validation and round-boundary GC are outside timing.
GC occurring during an operation remains included. Bounded caches preallocate entries
during setup; JITArray retains all generated values and has no eviction measurement.
Raw Array, Dict and tree rows store the supplied matrix directly; cache rows call the
generator on a missing value.

Batch policies use their default `batch_size=64` except in the central-leaf eviction
column described above. `:batch_age` uses its default retention window. Batch timings include
complete cleanup bursts and are amortized per request; individual cleanup requests cost more.

ArrayMap uses 4,096 slots and integer keys. Dictionary and tree cases use two-dimensional
coordinate keys and full signed-Int axes. Dictionary resident coordinates are nearby and
nonnegative. Both trees use `block_shape=(8, 8)` and `center=0.5`.
Initial tree generation and ordinary replacement workloads use 16 values in each of 32 neighboring leaves.
Ordinary replacement
moves keys by 128 along x into a different region. Central-leaf replacement uses nearby
coordinates outside the central leaf to fill the remaining capacity.

### Tree read sampling

Both read columns use the same populated tree: all 64 keys in the central leaf, plus
448 distinct keys sampled uniformly across a 4,096 × 4,096 coordinate region outside that leaf.
Sampling uses a fixed seed. There are no generation or eviction requests in either read column.

- **Read (spread out):** requests cycle evenly through the 448 scattered keys, then are shuffled.
- **Read (central leaf):** each of the 64 central-leaf keys receives 512 reads, shuffled in advance.
  Every request selects the level-zero shortcut in SubtreeRadixMap.

![Two sampling patterns: evenly spread requests on the left, requests only in the central leaf on the right](assets/tree-sampling.svg)

Schematic distributions: evenly spread requests (left) and requests confined to one central leaf
(right). Squares represent leaves; green highlights the central leaf. Equal-size dots represent requests. Both tree maps receive
identical benchmark sequences. Regenerate the SVG diagrams with `python docs/plot_tree_sampling.py`
(requires Matplotlib).

### ArrayMap

| Implementation | Read | Initial generation / insertion | Generate + evict |
|---|---:|---:|---:|
| Array | 3.6 ns | 3.8 ns | — |
| JITArray | 4.1 ns | 22.5 ns | — |
| ExactLRUArray | 10.8 ns | 12.1 ns | 15.2 ns |
| SampledLRUArray (`:sieve`) | 4.9 ns | 15.0 ns | 24.8 ns |
| SampledLRUArray (`:batch_reinsert`) | 4.0 ns | 14.0 ns | 22.7 ns |
| SampledLRUArray (`:batch_age`) | 5.0 ns | 14.3 ns | 20.2 ns |

### Dictionary (`:dict`)

Standalone Julia `Dict` versus DictMap-backed caches.

| Implementation | Read | Initial generation / insertion | Generate + evict / replace |
|---|---:|---:|---:|
| Dict | 12.0 ns | 87.7 ns | 130.3 ns |
| JITArray | 18.3 ns | 132.1 ns | — |
| ExactLRUArray | 29.3 ns | 124.2 ns | 167.7 ns |
| SampledLRUArray (`:sieve`) | 20.4 ns | 124.9 ns | 172.4 ns |
| SampledLRUArray (`:batch_reinsert`) | 20.3 ns | 113.7 ns | 173.8 ns |
| SampledLRUArray (`:batch_age`) | 21.0 ns | 116.4 ns | 163.3 ns |

### RadixMap (`:radixtree`)

| Implementation | Read (spread out) | Read (central leaf) | Initial generation / insertion | Generation / insertion (central leaf) | Generate + evict / replace | Generate + evict / replace (central leaf) |
|---|---:|---:|---:|---:|---:|---:|
| RadixMap | 35.6 ns | 20.8 ns | 54.2 ns | 33.5 ns | 66.0 ns | 43.5 ns |
| JITArray | 43.0 ns | 22.4 ns | 81.4 ns | 54.8 ns | — | — |
| ExactLRUArray | 67.7 ns | 38.1 ns | 72.7 ns | 57.6 ns | 110.1 ns | 70.3 ns |
| SampledLRUArray (`:sieve`) | 53.3 ns | 33.2 ns | 76.5 ns | 70.0 ns | 123.4 ns | 93.5 ns |
| SampledLRUArray (`:batch_reinsert`) | 55.6 ns | 28.8 ns | 68.2 ns | 54.4 ns | 91.4 ns | 75.1 ns |
| SampledLRUArray (`:batch_age`) | 52.1 ns | 29.8 ns | 65.6 ns | 59.0 ns | 92.5 ns | 81.2 ns |

### SubtreeRadixMap (`:subtree`)

| Implementation | Read (spread out) | Read (central leaf) | Initial generation / insertion | Generation / insertion (central leaf) | Generate + evict / replace | Generate + evict / replace (central leaf) |
|---|---:|---:|---:|---:|---:|---:|
| SubtreeRadixMap | 47.9 ns | 7.4 ns | 83.9 ns | 13.8 ns | 83.5 ns | 20.6 ns |
| JITArray | 57.1 ns | 9.6 ns | 115.3 ns | 37.7 ns | — | — |
| ExactLRUArray | 82.2 ns | 22.1 ns | 111.7 ns | 49.4 ns | 112.4 ns | 44.9 ns |
| SampledLRUArray (`:sieve`) | 69.9 ns | 14.0 ns | 114.4 ns | 67.2 ns | 116.8 ns | 51.0 ns |
| SampledLRUArray (`:batch_reinsert`) | 63.2 ns | 15.0 ns | 96.1 ns | 51.8 ns | 118.0 ns | 56.6 ns |
| SampledLRUArray (`:batch_age`) | 63.4 ns | 14.8 ns | 94.8 ns | 53.7 ns | 118.8 ns | 54.2 ns |

### Matrix operations without output allocations

Run these independently, with a maximum matrix dimension (default 16):

```sh
julia --startup-file=no --compiled-modules=existing --project=. -e 'include("benchmark.jl"); benchmarkmatrices()'
julia --startup-file=no --compiled-modules=existing --project=. -e 'include("benchmark.jl"); benchmarkmatrices(64)'
```

Sizes are powers of two from 2 through the specified maximum. Inputs are ordinary
`Matrix{Float64}`; OpenBLAS uses one thread. Addition uses `C .= A .+ B` and
multiplication uses `mul!(C, A, B)`, writing into the supplied output.

`eigh` calls LAPACK `DSYEVR`. Eigenvalues, eigenvectors and working buffers are allocated
before timing; workspace sizes are queried once per size. The solver writes into
supplied outputs. Copying the input into scratch is included because LAPACK overwrites it.

Each sample cycles through 64 input pairs for 4,096 operations and consumes one output
element in a scalar checksum. Fifteen rounds rotate operation order. Input preparation,
compilation, buffer allocation and round-boundary GC are excluded. Numerical checks
and zero-allocation assertions pass for all three operations. Times are per operation.

| Size | Addition | Multiplication | eigh |
|---|---:|---:|---:|
| 2×2 | 68.0 ns | 90.9 ns | 1.0 µs |
| 4×4 | 94.0 ns | 153.0 ns | 10.5 µs |
| 8×8 | 156.3 ns | 231.0 ns | 34.3 µs |
| 16×16 | 310.8 ns | 795.6 ns | 95.6 µs |
