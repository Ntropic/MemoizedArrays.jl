# Benchmarks

## Run

ArrayMap with all cache variants, followed by matrix operations:

```sh
julia --startup-file=no --compiled-modules=existing --project=. benchmark.jl --maps=array
```

Select cache variants with `--caches=jit,exact,clock,sieve`. Other maps remain available
through `--maps`.

## ArrayMap

Median time per request, Julia 1.12.7, AMD Ryzen AI 7 350, 2026-09-17.

| Implementation | Read | Initial generation / insertion | Generate + evict |
|---|---:|---:|---:|
| Array | 3.6 ns | 3.7 ns | — |
| JITArray | 3.9 ns | 18.4 ns | — |
| ExactLRUArray | 10.1 ns | 11.7 ns | 13.9 ns |
| SampledLRUArray (`:clock`) | 4.4 ns | 8.5 ns | 14.8 ns |
| SampledLRUArray (`:sieve`) | 4.3 ns | 12.4 ns | 19.2 ns |

4096 slots, capacity 512, fifteen rotated rounds with eight containers per sample.
Reads use 32,768 requests/container; generation and eviction use 512. The generator
returns an existing 4×4 matrix. Each request consumes its first element in a scalar
checksum; no output array is written. Setup, prefill and round-boundary GC are excluded.
Bounded caches preallocate entries during setup. JITArray has no eviction.

### ArrayMap backing storage: Array → Memory

ArrayMap now stores slots in `Memory`, keeps shape and strides directly, and returns
`MemoryRef` slots. The former Array-backed implementation and the new implementation
were measured in the same process with rotated execution order. Shape, indexing,
generator and cache policies were unchanged. Values are **Array-backed → Memory-backed**.

| Cache | Read | Initial generation | Generate + evict |
|---|---:|---:|---:|
| JITArray | 3.6 ns → 3.9 ns | 20.2 ns → 18.4 ns | — |
| ExactLRUArray | 10.7 ns → 10.1 ns | 12.6 ns → 11.7 ns | 14.8 ns → 13.9 ns |
| SampledLRUArray (`:clock`) | 4.8 ns → 4.4 ns | 9.0 ns → 8.5 ns | 17.8 ns → 14.8 ns |
| SampledLRUArray (`:sieve`) | 4.9 ns → 4.3 ns | 13.4 ns → 12.4 ns | 19.6 ns → 19.2 ns |

Most operations improved modestly in this run; JITArray reads were 0.3 ns slower.
This compares complete backing implementations, including their slot references and
metadata layout, rather than isolating the buffer type alone.

## Dict

Standalone Julia `Dict` versus our `DictMap` inside each cache. Run only this comparison:

```sh
julia --startup-file=no --compiled-modules=existing --project=. -e 'include("benchmark.jl"); benchmark(maps=(:dict,))'
```

| Implementation | Read | Initial generation / insertion | Generate + evict / replace |
|---|---:|---:|---:|
| Dict | 23.6 ns | 102.3 ns | 258.8 ns |
| JITArray | 20.5 ns | 122.7 ns | — |
| ExactLRUArray | 31.4 ns | 109.0 ns | 209.3 ns |
| SampledLRUArray (:clock) | 22.9 ns | 100.6 ns | 164.7 ns |
| SampledLRUArray (:sieve) | 22.8 ns | 116.1 ns | 181.6 ns |

Median time per request, using the same generator, request counts and rotated rounds as above.
All implementations use `(Int, Int)` keys, with 512 nearby resident coordinates. DictMap-backed
caches use full signed axis ranges; bounded caches have capacity 512. `Dict` inserts supplied
values directly and replaces by deleting a specified key and inserting another. Caches call
the generator on misses and choose their own eviction victim.

The ArrayMap optimizations also apply here: preallocated LRU entries, victim reuse, concrete
ExactLRU links, direct generated-value returns and conversion only when needed. DictMap retains
the incoming hash and insertion position across generation, rechecking the position if nested
generation changes the table. Probe indices use masked signed arithmetic without redundant
checked conversions. DictMap still allocates a stable map record for each inserted key and
occasionally rebuilds its table; preallocating cache entries does not remove those costs.

## Radix trees

```sh
julia --startup-file=no --compiled-modules=existing --project=. -e 'include("benchmark.jl"); benchmark(maps=(:radixtree,:subtreetree))'
```

Median time per request, Julia 1.12.7, AMD Ryzen AI 7 350, 2026-09-17.
Both maps use the same nearby two-dimensional coordinates, request counts, generator,
and rotated rounds as the Dict comparison. Each table includes raw map operations
and all four cache variants. These results do not cover scattered or large coordinates.

### RadixMap (`:radixtree`)

| Implementation | Read | Initial generation / insertion | Generate + evict / replace | Delete |
|---|---:|---:|---:|---:|
| RadixMap | 24.5 ns | 152.7 ns | 132.4 ns | 40.4 ns |
| JITArray | 35.1 ns | 154.1 ns | — | — |
| ExactLRUArray | 45.9 ns | 142.3 ns | 157.4 ns | — |
| SampledLRUArray (:clock) | 38.6 ns | 117.2 ns | 152.7 ns | — |
| SampledLRUArray (:sieve) | 40.8 ns | 157.4 ns | 152.6 ns | — |

### SubtreeRadixMap (`:subtreetree`)

| Implementation | Read | Initial generation / insertion | Generate + evict / replace | Delete |
|---|---:|---:|---:|---:|
| SubtreeRadixMap | 31.1 ns | 101.9 ns | 154.8 ns | 60.7 ns |
| JITArray | 39.3 ns | 115.6 ns | — | — |
| ExactLRUArray | 53.4 ns | 112.4 ns | 189.5 ns | — |
| SampledLRUArray (:clock) | 40.3 ns | 99.2 ns | 171.9 ns | — |
| SampledLRUArray (:sieve) | 41.9 ns | 112.2 ns | 183.2 ns | — |

`SubtreeRadixMap` selects a depth shortcut into one shared radix tree. Structural
changes on the zero-prefix path refresh those pointers. Initial insertion was faster
in this run; reads, deletion and replacement were slower than `RadixMap`.

### Central-leaf and scattered routing experiment

```sh
julia --startup-file=no --compiled-modules=existing --project=. experiments/treepositions.jl
```

Both maps contain the same 512 values: all 64 coordinates in `(-4:3, -4:3)` and
448 scattered coordinates across the signed `Int` range. Zigzag encoding places the
central coordinates in one leaf. The subtree shortcut is checked to point directly
to that leaf; the distant residents keep the full tree populated above it.
Each workload uses 32,768 randomized ready reads and 31 rotated rounds. All measured
loops allocate zero bytes, consume a scalar checksum, and write no output array.
These measure raw maps, not cache policies. Julia 1.12.7, 2026-09-17.

| Implementation / workload | Full read | Encoding precomputed | Encoding + depth precomputed |
|---|---:|---:|---:|
| RadixMap / central leaf | 31.5 ns | 28.3 ns | 32.6 ns |
| SubtreeRadixMap / central leaf | 11.4 ns | 9.9 ns | 7.7 ns |
| RadixMap / scattered | 52.3 ns | 47.2 ns | 51.1 ns |
| SubtreeRadixMap / scattered | 72.0 ns | 61.7 ns | 54.8 ns |

Precomputed depth supplies the shortcut index; loading its pointer and traversing the
tree remain timed. Precomputation changes the request representation and generated
loop, so differences are not exact additive instruction costs.

| Arithmetic / workload | Encoding | Depth selection | Leaf position | Combined |
|---|---:|---:|---:|---:|
| RadixMap / central leaf | 0.8 ns | 0.0 ns | 2.4 ns | 2.4 ns |
| SubtreeRadixMap / central leaf | 0.8 ns | 1.2 ns | 2.4 ns | 5.5 ns |
| RadixMap / scattered | 0.8 ns | 0.0 ns | 2.4 ns | 2.4 ns |
| SubtreeRadixMap / scattered | 0.8 ns | 1.2 ns | 2.4 ns | 5.6 ns |

Arithmetic-only loops consume integer checksums and measure throughput, not isolated
instruction latency. RadixMap's constant root selection is optimized away. Combined
arithmetic includes encoding, root selection and the bottom-leaf position, not addresses
at intervening branch levels. These timings must not be added to predict read latency.

| Direct central-leaf access | Read |
|---|---:|
| Compute position from encoded coordinates | 3.5 ns |
| Position precomputed | 3.5 ns |

Compiled leaf-position code uses two masks and one address calculation. Depth selection
uses OR, leading-zero count, and constant multiply/shift arithmetic; it has no integer
division instruction. Both operate on machine words regardless of how many bits are set.

The central shortcut makes full reads about 2.8× faster than starting at the main root.
Leaf-position arithmetic is not the principal overhead: removing it does not improve
the direct-leaf baseline. Precomputing encoding and depth reduces central subtree reads
from 11.4 ns to 7.7 ns, but generic node/prefix handling and pointer loads remain above
the 3.5 ns direct baseline. Scattered reads benefit substantially from precomputation,
although the changed loop layout prevents assigning the entire difference to arithmetic.

### TreeHint removal: paired cache comparison

The previous `TreeHint` struct and current tuple representation were compared in the
same process, using an isolated source copy differing only in that representation.
Each pair uses the nearby-coordinate workload above, fifteen rotated rounds and eight
containers per sample. These are cache operations: raw map reads bypass the prepared
lookup path affected by this change. Values below are medians, Julia 1.12.7, 2026-09-17.

#### RadixMap

| Implementation | Read | Initial generation | Generate + evict |
|---|---:|---:|---:|
| JITArray / TreeHint | 35.6 ns | 97.2 ns | — |
| JITArray / tuple | 35.2 ns | 98.6 ns | — |
| ExactLRUArray / TreeHint | 46.1 ns | 100.1 ns | 150.6 ns |
| ExactLRUArray / tuple | 46.6 ns | 100.8 ns | 148.0 ns |
| SampledLRUArray (:clock) / TreeHint | 43.3 ns | 93.6 ns | 146.1 ns |
| SampledLRUArray (:clock) / tuple | 43.9 ns | 92.8 ns | 165.7 ns |
| SampledLRUArray (:sieve) / TreeHint | 42.6 ns | 100.2 ns | 169.0 ns |
| SampledLRUArray (:sieve) / tuple | 41.7 ns | 96.8 ns | 161.8 ns |

#### SubtreeRadixMap

| Implementation | Read | Initial generation | Generate + evict |
|---|---:|---:|---:|
| JITArray / TreeHint | 42.5 ns | 133.0 ns | — |
| JITArray / tuple | 44.5 ns | 119.0 ns | — |
| ExactLRUArray / TreeHint | 53.5 ns | 109.0 ns | 189.0 ns |
| ExactLRUArray / tuple | 53.5 ns | 107.9 ns | 188.0 ns |
| SampledLRUArray (:clock) / TreeHint | 41.4 ns | 101.0 ns | 174.7 ns |
| SampledLRUArray (:clock) / tuple | 40.2 ns | 98.9 ns | 174.4 ns |
| SampledLRUArray (:sieve) / TreeHint | 42.1 ns | 107.5 ns | 179.7 ns |
| SampledLRUArray (:sieve) / tuple | 42.3 ns | 104.6 ns | 184.3 ns |

A targeted repeat checked the two largest differences from the first run:

| Repeated operation | TreeHint | Tuple |
|---|---:|---:|
| RadixMap CLOCK generate + evict | 149.5 ns | 155.5 ns |
| SubtreeRadixMap JIT initial generation | 123.8 ns | 124.4 ns |

The apparent JIT generation improvement disappeared; the CLOCK replacement slowdown
shrank from 13% to 4%. Ready reads in both repeated cache/map pairs allocated zero bytes
with either representation. This is a source simplification, not a demonstrated
consistent performance improvement.

## Matrix operations with supplied outputs

Run these independently of the cache benchmarks, with a maximum matrix dimension (default 16):

```sh
julia --startup-file=no --compiled-modules=existing --project=. -e 'include("benchmark.jl"); benchmarkmatrices()'
julia --startup-file=no --compiled-modules=existing --project=. -e 'include("benchmark.jl"); benchmarkmatrices(64)'
```

Sizes are powers of two from 2 through the specified maximum. Inputs are ordinary
`Matrix{Float64}`; OpenBLAS is configured for one thread. Addition uses `C .= A .+ B` and
multiplication uses `mul!(C, A, B)`, writing directly into the prepared output.

`eigh` calls LAPACK `DSYEVR`, the solver used by Julia's default symmetric eigendecomposition.
Eigenvalues, eigenvectors, floating-point workspace, integer workspace and support buffers are
allocated before timing; workspace sizes are queried once per size. The solver writes directly
into the supplied outputs, without result copying. Each call copies the input into prepared
scratch because LAPACK overwrites it; this input copy is included in the timing.

Each sample cycles through 64 input pairs for 4096 operations and consumes one output element
in a scalar checksum. Fifteen rounds rotate operation order. Input preparation, compilation,
buffer allocation and round-boundary GC are excluded. Checks cover addition, multiplication,
eigenvalue agreement, orthogonality and reconstruction. All three operations allocate zero bytes.
Times below are per operation.

| Size | Addition | Multiplication | eigh |
|---|---:|---:|---:|
| 2×2 | 58.1 ns | 88.5 ns | 1.0 µs |
| 4×4 | 154.4 ns | 267.3 ns | 14.5 µs |
| 8×8 | 183.5 ns | 305.3 ns | 31.5 µs |
| 16×16 | 542.5 ns | 783.3 ns | 93.5 µs |
