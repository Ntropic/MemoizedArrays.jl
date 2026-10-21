# MemoizedArrays.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Julia 1.12+](https://img.shields.io/badge/Julia-1.12%2B-9558B2)](https://julialang.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://ntropic.github.io/MemoizedArrays.jl/)
[![Tests](https://img.shields.io/github/actions/workflow/status/Ntropic/MemoizedArrays.jl/tests.yml?label=tests&logo=githubactions&logoColor=white)](https://github.com/Ntropic/MemoizedArrays.jl/actions/workflows/tests.yml)

**Compute array elements when you need them, and reuse the results.**

- Avoid repeating expensive computations.
- Generate only the coordinates you need.
- Limit memory usage via LRU eviction.
- Concurrently access and generate values.

## Install

Install from source via

```julia
using Pkg
Pkg.add(url="https://github.com/Ntropic/MemoizedArrays.jl")
```

## Basic usage

```julia
using MemoizedArrays

value_at(i, j) = sin(i) * cos(j)
values = MemoizedArray(value_at, ArrayMap{Float64}((100, 100)))

values[2, 3]  # Compute on first access.
values[2, 3]  # Reuses the stored result.
```

The map specifies the value type, domain and coordinate lookup; the cache controls retention.


## Core idea

Each array combines two components:

- **Cache:** generation and eviction strategy.
  - `MemoizedArray`: retain every generated value.
  - `LRUMemoizedArray`: bound retention and regenerate evicted values when needed.
- **Map:** coordinate routing and slot storage.
  - `ArrayMap`: fixed, dense storage.
  - `ExpandingArrayMap`: dense storage that grows above its upper bounds.
  - `RadixMap`: sparse tree storage.
  - `SubtreeRadixMap`: radix map with shortcuts for access near zero.
  - `DictMap`: hash-based sparse storage.

### LRU retention

```julia
retained = LRUMemoizedArray(value_at, ArrayMap{Float64}((100, 100)); capacity=512)
retained[2, 3]  # Keep at most 512 generated values.
```

### Radix map

```julia
sparse = MemoizedArray(value_at, RadixMap{Float64}(unbound(2); block_shape=(8, 8)))
sparse[-200, 300]  # Allocate storage only along requested tree paths.
```

### Concurrent access

Start Julia with `julia --threads=4`.

```julia
shared = LRUMemoizedArray(value_at, ArrayMap{Float64}((100, 100));
                  capacity=512, concurrent=true, queues=4)
handles = queue_handles(shared)
tasks = [Threads.@spawn handles[q][q, 3] for q in 1:4]
fetch.(tasks)
```
For automatic scoped assignment, use `with_queue`:

```julia
with_queue(shared) do caller
    caller[1, 3]
end
```
Do not mix automatic selection with explicit handles concurrently on the same cache.

## Development

Run the test suite with `julia --project=. -t 8 test/runtests.jl`.
`test/Behavior_test.jl` checks shared behavior across maps and cache modes;
`test/Concurrency_test.jl` holds focused race regressions. Shared fixtures and
allocation probes live in `test/helpers.jl`.
Focused implementation tests live in `test/internal/`: coordinate mapping,
growth and relocation, LRU bookkeeping, tree pruning, and dictionary collisions.
Concurrency regressions retain the explicit scheduling needed to reproduce races.

Run benchmarks with `julia --project=. -t 8 benchmark.jl`. See
[benchmark workloads and options](docs/src/benchmarks.md#reproduce-the-results)
for selecting cases and operations.
