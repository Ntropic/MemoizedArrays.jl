<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-dark.svg">
  <img alt="MemoizedArrays.jl" src="docs/src/assets/logo-light.svg" width="364">
</picture>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Julia 1.13+](https://img.shields.io/badge/Julia-1.13%2B-9558B2)](https://julialang.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://ntropic.github.io/MemoizedArrays.jl/stable/)
[![Tests](https://github.com/Ntropic/MemoizedArrays.jl/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Ntropic/MemoizedArrays.jl/actions/workflows/tests.yml)

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

`LRUMemoizedArray` limits retained values by evicting least recently used entries and regenerating them when needed.

```julia
retained = LRUMemoizedArray(value_at, ArrayMap{Float64}((100, 100)); capacity=512)
retained[2, 3]  # Keep at most 512 generated values.
```

### Radix map

`RadixMap` stores sparse coordinates in a tree that allocates storage along requested paths.

```julia
sparse = MemoizedArray(value_at, RadixMap{Float64}(unbound(2); block_shape=(8, 8)))
sparse[-200, 300]  # Allocate storage only along requested tree paths.
```

### Concurrent access

Set `concurrent=true` and provide one queue per active caller. For example, with
`julia --threads=n`, use `queues=n` and a capacity divisible by `n`:

```julia
n = Threads.nthreads(:default)
shared = LRUMemoizedArray(value_at, ArrayMap{Float64}((100, 100));
    capacity=128*n, concurrent=true, queues=n)
handles = queue_handles(shared)
tasks = [Threads.@spawn sum(handles[q][i, 3] for i in 1:100) for q in 1:n]
fetch.(tasks)
```

Alternatively, use `with_queue` to give each worker task a caller that it reuses
across many runs. The queue is claimed once and released when the block finishes:

```julia
tasks = [
    Threads.@spawn begin
        with_queue(shared) do caller
            [sum(caller[i, 3] for i in 1:100) for run in 1:100]
        end
    end for _ in 1:n
]
fetch.(tasks)
```
## Author

Michael Schilling
