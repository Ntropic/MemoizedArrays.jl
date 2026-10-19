# MemoizedArrays.jl

```@meta
DocTestSetup = :(using MemoizedArrays)
```

**Compute array elements when you need them, and reuse the results.**

- Avoid repeating expensive computations.
- Generate only the coordinates you access.
- Limit memory usage via LRU eviction.
- Concurrently access and generate values.

## Install

Install from source via

```julia
using Pkg
Pkg.add(url="https://github.com/Ntropic/MemoizedArrays.jl")
```

## Core idea

Each array combines three components:
- **Generator** the function or closure connecting the indexes of an array to their desired values.
- **Map:** coordinate lookup and storage.
- **Cache:** manages the generation and eviction of values.

Choose one of each below.

```@raw html
<iframe src="assets/overview.html" title="Cache and map choices"
    style="display: block; width: 100%; border: 0; background: transparent;"></iframe>
```

## Basic usage

```julia
using MemoizedArrays

value_at(i, j) = sin(i) * cos(j)
values = MemoizedArray(value_at, ArrayMap{Float64}((100, 100)))

values[2, 3]  # Compute on first access.
values[2, 3]  # Reuses the stored result.
```

The map (`ArrayMap`) specifies the type of value type, array type and size; the cache (`MemoizedArray`) controls generation of values, their retention and concurrency.

See [Examples](examples.md) for other cache and map combinations and functionalities. The [API reference](api.md) documents constructors, domains and array operations.
