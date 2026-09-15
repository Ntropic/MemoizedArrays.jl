# JITArrays.jl

**Generate once, reuse through array indexing.** Unbounded or LRU caching, dense or sparse storage,
shared generation across tasks. Julia 1.12+.

```julia
using JITArrays

a = JITArray{Int}((i, j) -> i + j, 100, 100)
b = LRUArray{Int}((i, j) -> i + j, -100:100, -100:100; capacity=512)
a[2, 3]   # Generate, then reuse.
```

[Usage](docs/src/index.md) · [API](docs/src/api.md) · [Benchmarks](docs/src/benchmarks.md)
