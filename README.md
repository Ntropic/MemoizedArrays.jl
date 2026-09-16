# JITArrays.jl

**Generate once, reuse through array indexing.** Unbounded or LRU caching, array or sparse storage,
synchronous generation. Julia 1.12+.

This version has no concurrency handling; use one caller per cache. [Concurrency ideas](spec/details/concurrency.md) are deferred.

```julia
using JITArrays

a = JITArray{Int}((i, j) -> i + j, 100, 100)
b = LRUArray{Int}((i, j) -> i + j, -100:100, -100:100; capacity=512)
a[2, 3]   # Generate, then reuse.
```

[Usage](docs/src/index.md) · [API](docs/src/api.md) · [Benchmarks](docs/src/benchmarks.md)

Run tests: `julia --startup-file=no --compiled-modules=existing --project=. test/runtests.jl`.
