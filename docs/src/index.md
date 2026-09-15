# JITArrays.jl

**Generate on demand. Reuse through ordinary array indexing.**

JITArrays caches expensive, index-dependent computations in memory: scalars, matrices, or other Julia
objects. Concurrent callers share generation; reading a cached entry allocates no memory.
Keep every generated value, or set a capacity and let the cache evict old entries.

Use standard indexing, slices, and lazy views, with asynchronous generation for batches.
Array storage suits fixed shapes; sparse storage supports huge signed coordinate domains without
allocating every position. Requires Julia 1.12 or later.

## Choose cache and storage

Choose these independently: **unbounded** or **bounded** caching, with any of the four storage backends.
Bounded caches offer sampled eviction (default) or exact LRU. Click a diagram node for its API page.

```@raw html
<iframe src="assets/overview.html" title="Cache and storage choices"
    style="width: 100%; height: 440px; border: 0;"></iframe>
```

## A few examples

### Compute once, read again

```jldoctest
julia> using JITArrays

julia> a = JITArray{Matrix{Float64}}((i, j) -> fill(Float64(i + j), 2, 2), 3, 4);

julia> a[2, 3] == fill(5.0, 2, 2)
true

julia> a[8] === a[2, 3]
true
```

### Bound the number of resident values

```jldoctest
julia> using JITArrays

julia> a = LRUArray{Int}(identity, 10; capacity=2, policy=:exact);

julia> a[1], a[2], a[1], a[3]
(1, 2, 1, 3)

julia> isready(a, 2)
false
```

### Generate sparse coordinates in the background

```jldoctest
julia> using JITArrays

julia> a = JITArray{Tuple{Int,Int}}(tuple, typemin(Int):typemax(Int), -100:100);

julia> job = generate!(a, ((-20, 3), (50, -1)));

julia> fetch(job) === a
true

julia> a[-20, 3]
(-20, 3)
```

Continue with the [API reference](api.md) for indexing, views, generation, cache policies, and storage choices.
