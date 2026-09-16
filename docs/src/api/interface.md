# [Array interface](@id indexing-generation)

```@meta
CurrentModule = JITArrays
```


Caches implement the read-only `AbstractArray` interface: scalar indexing, `CartesianIndex`, ranges,
index vectors, masks, iteration, and broadcasting. Entries are generated rather than assigned;
returned mutable values remain ordinary Julia objects.

Slices return ordinary arrays containing the selected values. Creating a view generates nothing;
reading through it may generate values. Views use their own axes and do not pin cached entries.
Evicted values may therefore be regenerated on later view access.

```jldoctest
julia> using JITArrays

julia> a = JITArray{Tuple{Int,Int}}(tuple, -3:3, 0:4);

julia> v = @view a[-2:2, 0:3];

julia> isready(a, -2, 0)
false

julia> v[1, 1]
(-2, 0)
```

Generation runs in the calling task. Generator exceptions propagate directly, with no cleanup or
recovery guarantee. Batched `generate!` runs in request order and returns the cache directly.

```@docs
isready
generate!
```

Keep batch requests unchanged until completion. A bounded cache may evict earlier batch results before
the batch finishes. Sampled-cache bulk selections and generation batches share one timestamp; the access
counter advances once by the number of requests, including repeated keys. Individual view reads update
recency separately.
