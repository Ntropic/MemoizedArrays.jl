# Cache API

```@meta
CurrentModule = JITArrays
```

Choose how long generated values remain resident. Both choices work with every [storage backend](storage.md).
Vector/tuple arguments use [`VT`](@ref).

Put the generator first, followed by dimensions or signed ranges. Tuple/vector domains also work.
Bounded constructors require `capacity` as a keyword. Legacy domain-first overloads accept
functions; callable objects use generator-first constructors. This order supports Julia's `do` syntax:

```jldoctest
julia> using JITArrays

julia> a = LRUArray{Int}(3, 4; capacity=4) do i, j
           i + j
       end;

julia> a[2, 3]
5
```

## [Unbounded caching](@id unbounded)

Keep all successfully generated values for reuse.

```@docs
JITArray
```

## [Bounded caching](@id bounded)

Limit the number of stored values. Values are generated before admission and eviction.
Full caches reuse evicted entry objects. Eviction does not mutate or invalidate values already returned.
`isready` neither generates values nor changes recency.

```@docs
LRUArray
```

| Policy | Behavior |
|---|---|
| `:sampled` (default) | Samples resident entries and evicts an approximately oldest value; cheaper ready reads. |
| `:exact` | Evicts the least recently accessed value; updates its access order on each read. |

```@docs
SampledLRUArray
ExactLRUArray
```

### Sampled-cache recency

A per-cache `UInt64` counter tracks accesses. Each scalar access advances it by one.
Batch requests share one stamp and advance the counter by their request count, including duplicates.
Empty batches leave the counter unchanged.
