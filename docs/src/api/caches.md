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

For ArrayMap-backed JITArray, `getready(cache, indices...)` reads a value that the caller
has already generated. It skips the presence check and generation path. Bounds checks
remain unless called with `@inbounds`.

```@docs
JITArray
getready
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
| `:exact` | Evicts the least recently accessed value; updates its access order on each read. |
| `:clock` (default) | Scans a circular resident vector, giving visited entries a second chance. |
| `:sieve` | Scans insertion order; reads mark entries as visited without reordering. |

```@docs
SampledLRUArray
ExactLRUArray
```

### Approximate LRU

`SampledLRUArray(...; policy=:clock)` and `SampledLRUArray(...; policy=:sieve)` are specializations
of the same cache type. Reads set a visited bit; neither policy uses access counters or random samples.
CLOCK stores residents in a circular vector; SIEVE keeps insertion order in a linked list.

### References

- **CLOCK:** F. J. Corbató, [*A Paging Experiment with the Multics System*](https://multicians.org/paging-experiment.pdf), in *In Honor of Philip M. Morse*, MIT Press, 1969, pp. 217–228. The circular scan gives referenced entries a second chance.
- **SIEVE:** Yazhuo Zhang, Juncheng Yang, Yao Yue, Ymir Vigfusson, and K. V. Rashmi, [*SIEVE is Simpler than LRU: an Efficient Turn-Key Eviction Algorithm for Web Caches*](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo), NSDI 2024. Hits mark entries visited; the eviction hand scans insertion order and clears visited bits.
