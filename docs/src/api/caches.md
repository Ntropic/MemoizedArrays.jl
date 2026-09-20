# Cache API

```@meta
CurrentModule = JITArrays
```

Choose how long generated values remain resident. All policies work with every [storage backend](storage.md).
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
Bounded caches reuse evicted entry objects. Eviction does not mutate or invalidate values already returned.
`isready` neither generates values nor changes recency.

```@docs
LRUArray
```

| Policy | Behavior |
|---|---|
| `:exact` | Evicts the least recently accessed value; updates its access order on each read. |
| `:sieve` (default) | Scans insertion order; reads mark entries as visited without reordering. |
| `:batch_reinsert` | Cleans an old batch; reused entries get another turn in the queue, and unvisited entries are evicted together. |
| `:batch_age` | Cleans an old batch; entries reused within a recent cleanup window survive, and older entries are evicted together. |

```@docs
ApproxLRUArray
ExactLRUArray
```

### Approximate LRU

`ApproxLRUArray` supports `:sieve`, `:batch_reinsert`, and `:batch_age`. These policies track reuse without
reordering entries on each read. None uses random sampling.

SIEVE scans insertion order when it needs a victim. It clears visited flags while scanning,
then evicts the first unvisited entry. A newly inserted entry starts unvisited.

The batch policies keep an insertion queue and a pool of reusable entries. When the pool is
empty, cleanup processes up to `batch_size` queued entries together. `:batch_reinsert` clears each
survivor's visited flag and appends it to the queue; `:batch_age` appends recent survivors without
changing their timestamps. New entries have no reuse mark until first read.

```julia
cache = LRUArray{Float64}(i -> sin(i), (100_000,);
    capacity=10_000, policy=:batch_reinsert, batch_size=64)

cache = LRUArray{Float64}(i -> sin(i), (100_000,);
    capacity=10_000, policy=:batch_age, batch_size=64, keep_epochs=160)
```

`batch_size` defaults to 64 and is capped at `capacity`. It groups residents by queue order,
independently of their coordinates or tree leaves. Cleanup repeats until it releases at least one
entry, so `batch_size` does not bound the total work of an individual request. Later insertions
consume the freed entries without another cleanup. Occupancy can drop below capacity between
cleanups, and the request that triggers cleanup pays for the whole batch.

For `:batch_age`, each processed batch advances an epoch. Reads record the current epoch;
an entry survives while its age is less than `keep_epochs`. The default is
`max(4, cld(capacity, min(batch_size, capacity)))`. These epochs count cleanup batches,
not elapsed time or reads. `keep_epochs` applies only to `:batch_age`.

All cache policies currently require synchronous access; concurrent access to a cache is unsupported.

### References

- **SIEVE:** Yazhuo Zhang, Juncheng Yang, Yao Yue, Ymir Vigfusson, and K. V. Rashmi, [*SIEVE is Simpler than LRU: an Efficient Turn-Key Eviction Algorithm for Web Caches*](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo), NSDI 2024. Cached reads mark entries visited; the eviction hand scans insertion order and clears visited flags.
