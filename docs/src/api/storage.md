# Storage API

```@meta
CurrentModule = JITArrays
```

Storage determines how coordinates locate entries; the [cache choice](caches.md) determines retention.
Vector/tuple arguments use [`VT`](@ref).

## Coordinate bounds

Pass ranges to set bounds, or omit them and supply `ndims` to accept every `Int` coordinate:

```julia
SparseJITArray{Float64}((i, j) -> Float64(i) + j; ndims=2)
SparseLRUArray{Float64}((i, j) -> Float64(i) + j; ndims=2, capacity=512,
    storage=:radixtree, block_shape=(4, 8))
```

Storage options: `:dict`, `:radixtree`, `:subtreetree`.
`AxisExtent` describes each dimension's inclusive bounds, independently of storage.

## [Array](@id array)

Pass an integer shape, such as `(100, 100)`. Allocate one reference slot per position and generate values
only when requested. This gives direct lookup for fixed domains whose reference map fits in memory.

## [Dictionary](@id dictionary)

Pass signed integer ranges; `storage=:dict` is the default. Hash coordinate tuples and allocate storage
as entries are requested. Suitable for scattered accesses over large domains.

## [Radix tree](@id radix-tree)

Use `storage=:radixtree` with `block_shape`. Start with the coordinate bits with the
highest binary place values, refining toward the requested entry. Allocate only touched branches and entry blocks.

## [Tree shortcuts by coordinate depth](@id subtree-radix-tree)

Use `storage=:subtreetree` with `block_shape`. Select a shortcut into one shared radix tree from the highest set bits
of the encoded coordinates, then use the same high-to-low traversal. Small coordinates skip
upper branches. Structural insertion and pruning refresh the shortcuts.

Signed coordinates use unsigned zigzag encoding, placing small positive and negative values together.
The shortcut array covers the platform's `Int` width; shortcuts allocate no extra tree nodes.

### Block shapes and coordinate bounds

`block_shape` sets both the child-reference block and the cached-entry block dimensions.
For example, `(8, 4)` gives 32 slots in each block and consumes 3 bits of the first coordinate
and 2 bits of the second per routing step. It does not specify the shape of a generated value.

Give one width per coordinate axis; each must be a power of two ≥ 2. Larger blocks reduce tree depth
but allocate more slots per node. LRU `capacity` independently limits resident values.

```jldoctest
julia> using JITArrays

julia> a = JITArray{Tuple{Int,Int}}(tuple, -100:100, -100:100;
           storage=:subtreetree, block_shape=(8, 4));

julia> a[-20, 3]
(-20, 3)
```

Omitted sparse axes use equal lengths centered around zero, with their product at most `typemax(Int)`.
The equal axis length is the largest that fits this limit, so default linear indexing cannot overflow.
Explicit sparse axes may each span the full signed `Int` domain. Coordinate access and finite selections remain
usable when global `size`, `length`, or linear-index calculations would overflow and throw `OverflowError`.
In one dimension, an integer denotes the signed coordinate; in higher dimensions, a single integer
uses column-major linear indexing.

```@docs
SparseJITArray
SparseLRUArray
```
