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
    storage=:msbtree, block_shape=(4, 8))
```

Storage options: `:dict`, `:msbtree`, `:lsbtree`.
`AxisExtent` describes each dimension's inclusive bounds, independently of storage.

## [Array](@id array)

Pass an integer shape, such as `(100, 100)`. Allocate one reference slot per position and generate values
only when requested. This gives direct lookup for fixed domains whose reference map fits in memory.

## [Dictionary](@id dictionary)

Pass signed integer ranges; `storage=:dict` is the default. Hash coordinate tuples and allocate storage
as entries are requested. Suitable for scattered accesses over large domains.

## [Most significant bit (MSB) tree](@id msb-tree)

Use `storage=:msbtree` with `block_shape`. Start with the coordinate bits with the
highest binary place values, refining toward the requested entry. Allocate only touched branches and entry blocks.

## [Least significant bit (LSB) tree](@id lsb-tree)

Use `storage=:lsbtree` with `block_shape`. Start with the coordinate bits with the
lowest binary place values: small coordinates have shallow paths, while larger coordinates need more levels. Distant
insertions do not deepen existing paths.

### Block shapes and coordinate bounds

`block_shape` sets both the child-reference block and the cached-entry block dimensions.
For example, `(8, 4)` gives 32 slots in each block and consumes 3 bits of the first coordinate
and 2 bits of the second per routing step. It does not specify the shape of a generated value.

Give one width per coordinate axis; each must be a power of two ≥ 2. Larger blocks reduce tree depth
but allocate more slots per node. LRU `capacity` independently limits resident values.

```jldoctest
julia> using JITArrays

julia> a = JITArray{Tuple{Int,Int}}(tuple, -100:100, -100:100;
           storage=:lsbtree, block_shape=(8, 4));

julia> a[-20, 3]
(-20, 3)
```

Sparse axes may each span the full signed `Int64` domain. Coordinate access and finite selections remain
usable when global `size`, `length`, or linear-index calculations would overflow and throw `OverflowError`.
In one dimension, an integer denotes the signed coordinate; in higher dimensions, a single integer
uses column-major linear indexing.

```@docs
SparseJITArray
SparseLRUArray
```
