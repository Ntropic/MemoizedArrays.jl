# Storage API

```@meta
CurrentModule = JITArrays
```

Storage determines how coordinates locate entries; the [cache choice](caches.md) determines retention.
Vector/tuple arguments use [`VT`](@ref).

## Domain and storage selection

Supply a domain as a tuple or vector of integer lengths, integer unit ranges, or `:`.
Lengths start at one; ranges retain their coordinates; `:` accepts every signed `Int` coordinate.
Mix axis forms as needed. `unbound(N)` creates `N` unbounded axes.

```julia
JITArray{Float64}((i, j) -> Float64(i) + j, (100, 100))
ExactLRUArray{Float64}((i, j) -> Float64(i) + j, (-10:10, 20); capacity=32)
ApproxLRUArray{Float64}((i, j) -> Float64(i) + j, unbound(2); capacity=512)
JITArray{Float64}((i, j) -> Float64(i) + j, (:, 1:100);
    storage=:radixtree, block_shape=(4, 8))
```

The domain is required; use `()` for a scalar cache. When storage is omitted, all finite axes
select `:array`; any unbounded axis selects `:dict`. Explicit storage is used or rejected,
never replaced. Storage selection does not change the requested eviction policy.

The existing `SparseJITArray` and `SparseLRUArray` conveniences default explicitly to `:dict`.
Their legacy `ndims` shorthand retains its finite, centered domain described below.

## [Array](@id array)

`storage=:array` allocates one reference slot per position and generates values only when requested.
Integer lengths and finite ranges are accepted, including offset ranges. Unbounded domains are
rejected for every cache policy; an LRU capacity does not reduce the slot array.

## [Dictionary](@id dictionary)

`storage=:dict` hashes coordinate tuples and allocates storage as entries are requested.
It accepts finite and unbounded domains and is the default when any axis is unbounded.

## [Radix tree](@id radix-tree)

Use `storage=:radixtree` with `block_shape`. Start with the coordinate bits with the
highest binary place values, refining toward the requested entry. Allocate only touched branches and entry blocks.

![A radix lookup follows selected slots through branch arrays into a leaf array, with dots indicating further levels above](../assets/radix-slots.svg)

Each branch contains an array of child-reference slots; each leaf contains an array of entry slots.
The angled grids expand one selected interior tile at each level, centering the child grid beneath it.
The ellipsis indicates additional levels above the illustrated branches.
Blue arrows trace a lookup to its value, dotted where they pass beneath a layer.
This example uses `block_shape=(4, 4)`, giving every
node a 4×4 array of slots.

## [Tree shortcuts by coordinate depth](@id subtree-radix-tree)

Use `storage=:subtree` with `block_shape`. Every domain axis must contain zero; otherwise construction errors.
Both tree maps use the same grid alignment, controlled by `center=0.5` (default) or `center=-0.5`.
Select a shortcut by the highest bits that differ from the encoded zero coordinate, then use the same
high-to-low traversal. Coordinates sharing high bits with zero skip upper branches.
The path to zero is computed once at construction; structural insertion and pruning refresh its shortcuts.
The shortcut array covers the platform's `Int` width; shortcuts allocate no extra tree nodes.

### Block shapes and coordinate bounds

`block_shape` sets both the child-reference block and the cached-entry block dimensions.
For example, `(8, 4)` gives 32 slots in each block and consumes 3 bits of the first coordinate
and 2 bits of the second per routing step. It does not specify the shape of a generated value.

Give one width per coordinate axis; each must be a power of two ≥ 2. Larger blocks reduce tree depth
but allocate more slots per node. LRU `capacity` independently limits resident values.

For width 8, `center=0.5` puts coordinates `-3:4` in the leaf containing zero;
`center=-0.5` instead uses `-4:3`. The scalar setting applies to every axis and is available
for all cache policies with either tree storage. Ancestors stay as close to that center as
their nested child boundaries allow. Domain bounds and generator coordinates are unchanged.
An explicit `center` is rejected for array and dictionary storage.

```jldoctest
julia> using JITArrays

julia> a = JITArray{Tuple{Int,Int}}(tuple, -100:100, -100:100;
           storage=:subtree, block_shape=(8, 4));

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
unbound
isunbound
```
