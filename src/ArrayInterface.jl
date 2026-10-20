# Julia array operations: axes, bounds, indexing, views and copying.
# Also provides domain normalization and coordinate routing.

"""`AxisExtent(lo, hi)` defines inclusive axis bounds; coordinates index themselves."""
struct AxisExtent{Full} <: AbstractUnitRange{Int}
    lo::Int
    hi::Int
    AxisExtent(lo::Int, hi::Int) = new{lo == typemin(Int) && hi == typemax(Int)}(lo, hi)
end
Base.first(axis::AxisExtent) = axis.lo
Base.last(axis::AxisExtent) = axis.hi
Base.firstindex(axis::AxisExtent) = axis.lo
Base.lastindex(axis::AxisExtent) = axis.hi
Base.axes(axis::AxisExtent) = (axis,)
Base.isempty(axis::AxisExtent) = axis.hi < axis.lo
Base.length(axis::AxisExtent) = checked_extent(axis)
Base.size(axis::AxisExtent) = (length(axis),)
Base.checkindex(::Type{Bool}, axis::AxisExtent, i::Real) = first(axis) <= i <= last(axis)
# Every Int is valid on a full axis; specialization removes the bounds comparisons.
Base.checkindex(::Type{Bool}, ::AxisExtent{true}, ::Int) = true
Base.@propagate_inbounds function Base.getindex(axis::AxisExtent, i::Int)
    @boundscheck if !checkindex(Bool, axis, i)
        throw(BoundsError(axis, i))
    end
    return i
end
Base.iterate(axis::AxisExtent) = isempty(axis) ? nothing : (first(axis), first(axis))
Base.iterate(axis::AxisExtent, state::Int) = state == last(axis) ? nothing : (state + 1, state + 1)

const SparseArray{V,N} = AbstractMemoizedArray{V,N,M} where {M<:SparseMap}

Base.axes(cache::SparseArray) = cache.map.domain
Base.size(cache::SparseArray) = map(length, cache.map.domain)
Base.size(cache::SparseArray, d::Int) = d > ndims(cache) ? 1 : length(cache.map.domain[d])
Base.length(cache::SparseArray) = domain_length(cache.map.domain)
Base.IndexStyle(::Type{<:AbstractMemoizedArray}) = IndexCartesian()
Base.similar(cache::AbstractMemoizedArray, ::Type{T}, dims::Dims) where T = Array{T}(undef, dims)

Base.@propagate_inbounds function Base.getindex(cache::AbstractMemoizedArray{V,N}, indices::Vararg{Int,N}) where {V,N}
    return cache_get(cache, indices)::V
end
Base.@propagate_inbounds function cache_get(cache::AbstractMemoizedArray{V}, indices::Tuple) where V
    lookup = map_lookup(cache, indices)
    return cache_get(cache, lookup, indices)::V
end
Base.@propagate_inbounds Base.getindex(cache::AbstractMemoizedArray, key::CartesianIndex) = cache[Tuple(key)...]
# Prevent AbstractArray's linear-assignment fallback; supported setters require
# exactly one coordinate per dimension.
Base.setindex!(cache::AbstractMemoizedArray, value, indices::Vararg{Int}) =
    throw(Base.CanonicalIndexError("setindex!",typeof(cache)))
Base.@propagate_inbounds function Base.getindex(cache::AbstractMemoizedArray{V,N}, indices...) where {V,N}
    length(indices) == N || throw(ArgumentError("supply one index per cache dimension; linear indexing is unsupported"))
    return array_index(cache, to_indices(cache, indices))
end
Base.@propagate_inbounds function Base.view(cache::AbstractMemoizedArray{V,N}, indices...) where {V,N}
    length(to_indices(cache, indices)) == N || throw(ArgumentError("supply one index per cache dimension; linear views are unsupported"))
    return invoke(Base.view, Tuple{AbstractArray,Vararg{Any}}, cache, indices...)
end

@inline ready_value(cache, stored) = stored.value
@inline ready_entry(cache, stored) = stored !== cache.map.empty
"""`isready(cache, indices...)` checks for a cached value without generating or changing recency."""
Base.@propagate_inbounds Base.isready(cache::AbstractMemoizedArray{V,N}, indices::Vararg{Int,N}) where {V,N} = ready_entry(cache, entry(cache, cache_key(cache, indices)))
Base.@propagate_inbounds function Base.isready(cache::AbstractMemoizedArray{V,N}, key::VT{N,Int}) where {V,N}
    @astuple N key
    return isready(cache, key...)
end
Base.@propagate_inbounds Base.isready(cache::AbstractMemoizedArray, key::CartesianIndex) = isready(cache, Tuple(key))

"""
    generate!(cache, indices...)
    generate!(cache, requests)

Return one value, or generate a batch synchronously and return the cache. Integer tuples denote one coordinate.
"""
Base.@propagate_inbounds generate!(cache::AbstractMemoizedArray, indices::Vararg{Int,N}) where N = cache[indices...]
Base.@propagate_inbounds generate!(cache::AbstractMemoizedArray{V,N}, key::NTuple{N,Int}) where {V,N} = cache[key...]
Base.@propagate_inbounds generate!(cache::AbstractMemoizedArray{V,0}, key::Tuple{}) where V = cache[]
Base.@propagate_inbounds generate!(cache::AbstractMemoizedArray, key::CartesianIndex) = cache[key]
function generate!(cache::AbstractMemoizedArray{V,N},
    requests::Union{VT{B,NTuple{N,Int}},AbstractVector{Int},AbstractVector{CartesianIndex{N}}}) where {V,N,B}
    for key in requests
        generate!(cache, key)
    end
    return cache
end

# Copy coordinate selections without requesting linear cache indices.
const CacheView = SubArray{<:Any,<:Any,<:AbstractMemoizedArray}
const CacheSource = Union{AbstractMemoizedArray,CacheView}
function Base.copyto!(dest::AbstractArray, source::CacheSource)
    cache, keys = source_keys(source)
    return copyvalues!(dest, cache, keys)
end
function Base.copyto!(dest::PermutedDimsArray, source::CacheSource)
    cache, keys = source_keys(source)
    return copyvalues!(dest, cache, keys)
end
Base.copy(source::CacheSource) = copyto!(Array{eltype(source)}(undef, size(source)), source)

# Helpers -------------------------------------------------------------------------------

"""`unbound(N)` returns a tuple of `N` unbounded axes (`:`)."""
function unbound(N::Integer)
    if N < 0
        throw(ArgumentError("dimension count must be nonnegative"))
    end
    return ntuple(_ -> Colon(), N)
end

"""`isunbound(axis)` tests whether an axis covers every supported integer coordinate."""
isunbound(::Colon) = true
isunbound(::Int) = false
isunbound(axis::AbstractUnitRange{Int}) = first(axis) == typemin(Int) && last(axis) == typemax(Int)

function axis_domain(axis::Int)
    if axis < 0
        throw(ArgumentError("dimensions must be nonnegative"))
    end
    return Base.OneTo(axis)
end
axis_domain(::Colon) = AxisExtent(typemin(Int), typemax(Int))
axis_domain(axis::AbstractUnitRange{Int}) = first(axis) == 1 ? Base.OneTo(last(axis)) : AxisExtent(first(axis), last(axis))
axis_domain(axis) = throw(ArgumentError("domain axes must be integer lengths, integer unit ranges, or :"))


# The precomputed offsets belong to the cache type, with no coordinate-time geometry work.
routing_offset(::Type) = ()
routing_offset(map) = routing_offset(typeof(map))
@generated function cache_offset(::AbstractMemoizedArray{V,N,M}) where {V,N,M}
    offset = routing_offset(M)
    return :(Val{$(QuoteNode(offset))}())
end
@inline cache_routing(cache, key) = cache_routing(cache, key, cache.map)
@inline cache_routing(cache, key, map) = map_routing(map, key)
@inline delete_entry!(cache, key) = delete_key!(cache.map, key)

# ArrayMap receives slot addresses; sparse maps receive coordinate tuples.
Base.@propagate_inbounds function cache_key(cache::SparseArray{V,N}, key::NTuple{N,Int}) where {V,N}
    @boundscheck checkbounds(cache, key...)
    return key
end
Base.@propagate_inbounds cache_key(cache::AbstractMemoizedArray, key::CartesianIndex) = cache_key(cache, Tuple(key))

# Coordinate selections use Base's lazy Cartesian index array.
Base.@propagate_inbounds array_index(cache::AbstractMemoizedArray, indices::Tuple{Vararg{Int}}) = cache[indices...]
Base.@propagate_inbounds function array_index(cache::AbstractMemoizedArray, indices::Tuple)
    _, source = source_keys(cache)
    keys = view(source, indices...)
    return copyvalues!(Array{eltype(cache)}(undef, size(keys)), cache, keys)
end
source_keys(cache::AbstractMemoizedArray) = (cache, CartesianIndices(axes(cache)))
function source_keys(source::CacheView)
    cache, keys = source_keys(parent(source))
    return cache, view(keys, parentindices(source)...)
end

# Extents can exceed Int even though both coordinate endpoints fit.
function checked_extent(axis::AxisExtent)
    extent = max(Int128(0), Int128(last(axis)) - Int128(first(axis)) + 1)
    if extent > typemax(Int)
        throw(OverflowError("axis extent exceeds Int; use coordinate indexing or a finite selection"))
    end
    return Int(extent)
end

function domain_length(ranges)
    if any(isempty, ranges)
        return 0
    end
    count = 1
    for axis in ranges
        count = Base.checked_mul(count, length(axis))
    end
    return count
end

@inline function entry(cache, key)
    slot = map_slot(cache.map, key, cache_routing(cache, key))
    return slot === nothing ? cache.map.empty : slot_entry(slot)
end

# Copy finite selections through the per-key cache path.
function copyvalues!(dest, cache, keys)
    if length(dest) < length(keys)
        throw(BoundsError(dest, eachindex(keys)))
    end
    if Base.mightalias(dest, keys)
        keys = copy(keys)
    end
    for (i, key) in zip(eachindex(dest), keys)
        @inbounds dest[i] = cache[key]
    end
    return dest
end
