"""`AxisExtent(lo, hi)` defines inclusive axis bounds; coordinates index themselves."""
struct AxisExtent <: AbstractUnitRange{Int}
    lo::Int
    hi::Int
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
Base.@propagate_inbounds function Base.getindex(axis::AxisExtent, i::Int)
    @boundscheck if !checkindex(Bool, axis, i)
        throw(BoundsError(axis, i))
    end
    return i
end
Base.iterate(axis::AxisExtent) = isempty(axis) ? nothing : (first(axis), first(axis))
Base.iterate(axis::AxisExtent, state::Int) = state == last(axis) ? nothing : (state + 1, state + 1)

const _IndexedArray{V,N,E} = Union{JITArray{V,N,E},ExactLRUArray{V,N,E},SampledLRUArray{V,N,E}}
const _UnboundedArray = JITArray
"""`LRUArray{V}(generator, dims...; capacity, policy=:sampled, kwargs...)`: policies `:sampled` (default), `:exact`."""
const LRUArray{V,N} = Union{ExactLRUArray{V,N},SampledLRUArray{V,N}}
const _SparseMap = Union{DictMap,MSBRadixMap,LSBRadixMap}
const _ArrayBacked{V,N} = Union{JITArray{V,N,E,M},ExactLRUArray{V,N,E,M},SampledLRUArray{V,N,E,M}} where {E,M<:AtomicMemory}
const _SparseArray{V,N} = Union{JITArray{V,N,E,M},ExactLRUArray{V,N,E,M},SampledLRUArray{V,N,E,M}} where {E,M<:_SparseMap}
"""
    SparseJITArray{V}(generator; ndims, storage=:dict, kwargs...)
    SparseJITArray{V}(generator, ranges...; kwargs...)

Cache signed coordinates without eviction. Omitted ranges cover all Int coordinates.
"""
const SparseJITArray{V,N} = JITArray{V,N,Entry{V},M} where {M<:_SparseMap}
"""
    SparseLRUArray{V}(generator; ndims, capacity, policy=:sampled, kwargs...)
    SparseLRUArray{V}(generator, ranges...; capacity, kwargs...)

Cache signed coordinates with eviction. Omitted ranges cover all Int coordinates.
"""
const SparseLRUArray{V,N} = Union{ExactLRUArray{V,N,E,M},SampledLRUArray{V,N,E,M}} where {E,M<:_SparseMap}

function LRUArray{V}(generator, domain::VT{N,T}; capacity::Int,
    policy::Symbol=:sampled, kwargs...) where {V,N,T<:Union{Int,AbstractUnitRange{Int}}}
    if policy === :sampled
        return SampledLRUArray{V}(generator, domain; capacity, kwargs...)
    elseif policy === :exact
        return ExactLRUArray{V}(generator, domain; capacity, kwargs...)
    end
    throw(ArgumentError("unknown LRU policy $policy; use :sampled or :exact"))
end
SparseJITArray{V}(generator, axes::VT{N,R}; storage=:dict, kwargs...) where {V,N,R<:AbstractUnitRange{Int}} =
    JITArray{V}(generator, axes; storage, kwargs...)
SparseLRUArray{V}(generator, axes::VT{N,R}; capacity::Int, storage=:dict, kwargs...) where
    {V,N,R<:AbstractUnitRange{Int}} =
    LRUArray{V}(generator, axes; capacity, storage, kwargs...)

SparseJITArray{V}(generator; ndims::Int, kwargs...) where V =
    SparseJITArray{V}(generator, int_axes(ndims); kwargs...)
SparseLRUArray{V}(generator; ndims::Int, kwargs...) where V =
    SparseLRUArray{V}(generator, int_axes(ndims); kwargs...)

function int_axes(ndims::Int)
    ndims < 0 && throw(ArgumentError("ndims must be nonnegative"))
    return ntuple(_ -> AxisExtent(typemin(Int), typemax(Int)), ndims)
end

LRUArray{V}(generator, dims::Int...; kwargs...) where V =
    LRUArray{V}(generator, dims; kwargs...)
LRUArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    LRUArray{V}(generator, (axis, axes...); kwargs...)
SparseJITArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    SparseJITArray{V}(generator, (axis, axes...); kwargs...)
SparseLRUArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    SparseLRUArray{V}(generator, (axis, axes...); kwargs...)

LRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} =
    LRUArray{V}(generator, domain; capacity, kwargs...)
SparseJITArray{V}(domain::VT{N,T}, generator::Function; kwargs...) where {V,N,T} =
    SparseJITArray{V}(generator, domain; kwargs...)
SparseLRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} =
    SparseLRUArray{V}(generator, domain; capacity, kwargs...)

Base.size(cache::_ArrayBacked) = cache.domain
Base.axes(cache::_SparseArray) = cache.domain
Base.size(cache::_SparseArray) = map(length, cache.domain)
Base.size(cache::_SparseArray, d::Int) = d > ndims(cache) ? 1 : length(cache.domain[d])
Base.length(cache::_SparseArray) = sparse_length(cache.domain)
Base.IndexStyle(::Type{<:_ArrayBacked}) = IndexLinear()
Base.IndexStyle(::Type{<:_SparseArray}) = IndexCartesian()
Base.similar(cache::AbstractJITArray, ::Type{T}, dims::Dims) where T = Array{T}(undef, dims)

Base.@propagate_inbounds function Base.getindex(cache::_IndexedArray{V,N}, indices::Vararg{Int,N}) where {V,N}
    key = cache_key(cache, indices)
    return cache_get(cache, key, nothing, indices)::V
end
Base.@propagate_inbounds function Base.getindex(cache::AbstractJITArray{V}, address::Int) where V
    return cache_get(cache, cache_key(cache, address))::V
end
Base.@propagate_inbounds Base.getindex(cache::AbstractJITArray, key::CartesianIndex) = cache[Tuple(key)...]
Base.@propagate_inbounds Base.getindex(cache::AbstractJITArray, indices...) =
    array_index(cache, to_indices(cache, indices))

"""`isready(cache, indices...)` checks for a cached value without generating or changing recency."""
Base.@propagate_inbounds function Base.isready(cache::_IndexedArray{V,N}, indices::Vararg{Int,N}) where {V,N}
    return ready_entry(entry(cache, cache_key(cache, indices)))
end
Base.@propagate_inbounds Base.isready(cache::AbstractJITArray, address::Int) =
    ready_entry(entry(cache, cache_key(cache, address)))
Base.@propagate_inbounds function Base.isready(cache::AbstractJITArray{V,N}, key::VT{N,Int}) where {V,N}
    @astuple N key
    return isready(cache, key...)
end
Base.@propagate_inbounds Base.isready(cache::AbstractJITArray, key::CartesianIndex) = isready(cache, Tuple(key))

"""
    generate!(cache, indices...)
    generate!(cache, requests)

Return one value, or a Task for a batch; `fetch(task)` returns the cache.
Keep batch requests unchanged until completion. Integer tuples denote one coordinate.
"""
Base.@propagate_inbounds generate!(cache::AbstractJITArray, indices::Vararg{Int,N}) where N = cache[indices...]
Base.@propagate_inbounds generate!(cache::AbstractJITArray{V,N}, key::NTuple{N,Int}) where {V,N} = cache[key...]
Base.@propagate_inbounds generate!(cache::AbstractJITArray{V,0}, key::Tuple{}) where V = cache[]
Base.@propagate_inbounds generate!(cache::AbstractJITArray, key::CartesianIndex) = cache[key]
function generate!(cache::AbstractJITArray{V,N},
    requests::Union{VT{B,NTuple{N,Int}},AbstractVector{Int},AbstractVector{CartesianIndex{N}}}) where {V,N,B}
    return Threads.@spawn begin
        stamp = batch_clock(cache, length(requests))
        Threads.@threads :dynamic for key in requests
            batch_value(cache, key, stamp)
        end
        return cache
    end
end

# Shared source-key forwarding preserves sampled timestamps through slices, views and reshapes.
const _CacheReshape = Base.ReshapedArray{<:Any,<:Any,<:AbstractJITArray}
const _CacheParent = Union{AbstractJITArray,_CacheReshape}
const _CacheView = SubArray{<:Any,<:Any,<:_CacheParent}
const _CacheSource = Union{_CacheParent,_CacheView}
Base.copyto!(dest::AbstractArray, source::_CacheSource) = copy_source!(dest, source)
Base.copyto!(dest::PermutedDimsArray, source::_CacheSource) = copy_source!(dest, source)
Base.copy(source::_CacheSource) = copyto!(Array{eltype(source)}(undef, size(source)), source)

# Helpers -------------------------------------------------------------------------------

# Preserve range-vector semantics when an empty vector becomes an empty tuple.
domain_storage(input, storage) = storage
domain_storage(input::AbstractVector{<:AbstractUnitRange{Int}}, ::Nothing) = :dict

axis_domain(input::NTuple{N,Int}, storage) where N = input
axis_domain(input::NTuple{N,R}, storage) where {N,R<:AbstractUnitRange{Int}} =
    map(r -> AxisExtent(first(r), last(r)), input)
axis_domain(::Tuple{}, storage) = ()
entry_keytype(::NTuple{N,Int}, storage) where N = Int
entry_keytype(::NTuple{N,AxisExtent}, storage) where N = NTuple{N,Int}
entry_keytype(::Tuple{}, storage) = storage === nothing ? Int : Tuple{}

# Select sparse storage without computing the domain's total extent.
make_map(::Type{T}, domain::NTuple{N,AxisExtent}, storage, block) where {T,N} =
    sparse_map(T, domain, storage, block)
function sparse_map(::Type{T}, domain::NTuple{N}, storage, block) where {T,N}
    if storage === nothing || storage === :dict
        if block !== nothing
            throw(ArgumentError("block shapes apply only to tree storage"))
        end
        return DictMap(T, Val(N), _EMPTY)
    elseif storage === :msbtree || storage === :lsbtree
        if block === nothing
            throw(ArgumentError("tree storage requires block_shape"))
        end
        if storage === :msbtree
            return MSBRadixMap(T, domain, block, _EMPTY)
        end
        return LSBRadixMap(T, domain, block, _EMPTY)
    end
    throw(ArgumentError("unknown storage $storage; use :dict, :msbtree, or :lsbtree"))
end
function make_map(::Type{T}, domain::NTuple{N,Int}, storage, block) where {T,N}
    if storage !== nothing || block !== nothing
        throw(ArgumentError("storage and block shapes require range axes"))
    end
    return empty_map(T, domain)
end
function make_map(::Type{T}, domain::Tuple{}, storage, block) where T
    if storage === nothing
        if block !== nothing
            throw(ArgumentError("storage and block shapes require range axes"))
        end
        return empty_map(T, domain)
    end
    return sparse_map(T, domain, storage, block)
end

# Convert only array domains to linear addresses; sparse maps keep coordinate tuples.
Base.@propagate_inbounds cache_key(cache::_ArrayBacked, key::Tuple) = LinearIndices(cache.domain)[key...]
Base.@propagate_inbounds function cache_key(cache::_ArrayBacked, address::Int)
    @boundscheck checkbounds(cache.map, address)
    return address
end
Base.@propagate_inbounds function cache_key(cache::_SparseArray{V,N}, key::NTuple{N,Int}) where {V,N}
    @boundscheck check_sparse_key(cache, key)
    return key
end
Base.@propagate_inbounds cache_key(cache::_SparseArray, address::Int) = sparse_coordinates(cache, address)
Base.@propagate_inbounds cache_key(cache::AbstractJITArray, key::CartesianIndex) = cache_key(cache, Tuple(key))
@inline generator_indices(cache::_ArrayBacked, key::Int) = @inbounds Tuple(CartesianIndices(cache.domain)[key])
@inline generator_indices(cache::_SparseArray, key::Tuple) = key
@inline generator_indices(cache, key, indices::Tuple) = indices
@inline generator_indices(cache, key, ::Nothing) = generator_indices(cache, key)
@inline entrytype(::_IndexedArray{V,N,E}) where {V,N,E} = E

# Scalar indexing follows Base; finite selections use a lazy key proxy.
Base.@propagate_inbounds function array_index(cache::AbstractJITArray, indices::Tuple{Vararg{Int}})
    if length(indices) == ndims(cache) || length(indices) == 1
        return cache[indices...]
    end
    return Base._getindex(IndexStyle(cache), cache, indices...)
end
Base.@propagate_inbounds function array_index(cache::AbstractJITArray, indices::Tuple)
    _, source = source_keys(cache)
    keys = view(source, indices...)
    return copy_values!(Array{eltype(cache)}(undef, size(keys)), cache, keys)
end
source_keys(cache::_ArrayBacked) = (cache, LinearIndices(cache))
source_keys(cache::_SparseArray) = (cache, _SparseCoordinates(cache.domain))
function source_keys(source::_CacheReshape)
    cache, keys = source_keys(parent(source))
    return cache, reshape(keys, size(source))
end
function source_keys(source::_CacheView)
    cache, keys = source_keys(parent(source))
    return cache, view(keys, parentindices(source)...)
end

batch_clock(::AbstractJITArray, count::Int) = nothing
@inline batch_clock(cache::SampledLRUArray, count::Int) = count == 0 ? UInt64(0) : clock_tick!(cache.clock, count)
@inline batch_value(cache::AbstractJITArray, key, stamp) = cache_get(cache, cache_key(cache, key), stamp)

# Extents can exceed Int even though both coordinate endpoints fit.
function checked_extent(axis::AxisExtent)
    extent = max(Int128(0), Int128(last(axis)) - Int128(first(axis)) + 1)
    if extent > typemax(Int)
        throw(OverflowError("axis extent exceeds Int; use coordinate indexing or a finite selection"))
    end
    return Int(extent)
end

function sparse_length(ranges)
    if any(isempty, ranges)
        return 0
    end
    count = 1
    for axis in ranges
        count = Base.checked_mul(count, length(axis))
    end
    return count
end

@inline function check_sparse_key(cache::_SparseArray{V,N}, key::NTuple{N,Int}) where {V,N}
    for i in 1:N
        if !checkindex(Bool, cache.domain[i], key[i])
            throw(BoundsError(cache, key))
        end
    end
    return nothing
end

Base.@propagate_inbounds function sparse_coordinates(cache::_SparseArray{V,N}, address::Int) where {V,N}
    count = length(cache)
    @boundscheck if !(1 <= address <= count)
        throw(BoundsError(cache, address))
    end
    return linear2sparse(cache.domain, address - 1)
end

@inline linear2sparse(ranges::Tuple{}, address::Int) = ()
@inline function linear2sparse(ranges::Tuple, address::Int)
    remaining, offset = divrem(address, length(first(ranges)))
    return (first(first(ranges)) + offset, linear2sparse(Base.tail(ranges), remaining)...)
end
Base.@propagate_inbounds function sparse_coordinates(cache::_SparseArray{V,1}, address::Int) where V
    @boundscheck check_sparse_key(cache, (address,))
    return (address,)
end
# A lazy coordinate array lets Base handle finite selectors and view composition.
struct _SparseCoordinates{N} <: AbstractArray{NTuple{N,Int},N}
    ranges::NTuple{N,AxisExtent}
end
Base.axes(keys::_SparseCoordinates) = keys.ranges
Base.size(keys::_SparseCoordinates) = map(length, keys.ranges)
Base.length(keys::_SparseCoordinates) = sparse_length(keys.ranges)
Base.IndexStyle(::Type{<:_SparseCoordinates}) = IndexCartesian()
Base.@propagate_inbounds function Base.getindex(keys::_SparseCoordinates{N}, key::Vararg{Int,N}) where N
    @boundscheck checkbounds(keys, key...)
    return key
end
