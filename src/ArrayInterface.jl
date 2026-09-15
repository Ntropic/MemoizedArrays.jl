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
Base.length(axis::AxisExtent) = _checked_extent(axis)
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

"""
    JITArray{V}(generator, dims...; kwargs...)

Cache values on demand without eviction. Accept sizes or coordinate ranges.
Storage: `:dict`, `:msbtree`, `:lsbtree`; trees require `block_shape`.
"""
struct JITArray{V,N,E,M,D,G} <: AbstractJITArray{V,N}
    map::M
    domain::D
    generator::G
end

"""
    ExactLRUArray{V}(generator, dims...; capacity, kwargs...)

Cache values with exact LRU eviction. Accept sizes or coordinate ranges.
Storage: `:dict`, `:msbtree`, `:lsbtree`; trees require `block_shape`.
"""
mutable struct ExactLRUArray{V,N,E,M,D,G} <: AbstractJITArray{V,N}
    const map::M
    const domain::D
    const generator::G
    oldest::Union{Nothing,E}
    newest::Union{Nothing,E}
    const capacity::Int
    count::Int
    const available::Threads.Condition
end

"""
    SampledLRUArray{V}(generator, dims...; capacity, clock=:lossy, kwargs...)

Cache values with sampled LRU eviction. Accept sizes or coordinate ranges.
Storage: `:dict`, `:msbtree`, `:lsbtree`; trees require `block_shape`.
Clocks: `:lossy`, `:coarse`, `:nanosecond`.
"""
mutable struct SampledLRUArray{V,N,E,M,D,G,Clock} <: AbstractJITArray{V,N}
    const map::M
    const domain::D
    const generator::G
    const active::Vector{E}
    ready::Int
    pending::Int
    const pool::Vector{Union{Nothing,E}}
    const ages::Vector{UInt64}
    pooled::Int
    rng::UInt64
    divisor::UInt64
    reciprocal::UInt64
    const clock::Clock
    const available::Threads.Condition
end

const _IndexedArray{V,N,E} = Union{JITArray{V,N,E},ExactLRUArray{V,N,E},SampledLRUArray{V,N,E}}
const _UnboundedArray = JITArray
"""`LRUArray{V}(generator, dims...; capacity, policy=:sampled, kwargs...)`: policies `:sampled`, `:exact`."""
const LRUArray{V,N} = Union{ExactLRUArray{V,N},SampledLRUArray{V,N}}
const _SparseMap = Union{DictMap,MSBRadixMap,LSBRadixMap}
const _DenseArray{V,N} = Union{JITArray{V,N,E,M},ExactLRUArray{V,N,E,M},SampledLRUArray{V,N,E,M}} where {E,M<:AtomicMemory}
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

function JITArray{V}(generator, domain::VT{N,T}; storage=nothing,
    block_shape=nothing) where {V,N,T<:Union{Int,AbstractUnitRange{Int}}}
    storage = _domain_storage(domain, storage)
    @astuple length(domain) domain
    return _construct(V, domain, generator, Entry, nothing, storage, block_shape, nothing)
end
function ExactLRUArray{V}(generator, domain::VT{N,T}; capacity::Int,
    storage=nothing, block_shape=nothing) where {V,N,T<:Union{Int,AbstractUnitRange{Int}}}
    storage = _domain_storage(domain, storage)
    @astuple length(domain) domain
    return _construct(V, domain, generator, ExactLRUEntry, capacity, storage, block_shape, nothing)
end
function SampledLRUArray{V}(generator, domain::VT{N,T}; capacity::Int, clock=:lossy,
    storage=nothing, block_shape=nothing) where {V,N,T<:Union{Int,AbstractUnitRange{Int}}}
    storage = _domain_storage(domain, storage)
    @astuple length(domain) domain
    return _construct(V, domain, generator, SampledLRUEntry, capacity, storage, block_shape, clock)
end
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
    SparseJITArray{V}(generator, _int_axes(ndims); kwargs...)
SparseLRUArray{V}(generator; ndims::Int, kwargs...) where V =
    SparseLRUArray{V}(generator, _int_axes(ndims); kwargs...)

function _int_axes(ndims::Int)
    ndims < 0 && throw(ArgumentError("ndims must be nonnegative"))
    return ntuple(_ -> AxisExtent(typemin(Int), typemax(Int)), ndims)
end

JITArray{V}(generator, dims::Int...; kwargs...) where V =
    JITArray{V}(generator, dims; kwargs...)
JITArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    JITArray{V}(generator, (axis, axes...); kwargs...)
ExactLRUArray{V}(generator, dims::Int...; kwargs...) where V =
    ExactLRUArray{V}(generator, dims; kwargs...)
ExactLRUArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    ExactLRUArray{V}(generator, (axis, axes...); kwargs...)
SampledLRUArray{V}(generator, dims::Int...; kwargs...) where V =
    SampledLRUArray{V}(generator, dims; kwargs...)
SampledLRUArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    SampledLRUArray{V}(generator, (axis, axes...); kwargs...)
LRUArray{V}(generator, dims::Int...; kwargs...) where V =
    LRUArray{V}(generator, dims; kwargs...)
LRUArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    LRUArray{V}(generator, (axis, axes...); kwargs...)
SparseJITArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    SparseJITArray{V}(generator, (axis, axes...); kwargs...)
SparseLRUArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    SparseLRUArray{V}(generator, (axis, axes...); kwargs...)

JITArray{V}(domain::VT{N,T}, generator::Function; kwargs...) where {V,N,T} =
    JITArray{V}(generator, domain; kwargs...)
ExactLRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} =
    ExactLRUArray{V}(generator, domain; capacity, kwargs...)
SampledLRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} =
    SampledLRUArray{V}(generator, domain; capacity, kwargs...)
LRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} =
    LRUArray{V}(generator, domain; capacity, kwargs...)
SparseJITArray{V}(domain::VT{N,T}, generator::Function; kwargs...) where {V,N,T} =
    SparseJITArray{V}(generator, domain; kwargs...)
SparseLRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} =
    SparseLRUArray{V}(generator, domain; capacity, kwargs...)

Base.size(cache::_DenseArray) = cache.domain
Base.axes(cache::_SparseArray) = cache.domain
Base.size(cache::_SparseArray) = map(length, cache.domain)
Base.size(cache::_SparseArray, d::Int) = d > ndims(cache) ? 1 : length(cache.domain[d])
Base.length(cache::_SparseArray) = _sparse_length(cache.domain)
Base.IndexStyle(::Type{<:_DenseArray}) = IndexLinear()
Base.IndexStyle(::Type{<:_SparseArray}) = IndexCartesian()
Base.similar(cache::AbstractJITArray, ::Type{T}, dims::Dims) where T = Array{T}(undef, dims)

Base.@propagate_inbounds function Base.getindex(cache::_IndexedArray{V,N}, indices::Vararg{Int,N}) where {V,N}
    key = _cache_key(cache, indices)
    return _cache_get(cache, key, nothing, indices)::V
end
Base.@propagate_inbounds function Base.getindex(cache::AbstractJITArray{V}, address::Int) where V
    return _cache_get(cache, _cache_key(cache, address))::V
end
Base.@propagate_inbounds Base.getindex(cache::AbstractJITArray, key::CartesianIndex) = cache[Tuple(key)...]
Base.@propagate_inbounds Base.getindex(cache::AbstractJITArray, indices...) =
    _array_index(cache, to_indices(cache, indices))

"""`isready(cache, indices...)` checks for a cached value without generating or changing recency."""
Base.@propagate_inbounds function Base.isready(cache::_IndexedArray{V,N}, indices::Vararg{Int,N}) where {V,N}
    return _ready_entry(_entry(cache, _cache_key(cache, indices)))
end
Base.@propagate_inbounds Base.isready(cache::AbstractJITArray, address::Int) =
    _ready_entry(_entry(cache, _cache_key(cache, address)))
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
        stamp = _batch_clock(cache, length(requests))
        Threads.@threads :dynamic for key in requests
            _batch_value(cache, key, stamp)
        end
        return cache
    end
end

# Shared source-key forwarding preserves sampled timestamps through slices, views and reshapes.
const _CacheReshape = Base.ReshapedArray{<:Any,<:Any,<:AbstractJITArray}
const _CacheParent = Union{AbstractJITArray,_CacheReshape}
const _CacheView = SubArray{<:Any,<:Any,<:_CacheParent}
const _CacheSource = Union{_CacheParent,_CacheView}
Base.copyto!(dest::AbstractArray, source::_CacheSource) = _copy_source!(dest, source)
Base.copyto!(dest::PermutedDimsArray, source::_CacheSource) = _copy_source!(dest, source)
Base.copy(source::_CacheSource) = copyto!(Array{eltype(source)}(undef, size(source)), source)

# Helpers -------------------------------------------------------------------------------

# Preserve range-vector semantics when an empty vector becomes an empty tuple.
_domain_storage(input, storage) = storage
_domain_storage(input::AbstractVector{<:AbstractUnitRange{Int}}, ::Nothing) = :dict

# Select concrete entry, map and container types once during construction.
function _construct(::Type{V}, input::Tuple, generator::G, kind, capacity, storage, block, clock) where {V,G}
    if capacity !== nothing
        if capacity < 1
            throw(ArgumentError("capacity must be positive"))
        end
    end
    domain = _domain(input, storage)
    N = length(domain)
    Key = _keytype(domain, storage)
    E = _entry_kind(kind, V, Key)
    map = _make_map(Union{Empty,E}, domain, storage, block)
    M, D = typeof(map), typeof(domain)
    if kind === Entry
        return JITArray{V,N,E,M,D,G}(map, domain, generator)
    elseif kind === ExactLRUEntry
        return ExactLRUArray{V,N,E,M,D,G}(
            map, domain, generator, nothing, nothing, capacity, 0, Threads.Condition())
    end
    clock = _cache_clock(clock)
    return SampledLRUArray{V,N,E,M,D,G,typeof(clock)}(map, domain, generator, Vector{E}(undef, capacity),
        0, 0, fill!(Vector{Union{Nothing,E}}(undef, 16), nothing), zeros(UInt64, 16), 0,
        0x123456789abcdef0, 0, 0, clock, Threads.Condition())
end

_entry_kind(::Type{Entry}, ::Type{V}, ::Type{Key}) where {V,Key} = Entry{V}
_entry_kind(::Type{ExactLRUEntry}, ::Type{V}, ::Type{Key}) where {V,Key} = ExactLRUEntry{V,Key}
_entry_kind(::Type{SampledLRUEntry}, ::Type{V}, ::Type{Key}) where {V,Key} = SampledLRUEntry{V,Key}
_domain(input::NTuple{N,Int}, storage) where N = input
_domain(input::NTuple{N,R}, storage) where {N,R<:AbstractUnitRange{Int}} =
    map(r -> AxisExtent(first(r), last(r)), input)
_domain(::Tuple{}, storage) = ()
_keytype(::NTuple{N,Int}, storage) where N = Int
_keytype(::NTuple{N,AxisExtent}, storage) where N = NTuple{N,Int}
_keytype(::Tuple{}, storage) = storage === nothing ? Int : Tuple{}

# Select sparse storage without computing the domain's total extent.
_make_map(::Type{T}, domain::NTuple{N,AxisExtent}, storage, block) where {T,N} =
    _sparse_map(T, domain, storage, block)
function _sparse_map(::Type{T}, domain::NTuple{N}, storage, block) where {T,N}
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
function _make_map(::Type{T}, domain::NTuple{N,Int}, storage, block) where {T,N}
    if storage !== nothing || block !== nothing
        throw(ArgumentError("storage and block shapes require range axes"))
    end
    return _empty_map(T, domain)
end
function _make_map(::Type{T}, domain::Tuple{}, storage, block) where T
    if storage === nothing
        if block !== nothing
            throw(ArgumentError("storage and block shapes require range axes"))
        end
        return _empty_map(T, domain)
    end
    return _sparse_map(T, domain, storage, block)
end

# Convert only dense domains to linear addresses; sparse maps keep coordinate tuples.
Base.@propagate_inbounds _cache_key(cache::_DenseArray, key::Tuple) = LinearIndices(cache.domain)[key...]
Base.@propagate_inbounds function _cache_key(cache::_DenseArray, address::Int)
    @boundscheck checkbounds(cache.map, address)
    return address
end
Base.@propagate_inbounds function _cache_key(cache::_SparseArray{V,N}, key::NTuple{N,Int}) where {V,N}
    @boundscheck _check_sparse_key(cache, key)
    return key
end
Base.@propagate_inbounds _cache_key(cache::_SparseArray, address::Int) = _sparse_coordinates(cache, address)
Base.@propagate_inbounds _cache_key(cache::AbstractJITArray, key::CartesianIndex) = _cache_key(cache, Tuple(key))
@inline _generator_indices(cache::_DenseArray, key::Int) = @inbounds Tuple(CartesianIndices(cache.domain)[key])
@inline _generator_indices(cache::_SparseArray, key::Tuple) = key
@inline _generator_indices(cache, key, indices::Tuple) = indices
@inline _generator_indices(cache, key, ::Nothing) = _generator_indices(cache, key)
@inline _entrytype(::_IndexedArray{V,N,E}) where {V,N,E} = E

# Scalar indexing follows Base; finite selections use a lazy key proxy.
Base.@propagate_inbounds function _array_index(cache::AbstractJITArray, indices::Tuple{Vararg{Int}})
    if length(indices) == ndims(cache) || length(indices) == 1
        return cache[indices...]
    end
    return Base._getindex(IndexStyle(cache), cache, indices...)
end
Base.@propagate_inbounds function _array_index(cache::AbstractJITArray, indices::Tuple)
    _, source = _source_keys(cache)
    keys = view(source, indices...)
    return _copy_values!(Array{eltype(cache)}(undef, size(keys)), cache, keys)
end
_source_keys(cache::_DenseArray) = (cache, LinearIndices(cache))
_source_keys(cache::_SparseArray) = (cache, _SparseCoordinates(cache.domain))
function _source_keys(source::_CacheReshape)
    cache, keys = _source_keys(parent(source))
    return cache, reshape(keys, size(source))
end
function _source_keys(source::_CacheView)
    cache, keys = _source_keys(parent(source))
    return cache, view(keys, parentindices(source)...)
end

_batch_clock(::AbstractJITArray, count::Int) = nothing
@inline _batch_clock(cache::SampledLRUArray, count::Int) = count == 0 ? UInt64(0) : _clock_tick!(cache.clock, count)
@inline _batch_value(cache::AbstractJITArray, key, stamp) = _cache_get(cache, _cache_key(cache, key), stamp)

# Extents can exceed Int even though both coordinate endpoints fit.
function _checked_extent(axis::AxisExtent)
    extent = max(Int128(0), Int128(last(axis)) - Int128(first(axis)) + 1)
    if extent > typemax(Int)
        throw(OverflowError("axis extent exceeds Int; use coordinate indexing or a finite selection"))
    end
    return Int(extent)
end

function _sparse_length(ranges)
    if any(isempty, ranges)
        return 0
    end
    count = 1
    for axis in ranges
        count = Base.checked_mul(count, length(axis))
    end
    return count
end

@inline function _check_sparse_key(cache::_SparseArray{V,N}, key::NTuple{N,Int}) where {V,N}
    for i in 1:N
        if !checkindex(Bool, cache.domain[i], key[i])
            throw(BoundsError(cache, key))
        end
    end
    return nothing
end

Base.@propagate_inbounds function _sparse_coordinates(cache::_SparseArray{V,N}, address::Int) where {V,N}
    count = length(cache)
    @boundscheck if !(1 <= address <= count)
        throw(BoundsError(cache, address))
    end
    return _linear2sparse(cache.domain, address - 1)
end

@inline _linear2sparse(ranges::Tuple{}, address::Int) = ()
@inline function _linear2sparse(ranges::Tuple, address::Int)
    remaining, offset = divrem(address, length(first(ranges)))
    return (first(first(ranges)) + offset, _linear2sparse(Base.tail(ranges), remaining)...)
end
Base.@propagate_inbounds function _sparse_coordinates(cache::_SparseArray{V,1}, address::Int) where V
    @boundscheck _check_sparse_key(cache, (address,))
    return (address,)
end
# A lazy coordinate array lets Base handle finite selectors and view composition.
struct _SparseCoordinates{N} <: AbstractArray{NTuple{N,Int},N}
    ranges::NTuple{N,AxisExtent}
end
Base.axes(keys::_SparseCoordinates) = keys.ranges
Base.size(keys::_SparseCoordinates) = map(length, keys.ranges)
Base.length(keys::_SparseCoordinates) = _sparse_length(keys.ranges)
Base.IndexStyle(::Type{<:_SparseCoordinates}) = IndexCartesian()
Base.@propagate_inbounds function Base.getindex(keys::_SparseCoordinates{N}, key::Vararg{Int,N}) where N
    @boundscheck checkbounds(keys, key...)
    return key
end
