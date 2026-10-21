struct ArrayMapConfig{V,D} <: MapConfig{V}
    domain::D
end

# Array storage: fixed slots with direct reads, writes and resets.
# Slot addresses survive eviction; capacity and recency belong to the cache.

"""
    ArrayMap{T}(domain)

Dense slot storage for a finite domain. Pass the result to MemoizedArray or LRUMemoizedArray.
"""
struct ArrayMap{T,N}
    reference::MemoryRef{T}
    shape::NTuple{N,Int}
    strides::NTuple{N,Int}
    empty::T
end

ArrayMap{V}(domain::VT) where V = ArrayMapConfig{V,typeof(Tuple(domain))}(Tuple(domain))

# Fixed atomic slots share ArrayMap indexing and coordinate generation.
# Queues own retention metadata.
struct ConcurrentArrayMap{T,N}
    reference::GenericMemoryRef{:atomic,T,Core.CPU}
    shape::NTuple{N,Int}
    strides::NTuple{N,Int}
    empty::Empty
end
const DenseMap{T,N} = Union{ArrayMap{T,N},ConcurrentArrayMap{T,N}}
const ArrayBacked{V,N} = AbstractMemoizedArray{V,N,M} where {M<:DenseMap}

function ArrayMap(::Type{T}, shape, empty=EMPTY) where T
    dimensions = Tuple(shape)
    slots = empty_map(T, dimensions, empty)
    return ArrayMap{T,length(dimensions)}(memoryref(slots), dimensions, Base.size_to_strides(1, dimensions...), empty)
end

# Internal tree blocks have already passed block_length validation.
function ArrayMap(::Type{T}, dimensions::NTuple{N,Int}, empty, ::Val{:nocheck}) where {T,N}
    slots = empty_map(T, dimensions, empty, Val(false), Val(:nocheck))
    return ArrayMap{T,N}(memoryref(slots), dimensions, Base.size_to_strides(1, dimensions...), empty)
end

function make_cache_map(kind, ::Type{V}, config::ArrayMapConfig;
    concurrent::Bool=false) where V
    domain = map(axis_domain, config.domain)
    any(isunbound, domain) && throw(ArgumentError("ArrayMap requires a finite domain"))
    if concurrent
        return concurrent_array_map(kind, V, map(length, domain)), domain
    end
    E, empty = map_entry(kind, V, Int, config)
    return ArrayMap(E, map(length, domain), empty), domain
end

function ConcurrentArrayMap(::Type{T}, shape) where T
    dimensions = Tuple(shape)
    slots = empty_map(T, dimensions, EMPTY, Val(true))
    return ConcurrentArrayMap{T,length(dimensions)}(memoryref(slots), dimensions,
        Base.size_to_strides(1, dimensions...), EMPTY)
end

@inline slot_address(::DenseMap, key::Int) = key

@inline function slot_address(map::DenseMap{T,N}, key::NTuple{N,Int}) where {T,N}
    address = N == 0 ? 1 : key[1]
    @inbounds for i in 2:N
        address += (key[i] - 1) * map.strides[i]
    end
    return address
end

@inline map_routing(::DenseMap, key) = nothing
@inline map_slot(map::DenseMap, key, routing) = @inbounds memoryref(map.reference, slot_address(map, key))
@inline map_reserve!(map::DenseMap, key, routing) = map_slot(map, key, routing)
@inline reserve!(map::DenseMap, key, routing, slot, hint) = slot
map_retain!(::DenseMap, slot) = nothing

@inline delete_key!(map::DenseMap, key) = slot_publish!(map_slot(map, key, nothing), map.empty)

Base.@propagate_inbounds function cache_key(cache::ArrayBacked, key::Tuple)
    @boundscheck checkbounds(cache, key...)
    return slot_address(cache.map, map((i, axis) -> i - first(axis) + 1, key, axes(cache)))
end

Base.@propagate_inbounds function map_lookup(cache::ArrayBacked, indices::Tuple)
    @boundscheck checkbounds(cache, indices...)
    key = slot_address(cache.map, map((i, axis) -> i - first(axis) + 1, indices, axes(cache)))
    return key, nothing, map_slot(cache.map, key, nothing), nothing
end

Base.size(cache::ArrayBacked) = cache.map.shape

Base.axes(cache::ArrayBacked) = cache.domain


function concurrent_array_map(kind, ::Type{V}, shape) where V
    S = concurrent_value_type(V)
    if kind === MemoizedEntry
        return ConcurrentArrayMap(Union{Empty,Generating,GenerationFailure,S}, shape)
    end
    Stored = SharedLRUEntry{S,Int}
    return ConcurrentArrayMap(Union{Empty,Generating,GenerationFailure,GenerateWait{S},Stored}, shape)
end

@inline remove_mapping!(map::ConcurrentArrayMap, entry) =
    slot_publish!(map_slot(map, (@atomic :acquire entry.key), nothing), EMPTY)
queue_key(::ConcurrentArrayMap) = 0


@inline assignment_slot(::ConcurrentArrayMap,key,routing,reference,hint) = reference
