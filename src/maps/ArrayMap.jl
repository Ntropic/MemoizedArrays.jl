# Array storage: fixed slots with direct reads, writes and resets.
# Slot addresses survive eviction; capacity and recency belong to the cache.

struct ArrayMap{T,N}
    reference::MemoryRef{T}
    shape::NTuple{N,Int}
    strides::NTuple{N,Int}
    empty::T
end

function ArrayMap(::Type{T}, shape, empty=EMPTY) where T
    dimensions = Tuple(shape)
    slots = empty_map(T, dimensions, empty)
    return ArrayMap{T,length(dimensions)}(memoryref(slots), dimensions, Base.size_to_strides(1, dimensions...), empty)
end

@inline slotaddress(::ArrayMap, key::Int) = key

@inline function slotaddress(map::ArrayMap{T,N}, key::NTuple{N,Int}) where {T,N}
    address = N == 0 ? 1 : key[1]
    @inbounds for i in 2:N
        address += (key[i] - 1) * map.strides[i]
    end
    return address
end

# Allocate and initialize slots for maps and tree blocks.
function empty_map(::Type{T}, shape, empty=EMPTY) where T
    count = 1
    for dimension in shape
        if dimension < 0
            throw(ArgumentError("dimensions must be nonnegative"))
        end
        count = Base.checked_mul(count, dimension)
    end
    map = Memory{T}(undef, count)
    for address in eachindex(map)
        map[address] = empty
    end
    return map
end

# Slot references share read/write operations with dictionary records and tree nodes.
@inline slot_entry(slot::MemoryRef{T}) where T = @inbounds slot[]
@inline slot_publish!(slot::MemoryRef{T}, value) where T = (@inbounds slot[] = value)
@inline store_hash(::ArrayMap, key) = nothing
@inline store_lookup(map::ArrayMap, key, routing) = @inbounds memoryref(map.reference, slotaddress(map, key))
@inline store_reserve!(map::ArrayMap, key, routing) = store_lookup(map, key, routing)
@inline prepare(map, key, routing) = (store_lookup(map, key, routing), nothing)
@inline preparedentry(slot, hint) = slot_entry(slot)
@inline reserve!(map::ArrayMap, key, routing, slot, hint) = slot
store_retain!(::ArrayMap, slot) = nothing
@inline store_delete!(map::ArrayMap, key, slot) = slot_publish!(slot, map.empty)

@inline delete_key!(map::ArrayMap, key) = slot_publish!(store_lookup(map, key, nothing), map.empty)
