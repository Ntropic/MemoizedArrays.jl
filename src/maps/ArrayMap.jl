# Array storage: fixed slots with direct reads, writes and resets.
# Slot addresses survive eviction; capacity and recency belong to the cache.

struct ArrayMap{T}
    slots::Memory{T}
end

ArrayMap(::Type{T}, shape) where T = ArrayMap(empty_map(T, shape))

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
@inline slot_entry(slot::MemoryRef) = @inbounds slot[]
@inline slot_publish!(slot::MemoryRef, value) = (@inbounds slot[] = value)
@inline store_hash(::ArrayMap, key) = nothing
@inline store_lookup(map::ArrayMap, key::Int, routing) = @inbounds memoryref(map.slots, key)
@inline store_reserve!(map::ArrayMap, key, routing) = store_lookup(map, key, routing)
@inline prepare(map, key, routing) = (store_lookup(map, key, routing), nothing)
@inline preparedentry(slot, hint) = slot_entry(slot)
@inline reserve!(map::ArrayMap, key, routing, slot, hint) = slot
store_retain!(::ArrayMap, slot) = nothing
@inline store_delete!(::ArrayMap, key, slot) = slot_publish!(slot, EMPTY)

@inline delete_key!(map::ArrayMap, key) = (@inbounds map.slots[key] = EMPTY)
