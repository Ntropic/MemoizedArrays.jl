# Serial dictionary entries carry their current bucket, so eviction needs no probe.
struct DictLocation{N}
    coordinates::NTuple{N,Int}
    hash::UInt
    address::Int
end

mutable struct LRUDictMap{E,N,D} <: SparseMap{E,N}
    table::Memory{Union{HashMarker,E}}
    count::Int
    used::Int
    insertions::UInt
    const empty::E
    const domain::D
end

lru_hash_table(::Type{E}, count::Int) where E = empty_map(Union{HashMarker,E}, (count,), HASH_EMPTY)

function make_cache_map(::Type{SerialLRUEntry}, ::Type{V}, config::DictMapConfig;
    concurrent::Bool=false) where V
    if concurrent
        throw(ArgumentError("serial dictionary entries do not support concurrent=true"))
    end
    domain = map(axis -> AxisExtent(first(axis), last(axis)), map(axis_domain, config.domain))
    E, empty = map_entry(SerialLRUEntry, V, DictLocation{length(domain)}, config)
    store = LRUDictMap{E,length(domain),typeof(domain)}(lru_hash_table(E, 64), 0, 0, UInt(0), empty, domain)
    return store, domain
end

@inline map_routing(::LRUDictMap, key) = hash(key)
@inline slot_entry(slot::SerialLRUEntry) = slot

Base.@propagate_inbounds function map_lookup(cache::AbstractMemoizedArray{V,N,M}, key::NTuple{N,Int}) where {V,N,M<:LRUDictMap}
    @boundscheck checkbounds(cache, key...)
    hashed = map_routing(cache.map, key)
    entry, hint = prepare(cache.map, key, hashed)
    return key, hashed, entry, hint
end

@inline function hash_find(table::Memory{Union{HashMarker,SerialLRUEntry{V,DictLocation{N}}}},
    key::NTuple{N,Int}, hashed::UInt) where {V,N}
    mask = length(table) - 1
    address = ((hashed % Int) & mask) + 1
    vacant = 0
    while true
        entry = @inbounds table[address]
        if entry isa HashMarker
            if entry === HASH_EMPTY
                return nothing, vacant == 0 ? address : vacant
            end
            if vacant == 0
                vacant = address
            end
        else
            location = entry.key
            if (N == 2 || location.hash == hashed) && location.coordinates == key
                return entry, address
            end
        end
        address = (address & mask) + 1
    end
end

@inline map_slot(store::LRUDictMap, key, hashed::UInt) = first(hash_find(store.table, key, hashed))

@inline function prepare(store::LRUDictMap, key, hashed::UInt)
    table = store.table
    entry, address = hash_find(table, key, hashed)
    return entry, (table, address, store.insertions)
end

function hash_rebuild!(store::LRUDictMap{E}, count::Int) where E
    old = store.table
    table = lru_hash_table(E, count)
    for i in eachindex(old)
        entry = @inbounds old[i]
        if entry isa E
            location = entry.key
            _, address = hash_find(table, location.coordinates, location.hash)
            entry.key = DictLocation(location.coordinates, location.hash, address)
            @inbounds table[address] = entry
        end
    end
    store.used = store.count
    store.table = table
    return table
end

@inline function delete_key!(store::LRUDictMap, location::DictLocation)
    @inbounds store.table[location.address] = HASH_DELETED
    store.count -= 1
    return nothing
end

# Generation and conversion finish before admission. Nested requests may have
# inserted this key, changed the saved bucket, or changed the current victim.
# Separate arguments keep the caller's lookup tuple from escaping this cold call.
@noinline function dict_lru_insert!(cache, key, hashed, table, address, insertions, value)
    store = cache.map
    entry = nothing
    if table !== store.table || insertions != store.insertions
        table = store.table
        entry, address = hash_find(table, key, hashed)
    end
    if entry !== nothing
        entry.value = value
        queue_touch!(cache.eviction, entry.id)
        return value
    end
    stored = serial_insert!(cache, cache.eviction, DictLocation(key, hashed, address), value)
    count = length(table)
    previous = @inbounds table[address]
    # Eviction already reduced the resident count; replacing a full cache must
    # not grow the table to make room for a temporary capacity-plus-one entry.
    if previous === HASH_EMPTY && store.used + 1 > 3(count ÷ 4)
        if store.count + 1 > 3(count ÷ 4)
            count = Base.checked_mul(count, 4)
        end
        table = hash_rebuild!(store, count)
        _, address = hash_find(table, key, hashed)
        previous = HASH_EMPTY
    end
    stored.key = DictLocation(key, hashed, address)
    @inbounds table[address] = stored
    store.count += 1
    store.insertions += UInt(1)
    if previous === HASH_EMPTY
        store.used += 1
    end
    return value
end
