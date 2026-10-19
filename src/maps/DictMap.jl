struct DictMapConfig{V,D} <: MapConfig{V}
    domain::D
end

# Dictionary storage: map coordinate tuples to stable records using a hash table.
# Resizing preserves records. Serial Memoized storage never evicts them.

function make_cache_map(kind, ::Type{V}, config::DictMapConfig;
    concurrent::Bool=false, capacity::Int=0) where V
    domain = map(axis -> AxisExtent(first(axis), last(axis)), map(axis_domain, config.domain))
    if concurrent
        return concurrent_dict_map(kind, V, domain; capacity), domain
    end
    E, empty = map_entry(kind, V, NTuple{length(domain),Int}, config)
    return DictMap(E, domain, empty), domain
end

@enum HashMarker::UInt8 HASH_EMPTY HASH_DELETED

# Records stay at the same location when the probe table grows.
mutable struct HashRecord{T,N}
    const key::NTuple{N,Int}
    const hash::UInt
    value::T
end

"""
    DictMap{T}(domain)

Hash-based sparse lookup for finite or unbounded domains.
"""
mutable struct DictMap{T,N,D,E} <: SparseMap{T,N}
    table::Memory{Union{HashMarker,HashRecord{T,N}}}
    count::Int
    const empty::E
    const domain::D
end

DictMap{V}(domain::VT) where V = DictMapConfig{V,typeof(Tuple(domain))}(Tuple(domain))

hash_table(::Type{T}, ::Val{N}, count::Int) where {T,N} = empty_map(Union{HashMarker,HashRecord{T,N}}, (count,), HASH_EMPTY)

DictMap(::Type{T}, domain::NTuple{N,Any}, empty) where {T,N} = DictMap{T,N,typeof(domain),typeof(empty)}(hash_table(T, Val(N), 64), 0, empty, domain)

@inline map_routing(::DictMap, key) = hash(key)
Base.@propagate_inbounds function map_lookup(cache::AbstractMemoizedArray{V,N,M}, key::NTuple{N,Int}) where {V,N,M<:DictMap}
    @boundscheck checkbounds(cache, key...)
    routing = map_routing(cache.map, key)
    slot, hint = prepare(cache.map, key, routing)
    return key, routing, slot, hint
end

@inline slot_entry(slot::HashRecord) = slot.value
@inline slot_publish!(slot::HashRecord, value) = (slot.value = value)
map_retain!(::DictMap, slot::HashRecord) = nothing

# Find a record or the first empty slot in its probe chain. Two-coordinate
# keys compare directly; other dimensions use the cached hash as a prefilter.
@inline function hash_find(table::Memory, key::NTuple{N,Int}, hashed::UInt) where {N}
    # Power-of-two table sizes keep masked signed indices within the table.
    mask = length(table) - 1
    address = ((hashed % Int) & mask) + 1
    while true
        entry = @inbounds table[address]
        if entry isa HashMarker
            return nothing, address
        elseif (N == 2 || entry.hash == hashed) && entry.key == key
            return entry, address
        end
        address = (address & mask) + 1
    end
end

@inline function map_slot(store::DictMap, key, hashed::UInt)
    table = store.table
    return first(hash_find(table, key, hashed))
end

@inline function prepare(store::DictMap, key, hashed::UInt)
    table = store.table
    record, address = hash_find(table, key, hashed)
    return record, (table, address, store.count)
end

# Nested insertions can invalidate a saved position.
@noinline function reserve!(store::DictMap, key, hashed, record, hint, initial=store.empty)
    table, address, count = hint
    if table !== store.table || count != store.count
        return map_reserve!(store, key, hashed, initial)
    end
    return hashreserve!(store, key, hashed, table, record, address, initial)
end

# Rebuild the probe table, retaining its records.
function hash_rebuild!(store::DictMap{T,N}, count::Int) where {T,N}
    old = store.table
    table = hash_table(T, Val(N), count)
    for i in eachindex(old)
        entry = @inbounds old[i]
        if entry isa HashRecord
            _, address = hash_find(table, entry.key, entry.hash)
            @inbounds table[address] = entry
        end
    end
    store.table = table
    return table
end

# Reserve a stable missing record.
function map_reserve!(store::DictMap{T,N}, key, hashed::UInt, initial=store.empty) where {T,N}
    table = store.table
    entry, address = hash_find(table, key, hashed)
    return hashreserve!(store, key, hashed, table, entry, address, initial)
end

@inline function hashreserve!(store::DictMap{T,N}, key, hashed, table, entry, address,
    initial=store.empty) where {T,N}
    if entry !== nothing
        return entry
    end
    count = length(table)
    if store.count + 1 > 3(count ÷ 4)
        table = hash_rebuild!(store, Base.checked_mul(count, 4))
        _, address = hash_find(table, key, hashed)
    end
    record = HashRecord{T,N}(key, hashed, initial)
    @inbounds table[address] = record
    store.count += 1
    return record
end
