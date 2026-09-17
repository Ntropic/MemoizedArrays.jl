# Dictionary storage: map coordinate tuples to stable records using a hash table.
# Resizing preserves records; deletion leaves tombstones for probing.

@enum HashMarker::UInt8 HASH_EMPTY HASH_DELETED

# Records stay at the same location when the probe table grows.
mutable struct HashRecord{T,N}
    const key::NTuple{N,Int}
    const hash::UInt
    value::T
end

mutable struct DictMap{T,N,D,E} <: SparseMap{T,N}
    table::Memory{Union{HashMarker,HashRecord{T,N}}}
    count::Int
    used::Int
    insertions::UInt
    const empty::E
    const domain::D
end

hash_table(::Type{T}, ::Val{N}, count::Int) where {T,N} = empty_map(Union{HashMarker,HashRecord{T,N}}, (count,), HASH_EMPTY)
DictMap(::Type{T}, ::Val{N}, empty) where {T,N} = DictMap(T, int_axes(N), empty)

DictMap(::Type{T}, domain::NTuple{N,Any}, empty) where {T,N} = DictMap{T,N,typeof(domain),typeof(empty)}(hash_table(T, Val(N), 64), 0, 0, UInt(0), empty, domain)

@inline store_hash(::DictMap, key) = hash(key)
@inline slot_entry(slot::HashRecord) = slot.value
@inline slot_publish!(slot::HashRecord, value) = (slot.value = value)
store_retain!(::DictMap, slot::HashRecord) = nothing

# Find a record or the first reusable slot in its probe chain.
@inline function hash_find(table::Memory, key, hashed::UInt)
    # Power-of-two table sizes keep masked signed indices within the table.
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
        elseif entry.hash == hashed && entry.key == key
            return entry, address
        end
        address = (address & mask) + 1
    end
end

@inline function store_lookup(store::DictMap, key, hashed::UInt)
    table = store.table
    return first(hash_find(table, key, hashed))
end

@inline function prepare(store::DictMap, key, hashed::UInt)
    table = store.table
    record, address = hash_find(table, key, hashed)
    return record, (table, address, store.insertions)
end

# Insertions can invalidate a saved position; eviction only leaves tombstones.
@noinline function reserve!(store::DictMap, key, hashed, record, hint)
    table, address, insertions = hint
    if table !== store.table || insertions != store.insertions ||
        (record !== nothing && (@inbounds table[address]) !== record)
        return store_reserve!(store, key, hashed)
    end
    return hashreserve!(store, key, hashed, table, record, address)
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
    store.used = store.count
    store.table = table
    return table
end

# Reserve a stable missing record.
function store_reserve!(store::DictMap{T,N}, key, hashed::UInt) where {T,N}
    table = store.table
    entry, address = hash_find(table, key, hashed)
    return hashreserve!(store, key, hashed, table, entry, address)
end

@inline function hashreserve!(store::DictMap{T,N}, key, hashed, table, entry, address) where {T,N}
    if entry !== nothing
        return entry
    end
    count = length(table)
    previous = @inbounds table[address]
    # Reusing a tombstone does not increase occupied table space.
    if previous === HASH_EMPTY && store.used + 1 > 3(count ÷ 4)
        if store.count + 1 > 3(count ÷ 4)
            count = Base.checked_mul(count, 4)
        end
        table = hash_rebuild!(store, count)
        _, address = hash_find(table, key, hashed)
        previous = HASH_EMPTY
    end
    record = HashRecord{T,N}(key, hashed, store.empty)
    @inbounds table[address] = record
    store.count += 1
    store.insertions += UInt(1)
    if previous === HASH_EMPTY
        store.used += 1
    end
    return record
end

# Remove only the current record, preserving probe chains with a tombstone.
function store_delete!(store::DictMap, key, slot::HashRecord)
    table = store.table
    entry, address = hash_find(table, key, slot.hash)
    if entry !== slot
        error("attempted to remove a detached cache record")
    end
    return hash_delete!(store, table, slot, address)
end

# Evict a record with one probe.
@inline function delete_key!(store::DictMap, key)
    table = store.table
    record, address = hash_find(table, key, hash(key))
    return hash_delete!(store, table, record, address)
end

@inline function hash_delete!(store, table, record, address)
    slot_publish!(record, store.empty)
    @inbounds table[address] = HASH_DELETED
    store.count -= 1
    return nothing
end
