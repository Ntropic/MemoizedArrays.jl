struct _HashMarker
    deleted::Bool
end
const _HASH_EMPTY = _HashMarker(false)
const _HASH_DELETED = _HashMarker(true)

# Records stay at the same location when the probe table grows.
mutable struct _HashRecord{T,N}
    const key::NTuple{N,Int}
    const hash::UInt
    @atomic value::T
end

struct _HashTable{T,N}
    slots::AtomicMemory{Union{_HashMarker,_HashRecord{T,N}}}
end

mutable struct DictMap{T,N}
    @atomic table::_HashTable{T,N}
    count::Int
    used::Int
    empty::Empty
    const guard::ReentrantLock
end

# Allocate probe slots without allocating key records.
function _hash_table(::Type{T}, ::Val{N}, count::Int) where {T,N}
    slots = _empty_map(Union{_HashMarker,_HashRecord{T,N}}, (count,), _HASH_EMPTY)
    return _HashTable{T,N}(slots)
end

DictMap(::Type{T}, ::Val{N}, empty::Empty) where {T,N} =
    DictMap{T,N}(_hash_table(T, Val(N), 64), 0, 0, empty, ReentrantLock())

@inline _store_hash(::DictMap, key) = hash(key)
@inline _slot_entry(slot::_HashRecord) = @atomic :acquire slot.value
@inline _slot_publish!(slot::_HashRecord, value) = (@atomic :release slot.value = value)
@inline _slot_claim!(slot::_HashRecord, empty, pending) =
    @atomicreplace :acquire_release :acquire slot.value empty => pending
_store_retain!(::DictMap, slot::_HashRecord) = nothing

# Probe one captured table; writers publish complete records atomically.
@inline function _hash_find(table::_HashTable, key, hashed::UInt)
    mask = UInt(length(table.slots) - 1)
    address = Int(hashed & mask) + 1
    vacant = 0
    while true
        entry = @inbounds @atomic :acquire table.slots[address]
        if entry isa _HashMarker
            if !entry.deleted
                return nothing, vacant == 0 ? address : vacant
            end
            if vacant == 0
                vacant = address
            end
        elseif entry.hash == hashed && entry.key == key
            return entry, address
        end
        address = Int(UInt(address) & mask) + 1
    end
end

@inline function _store_lookup(store::DictMap, key, hashed::UInt)
    table = @atomic :acquire store.table
    return first(_hash_find(table, key, hashed))
end

# Rebuild under the structural lock; old readers retain their table and shared records.
function _hash_rebuild!(store::DictMap{T,N}, count::Int) where {T,N}
    old = @atomic :acquire store.table
    table = _hash_table(T, Val(N), count)
    for i in eachindex(old.slots)
        entry = @inbounds @atomic :acquire old.slots[i]
        if entry isa _HashRecord
            _, address = _hash_find(table, entry.key, entry.hash)
            @inbounds @atomic :release table.slots[address] = entry
        end
    end
    store.used = store.count
    @atomic :release store.table = table
    return table
end

# Reserve a stable missing record; caller holds the structural lock.
function _store_reserve!(store::DictMap{T,N}, key, hashed::UInt) where {T,N}
    table = @atomic :acquire store.table
    entry, address = _hash_find(table, key, hashed)
    if entry !== nothing
        return entry
    end
    count = length(table.slots)
    if store.used + 1 > 3(count ÷ 4)
        if store.count + 1 > 3(count ÷ 4)
            count = Base.checked_mul(count, 2)
        end
        table = _hash_rebuild!(store, count)
        _, address = _hash_find(table, key, hashed)
    end
    previous = @inbounds @atomic :acquire table.slots[address]
    record = _HashRecord{T,N}(key, hashed, store.empty)
    @inbounds @atomic :release table.slots[address] = record
    store.count += 1
    if previous === _HASH_EMPTY
        store.used += 1
    end
    return record
end

# Remove only the current record, preserving probe chains with a tombstone.
function _store_delete!(store::DictMap, key, slot::_HashRecord)
    table = @atomic :acquire store.table
    entry, address = _hash_find(table, key, slot.hash)
    if entry !== slot
        error("attempted to remove a detached cache record")
    end
    return _hash_delete!(store, table, slot, address)
end

# Eviction needs one probe; captured failure slots still use the identity check above.
function _delete_key!(store::DictMap, key)
    table = @atomic :acquire store.table
    record, address = _hash_find(table, key, hash(key))
    return _hash_delete!(store, table, record, address)
end

@inline function _hash_delete!(store, table, record, address)
    _slot_publish!(record, store.empty)
    @inbounds @atomic :release table.slots[address] = _HASH_DELETED
    store.count -= 1
    return nothing
end

_store_reserve!(store::DictMap, key, hashed::UInt, ::Val) = _store_reserve!(store, key, hashed)
