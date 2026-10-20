# Readers retain stable records across partition growth. Insertion, rehashing
# and removal lock only the key's partition; generation and reads take no lock.
mutable struct ConcurrentHashRecord{T,N}
    const key::NTuple{N,Int}
    const hash::UInt
    @atomic value::T
    address::Int
end

mutable struct ConcurrentHashPartition{T,N}
    const table::GenericMemoryRef{:atomic,AtomicMemory{Union{HashMarker,ConcurrentHashRecord{T,N}}},Core.CPU}
    count::Int
    used::Int
    const structural::ReentrantLock
end

struct ConcurrentDictMap{T,N,D,R} <: SparseMap{T,N}
    # Readers load the selected table directly, without visiting writer metadata.
    tables::AtomicMemory{AtomicMemory{Union{HashMarker,ConcurrentHashRecord{T,N}}}}
    partitions::Memory{ConcurrentHashPartition{T,N}}
    empty::Empty
    domain::D
    # Only the owner of an entry's editing state accesses its registry slot.
    registry::R
end

concurrent_hash_table(::Type{T}, ::Val{N}, count::Int) where {T,N} =
    empty_map(Union{HashMarker,ConcurrentHashRecord{T,N}}, (count,), HASH_EMPTY, Val(true))

function concurrent_dict_map(kind, ::Type{V}, domain::NTuple{N,Any}; capacity::Int=0) where {V,N}
    S = concurrent_value_type(V)
    if kind === MemoizedEntry
        T = Union{Empty,Generating,GenerationFailure,S}
        registry = nothing
    else
        T = Union{Empty,Generating,GenerationFailure,GenerateWait{S},Relocated,SharedLRUEntry{S,NTuple{N,Int}}}
        registry = Memory{ConcurrentHashRecord{T,N}}(undef, capacity)
    end
    count = nextpow(2, 4Threads.nthreads(:default))
    capacity > 0 && (count = min(count, prevpow(2, capacity)))
    tables = AtomicMemory{AtomicMemory{Union{HashMarker,ConcurrentHashRecord{T,N}}}}(undef, count)
    partitions = Memory{ConcurrentHashPartition{T,N}}(undef, count)
    for i in eachindex(partitions)
        reference = memoryref(tables, i)
        slot_publish!(reference, concurrent_hash_table(T, Val(N), max(4, 64 ÷ count)))
        partitions[i] = ConcurrentHashPartition{T,N}(reference, 0, 0, ReentrantLock())
    end
    return ConcurrentDictMap{T,N,typeof(domain),typeof(registry)}(tables, partitions, EMPTY, domain, registry)
end

@inline map_routing(::ConcurrentDictMap, key) = hash(key)
@inline hash_partition_index(store::ConcurrentDictMap, hashed::UInt) =
    Int((hashed >> (4sizeof(UInt))) & UInt(length(store.tables) - 1)) + 1
@inline hash_partition(store::ConcurrentDictMap, hashed::UInt) =
    @inbounds store.partitions[hash_partition_index(store, hashed)]
Base.@propagate_inbounds function map_lookup(cache::AbstractMemoizedArray{V,N,M},
    key::NTuple{N,Int}) where {V,N,M<:ConcurrentDictMap}
    @boundscheck checkbounds(cache, key...)
    routing = map_routing(cache.map, key)
    slot, hint = prepare(cache.map, key, routing)
    return key, routing, slot, hint
end

@inline slot_entry(slot::ConcurrentHashRecord) = @atomic :acquire slot.value
@inline slot_publish!(slot::ConcurrentHashRecord, value) = (@atomic :release slot.value = value)
@inline claim_slot!(slot::ConcurrentHashRecord, expected, desired) =
    @atomicreplace :acquire_release :acquire slot.value expected => desired
map_retain!(::ConcurrentDictMap, slot::ConcurrentHashRecord) = nothing

@inline function hash_find(table::AtomicMemory, key::NTuple{N,Int}, hashed::UInt) where N
    mask = length(table) - 1
    address = ((hashed % Int) & mask) + 1
    vacant = 0
    while true
        entry = @inbounds @atomic :acquire table[address]
        if entry isa HashMarker
            if entry === HASH_EMPTY
                return nothing, vacant == 0 ? address : vacant
            end
            if vacant == 0
                vacant = address
            end
        elseif (N == 2 || entry.hash == hashed) && entry.key == key
            return entry, address
        end
        address = (address & mask) + 1
    end
end

@inline function map_slot(store::ConcurrentDictMap, key, hashed::UInt)
    table = @inbounds @atomic :acquire store.tables[hash_partition_index(store, hashed)]
    return first(hash_find(table, key, hashed))
end

@inline prepare(store::ConcurrentDictMap, key, hashed::UInt) = (map_slot(store, key, hashed), nothing)
@inline reserve!(store::ConcurrentDictMap, key, hashed, record, hint) = map_reserve!(store, key, hashed)

# The structural lock protects bucket addresses as well as the current table.
# Old snapshots stay valid until their readers release the corresponding refs.
function hash_rebuild!(store::ConcurrentHashPartition{T,N}, count::Int) where {T,N}
    old = slot_entry(store.table)
    table = concurrent_hash_table(T, Val(N), count)
    for i in eachindex(old)
        record = @inbounds @atomic :acquire old[i]
        if record isa ConcurrentHashRecord
            _, address = hash_find(table, record.key, record.hash)
            record.address = address
            @inbounds @atomic :release table[address] = record
        end
    end
    store.used = store.count
    slot_publish!(store.table, table)
    return table
end

@noinline function map_reserve!(store::ConcurrentDictMap{T,N}, key, hashed::UInt) where {T,N}
    partition = hash_partition(store, hashed)
    lock(partition.structural) do
        table = slot_entry(partition.table)
        record, address = hash_find(table, key, hashed)
        if record !== nothing
            return record
        end
        count = length(table)
        previous = @inbounds @atomic :acquire table[address]
        if previous === HASH_EMPTY && partition.used + 1 > 3(count ÷ 4)
            if partition.count + 1 > 3(count ÷ 4)
                count = Base.checked_mul(count, 4)
            end
            table = hash_rebuild!(partition, count)
            _, address = hash_find(table, key, hashed)
            previous = HASH_EMPTY
        end
        record = ConcurrentHashRecord{T,N}(key, hashed, EMPTY, address)
        @inbounds @atomic :release table[address] = record
        partition.count += 1
        if previous === HASH_EMPTY
            partition.used += 1
        end
        return record
    end
end

# A cold LRU request carries the map so a deleted record can be followed to a
# replacement. The detached record remains RELOCATED and cannot be resurrected.
struct DictSlot{T,N,D,R}
    map::ConcurrentDictMap{T,N,D,R}
    record::ConcurrentHashRecord{T,N}
end

@noinline function relocated_slot(slot::DictSlot)
    record = map_reserve!(slot.map, slot.record.key, slot.record.hash)
    return DictSlot(slot.map, record)
end

@inline function slot_entry(slot::DictSlot)
    stored = slot_entry(slot.record)
    while stored === RELOCATED
        slot = relocated_slot(slot)
        stored = slot_entry(slot.record)
    end
    return stored
end

@inline function claim_slot!(slot::DictSlot, expected, desired)
    claimed = claim_slot!(slot.record, expected, desired)
    if claimed.old === RELOCATED
        return relocated_claim!(slot, expected, desired)
    end
    return claimed
end

@noinline function relocated_claim!(slot::DictSlot, expected, desired)
    while true
        slot = relocated_slot(slot)
        claimed = claim_slot!(slot.record, expected, desired)
        if claimed.old !== RELOCATED
            return claimed
        end
    end
end

@inline function slot_exchange!(slot::DictSlot, stored, previous)
    while true
        claimed = claim_slot!(slot.record, previous, stored)
        if claimed.success
            if stored isa SharedLRUEntry
                # Publication precedes releasing the exclusively held entry
                # state; the next owner then acquires this registry assignment.
                @inbounds slot.map.registry[stored.id] = slot.record
            end
            return previous
        elseif claimed.old === RELOCATED
            slot = relocated_slot(slot)
        else
            previous = claimed.old
        end
    end
end

# The caller exclusively owns entry.state in editing mode. No other caller can
# replace this mapping or registry slot until that ownership is released.
function remove_mapping!(store::ConcurrentDictMap, entry)
    record = @inbounds store.registry[entry.id]
    partition = hash_partition(store, record.hash)
    lock(partition.structural) do
        table = slot_entry(partition.table)
        slot_publish!(record, RELOCATED)
        @inbounds @atomic :release table[record.address] = HASH_DELETED
        partition.count -= 1
    end
    return nothing
end

queue_key(::ConcurrentDictMap{T,N}) where {T,N} = ntuple(_ -> 0, Val(N))


function assignment_slot(map::ConcurrentDictMap,key,hashed,record,hint)
    record === nothing && (record = map_reserve!(map,key,hashed))
    return DictSlot(map,record)
end
