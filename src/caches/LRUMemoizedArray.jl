# Bounded cache construction and access. Queue code owns order and entry reuse.
"""
    LRUMemoizedArray(generator, map; capacity, concurrent=false, queues=nothing)

Generate values on demand with bounded retention. Capacity must be positive and fit UInt32 entry IDs.
Serial access uses one exact LRU queue. Concurrent access uses queue handles;
queues default to `Threads.nthreads(:default)` and their positive count must divide capacity.
Ordering is exact within each queue and approximate across queues.
Manual assignment creates a pinned entry outside the pool and capacity; pinned reads
skip queue updates. Concurrent assignment uses a queue handle.
"""
struct LRUMemoizedArray{V,N,M,G,D,Q,Concurrent} <: AbstractMemoizedArray{V,N,M}
    map::M
    generator::G
    domain::D
    eviction::Q
end

Base.@constprop :aggressive function LRUMemoizedArray(generator, config::MapConfig; capacity::Int,
    concurrent::Bool=false, queues::Union{Nothing,Int}=nothing)
    return LRUMemoizedArray(generator, config, Val(concurrent); capacity, queues)
end

function LRUMemoizedArray(generator, config::MapConfig{V}, ::Val{Concurrent}; capacity::Int,
    queues::Union{Nothing,Int}=nothing) where {V,Concurrent}
    if capacity < 1
        throw(ArgumentError("capacity must be positive"))
    end
    if capacity > typemax(UInt32)
        throw(ArgumentError("LRU capacity must fit UInt32 entry IDs"))
    end
    if Concurrent
        count = queues === nothing ? Threads.nthreads(:default) : queues
        if count < 1 || capacity < count || capacity % count != 0
            throw(ArgumentError("capacity must be divisible by the positive queue count and at least as large; supply queues explicitly if the default is unsuitable"))
        end
        if !(config isa Union{ArrayMapConfig,ExpandingArrayMapConfig,DictMapConfig,RadixMapConfig,SubtreeRadixMapConfig})
            throw(ArgumentError("unsupported concurrent LRUMemoizedArray config"))
        end
        map, domain = if config isa Union{DictMapConfig,RadixMapConfig,SubtreeRadixMapConfig}
            make_cache_map(SharedLRUEntry, V, config; concurrent=true, capacity)
        else
            make_cache_map(SharedLRUEntry, V, config; concurrent=true)
        end
        quota = capacity ÷ count
        key = queue_key(map)
        E = SharedLRUEntry{concurrent_value_type(V),typeof(key)}
        eviction = MultiQueue([PrivateLRUQueue(E, key, i, quota, capacity, count) for i in 1:count])
    else
        if queues !== nothing && queues != 1
            throw(ArgumentError("serial LRUMemoizedArray uses exactly one queue"))
        end
        map, domain = make_cache_map(SerialLRUEntry, V, config)
        eviction = SerialLRUQueue(typeof(map.empty), capacity)
    end
    return LRUMemoizedArray{V,length(domain),typeof(map),typeof(generator),typeof(domain),typeof(eviction),Concurrent}(
        map, generator, domain, eviction)
end

resident_count(cache::LRUMemoizedArray) = resident_count(cache.eviction)
resident_count(queue::SerialLRUQueue) = queue.count

@inline function cache_get(cache::LRUMemoizedArray{V,N,M}, lookup, indices::Tuple) where {V,N,M<:Union{ArrayMap,ExpandingArrayMap,TreeMap}}
    key, routing, slot, hint = lookup
    if slot !== nothing
        stored = prepared_entry(slot, hint)
        if stored !== cache.map.empty
            stored.explicit && return stored.value
            queue_touch!(cache.eviction, stored.id)
            return stored.value
        end
    end
    generated = cache.generator(indices...)
    value = generated isa V ? generated : (convert(V, generated)::V)
    # Nested requests may grow the map; reserve before eviction can change it.
    slot = reserve!(cache.map, key, routing, slot, hint)
    stored = slot_entry(slot)
    if stored !== cache.map.empty
        # A nested generator or conversion may have completed this key already.
        stored.explicit && return stored.value
        stored.value = value
        queue_touch!(cache.eviction, stored.id)
        return value
    end
    # Keep a tree destination alive when eviction empties that leaf.
    map_retain!(cache.map, slot)
    stored = serial_insert!(cache, cache.eviction, key, value)
    slot_publish!(slot, stored)
    return value
end

@inline function cache_get(cache::LRUMemoizedArray{V,N,M}, lookup, indices::Tuple) where {V,N,M<:LRUDictMap}
    key, hashed, stored, hint = lookup
    if stored !== nothing
        stored.explicit && return stored.value
        queue_touch!(cache.eviction, stored.id)
        return stored.value
    end
    generated = cache.generator(indices...)
    value = generated isa V ? generated : (convert(V, generated)::V)
    return dict_lru_insert!(cache, key, hashed, hint..., value)
end

struct QueueHandle{V,N,M<:Union{ConcurrentArrayMap,ConcurrentExpandingArrayMap,ConcurrentDictMap,ConcurrentTreeMap},C,Q} <: AbstractMemoizedArray{V,N,M}
    cache::C
    queue::Q
end

"""
    queue_handle(cache::LRUMemoizedArray, i)

Select concurrent queue `i`. Each queue must have only one active caller;
different queues can be used concurrently. Selection does not acquire a lock.
"""
function queue_handle(cache::LRUMemoizedArray{V,N,M,G,D,Q,true}, i::Integer) where {V,N,M,G,D,Q}
    checkbounds(cache.eviction.queues, i)
    queue = @inbounds cache.eviction.queues[i]
    return QueueHandle{V,N,M,typeof(cache),typeof(queue)}(cache, queue)
end

"""
    queue_handles(cache::LRUMemoizedArray)

Return all concurrent queue handles as a tuple.
"""
function queue_handles(cache::LRUMemoizedArray{V,N,M,G,D,Q,true}) where {V,N,M,G,D,Q}
    return ntuple(i -> queue_handle(cache, i), length(cache.eviction.queues))
end

"""
    with_queue(f, cache::LRUMemoizedArray)

Call `f(handle)` with an available concurrent queue; release it on return or exception.
Do not use explicit handles concurrently with automatic selection on the same cache.
"""
function with_queue(f::F, cache::LRUMemoizedArray{V,N,M,G,D,Q,true}) where {F,V,N,M,G,D,Q}
    claims = cache.eviction.claims
    while true
        for i in eachindex(claims)
            if trylock(@inbounds claims[i])
                try
                    return f(queue_handle(cache, i))
                finally
                    unlock(@inbounds claims[i])
                end
            end
        end
        yield()
    end
end
Base.size(handle::QueueHandle) = size(handle.cache)
Base.axes(handle::QueueHandle) = axes(handle.cache)
Base.@propagate_inbounds cache_key(handle::QueueHandle, indices::Tuple) = cache_key(handle.cache, indices)
Base.@propagate_inbounds map_lookup(handle::QueueHandle, indices::Tuple) = map_lookup(handle.cache, indices)
@inline entry(handle::QueueHandle, key) = entry(handle.cache, key)
@inline ready_entry(handle::QueueHandle, stored) = ready_entry(handle.cache, stored)
@inline ready_entry(cache::LRUMemoizedArray{V,N,M,G,D,Q,true}, stored) where {V,N,M,G,D,Q} =
    stored isa managed_stored_type(V, cache.eviction.queues[1])
cache_get(::LRUMemoizedArray{V,N,M,G,D,Q,true}, key, indices::Tuple) where {V,N,M,G,D,Q} =
    throw(ArgumentError("use queue_handle(cache, i) or with_queue(f, cache) to request values"))
@inline function cache_get(handle::QueueHandle, lookup, indices::Tuple)
    key, _, ref, _ = lookup
    return private_get(handle.cache, handle.queue, ref, key, indices)
end

@inline function cache_get(handle::QueueHandle{V,N,M}, lookup, indices::Tuple) where {V,N,M<:ConcurrentDictMap}
    key, hashed, record, _ = lookup
    if record === nothing
        return dict_handle_miss!(handle, key, hashed, indices)
    end
    return private_get(handle.cache, handle.queue, record, key, indices, slot_entry(record), record)
end

@noinline function dict_handle_miss!(handle, key, hashed, indices)
    record = map_reserve!(handle.cache.map, key, hashed)
    return private_get(handle.cache, handle.queue, record, key, indices, slot_entry(record), record)
end

Base.axes(handle::QueueHandle{V,N,M}) where {V,N,M<:Union{ConcurrentDictMap,ConcurrentTreeMap}} = axes(handle.cache)
Base.size(handle::QueueHandle{V,N,M}) where {V,N,M<:Union{ConcurrentDictMap,ConcurrentTreeMap}} = size(handle.cache)
Base.size(handle::QueueHandle{V,N,M}, d::Int) where {V,N,M<:Union{ConcurrentDictMap,ConcurrentTreeMap}} = size(handle.cache,d)
Base.length(handle::QueueHandle{V,N,M}) where {V,N,M<:Union{ConcurrentDictMap,ConcurrentTreeMap}} = length(handle.cache)
Base.@propagate_inbounds cache_key(handle::QueueHandle{V,N,M}, key::NTuple{N,Int}) where {V,N,M<:Union{ConcurrentDictMap,ConcurrentTreeMap}} =
    cache_key(handle.cache,key)
Base.@propagate_inbounds map_lookup(handle::QueueHandle{V,N,M}, key::NTuple{N,Int}) where {V,N,M<:Union{ConcurrentDictMap,ConcurrentTreeMap}} =
    map_lookup(handle.cache,key)

@inline function cache_get(handle::QueueHandle{V,N,M}, lookup, indices::Tuple) where {V,N,M<:ConcurrentTreeMap}
    key, routing, ref, leaf = lookup
    if ref === nothing
        return tree_handle_miss!(handle, key, routing, indices)
    end
    return private_get(handle.cache, handle.queue, ref, key, indices, slot_entry(ref), leaf)
end

@noinline function tree_handle_miss!(handle, key, routing, indices)
    slot = map_reserve!(handle.cache.map, key, routing, Val(true))
    return private_get(handle.cache, handle.queue, slot, key, indices)
end

Base.size(handle::QueueHandle{V,N,M}) where {V,N,M<:ConcurrentExpandingArrayMap} = size(handle.cache)
Base.axes(handle::QueueHandle{V,N,M}) where {V,N,M<:ConcurrentExpandingArrayMap} = axes(handle.cache)
@inline entry(handle::QueueHandle{V,N,M}, key) where {V,N,M<:ConcurrentExpandingArrayMap} = entry(handle.cache, key)
@inline cache_key(handle::QueueHandle{V,N,M}, indices::NTuple{N,Int}) where {V,N,M<:ConcurrentExpandingArrayMap} =
    cache_key(handle.cache, indices)
@inline map_lookup(handle::QueueHandle{V,N,M}, indices::NTuple{N,Int}) where {V,N,M<:ConcurrentExpandingArrayMap} =
    map_lookup(handle.cache, indices)

@inline function cache_get(handle::QueueHandle{V,N,M}, lookup, indices::Tuple) where {V,N,M<:ConcurrentExpandingArrayMap}
    key, allocation, reference, _ = lookup
    if reference === nothing
        return expanding_request!(handle, key, indices)
    end
    return private_get(handle.cache, handle.queue, reference, key, indices, slot_entry(reference), allocation)
end

@noinline function expanding_request!(handle, key, indices)
    allocation = grow_allocation!(handle.cache.map, key)
    reference = allocation_slot(allocation, key)
    return private_get(handle.cache, handle.queue, reference, key, indices, slot_entry(reference), allocation)
end

# Exchange pairs with waiter registration. Joined callers retain their outcome
# even when another request immediately evicts and recycles the resident entry.
@inline function managed_publish!(ref, stored, outcome)
    previous = slot_exchange!(ref, stored, GENERATING)
    if previous isa GenerateWait
        slot_publish!(previous.reference, outcome)
    end
    return nothing
end

@inline managed_publish!(store, ref, stored, outcome) = managed_publish!(ref, stored, outcome)
@inline claim_slot!(store, key, ref, expected, desired) = claim_slot!(ref, expected, desired)
@inline claimed_slot(ref, claim) = ref

@inline function private_get(cache::LRUMemoizedArray{V},q,ref,key,indices,stored=slot_entry(ref),allocation=nothing) where V
    if stored isa managed_stored_type(V,q)
        stored.explicit && return decode_value(V, (@atomic :acquire stored.value))::V
        index = @inbounds memoryref(q.positions,stored.id)[]
        if index != 0
            # A fresh slot load follows this queue's earlier validated access.
            # Its remembered incarnation therefore also identifies the key.
            version = @inbounds memoryref(q.incarnations,index)[]
            # Acquiring a replacement payload must precede the state check.
            value = @atomic :acquire stored.value
            valid = private_state(stored) == version
        else
            valid = false
        end
        if !valid
            valid,value,version = private_snapshot(stored,key)
            if !valid
                return private_miss!(cache,q,ref,key,indices,stored,allocation)::V
            end
            if index == 0
                private_add!(cache,q,stored,version)
                return decode_value(V,value)::V
            end
            @inbounds memoryref(q.incarnations,index)[] = version
        end
        queue_touch!(q,index)
        private_mark!(q,stored,version)
        return decode_value(V,value)::V
    end
    return private_miss!(cache,q,ref,key,indices,stored,allocation)::V
end

@inline private_miss!(cache,q,ref,key,indices,stored,::Nothing) =
    private_miss!(cache,q,ref,key,indices,stored)

@noinline function private_miss!(cache,q,ref,key,indices,stored,allocation::ExpandingAllocation)
    slot = ExpandingSlot(cache.map,allocation,ref,key)
    return @inline private_miss!(cache,q,slot,key,indices,stored)
end

@noinline function private_miss!(cache,q,ref,key,indices,stored,record::ConcurrentHashRecord)
    slot = DictSlot(cache.map,record)
    return @inline private_miss!(cache,q,slot,key,indices,stored)
end

@noinline function private_miss!(cache,q,ref,key,indices,stored,leaf::ConcurrentTreeLeaf)
    slot = ConcurrentLeafSlot(leaf,ref)
    # Keep the slot as a call argument across generator exception handling.
    return private_miss!(cache,q,slot,key,indices,stored)
end

@inline function private_generate!(cache::LRUMemoizedArray{V},q,ref,key,indices) where V
    # Generation runs before entry ownership or queue mutation, including nested
    # generator calls. Failed generations preserve the existing residents.
    value = try
        generated = cache.generator(indices...)
        generated isa V ? generated : (convert(V,generated)::V)
    catch error
        failure = GenerationFailure(error)
        managed_publish!(cache.map,ref,failure,failure)
        rethrow()
    end
    outcome = encode_value(V,value)
    entry,index,version = private_reserve!(cache,q)
    @atomic :release entry.key = key
    @atomic :release entry.value = outcome
    if index == 0
        private_remember!(cache,q,entry,version,Val(true))
    else
        # Reuse the victim's node directly: membership and count stay intact.
        @inbounds memoryref(q.incarnations,index)[] = version
        queue_touch!(q,index)
        private_mark!(q,entry,version)
    end
    # Publish while the incarnation remains exclusively claimed. Registered
    # waiters retain the value independently of subsequent entry reuse.
    managed_publish!(cache.map,ref,entry,outcome)
    @atomic :release entry.state = version
    return value
end

@noinline function private_miss!(cache::LRUMemoizedArray{V},q,ref,key,indices,stored) where V
    if stored === EMPTY
        claim = claim_slot!(cache.map,key,ref,EMPTY,GENERATING)
        claim.success && return private_generate!(cache,q,claimed_slot(ref,claim),key,indices)
        stored = claim.old
    end
    return private_contended!(cache,q,ref,key,indices,stored)::V
end

@noinline function private_contended!(cache::LRUMemoizedArray{V},q,ref,key,indices,stored) where V
    S = concurrent_value_type(V)
    while true
        if stored === EMPTY
            claim = claim_slot!(cache.map,key,ref,EMPTY,GENERATING)
            claim.success && return private_generate!(cache,q,claimed_slot(ref,claim),key,indices)
            stored = claim.old
            continue
        elseif stored === GENERATING
            pending = GenerateWait{S}()
            claim = claim_slot!(cache.map,key,ref,GENERATING,pending)
            claim.success && return wait_generated(V,pending.reference)
            stored = claim.old
            continue
        elseif stored isa GenerateWait{S}
            return wait_generated(V,stored.reference)
        elseif stored isa GenerationFailure
            throw(stored.error)
        elseif stored === RELOCATED
            stored = slot_entry(ref)
            continue
        elseif stored isa managed_stored_type(V,q)
            stored.explicit && return decode_value(V, (@atomic :acquire stored.value))::V
            valid,value,version = private_snapshot(stored,key)
            if valid
                private_remember!(cache,q,stored,version)
                return decode_value(V,value)::V
            end
        end
        yield()
        stored = slot_entry(ref)
    end
end


# Conversion precedes lookup: user-defined conversion may itself access the cache.
Base.@propagate_inbounds function Base.setindex!(cache::LRUMemoizedArray{V,N,M,G,D,Q,false}, value,
    indices::Vararg{Int,N}) where {V,N,M,G,D,Q}
    value = value isa V ? value : (convert(V,value)::V)
    return cache_set!(cache,value,map_lookup(cache,indices))
end

Base.@propagate_inbounds Base.setindex!(cache::LRUMemoizedArray, value, key::CartesianIndex) =
    setindex!(cache,value,Tuple(key)...)

function cache_set!(cache::LRUMemoizedArray, value, lookup)
    key,routing,slot,hint = lookup
    stored = typeof(cache.map.empty)(UInt32(0),true)
    stored.key = key
    stored.value = value
    slot = reserve!(cache.map,key,routing,slot,hint)
    previous = slot_entry(slot)
    if previous === cache.map.empty
        map_retain!(cache.map,slot)
    elseif !previous.explicit
        queue_unlink!(cache.eviction,previous.id)
    end
    slot_publish!(slot,stored)
    return value
end

function cache_set!(cache::LRUMemoizedArray{V,N,M}, value, lookup) where {V,N,M<:LRUDictMap}
    return dict_lru_set!(cache,value,lookup)
end

Base.@propagate_inbounds function Base.setindex!(handle::QueueHandle{V,N}, value,
    indices::Vararg{Int,N}) where {V,N}
    value = value isa V ? value : (convert(V,value)::V)
    key,routing,reference,hint = map_lookup(handle,indices)
    slot = assignment_slot(handle.cache.map,key,routing,reference,hint)
    return private_set!(handle.cache,handle.queue,slot,key,value)
end

Base.@propagate_inbounds Base.setindex!(handle::QueueHandle, value, key::CartesianIndex) =
    setindex!(handle,value,Tuple(key)...)

function private_set!(cache::LRUMemoizedArray{V},q,slot,key,value) where V
    S = concurrent_value_type(V)
    entry = managed_stored_type(V,q)(true,UInt32(0),UInt64(0),UInt64(0),key,encode_value(V,value))
    while true
        previous = slot_entry(slot)
        if previous === GENERATING
            pending = GenerateWait{S}()
            claim = claim_slot!(cache.map,key,slot,GENERATING,pending)
            slot = claimed_slot(slot,claim)
            claim.success && wait_generated(V,pending.reference)
        elseif previous isa GenerateWait{S}
            wait_generated(V,previous.reference)
        elseif previous isa managed_stored_type(V,q) && !previous.explicit
            valid,_,version = private_snapshot(previous,key)
            if valid
                owned = @atomicreplace :acquire_release :acquire previous.state version => (version | 1)
                if owned.success
                    claim = claim_slot!(cache.map,key,slot,previous,entry)
                    slot = claimed_slot(slot,claim)
                    if claim.success
                        # Other queues keep their references; incarnation checks reject them.
                        index = @inbounds memoryref(q.positions,previous.id)[]
                        index != 0 && private_unlink!(q,index)
                        private_free!(q,previous,version)
                        return value
                    end
                    @atomic :release previous.state = version
                end
            end
        else
            claim = claim_slot!(cache.map,key,slot,previous,entry)
            slot = claimed_slot(slot,claim)
            claim.success && return value
        end
        yield()
    end
end
