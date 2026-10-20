# Exclusive callers own queue links. Shared entries carry no list pointers.
# State: incarnation in the high bits; low bits 0 = live, 1 = editing, 2 = free.
# Last access: the same incarnation plus queue number, used only as a recency hint.
mutable struct SharedLRUEntry{S,K}
    const explicit::Bool
    const id::UInt32
    @atomic state::UInt64
    @atomic last_access::UInt64
    @atomic key::K
    @atomic value::Union{Empty,S}
end

mutable struct PrivateLRUQueue{E}
    const number::UInt32
    const step::UInt64
    const capacity::Int
    count::Int
    const entries::Vector{E}
    const incarnations::MemoryRef{UInt64}
    # Two adjacent words per node; linear access keeps the fixed stride in code.
    const links::MemoryRef{UInt32}
    const positions::MemoryRef{UInt32}
    oldest::UInt32
    newest::UInt32
    vacant::UInt32
    # Stable registry for finding free entries, never a shared mutable allocator.
    const pool::Vector{E}
    const free::Vector{E}
    free_count::Int
    # Separate adjacent queues' mutable fields.
    const padding::NTuple{8,UInt64}
end

function PrivateLRUQueue(::Type{E}, key, number, quota, capacity, queues) where E
    step = UInt64(nextpow(2,max(4,queues+1)))
    pool = [E(false,UInt32((number-1)*quota+i),2,0,key,EMPTY) for i in 1:quota]
    links = Memory{UInt32}(undef,2quota)
    fill!(links,0)
    for i in 1:quota
        links[2i] = UInt32(i == quota ? 0 : i+1)
    end
    return PrivateLRUQueue{E}(UInt32(number),step,quota,0,Vector{E}(undef,quota),
        memoryref(fill!(Memory{UInt64}(undef,quota),0)),memoryref(links),
        memoryref(fill!(Memory{UInt32}(undef,capacity),0)),0,0,1,pool,copy(pool),quota,ntuple(_->UInt64(0),8))
end

@inline managed_stored_type(::Type, queue::PrivateLRUQueue) = eltype(queue.entries)
@inline private_state(entry) = @atomic :acquire entry.state
@inline private_live(state) = state != 0 && state & 3 == 0

@inline function queue_touch!(q,index)
    @inbounds if index != q.newest
        previous,next = memoryref(q.links,2Int(index)-1)[],memoryref(q.links,2Int(index))[]
        if index == q.oldest
            q.oldest = next
            memoryref(q.links,2Int(next)-1)[] = next
        else
            memoryref(q.links,2Int(previous))[] = next
            memoryref(q.links,2Int(next)-1)[] = previous
        end
        memoryref(q.links,2Int(q.newest))[] = index
        memoryref(q.links,2Int(index)-1)[] = q.newest
        memoryref(q.links,2Int(index))[] = index
        q.newest = index
    end
    return nothing
end

# Remove only local membership; the vacated link becomes the local free-index list.
@inline function queue_unlink!(q,index)
    @inbounds begin
        previous,next = memoryref(q.links,2Int(index)-1)[],memoryref(q.links,2Int(index))[]
        if q.count == 1
            q.oldest = q.newest = 0
        elseif index == q.oldest
            q.oldest = next
            memoryref(q.links,2Int(next)-1)[] = next
        elseif index == q.newest
            q.newest = previous
            memoryref(q.links,2Int(previous))[] = previous
        else
            memoryref(q.links,2Int(previous))[] = next
            memoryref(q.links,2Int(next)-1)[] = previous
        end
        memoryref(q.links,2Int(index)-1)[] = 0
        memoryref(q.links,2Int(index))[] = q.vacant
        q.vacant = index
        q.count -= 1
    end
    return nothing
end

@inline function private_unlink!(q,index)
    @inbounds memoryref(q.positions,q.entries[index].id)[] = 0
    return queue_unlink!(q,index)
end

# No reader lock. Both payload fields are atomic, and their release/acquire
# accesses make a recycled payload visible only with a changed incarnation.
@inline function private_snapshot(entry,key)
    version = private_state(entry)
    private_live(version) || return (false,EMPTY,version)
    location = @atomic :acquire entry.key
    value = @atomic :acquire entry.value
    return (location == key && private_state(entry) == version,value,version)
end

@inline function private_reclaim!(cache,q,entry,version)
    private_state(entry) == version || return false
    recent = @atomic :monotonic entry.last_access
    mask = q.step-1
    # A delayed read may leave a hint for an older incarnation. Such a hint
    # cannot protect the replacement indefinitely; only current hints count.
    if recent & ~mask == version && recent & mask != q.number
        return false
    end
    claimed = @atomicreplace :acquire_release :acquire entry.state version => (version | 1)
    claimed.success || return false
    # The incarnation claim grants the sole right to remove this mapping.
    # A replacement generator cannot claim the slot until this release-store.
    remove_mapping!(cache.map,entry)
    return true
end

@inline function private_free!(q,entry,version)
    @atomic :release entry.value = EMPTY
    @atomic :release entry.state = version | 2
    # A bounded local hint list; the immutable pool registry also finds free
    # entries if another queue has already collected more than its own quota.
    if q.free_count < q.capacity
        q.free_count += 1
        @inbounds q.free[q.free_count] = entry
    end
    return nothing
end

@inline function private_remember!(cache,q,entry,version,::Val{InlineAppend}=Val(false)) where InlineAppend
    index = @inbounds memoryref(q.positions,entry.id)[]
    if index != 0
        # Stable IDs always name the same object; only its incarnation changes.
        # Ordinary reads need neither a reference write nor a GC write barrier.
        @inbounds if memoryref(q.incarnations,index)[] != version
            memoryref(q.incarnations,index)[] = version
        end
        queue_touch!(q,index)
        private_mark!(q,entry,version)
        return nothing
    end
    # Generation inlines the short append; reads keep new membership cold.
    if InlineAppend
        if q.count < q.capacity
            return private_append!(q,entry,version)
        end
    end
    return private_add!(cache,q,entry,version)
end

@inline function private_mark!(q,entry,version)
    hint = version | UInt64(q.number)
    if (@atomic :monotonic entry.last_access) != hint
        @atomic :monotonic entry.last_access = hint
    end
    return nothing
end

@inline function private_append!(q,entry,version)
    index = q.vacant
    @inbounds begin
        q.vacant = memoryref(q.links,2Int(index))[]
        if q.count == 0
            q.oldest = index
            memoryref(q.links,2Int(index)-1)[] = index
            memoryref(q.links,2Int(index))[] = index
        else
            memoryref(q.links,2Int(q.newest))[] = index
            memoryref(q.links,2Int(index)-1)[] = q.newest
            memoryref(q.links,2Int(index))[] = index
        end
        q.newest = index
        q.count += 1
    end
    @inbounds memoryref(q.positions,entry.id)[] = index
    @inbounds q.entries[index] = entry
    @inbounds memoryref(q.incarnations,index)[] = version
    private_mark!(q,entry,version)
    return nothing
end

@noinline function private_add!(cache,q,entry,version)
    if q.count < q.capacity
        return private_append!(q,entry,version)
    end
    index = q.oldest
    old = @inbounds q.entries[index]
    oldversion = @inbounds memoryref(q.incarnations,index)[]
    if private_reclaim!(cache,q,old,oldversion)
        private_free!(q,old,oldversion)
    end
    # Replace the oldest membership in place; count and vacant stay valid.
    @inbounds memoryref(q.positions,old.id)[] = 0
    queue_touch!(q,index)
    @inbounds memoryref(q.positions,entry.id)[] = index
    @inbounds q.entries[index] = entry
    @inbounds memoryref(q.incarnations,index)[] = version
    # Deliberately approximate: no CAS/retry when another reader wins recency.
    # Stamp the hint so delayed accesses cannot protect a different incarnation.
    private_mark!(q,entry,version)
    return nothing
end

@inline function private_claim_free!(q,entry)
    # Incarnations start at step; zero can report a failed ownership claim.
    state = private_state(entry)
    state & 3 == 2 || return UInt64(0)
    version = Base.checked_add(state & ~(q.step-1),q.step)
    claimed = @atomicreplace :acquire_release :acquire entry.state state => (version | 1)
    return claimed.success ? version : UInt64(0)
end

@inline function private_reserve!(cache,q)
    while q.free_count > 0
        entry = @inbounds q.free[q.free_count]
        q.free_count -= 1
        version = private_claim_free!(q,entry)
        version != 0 && return (entry,UInt32(0),version)
    end
    while q.count > 0
        index = q.oldest
        entry = @inbounds q.entries[index]
        version = @inbounds memoryref(q.incarnations,index)[]
        nextversion = Base.checked_add(version,q.step)
        reclaimed = private_reclaim!(cache,q,entry,version)
        if reclaimed
            @atomic :release entry.state = nextversion | 1
            return (entry,index,nextversion)
        end
        private_unlink!(q,index)
    end
    entry,version = private_find_free!(cache,q)
    return (entry,UInt32(0),version)
end

@noinline function private_find_free!(cache,q)
    # Cold path: local victims were recently used elsewhere. Claim a free entry
    # from a fixed registry; no shared count, allocator lock, or new entry object.
    while true
        for queue in cache.eviction.queues, entry in queue.pool
            version = private_claim_free!(q,entry)
            version != 0 && return (entry,version)
        end
        yield() # another caller may currently hold the remaining free entries
    end
end

# Diagnostic only, after callers have stopped. Queue counts include duplicates.
function resident_count(management::MultiQueue)
    return sum(private_live(private_state(entry)) for q in management.queues for entry in q.pool)
end
