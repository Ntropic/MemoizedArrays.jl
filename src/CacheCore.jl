# Wake registered waiters while holding their condition lock.
@noinline function notify_waiters(condition::Threads.Condition)
    lock(condition)
    try
        notify(condition; all=true)
    finally
        unlock(condition)
    end
    return nothing
end

# Publish an attempt's outcome before notifying its captured waiters.
@inline function finish!(entry::AbstractEntry, state)
    previous = @atomicswap :acquire_release entry.state = state
    if previous isa Threads.Condition
        notify_waiters(previous)
    end
    return nothing
end

# Install a condition only when a caller actually needs to wait.
@noinline function wait_entry(entry::AbstractEntry)
    state = @atomic :acquire entry.state
    if state === nothing
        candidate = Threads.Condition()
        claim = @atomicreplace :acquire_release :acquire entry.state nothing => candidate
        if claim.success
            state = candidate
        else
            state = claim.old
        end
    end
    if state isa Threads.Condition
        condition = state
        lock(condition)
        try
            while true
                state = @atomic :acquire entry.state
                if state === true
                    break
                elseif state === _EMPTY
                    break
                end
                wait(condition)
            end
        finally
            unlock(condition)
        end
    end
    if state === _EMPTY
        error("generation did not produce a value; retry the cache access")
    end
    return entry.value
end

# Only a completed entry exposes its ordinary value field.
@inline ready_entry(::Union{Nothing,Empty}) = false
@inline ready_entry(entry::AbstractEntry) = (@atomic :acquire entry.state) === true
@inline new_entry(::Type{E}, key) where E<:Union{ExactLRUEntry,SampledLRUEntry} = E(key)

# One map lookup feeds readiness checks and scalar access.
@inline function entry(cache, key)
    slot = store_lookup(cache.map, key, store_hash(cache.map, key))
    if slot === nothing
        return _EMPTY
    end
    return slot_entry(slot)
end

# Unbounded ready reads never take a structural lock.
@inline function cache_get(cache::_UnboundedArray{V}, key, stamp=nothing, indices=nothing) where V
    routing = store_hash(cache.map, key)
    slot = store_lookup(cache.map, key, routing)
    if slot === nothing
        slot = reserve_unbounded!(cache.map, key, routing)
        return get_missing(cache, slot, key, indices, slot_entry(slot))::V
    end
    entry = slot_entry(slot)
    if ready_entry(entry)
        return entry.value::V
    end
    return get_missing(cache, slot, key, indices, entry)::V
end

# Dictionary records require a structural lock; tree and array slots publish locally.
@inline reserve_unbounded!(map, key, routing) = store_reserve!(map, key, routing, Val(false))
function reserve_unbounded!(map::DictMap, key, routing)
    lock(map.guard)
    try
        return store_reserve!(map, key, routing)
    finally
        unlock(map.guard)
    end
end

# Claim a missing location; preserve the same entry through completion.
@inline function get_missing(cache::_UnboundedArray{V}, slot, key, indices, entry) where V
    while true
        if entry isa Entry{V}
            return wait_entry(entry)
        end
        candidate = Entry{V}()
        claim = slot_claim!(slot, _EMPTY, candidate)
        if claim.success
            try
                candidate.value = cache.generator(generator_indices(cache, key, indices)...)
            catch exception
                slot_publish!(slot, _EMPTY)
                finish!(candidate, _EMPTY)
                rethrow()
            end
            finish!(candidate, true)
            return candidate.value
        end
        entry = claim.old
    end
end

# Exact recency always uses the bookkeeping lock; sampled completed hits only update their timestamp.
@inline cache_get(cache::ExactLRUArray, key, stamp=nothing, indices=nothing) =
    bounded_get(cache, key, store_hash(cache.map, key), stamp, indices)
@inline function cache_get(cache::SampledLRUArray{V}, key, stamp=nothing, indices=nothing) where V
    routing = store_hash(cache.map, key)
    slot = store_lookup(cache.map, key, routing)
    if slot !== nothing
        entry = slot_entry(slot)
        if ready_entry(entry)
            return sampled_touch(cache, entry, stamp)::V
        end
    end
    return bounded_get(cache, key, routing, stamp, indices)::V
end

# Claims and eviction share a lock; generation and entry waiting happen outside it.
@inline function bounded_get(cache::LRUArray{V}, key, routing, stamp, indices) where V
    owner = false
    E = entrytype(cache)
    local entry, slot
    lock(cache.available)
    try
        while true
            slot = store_lookup(cache.map, key, routing)
            if slot !== nothing
                observed = slot_entry(slot)
                if observed isa E
                    if ready_entry(observed)
                        return policy_touch(cache, observed, stamp)::V
                    end
                    entry = observed
                    break
                end
            end
            if policy_full(cache, E)
                if policy_no_ready(cache, E)
                    wait(cache.available)
                    continue
                end
                policy_evict!(cache, E)
                slot = nothing # Eviction may prune the previously located block.
            end
            entry = new_entry(E, key)
            if slot === nothing
                slot = store_reserve!(cache.map, key, routing)
            end
            store_retain!(cache.map, slot)
            policy_claim!(cache, entry)
            slot_publish!(slot, entry)
            owner = true
            break
        end
    finally
        unlock(cache.available)
    end
    if !owner
        wait_entry(entry)
        return wait_value(cache, entry, stamp)::V
    end
    local previous
    try
        entry.value = cache.generator(generator_indices(cache, key, indices)...)
        lock(cache.available)
        try
            policy_add!(cache, entry, stamp)
            previous = @atomicswap :acquire_release entry.state = true
            notify(cache.available; all=true)
        finally
            unlock(cache.available)
        end
    catch exception
        if !ready_entry(entry)
            fail_generation!(cache, key, slot, entry)
        end
        rethrow()
    end
    if previous isa Threads.Condition
        notify_waiters(previous)
    end
    return entry.value::V
end

# Reset only unfinished attempts; notification after success never rolls back a published entry.
function fail_generation!(cache, key, slot, entry)
    lock(cache.available)
    try
        store_delete!(cache.map, key, slot)
        policy_release!(cache, entry)
        notify(cache.available; all=true)
    finally
        unlock(cache.available)
    end
    finish!(entry, _EMPTY)
    return nothing
end

# Captured exact waiters must not reorder entries that may already have been evicted.
@inline wait_value(cache, entry::ExactLRUEntry, stamp) = entry.value
@inline wait_value(cache, entry::SampledLRUEntry, stamp) = sampled_touch(cache, entry, stamp)
# Remove a retained key through its storage backend.
delete_entry!(cache, entry) = delete_key!(cache.map, entry.key)
function delete_key!(store, key)
    routing = store_hash(store, key)
    slot = store_lookup(store, key, routing)
    return store_delete!(store, key, slot)
end

# Copy finite selections with one clock advance and the common per-key path.
function copy_values!(dest, cache, keys)
    if length(dest) < length(keys)
        throw(BoundsError(dest, eachindex(keys)))
    end
    if Base.mightalias(dest, keys)
        keys = copy(keys)
    end
    stamp = batch_clock(cache, length(keys))
    for (i, key) in zip(eachindex(dest), keys)
        @inbounds dest[i] = batch_value(cache, key, stamp)
    end
    return dest
end

# Views and reshapes supply their selected keys to the copy loop.
function copy_source!(dest, source)
    cache, keys = source_keys(source)
    return copy_values!(dest, cache, keys)
end
