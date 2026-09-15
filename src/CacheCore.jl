# Wake registered waiters while holding their condition lock.
@noinline function _notify_waiters(condition::Threads.Condition)
    lock(condition)
    try
        notify(condition; all=true)
    finally
        unlock(condition)
    end
    return nothing
end

# Publish an attempt's outcome before notifying its captured waiters.
@inline function _finish!(entry::AbstractEntry, state)
    previous = @atomicswap :acquire_release entry.state = state
    if previous isa Threads.Condition
        _notify_waiters(previous)
    end
    return nothing
end

# Install a condition only when a caller actually needs to wait.
@noinline function _wait_entry(entry::AbstractEntry)
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
@inline _ready_entry(::Union{Nothing,Empty}) = false
@inline _ready_entry(entry::AbstractEntry) = (@atomic :acquire entry.state) === true
@inline _new_entry(::Type{E}, key) where E<:Union{ExactLRUEntry,SampledLRUEntry} = E(key)

# One map lookup feeds readiness checks and scalar access.
@inline function _entry(cache, key)
    slot = _store_lookup(cache.map, key, _store_hash(cache.map, key))
    if slot === nothing
        return _EMPTY
    end
    return _slot_entry(slot)
end

# Unbounded ready reads never take a structural lock.
@inline function _cache_get(cache::_UnboundedArray{V}, key, stamp=nothing, indices=nothing) where V
    routing = _store_hash(cache.map, key)
    slot = _store_lookup(cache.map, key, routing)
    if slot === nothing
        slot = _reserve_unbounded!(cache.map, key, routing)
        return _get_missing(cache, slot, key, indices, _slot_entry(slot))::V
    end
    entry = _slot_entry(slot)
    if _ready_entry(entry)
        return entry.value::V
    end
    return _get_missing(cache, slot, key, indices, entry)::V
end

# Dictionary records require a structural lock; tree and dense slots publish locally.
@inline _reserve_unbounded!(map, key, routing) = _store_reserve!(map, key, routing, Val(false))
function _reserve_unbounded!(map::DictMap, key, routing)
    lock(map.guard)
    try
        return _store_reserve!(map, key, routing)
    finally
        unlock(map.guard)
    end
end

# Claim a missing location; preserve the same entry through completion.
@inline function _get_missing(cache::_UnboundedArray{V}, slot, key, indices, entry) where V
    while true
        if entry isa Entry{V}
            return _wait_entry(entry)
        end
        candidate = Entry{V}()
        claim = _slot_claim!(slot, _EMPTY, candidate)
        if claim.success
            try
                candidate.value = cache.generator(_generator_indices(cache, key, indices)...)
            catch exception
                _slot_publish!(slot, _EMPTY)
                _finish!(candidate, _EMPTY)
                rethrow()
            end
            _finish!(candidate, true)
            return candidate.value
        end
        entry = claim.old
    end
end

# Exact recency always uses the bookkeeping lock; sampled completed hits only update their timestamp.
@inline _cache_get(cache::ExactLRUArray, key, stamp=nothing, indices=nothing) =
    _bounded_get(cache, key, _store_hash(cache.map, key), stamp, indices)
@inline function _cache_get(cache::SampledLRUArray{V}, key, stamp=nothing, indices=nothing) where V
    routing = _store_hash(cache.map, key)
    slot = _store_lookup(cache.map, key, routing)
    if slot !== nothing
        entry = _slot_entry(slot)
        if _ready_entry(entry)
            return _sampled_touch(cache, entry, stamp)::V
        end
    end
    return _bounded_get(cache, key, routing, stamp, indices)::V
end

# Claims and eviction share a lock; generation and entry waiting happen outside it.
@inline function _bounded_get(cache::LRUArray{V}, key, routing, stamp, indices) where V
    owner = false
    E = _entrytype(cache)
    local entry, slot
    lock(cache.available)
    try
        while true
            slot = _store_lookup(cache.map, key, routing)
            if slot !== nothing
                observed = _slot_entry(slot)
                if observed isa E
                    if _ready_entry(observed)
                        return _policy_touch(cache, observed, stamp)::V
                    end
                    entry = observed
                    break
                end
            end
            if _policy_full(cache, E)
                if _policy_no_ready(cache, E)
                    wait(cache.available)
                    continue
                end
                _policy_evict!(cache, E)
                slot = nothing # Eviction may prune the previously located block.
            end
            entry = _new_entry(E, key)
            if slot === nothing
                slot = _store_reserve!(cache.map, key, routing)
            end
            _store_retain!(cache.map, slot)
            _policy_claim!(cache, entry)
            _slot_publish!(slot, entry)
            owner = true
            break
        end
    finally
        unlock(cache.available)
    end
    if !owner
        _wait_entry(entry)
        return _wait_value(cache, entry, stamp)::V
    end
    local previous
    try
        entry.value = cache.generator(_generator_indices(cache, key, indices)...)
        lock(cache.available)
        try
            _policy_add!(cache, entry, stamp)
            previous = @atomicswap :acquire_release entry.state = true
            notify(cache.available; all=true)
        finally
            unlock(cache.available)
        end
    catch exception
        if !_ready_entry(entry)
            _fail_generation!(cache, key, slot, entry)
        end
        rethrow()
    end
    if previous isa Threads.Condition
        _notify_waiters(previous)
    end
    return entry.value::V
end

# Reset only unfinished attempts; notification after success never rolls back a published entry.
function _fail_generation!(cache, key, slot, entry)
    lock(cache.available)
    try
        _store_delete!(cache.map, key, slot)
        _policy_release!(cache, entry)
        notify(cache.available; all=true)
    finally
        unlock(cache.available)
    end
    _finish!(entry, _EMPTY)
    return nothing
end

# Captured exact waiters must not reorder entries that may already have been evicted.
@inline _wait_value(cache, entry::ExactLRUEntry, stamp) = entry.value
@inline _wait_value(cache, entry::SampledLRUEntry, stamp) = _sampled_touch(cache, entry, stamp)
# Remove a retained key through its storage backend.
_delete_entry!(cache, entry) = _delete_key!(cache.map, entry.key)
function _delete_key!(store, key)
    routing = _store_hash(store, key)
    slot = _store_lookup(store, key, routing)
    return _store_delete!(store, key, slot)
end

# Copy finite selections with one clock advance and the common per-key path.
function _copy_values!(dest, cache, keys)
    if length(dest) < length(keys)
        throw(BoundsError(dest, eachindex(keys)))
    end
    if Base.mightalias(dest, keys)
        keys = copy(keys)
    end
    stamp = _batch_clock(cache, length(keys))
    for (i, key) in zip(eachindex(dest), keys)
        @inbounds dest[i] = _batch_value(cache, key, stamp)
    end
    return dest
end

# Views and reshapes supply their selected keys to the copy loop.
function _copy_source!(dest, source)
    cache, keys = _source_keys(source)
    return _copy_values!(dest, cache, keys)
end
