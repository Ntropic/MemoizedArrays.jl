mutable struct ExactLRUEntry{V,Key} <: AbstractEntry{V}
    @atomic state::Union{Nothing,Threads.Condition,Bool,Empty}
    key::Key
    previous::Union{Nothing,ExactLRUEntry{V,Key}}
    next::Union{Nothing,ExactLRUEntry{V,Key}}
    value::V
    ExactLRUEntry{V,Key}(key) where {V,Key} = new{V,Key}(nothing, key, nothing, nothing)
end

# Remove a ready entry from the access-order list; caller holds the bookkeeping lock.
@inline function _unlink!(cache, entry::ExactLRUEntry)
    previous, next = entry.previous, entry.next
    if previous === nothing
        cache.oldest = next
    else
        previous.next = next
    end
    if next === nothing
        cache.newest = previous
    else
        next.previous = previous
    end
    entry.previous = nothing
    entry.next = nothing
    return nothing
end

# Append a detached ready entry as newest; caller holds the bookkeeping lock.
@inline function _append!(cache, entry::ExactLRUEntry)
    entry.previous = cache.newest
    entry.next = nothing
    if cache.newest === nothing
        cache.oldest = entry
    else
        cache.newest.next = entry
    end
    cache.newest = entry
    return nothing
end

@inline _policy_full(cache, ::Type{<:ExactLRUEntry}) = cache.count == cache.capacity
@inline _policy_no_ready(cache, ::Type{<:ExactLRUEntry}) = cache.oldest === nothing
@inline _policy_add!(cache, entry::ExactLRUEntry, stamp) = _append!(cache, entry)

@inline function _policy_touch(cache, entry::ExactLRUEntry, stamp)
    if cache.newest !== entry
        _unlink!(cache, entry)
        _append!(cache, entry)
    end
    return entry.value
end

function _policy_claim!(cache, entry::ExactLRUEntry)
    cache.count += 1
    return nothing
end

function _policy_release!(cache, entry::ExactLRUEntry)
    cache.count -= 1
    return nothing
end

function _policy_evict!(cache, ::Type{<:ExactLRUEntry})
    entry = cache.oldest
    _unlink!(cache, entry)
    cache.count -= 1
    _delete_entry!(cache, entry)
    return nothing
end
