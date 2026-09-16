# Exact LRU cache, entry and constructors; maintains access order for eviction.

mutable struct ExactLRUEntry{V,Key} <: AbstractEntry{V}
    key::Key
    previous::Union{Nothing,ExactLRUEntry{V,Key}}
    next::Union{Nothing,ExactLRUEntry{V,Key}}
    value::V
    ExactLRUEntry{V,Key}(key, value) where {V,Key} = new{V,Key}(key, nothing, nothing, value)
end

"""
    ExactLRUArray{V}(generator, dims...; capacity, kwargs...)

Cache values with exact LRU eviction. Accept sizes or coordinate ranges.
Sparse storage: `:dict` (default), `:msbtree`, `:lsbtree`; trees require `block_shape`.
"""
mutable struct ExactLRUArray{V,N,M,D,G,Key} <: AbstractJITArray{V,N,M}
    const map::M
    const domain::D
    const generator::G
    oldest::Union{Nothing,ExactLRUEntry{V,Key}}
    newest::Union{Nothing,ExactLRUEntry{V,Key}}
    const capacity::Int
    count::Int
end

function ExactLRUArray{V}(generator, domain::VT{N,T}; capacity::Int,
    storage=nothing, block_shape=nothing) where {V,N,T<:Union{Int,AbstractUnitRange{Int}}}
    storage = domain_storage(domain, storage)
    @astuple length(domain) domain
    return construct(ExactLRUArray{V}, domain, generator, capacity, storage, block_shape)
end

ExactLRUArray{V}(generator, dims::Int...; kwargs...) where V = ExactLRUArray{V}(generator, dims; kwargs...)

ExactLRUArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V = ExactLRUArray{V}(generator, (axis, axes...); kwargs...)

ExactLRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} = ExactLRUArray{V}(generator, domain; capacity, kwargs...)

function construct(::Type{ExactLRUArray{V}}, input::Tuple, generator::G, capacity::Int,
    storage, block) where {V,G}
    capacity < 1 && throw(ArgumentError("capacity must be positive"))
    domain = axis_domain(input, storage)
    Key = entry_keytype(domain, storage)
    E = ExactLRUEntry{V,Key}
    map = make_map(Union{Empty,E}, domain, storage, block)
    return ExactLRUArray{V,length(domain),typeof(map),typeof(domain),G,Key}(
        map, domain, generator, nothing, nothing, capacity, 0)
end

# Remove a ready entry from the access-order list.
@inline function unlink!(cache, entry::ExactLRUEntry)
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

# Append a detached ready entry as newest.
@inline function append_entry!(cache, entry::ExactLRUEntry)
    newest = cache.newest
    entry.previous = newest
    entry.next = nothing
    if newest === nothing
        cache.oldest = entry
    else
        newest.next = entry
    end
    cache.newest = entry
    return nothing
end


@inline function policy_touch(cache, entry::ExactLRUEntry, stamp)
    if cache.newest !== entry
        unlink!(cache, entry)
        append_entry!(cache, entry)
    end
    return entry.value
end

function evict!(cache::ExactLRUArray)
    entry = cache.oldest
    unlink!(cache, entry)
    cache.count -= 1
    delete_key!(cache.map, entry.key)
    return entry
end

@inline entrytype(::ExactLRUArray{V,N,M,D,G,Key}) where {V,N,M,D,G,Key} = ExactLRUEntry{V,Key}

@inline function cacheget(cache::ExactLRUArray{V,N,M,D,G,Key}, key, stamp=nothing, indices=nothing) where {V,N,M,D,G,Key}
    routing = store_hash(cache.map, key)
    slot, hint = prepare(cache.map, key, routing)
    if slot !== nothing
        stored = preparedentry(slot, hint)
        if stored isa ExactLRUEntry{V,Key}
            return policy_touch(cache, stored, stamp)
        end
    end
    value = convert(V, cache.generator(generator_indices(cache, key, indices)...))
    if cache.count == cache.capacity
        stored = evict!(cache)
        stored.key = key
        stored.value = value
    else
        stored = ExactLRUEntry{V,Key}(key, value)
    end
    slot = reserve!(cache.map, key, routing, slot, hint)
    store_retain!(cache.map, slot)
    slot_publish!(slot, stored)
    append_entry!(cache, stored)
    cache.count += 1
    return stored.value
end
