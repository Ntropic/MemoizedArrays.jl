# Exact LRU cache, entry and constructors; maintains access order for eviction.

mutable struct ExactLRUEntry{V,Key} <: AbstractEntry{V}
    previous::ExactLRUEntry{V,Key}
    next::ExactLRUEntry{V,Key}
    key::Key
    value::V
    function ExactLRUEntry{V,Key}() where {V,Key}
        e = new{V,Key}()
        e.previous = e
        e.next = e
        return e
    end
end

"""
    ExactLRUArray{V}(generator, domain::VT{N,T}; capacity, kwargs...)

Cache values with exact LRU eviction. Domain axes are lengths, integer ranges, or `:`.
Omitted storage selects `:array` for finite domains and `:dict` otherwise; trees require `block_shape`.
"""
mutable struct ExactLRUArray{V,N,M,G,Key,D,Offset} <: AbstractJITArray{V,N,M}
    const capacity::Int
    count::Int
    const entries::Vector{ExactLRUEntry{V,Key}}
    const root::ExactLRUEntry{V,Key}
    const domain::D
    const map::M
    const generator::G
end

function ExactLRUArray{V}(generator::G, domain::VT{N,T}; capacity::Int,
    storage=nothing, block_shape=nothing, center=nothing) where {V,G,N,T}
    if capacity < 1
        throw(ArgumentError("capacity must be positive"))
    end
    map, domain = make_cache_map(ExactLRUEntry, V, domain, storage, block_shape, center)
    root = map.empty
    E = typeof(root)
    Key = fieldtype(E, :key)
    offset = routing_offset(map)
    return ExactLRUArray{V,length(domain),typeof(map),G,Key,typeof(domain),offset}(
        capacity, 0, preallocate(E, capacity), root, domain, map, generator)
end

@inline cache_offset(::ExactLRUArray{V,N,M,G,K,D,O}) where {V,N,M,G,K,D,O} = Val(O)

ExactLRUArray{V}(generator, axis::Union{Int,AbstractUnitRange{Int}}, axes::Union{Int,AbstractUnitRange{Int}}...; kwargs...) where V = ExactLRUArray{V}(generator, (axis, axes...); kwargs...)
ExactLRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} = ExactLRUArray{V}(generator, domain; capacity, kwargs...)

# Circular list: root.next is oldest, root.previous is newest.
@inline function policy_touch(cache, entry::ExactLRUEntry)
    root = cache.root
    newest = root.previous
    if newest !== entry
        previous, next = entry.previous, entry.next
        previous.next = next
        next.previous = previous
        entry.previous = newest
        entry.next = root
        newest.next = entry
        root.previous = entry
    end
    return entry.value
end

@inline entrytype(::ExactLRUArray{V,N,M,G,Key}) where {V,N,M,G,Key} = ExactLRUEntry{V,Key}

@inline function cacheget(cache::ExactLRUArray{V,N,M,G,Key}, key, indices=nothing) where {V,N,M,G,Key}
    routing = cache_routing(cache, key)
    slot, hint = prepare(cache.map, key, routing)
    if slot !== nothing
        stored = preparedentry(slot, hint)
        if stored !== cache.root
            return policy_touch(cache, stored)
        end
    end
    generated = cache.generator(generator_indices(cache, key, indices)...)
    value = generated isa V ? generated : convert(V, generated)
    if cache.count == cache.capacity
        stored = cache.root.next
        delete_entry!(cache, stored.key)
        stored.key = key
        stored.value = value
        policy_touch(cache, stored)
    else
        root = cache.root
        newest = root.previous
        @inbounds stored = cache.entries[cache.count + 1]
        stored.key, stored.value = key, value
        stored.previous, stored.next = newest, root
        newest.next = stored
        root.previous = stored
        cache.count += 1
    end
    slot = reserve!(cache.map, key, routing, slot, hint)
    store_retain!(cache.map, slot)
    slot_publish!(slot, stored)
    return value
end
