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
    ExactLRUArray{V}(generator, dims...; capacity, kwargs...)

Cache values with exact LRU eviction. Accept sizes or coordinate ranges.
Sparse storage: `:dict` (default), `:radixtree`, `:subtreetree`; trees require `block_shape`.
"""
mutable struct ExactLRUArray{V,N,M,G,Key} <: AbstractJITArray{V,N,M}
    const capacity::Int
    count::Int
    const entries::Vector{ExactLRUEntry{V,Key}}
    const root::ExactLRUEntry{V,Key}
    const map::M
    const generator::G
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
    root = E()
    map = make_map(E, domain, storage, block, root)
    return ExactLRUArray{V,length(domain),typeof(map),G,Key}(
        capacity, 0, preallocate(E, capacity), root, map, generator)
end

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
    routing = store_hash(cache.map, key)
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
        delete_key!(cache.map, stored.key)
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
