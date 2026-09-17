# Approximate LRU caches: CLOCK scans resident slots; SIEVE scans insertion order.
# Policy and field layouts are type parameters; reads only mark entries as visited.

mutable struct ClockEntry{V,Key} <: AbstractEntry{V}
    key::Key
    visited::Bool
    value::V
    ClockEntry{V,Key}() where {V,Key} = new{V,Key}()
end

mutable struct SieveEntry{V,Key} <: AbstractEntry{V}
    key::Key
    previous::Union{Nothing,SieveEntry{V,Key}}
    next::Union{Nothing,SieveEntry{V,Key}}
    visited::Bool
    value::V
    SieveEntry{V,Key}() where {V,Key} = new{V,Key}()
end

"""`SampledLRUArray{V}(generator, dims...; capacity, policy=:clock, kwargs...)`: policies `:clock` (default), `:sieve`."""
mutable struct SampledLRUArray{V,N,M,G,Policy,E,A,L,H} <: AbstractJITArray{V,N,M}
    const capacity::Int
    count::Int
    const active::A
    oldest::L
    newest::L
    hand::H
    const map::M
    const generator::G
end

function SampledLRUArray{V}(generator::G, domain::VT{N,T}; capacity::Int, policy::Symbol=:clock,
    storage=nothing, block_shape=nothing) where {V,G,N,T<:Union{Int,AbstractUnitRange{Int}}}
    capacity < 1 && throw(ArgumentError("capacity must be positive"))
    storage = domain_storage(domain, storage)
    @astuple length(domain) domain
    domain = axis_domain(domain, storage)
    Key = entry_keytype(domain, storage)
    if policy === :clock
        E = ClockEntry{V,Key}
        active, oldest, newest, hand = preallocate(E, capacity), nothing, nothing, 1
        L, H = Nothing, Int
    elseif policy === :sieve
        E = SieveEntry{V,Key}
        active, oldest, newest, hand = preallocate(E, capacity), nothing, nothing, nothing
        L = H = Union{Nothing,E}
    else
        throw(ArgumentError("unknown approximate LRU policy $policy; use :clock or :sieve"))
    end
    map = make_map(Union{Empty,E}, domain, storage, block_shape)
    return SampledLRUArray{V,length(domain),typeof(map),G,policy,E,typeof(active),L,H}(
        capacity, 0, active, oldest, newest, hand, map, generator)
end

SampledLRUArray{V}(generator, dims::Int...; kwargs...) where V = SampledLRUArray{V}(generator, dims; kwargs...)
SampledLRUArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V = SampledLRUArray{V}(generator, (axis, axes...); kwargs...)
SampledLRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} = SampledLRUArray{V}(generator, domain; capacity, kwargs...)
@inline entrytype(::SampledLRUArray{V,N,M,G,P,E}) where {V,N,M,G,P,E} = E

# CLOCK: F. J. Corbató, A Paging Experiment with the Multics System (1969).
# https://multicians.org/paging-experiment.pdf
function victim!(cache::SampledLRUArray{V,N,M,G,:clock}) where {V,N,M,G}
    @inbounds while true
        index = cache.hand
        cache.hand = index == cache.capacity ? 1 : index + 1
        stored = cache.active[index]
        if !stored.visited
            return index
        end
        stored.visited = false
    end
end

@inline function cacheget(cache::SampledLRUArray{V,N,M,G,:clock,E}, key, indices=nothing) where {V,N,M,G,E}
    routing = store_hash(cache.map, key)
    slot, hint = prepare(cache.map, key, routing)
    if slot !== nothing
        stored = preparedentry(slot, hint)
        if stored isa E
            if !stored.visited
                stored.visited = true
            end
            return stored.value
        end
    end
    generated = cache.generator(generator_indices(cache, key, indices)...)
    value = generated isa V ? generated : convert(V, generated)
    if cache.count == cache.capacity
        index = victim!(cache)
        @inbounds stored = cache.active[index]
        delete_key!(cache.map, stored.key)
        stored.key, stored.value, stored.visited = key, value, true
    else
        @inbounds stored = cache.active[cache.count + 1]
        stored.key, stored.value, stored.visited = key, value, true
        cache.count += 1
    end
    slot = reserve!(cache.map, key, routing, slot, hint)
    store_retain!(cache.map, slot)
    slot_publish!(slot, stored)
    return value
end

# SIEVE: Zhang et al., SIEVE is Simpler than LRU (NSDI 2024).
# https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo
function victim!(cache::SampledLRUArray{V,N,M,G,:sieve}) where {V,N,M,G}
    stored = cache.hand === nothing ? cache.oldest : cache.hand
    while stored.visited
        stored.visited = false
        stored = stored.next === nothing ? cache.oldest : stored.next
    end
    cache.hand = stored.next
    return stored
end

@inline function cacheget(cache::SampledLRUArray{V,N,M,G,:sieve,E}, key, indices=nothing) where {V,N,M,G,E}
    routing = store_hash(cache.map, key)
    slot, hint = prepare(cache.map, key, routing)
    if slot !== nothing
        stored = preparedentry(slot, hint)
        if stored isa E
            if !stored.visited
                stored.visited = true
            end
            return stored.value
        end
    end
    generated = cache.generator(generator_indices(cache, key, indices)...)
    value = generated isa V ? generated : convert(V, generated)
    if cache.count == cache.capacity
        stored = victim!(cache)
        delete_key!(cache.map, stored.key)
        if stored.previous === nothing
            cache.oldest = stored.next
        else
            stored.previous.next = stored.next
        end
        if stored.next === nothing
            cache.newest = stored.previous
        else
            stored.next.previous = stored.previous
        end
        stored.key, stored.value, stored.visited = key, value, false
        stored.previous, stored.next = cache.newest, nothing
    else
        @inbounds stored = cache.active[cache.count + 1]
        stored.key, stored.value, stored.visited = key, value, false
        stored.previous, stored.next = cache.newest, nothing
        cache.count += 1
    end
    if cache.newest === nothing
        cache.oldest = stored
    else
        cache.newest.next = stored
    end
    cache.newest = stored
    slot = reserve!(cache.map, key, routing, slot, hint)
    store_retain!(cache.map, slot)
    slot_publish!(slot, stored)
    return value
end
