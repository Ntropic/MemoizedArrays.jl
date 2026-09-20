# Approximate policies reuse pooled entries; reads update only entry recency.

mutable struct SieveEntry{V,Key} <: AbstractEntry{V}
    key::Key
    previous::Union{Nothing,SieveEntry{V,Key}}
    next::Union{Nothing,SieveEntry{V,Key}}
    visited::Bool
    value::V
    SieveEntry{V,Key}() where {V,Key} = new{V,Key}()
end

mutable struct BatchEntry{R,V,Key} <: AbstractEntry{V}
    key::Key
    recent::R
    value::V
    BatchEntry{R,V,Key}() where {R,V,Key} = new{R,V,Key}()
end

mutable struct BatchState{E}
    const queue::Memory{E}
    head::Int
    tail::Int
    nfree::Int
    const blocksize::Int
    epoch::UInt
    const keep_epochs::UInt
end

"""
    ApproxLRUArray{V}(generator, domain; capacity, policy=:sieve, batch_size=nothing, keep_epochs=nothing, kwargs...)

Approximate LRU with `:sieve` (default), `:batch_reinsert`, or `:batch_age`. Batch policies process
`batch_size` entries at a time (default 64, limited by capacity). Age retention uses
`keep_epochs` processed blocks; its default is at least four blocks and one pool pass.
"""
mutable struct ApproxLRUArray{V,N,M,G,Policy,E,A,L,H,D,Offset,R} <: AbstractJITArray{V,N,M}
    const capacity::Int
    count::Int
    const active::A
    oldest::L
    newest::L
    hand::H
    const recency::R
    const domain::D
    const map::M
    const generator::G
end

function ApproxLRUArray{V}(generator::G, domain::VT{N,T}; capacity::Int, policy::Symbol=:sieve,
    storage=nothing, block_shape=nothing, center=nothing, batch_size::Union{Nothing,Int}=nothing,
    keep_epochs::Union{Nothing,Int}=nothing) where {V,G,N,T}
    if capacity < 1
        throw(ArgumentError("capacity must be positive"))
    end
    if !(policy in (:sieve, :batch_reinsert, :batch_age))
        throw(ArgumentError("unknown approximate LRU policy $policy; use :sieve, :batch_reinsert, or :batch_age"))
    end
    if batch_size !== nothing && (policy === :sieve || batch_size < 1)
        throw(ArgumentError("batch_size must be positive and applies only to :batch_reinsert or :batch_age"))
    end
    if keep_epochs !== nothing && (policy !== :batch_age || keep_epochs < 1)
        throw(ArgumentError("keep_epochs must be positive and applies only to :batch_age"))
    end
    kind = policy === :sieve ? SieveEntry : policy === :batch_reinsert ? BatchEntry{Bool} : BatchEntry{UInt}
    map, domain = make_cache_map(kind, V, domain, storage, block_shape, center)
    Key = map isa ArrayMap ? Int : NTuple{length(domain),Int}
    if policy === :sieve
        E = SieveEntry{V,Key}
        recency = nothing
        L = H = Union{Nothing,E}
    else
        E = kind{V,Key}
        blocksize = min(capacity, batch_size === nothing ? 64 : batch_size)
        window = keep_epochs === nothing ? max(4, cld(capacity, blocksize)) : keep_epochs
        recency = BatchState(Memory{E}(undef, capacity), 1, 1, capacity, blocksize, UInt(1), UInt(window))
        L = H = Nothing
    end
    active = preallocate(E, capacity)
    offset = routing_offset(map)
    return ApproxLRUArray{V,length(domain),typeof(map),G,policy,E,typeof(active),L,H,typeof(domain),offset,typeof(recency)}(
        capacity, 0, active, nothing, nothing, nothing, recency, domain, map, generator)
end

@inline cache_offset(::ApproxLRUArray{V,N,M,G,P,E,A,L,H,D,O}) where {V,N,M,G,P,E,A,L,H,D,O} = Val(O)

ApproxLRUArray{V}(generator, axis::Union{Int,AbstractUnitRange{Int}}, axes::Union{Int,AbstractUnitRange{Int}}...; kwargs...) where V = ApproxLRUArray{V}(generator, (axis, axes...); kwargs...)
ApproxLRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} = ApproxLRUArray{V}(generator, domain; capacity, kwargs...)
@inline entrytype(::ApproxLRUArray{V,N,M,G,P,E}) where {V,N,M,G,P,E} = E

@inline function batchappend!(cache, stored)
    state = cache.recency
    @inbounds state.queue[state.tail] = stored
    state.tail = state.tail == cache.capacity ? 1 : state.tail + 1
    return nothing
end

@inline function batchtouch!(state, stored::BatchEntry{Bool})
    if !stored.recent
        stored.recent = true
    end
    return nothing
end
@inline function batchtouch!(state, stored::BatchEntry{UInt})
    if stored.recent != state.epoch
        stored.recent = state.epoch
    end
    return nothing
end
@inline batchrecent(state, stored::BatchEntry{Bool}) = stored.recent
@inline batchrecent(state, stored::BatchEntry{UInt}) =
    stored.recent != 0 && state.epoch - stored.recent < state.keep_epochs
@inline batchreset!(stored::BatchEntry{Bool}) = (stored.recent = false)
@inline batchreset!(stored::BatchEntry{UInt}) = nothing

function batchclean!(cache)
    state = cache.recency
    # A block bounds one pass, not total cleanup: reused entries may survive several passes.
    while state.nfree == 0
        for _ in 1:min(state.blocksize, cache.count)
            @inbounds stored = state.queue[state.head]
            state.head = state.head == cache.capacity ? 1 : state.head + 1
            if batchrecent(state, stored)
                batchreset!(stored)
                batchappend!(cache, stored)
            else
                delete_entry!(cache, stored.key)
                state.nfree += 1
                @inbounds cache.active[state.nfree] = stored
                cache.count -= 1
            end
        end
        state.epoch += UInt(1)
    end
    return nothing
end

@inline function cacheget(cache::ApproxLRUArray{V,N,M,G,P,E}, key, indices=nothing) where {V,N,M,G,P,E<:BatchEntry}
    routing = cache_routing(cache, key)
    slot, hint = prepare(cache.map, key, routing)
    if slot !== nothing
        stored = preparedentry(slot, hint)
        if stored !== cache.map.empty
            batchtouch!(cache.recency, stored)
            return stored.value
        end
    end
    generated = cache.generator(generator_indices(cache, key, indices)...)
    value = generated isa V ? generated : convert(V, generated)
    if cache.map isa TreeMap
        # Retain the destination before cleanup so evicting its last old value cannot prune it.
        slot = reserve!(cache.map, key, routing, slot, hint)
        store_retain!(cache.map, slot)
    end
    if cache.recency.nfree == 0
        batchclean!(cache)
    end
    @inbounds stored = cache.active[cache.recency.nfree]
    cache.recency.nfree -= 1
    stored.key, stored.value = key, value
    stored.recent = P === :batch_reinsert ? false : UInt(0)
    batchappend!(cache, stored)
    cache.count += 1
    if !(cache.map isa TreeMap)
        slot = reserve!(cache.map, key, routing, slot, hint)
        store_retain!(cache.map, slot)
    end
    slot_publish!(slot, stored)
    return value
end

# SIEVE: Zhang et al., SIEVE is Simpler than LRU (NSDI 2024).
# https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo
function victim!(cache::ApproxLRUArray{V,N,M,G,:sieve}) where {V,N,M,G}
    stored = cache.hand === nothing ? cache.oldest : cache.hand
    while stored.visited
        stored.visited = false
        stored = stored.next === nothing ? cache.oldest : stored.next
    end
    cache.hand = stored.next
    return stored
end

@inline function cacheget(cache::ApproxLRUArray{V,N,M,G,:sieve,E}, key, indices=nothing) where {V,N,M,G,E<:SieveEntry}
    routing = cache_routing(cache, key)
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
        delete_entry!(cache, stored.key)
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
