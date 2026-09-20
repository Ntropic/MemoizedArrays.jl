# Unbounded cache, entry, constructors and access; defines the shared entry base.

struct Empty end
const EMPTY = Empty()

abstract type AbstractEntry{V} end

preallocate(::Type{E}, capacity) where E = [E() for _ in 1:capacity]

struct JITEntry{V} <: AbstractEntry{V}
    value::V
end

# Mutable identity distinguishes the shared empty entry from every generated value.
mutable struct ArrayEntry{V} <: AbstractEntry{V}
    const value::V
    ArrayEntry{V}() where V = new{V}()
    ArrayEntry{V}(value) where V = new{V}(value)
end

"""
    JITArray{V}(generator, domain::VT{N,T}; kwargs...)

Cache values on demand without eviction. Domain axes are lengths, integer ranges, or `:`.
Omitted storage selects `:array` for finite domains and `:dict` otherwise.
Explicit storage: `:array`, `:dict`, `:radixtree`, `:subtree`; trees require `block_shape`.
Trees accept `center=0.5` (default) or `center=-0.5` to align the leaf containing zero.
"""
struct JITArray{V,N,M,D,G,Offset} <: AbstractJITArray{V,N,M}
    map::M
    domain::D
    generator::G
end

function JITArray{V}(generator::G, domain::VT{N,T}; storage=nothing,
    block_shape=nothing, center=nothing) where {V,G,N,T}
    map, domain = make_cache_map(JITEntry, V, domain, storage, block_shape, center)
    offset = routing_offset(map)
    return JITArray{V,length(domain),typeof(map),typeof(domain),G,offset}(map, domain, generator)
end

@inline cache_offset(::JITArray{V,N,M,D,G,Offset}) where {V,N,M,D,G,Offset} = Val(Offset)

JITArray{V}(generator, axis::Union{Int,AbstractUnitRange{Int}}, axes::Union{Int,AbstractUnitRange{Int}}...; kwargs...) where V = JITArray{V}(generator, (axis, axes...); kwargs...)
JITArray{V}(domain::VT{N,T}, generator::Function; kwargs...) where {V,N,T} = JITArray{V}(generator, domain; kwargs...)

@inline entrytype(cache::JITArray{V}) where V = cache.map.empty === EMPTY ? JITEntry{V} : typeof(cache.map.empty)

# Pass encoded coordinates as scalars; recover generator coordinates only on a miss.
# Inline generation into an existing leaf; keep structural misses out of the read path.
@noinline function treemiss!(cache::JITArray{V,N,M,D,G,Offset}, leaf, parent, address,
    encoded::Vararg{UInt,N}) where {V,N,M,D,G,Offset}
    map = cache.map
    slot = if leaf === nothing
        link = parent === nothing ? rootlink(map) : store_lookup(parent.children, address, nothing)
        treereserve!(map, encoded, link, parent, address)
    else
        LeafSlot(leaf, address)
    end
    key = ntuple(i -> reinterpret(Int, encoded[i] - Offset[i]), Val(N))
    stored = entrytype(cache)(cache.generator(key...))
    store_retain!(map, slot)
    slot_publish!(slot, stored)
    return stored.value
end

@inline function cacheget(cache::JITArray{V}, key, indices=nothing) where V
    routing = cache_routing(cache, key)
    slot, hint = prepare(cache.map, key, routing)
    if slot !== nothing
        stored = if cache.map isa SubtreeRadixMap
            slot_entry(store_lookup(slot.values, hint[3], nothing), cache.map)
        else
            preparedentry(slot, hint)
        end
        if stored !== cache.map.empty
            if cache.map isa SubtreeRadixMap
                # Only the sentinel has an uninitialized value field.
                assume_initialized(isdefined(stored, :value))
            end
            return stored.value
        end
        if cache.map isa TreeMap
            encoded = cache.map isa SubtreeRadixMap ? routing[1] : routing
            return @inline treemiss!(cache, slot, nothing, hint[3], encoded...)
        end
    end
    if cache.map isa TreeMap
        encoded = cache.map isa SubtreeRadixMap ? routing[1] : routing
        return treemiss!(cache, slot, hint[2], hint[3], encoded...)
    end
    stored = entrytype(cache)(cache.generator(generator_indices(cache, key, indices)...))
    slot = reserve!(cache.map, key, routing, slot, hint)
    store_retain!(cache.map, slot)
    slot_publish!(slot, stored)
    return stored.value
end
