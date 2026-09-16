# Unbounded cache, entry, constructors and access; defines the shared entry base.

struct Empty end
const EMPTY = Empty()

abstract type AbstractEntry{V} end

struct JITEntry{V} <: AbstractEntry{V}
    value::V
end

"""
    JITArray{V}(generator, dims...; kwargs...)

Cache values on demand without eviction. Accept sizes or coordinate ranges.
Sparse storage: `:dict` (default), `:msbtree`, `:lsbtree`; trees require `block_shape`.
"""
struct JITArray{V,N,M,D,G} <: AbstractJITArray{V,N,M}
    map::M
    domain::D
    generator::G
end

function JITArray{V}(generator, domain::VT{N,T}; storage=nothing,
    block_shape=nothing) where {V,N,T<:Union{Int,AbstractUnitRange{Int}}}
    storage = domain_storage(domain, storage)
    @astuple length(domain) domain
    return construct(JITArray{V}, domain, generator, storage, block_shape)
end

JITArray{V}(generator, dims::Int...; kwargs...) where V = JITArray{V}(generator, dims; kwargs...)

JITArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V = JITArray{V}(generator, (axis, axes...); kwargs...)

JITArray{V}(domain::VT{N,T}, generator::Function; kwargs...) where {V,N,T} = JITArray{V}(generator, domain; kwargs...)

function construct(::Type{JITArray{V}}, input::Tuple, generator::G, storage, block) where {V,G}
    domain = axis_domain(input, storage)
    E = JITEntry{V}
    map = make_map(Union{Empty,E}, domain, storage, block)
    return JITArray{V,length(domain),typeof(map),typeof(domain),G}(map, domain, generator)
end

@inline entrytype(::JITArray{V}) where V = JITEntry{V}

@inline function cacheget(cache::JITArray{V}, key, stamp=nothing, indices=nothing) where V
    routing = store_hash(cache.map, key)
    slot, hint = prepare(cache.map, key, routing)
    if slot !== nothing
        stored = preparedentry(slot, hint)
        if stored isa JITEntry{V}
            return stored.value
        end
    end
    stored = JITEntry{V}(cache.generator(generator_indices(cache, key, indices)...))
    slot = reserve!(cache.map, key, routing, slot, hint)
    store_retain!(cache.map, slot)
    slot_publish!(slot, stored)
    return stored.value
end
