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
    JITArray{V}(generator, dims...; kwargs...)

Cache values on demand without eviction. Accept sizes or coordinate ranges.
Sparse storage: `:dict` (default), `:radixtree`, `:subtreetree`; trees require `block_shape`.
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
    if (domain isa Tuple{Vararg{Int}} && storage === nothing) || storage === :radixtree || storage === :subtreetree
        E = ArrayEntry{V}
        map = make_map(E, domain, storage, block, E())
    else
        map = make_map(Union{Empty,JITEntry{V}}, domain, storage, block)
    end
    return JITArray{V,length(domain),typeof(map),typeof(domain),G}(map, domain, generator)
end

@inline entrytype(cache::JITArray{V}) where V = cache.map.empty === EMPTY ? JITEntry{V} : typeof(cache.map.empty)

@inline function cacheget(cache::JITArray{V}, key, indices=nothing) where V
    routing = store_hash(cache.map, key)
    if cache.map isa RadixMap
        slot, hint = store_reserve!(cache.map, key, routing), nothing
    else
        slot, hint = prepare(cache.map, key, routing)
    end
    if slot !== nothing
        stored = preparedentry(slot, hint)
        if stored !== cache.map.empty
            return stored.value
        end
    end
    stored = entrytype(cache)(cache.generator(generator_indices(cache, key, indices)...))
    if !(cache.map isa RadixMap)
        slot = reserve!(cache.map, key, routing, slot, hint)
    end
    store_retain!(cache.map, slot)
    slot_publish!(slot, stored)
    return stored.value
end
