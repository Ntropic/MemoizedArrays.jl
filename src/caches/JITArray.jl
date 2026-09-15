struct Empty end
const _EMPTY = Empty()

abstract type AbstractEntry{V} end

mutable struct Entry{V} <: AbstractEntry{V}
    @atomic state::Union{Nothing,Threads.Condition,Bool,Empty}
    value::V
    Entry{V}() where V = new{V}(nothing)
end

"""
    JITArray{V}(generator, dims...; kwargs...)

Cache values on demand without eviction. Accept sizes or coordinate ranges.
Sparse storage: `:dict` (default), `:msbtree`, `:lsbtree`; trees require `block_shape`.
"""
struct JITArray{V,N,E,M,D,G} <: AbstractJITArray{V,N}
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

JITArray{V}(generator, dims::Int...; kwargs...) where V =
    JITArray{V}(generator, dims; kwargs...)

JITArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    JITArray{V}(generator, (axis, axes...); kwargs...)

JITArray{V}(domain::VT{N,T}, generator::Function; kwargs...) where {V,N,T} =
    JITArray{V}(generator, domain; kwargs...)

function construct(::Type{JITArray{V}}, input::Tuple, generator::G, storage, block) where {V,G}
    domain = axis_domain(input, storage)
    E = Entry{V}
    map = make_map(Union{Empty,E}, domain, storage, block)
    return JITArray{V,length(domain),E,typeof(map),typeof(domain),G}(map, domain, generator)
end
