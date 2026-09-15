mutable struct ExactLRUEntry{V,Key} <: AbstractEntry{V}
    @atomic state::Union{Nothing,Threads.Condition,Bool,Empty}
    key::Key
    previous::Union{Nothing,ExactLRUEntry{V,Key}}
    next::Union{Nothing,ExactLRUEntry{V,Key}}
    value::V
    ExactLRUEntry{V,Key}(key) where {V,Key} = new{V,Key}(nothing, key, nothing, nothing)
end

"""
    ExactLRUArray{V}(generator, dims...; capacity, kwargs...)

Cache values with exact LRU eviction. Accept sizes or coordinate ranges.
Sparse storage: `:dict` (default), `:msbtree`, `:lsbtree`; trees require `block_shape`.
"""
mutable struct ExactLRUArray{V,N,E,M,D,G} <: AbstractJITArray{V,N}
    const map::M
    const domain::D
    const generator::G
    oldest::Union{Nothing,E}
    newest::Union{Nothing,E}
    const capacity::Int
    count::Int
    const available::Threads.Condition
end

function ExactLRUArray{V}(generator, domain::VT{N,T}; capacity::Int,
    storage=nothing, block_shape=nothing) where {V,N,T<:Union{Int,AbstractUnitRange{Int}}}
    storage = domain_storage(domain, storage)
    @astuple length(domain) domain
    return construct(ExactLRUArray{V}, domain, generator, capacity, storage, block_shape)
end

ExactLRUArray{V}(generator, dims::Int...; kwargs...) where V =
    ExactLRUArray{V}(generator, dims; kwargs...)

ExactLRUArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V =
    ExactLRUArray{V}(generator, (axis, axes...); kwargs...)

ExactLRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} =
    ExactLRUArray{V}(generator, domain; capacity, kwargs...)

function construct(::Type{ExactLRUArray{V}}, input::Tuple, generator::G, capacity::Int,
    storage, block) where {V,G}
    capacity < 1 && throw(ArgumentError("capacity must be positive"))
    domain = axis_domain(input, storage)
    E = ExactLRUEntry{V,entry_keytype(domain, storage)}
    map = make_map(Union{Empty,E}, domain, storage, block)
    return ExactLRUArray{V,length(domain),E,typeof(map),typeof(domain),G}(
        map, domain, generator, nothing, nothing, capacity, 0, Threads.Condition())
end

# Remove a ready entry from the access-order list; caller holds the bookkeeping lock.
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

# Append a detached ready entry as newest; caller holds the bookkeeping lock.
@inline function append_entry!(cache, entry::ExactLRUEntry)
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

@inline policy_full(cache, ::Type{<:ExactLRUEntry}) = cache.count == cache.capacity
@inline policy_no_ready(cache, ::Type{<:ExactLRUEntry}) = cache.oldest === nothing
@inline policy_add!(cache, entry::ExactLRUEntry, stamp) = append_entry!(cache, entry)

@inline function policy_touch(cache, entry::ExactLRUEntry, stamp)
    if cache.newest !== entry
        unlink!(cache, entry)
        append_entry!(cache, entry)
    end
    return entry.value
end

function policy_claim!(cache, entry::ExactLRUEntry)
    cache.count += 1
    return nothing
end

function policy_release!(cache, entry::ExactLRUEntry)
    cache.count -= 1
    return nothing
end

function policy_evict!(cache, ::Type{<:ExactLRUEntry})
    entry = cache.oldest
    unlink!(cache, entry)
    cache.count -= 1
    delete_entry!(cache, entry)
    return nothing
end
