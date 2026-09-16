# Sampled LRU cache, entry and constructors; tracks recency and selects eviction victims.

# Sampled LRU with candidate retention: Psounis & Prabhakar (2002), https://doi.org/10.1109/TNET.2002.801414
# Five samples and a persistent 16-candidate pool follow https://github.com/redis/redis/blob/8.2/src/evict.c.

mutable struct SampledLRUEntry{V,Key} <: AbstractEntry{V}
    key::Key
    slot::Int
    version::UInt64
    accessed::UInt64
    value::V
    SampledLRUEntry{V,Key}(key, value) where {V,Key} = new{V,Key}(key, 0, UInt64(0), UInt64(0), value)
end

"""
    SampledLRUArray{V}(generator, dims...; capacity, kwargs...)

Cache values with sampled LRU eviction. Accept sizes or coordinate ranges.
Sparse storage: `:dict` (default), `:msbtree`, `:lsbtree`; trees require `block_shape`.
"""
mutable struct SampledLRUArray{V,N,M,D,G,Key} <: AbstractJITArray{V,N,M}
    const map::M
    const domain::D
    const generator::G
    const active::Vector{SampledLRUEntry{V,Key}}
    ready::Int
    const pool::Vector{Union{Nothing,SampledLRUEntry{V,Key}}}
    const ages::Vector{UInt64}
    const versions::Vector{UInt64}
    pooled::Int
    rng::UInt64
    divisor::UInt64
    reciprocal::UInt64
    counter::UInt64
end

function SampledLRUArray{V}(generator, domain::VT{N,T}; capacity::Int,
    storage=nothing, block_shape=nothing) where {V,N,T<:Union{Int,AbstractUnitRange{Int}}}
    storage = domain_storage(domain, storage)
    @astuple length(domain) domain
    return construct(SampledLRUArray{V}, domain, generator, capacity, storage, block_shape)
end

SampledLRUArray{V}(generator, dims::Int...; kwargs...) where V = SampledLRUArray{V}(generator, dims; kwargs...)

SampledLRUArray{V}(generator, axis::AbstractUnitRange{Int}, axes::AbstractUnitRange{Int}...; kwargs...) where V = SampledLRUArray{V}(generator, (axis, axes...); kwargs...)

SampledLRUArray{V}(domain::VT{N,T}, capacity::Int, generator::Function; kwargs...) where {V,N,T} = SampledLRUArray{V}(generator, domain; capacity, kwargs...)

function construct(::Type{SampledLRUArray{V}}, input::Tuple, generator::G, capacity::Int,
    storage, block) where {V,G}
    capacity < 1 && throw(ArgumentError("capacity must be positive"))
    domain = axis_domain(input, storage)
    Key = entry_keytype(domain, storage)
    E = SampledLRUEntry{V,Key}
    map = make_map(Union{Empty,E}, domain, storage, block)
    return SampledLRUArray{V,length(domain),typeof(map),typeof(domain),G,Key}(
        map, domain, generator, Vector{E}(undef, capacity),
        0, fill!(Vector{Union{Nothing,E}}(undef, 16), nothing), zeros(UInt64, 16), zeros(UInt64, 16), 0,
        0x123456789abcdef0, 0, 0, 0)
end

# Resolve scalar timestamps only at access/publication; batches already carry their stamp.
@inline accessstamp(cache, ::Nothing) = (cache.counter += UInt64(1))
@inline accessstamp(cache, stamp::UInt64) = stamp
@inline batchstamp(cache::SampledLRUArray, count::Int) = (cache.counter += UInt64(count))

# Record access using the access counter; skip unchanged timestamps.
@inline function sampled_touch(cache, entry::SampledLRUEntry, stamp)
    now = accessstamp(cache, stamp)
    previous = entry.accessed
    if previous != now
        entry.accessed = now
    end
    return entry.value
end

function policy_add!(cache, entry::SampledLRUEntry, stamp)
    entry.accessed = accessstamp(cache, stamp)
    cache.ready += 1
    entry.slot = cache.ready
    @inbounds cache.active[entry.slot] = entry
    return nothing
end

function evict!(cache::SampledLRUArray)
    entry = sampled_victim!(cache)
    sampled_detach!(cache, entry)
    delete_key!(cache.map, entry.key)
    return entry
end

# Remove a sampled resident from the compact active array.
function sampled_detach!(cache, entry::SampledLRUEntry)
    @inbounds last = cache.active[cache.ready]
    if last !== entry
        @inbounds cache.active[entry.slot] = last
        last.slot = entry.slot
    end
    cache.ready -= 1
    # Invalidate candidates from this residency without scanning the pool.
    entry.version += UInt64(1)
    return nothing
end

# Advance the cache-local sampler for eviction.
@inline function sampled_random!(cache)
    x = cache.rng
    x ⊻= x << 13
    x ⊻= x >> 7
    x ⊻= x << 17
    cache.rng = x
    return x
end

# Exact remainder, using a reciprocal cached when the resident count changes.
@inline function sample_remainder(x::UInt64, n::UInt64, reciprocal::UInt64)
    q = UInt64(widemul(x, reciprocal) >>> 64)
    r = x - q * n
    if r >= n
        r -= n
    end
    return r
end

# Full pool: discard its minimum and shift only the affected prefix.
@inline function sampled_pooladd!(c, e, age)
    @inbounds begin
        if c.pooled == 16
            if age <= c.ages[1]
                return nothing
            end
            i = 1
            while i < 16
                if c.ages[i + 1] > age
                    break
                end
                c.ages[i] = c.ages[i + 1]
                c.versions[i] = c.versions[i + 1]
                c.pool[i] = c.pool[i + 1]
                i += 1
            end
        else
            i = c.pooled + 1
            while i > 1
                if c.ages[i - 1] <= age
                    break
                end
                c.ages[i] = c.ages[i - 1]
                c.versions[i] = c.versions[i - 1]
                c.pool[i] = c.pool[i - 1]
                i -= 1
            end
            c.pooled += 1
        end
        c.ages[i] = age
        c.versions[i] = e.version
        c.pool[i] = e
    end
    return nothing
end

# Sample resident entries and pop the oldest surviving candidate.
function sampled_victim!(c)
    n = UInt64(c.ready)
    if c.divisor != n
        c.divisor = n
        c.reciprocal = typemax(UInt64) ÷ n
    end
    @inbounds while true
        now = c.counter
        for _ in 1:5
            index = Int(sample_remainder(sampled_random!(c), n, c.reciprocal)) + 1
            e = c.active[index]
            last = e.accessed
            sampled_pooladd!(c, e, now >= last ? now - last : UInt64(0))
        end
        while c.pooled > 0
            e = c.pool[c.pooled]::eltype(c.active)
            version = c.versions[c.pooled]
            c.pool[c.pooled] = nothing
            c.pooled -= 1
            if e.version == version
                return e
            end
        end
    end
end

@inline entrytype(::SampledLRUArray{V,N,M,D,G,Key}) where {V,N,M,D,G,Key} = SampledLRUEntry{V,Key}

@inline function cacheget(cache::SampledLRUArray{V,N,M,D,G,Key}, key, stamp=nothing, indices=nothing) where {V,N,M,D,G,Key}
    routing = store_hash(cache.map, key)
    slot, hint = prepare(cache.map, key, routing)
    if slot !== nothing
        stored = preparedentry(slot, hint)
        if stored isa SampledLRUEntry{V,Key}
            return sampled_touch(cache, stored, stamp)
        end
    end
    value = convert(V, cache.generator(generator_indices(cache, key, indices)...))
    if cache.ready == length(cache.active)
        stored = evict!(cache)
        stored.key = key
        stored.value = value
    else
        stored = SampledLRUEntry{V,Key}(key, value)
    end
    slot = reserve!(cache.map, key, routing, slot, hint)
    store_retain!(cache.map, slot)
    slot_publish!(slot, stored)
    policy_add!(cache, stored, stamp)
    return stored.value
end
