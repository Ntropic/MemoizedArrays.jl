# Sampled LRU with candidate retention: Psounis & Prabhakar (2002), https://doi.org/10.1109/TNET.2002.801414
# Five samples and a persistent 16-candidate pool follow https://github.com/redis/redis/blob/8.2/src/evict.c.

mutable struct SampledLRUEntry{V,Key} <: AbstractEntry{V}
    @atomic state::Union{Nothing,Threads.Condition,Bool,Empty}
    key::Key
    slot::Int
    live::Bool
    @atomic accessed::UInt64
    value::V
    SampledLRUEntry{V,Key}(key) where {V,Key} = new{V,Key}(nothing, key, 0, false, UInt64(0))
end

# Resolve scalar timestamps only at access/publication; batches already carry their stamp.
@inline _access_stamp(cache, ::Nothing) = _clock_tick!(cache.clock)
@inline _access_stamp(cache, stamp::UInt64) = stamp

# Record access using the selected clock; skip unchanged timestamps.
@inline function _sampled_touch(cache, entry::SampledLRUEntry, stamp)
    now = _access_stamp(cache, stamp)
    previous = @atomic :monotonic entry.accessed
    if previous != now
        @atomic :monotonic entry.accessed = now
    end
    return entry.value
end

@inline _policy_full(cache, ::Type{<:SampledLRUEntry}) = cache.ready + cache.pending == length(cache.active)
@inline _policy_no_ready(cache, ::Type{<:SampledLRUEntry}) = cache.ready == 0
@inline _policy_touch(cache, entry::SampledLRUEntry, stamp) = _sampled_touch(cache, entry, stamp)

function _policy_claim!(cache, entry::SampledLRUEntry)
    cache.pending += 1
    return nothing
end

function _policy_release!(cache, entry::SampledLRUEntry)
    cache.pending -= 1
    return nothing
end

function _policy_add!(cache, entry::SampledLRUEntry, stamp)
    @atomic :monotonic entry.accessed = _access_stamp(cache, stamp)
    cache.ready += 1
    entry.slot = cache.ready
    entry.live = true
    @inbounds cache.active[entry.slot] = entry
    cache.pending -= 1
    return nothing
end

function _policy_evict!(cache, ::Type{<:SampledLRUEntry})
    entry = _sampled_victim!(cache)
    _sampled_detach!(cache, entry)
    _delete_entry!(cache, entry)
    return nothing
end

# Remove a sampled resident from the compact active array; caller holds its lock.
function _sampled_detach!(cache, entry::SampledLRUEntry)
    @inbounds last = cache.active[cache.ready]
    if last !== entry
        @inbounds cache.active[entry.slot] = last
        last.slot = entry.slot
    end
    cache.ready -= 1
    entry.live = false
    return nothing
end

# Advance the cache-local sampler while holding the bookkeeping lock.
@inline function _sampled_random!(cache)
    x = cache.rng
    x ⊻= x << 13
    x ⊻= x >> 7
    x ⊻= x << 17
    cache.rng = x
    return x
end

# Exact remainder, using a reciprocal cached when the resident count changes.
@inline function _sample_remainder(x::UInt64, n::UInt64, reciprocal::UInt64)
    q = UInt64(widemul(x, reciprocal) >>> 64)
    r = x - q * n
    if r >= n
        r -= n
    end
    return r
end

# Full pool: discard its minimum and shift only the affected prefix.
@inline function _sampled_pooladd!(c, e, age)
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
                c.pool[i] = c.pool[i - 1]
                i -= 1
            end
            c.pooled += 1
        end
        c.ages[i] = age
        c.pool[i] = e
    end
    return nothing
end

# Sample resident entries and pop the oldest surviving candidate.
function _sampled_victim!(c)
    n = UInt64(c.ready)
    if c.divisor != n
        c.divisor = n
        c.reciprocal = typemax(UInt64) ÷ n
    end
    @inbounds while true
        now = _clock_now(c.clock)
        for _ in 1:5
            index = Int(_sample_remainder(_sampled_random!(c), n, c.reciprocal)) + 1
            e = c.active[index]
            last = @atomic :monotonic e.accessed
            _sampled_pooladd!(c, e, now >= last ? now - last : UInt64(0))
        end
        while c.pooled > 0
            e = c.pool[c.pooled]::eltype(c.active)
            c.pool[c.pooled] = nothing
            c.pooled -= 1
            if e.live
                return e
            end
        end
    end
end
