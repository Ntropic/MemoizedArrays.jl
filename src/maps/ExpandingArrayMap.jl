struct ExpandingArrayMapConfig{V,D,I} <: MapConfig{V}
    domain::D
    increment::I
end

expansion_increments(increment::Int, ::Val{N}) where N = ntuple(_ -> increment, Val(N))
function expansion_increments(increment::Tuple, ::Val{N}) where N
    length(increment) == N || throw(DimensionMismatch("expected $N axis increments"))
    return increment
end

# Contiguous storage. Expansion preserves entries, but invalidates slot addresses.
"""
    ExpandingArrayMap{T}(domain; increment)

Contiguous storage with fixed lower bounds and growing upper bounds.
`increment` is a positive `Int` for all axes, or a tuple/vector of positive `Int`s,
one per axis. Growth adds multiples to the current size; existing values are preserved.
"""
mutable struct ExpandingArrayMap{T,N}
    reference::MemoryRef{T}
    shape::NTuple{N,Int}
    strides::NTuple{N,Int}
    const lower::NTuple{N,Int}
    const increment::NTuple{N,Int}
    const empty::T
end

function ExpandingArrayMap{V}(domain::VT;
    increment::Union{Int,Tuple{Vararg{Int}},AbstractVector{Int}}) where V
    domain = Tuple(domain)
    steps = expansion_increments(increment isa Int ? increment : Tuple(increment), Val(length(domain)))
    any(<(1), steps) && throw(ArgumentError("increments must be positive"))
    increment isa Int && increment < 1 && throw(ArgumentError("increments must be positive"))
    return ExpandingArrayMapConfig{V,typeof(domain),typeof(steps)}(domain, steps)
end

function ExpandingArrayMap(::Type{T}, shape::NTuple{N,Int}, lower::NTuple{N,Int},
    empty::T, increment) where {T,N}
    steps = expansion_increments(increment, Val(N))
    slots = empty_map(T, shape, empty)
    return ExpandingArrayMap{T,N}(memoryref(slots), shape,
        Base.size_to_strides(1, shape...), lower, steps, empty)
end

function make_cache_map(kind, ::Type{V}, config::ExpandingArrayMapConfig;
    concurrent::Bool=false) where V
    domain = map(axis_domain, config.domain)
    if any(isunbound, domain)
        throw(ArgumentError("ExpandingArrayMap requires a finite initial domain"))
    end
    if concurrent
        S = concurrent_value_type(V)
        Stored = if kind === MemoizedEntry
            Union{Empty,Generating,GenerationFailure,Relocated,S}
        else
            Union{Empty,Generating,GenerationFailure,GenerateWait{S},Relocated,SharedLRUEntry{S,NTuple{length(domain),Int}}}
        end
        return ConcurrentExpandingArrayMap(Stored, map(length, domain), map(first, domain), config.increment), domain
    end
    E, empty = map_entry(kind, V, NTuple{length(domain),Int}, config)
    return ExpandingArrayMap(E, map(length, domain), map(first, domain), empty, config.increment), domain
end

# Allocate and copy before changing the map; failure leaves the old allocation intact.
function expand_to!(map::ExpandingArrayMap{T,N}, shape::NTuple{N,Int}) where {T,N}
    if any(shape .< map.shape)
        throw(ArgumentError("ExpandingArrayMap cannot shrink"))
    end
    if shape == map.shape
        return map
    end
    for i in 1:N
        Base.checked_add(map.lower[i], shape[i] - 1)
    end
    slots = empty_map(T, shape, map.empty)
    strides = Base.size_to_strides(1, shape...)
    previous = map.reference.mem
    if all(i -> shape[i] == map.shape[i], 1:N-1)
        copyto!(slots, 1, previous, 1, length(previous))
    elseif !isempty(previous)
        # Copy intact first-axis rows into their new positions.
        width = map.shape[1]
        source = 1
        for coordinate in CartesianIndices(Base.tail(map.shape))
            destination = 1
            @inbounds for i in 2:N
                destination += (coordinate[i-1] - 1) * strides[i]
            end
            copyto!(slots, destination, previous, source, width)
            source += width
        end
    end
    map.reference = memoryref(slots)
    map.shape = shape
    map.strides = strides
    return map
end

@inline function slot_address(map::ExpandingArrayMap{T,N}, key::NTuple{N,Int}) where {T,N}
    address = 1
    @inbounds for i in 1:N
        address += (key[i] - 1) * map.strides[i]
    end
    return address
end

@inline map_routing(::ExpandingArrayMap, key) = nothing
@inline function map_slot(map::ExpandingArrayMap{T,N}, key::NTuple{N,Int}, routing) where {T,N}
    if any(i -> reinterpret(UInt, key[i] - 1) >= UInt(map.shape[i]), 1:N)
        return nothing
    end
    return @inbounds memoryref(map.reference, slot_address(map, key))
end
@inline prepare(map::ExpandingArrayMap, key, routing) = (map_slot(map, key, routing), nothing)
map_retain!(::ExpandingArrayMap, slot) = nothing

# Resident entries keep local coordinates, which survive changed strides. Their
# coordinates are already valid; only the current linear address is needed.
@inline function delete_key!(map::ExpandingArrayMap, key)
    slot = @inbounds memoryref(map.reference, slot_address(map, key))
    return slot_publish!(slot, map.empty)
end

function map_reserve!(map::ExpandingArrayMap{T,N}, key::NTuple{N,Int}, routing) where {T,N}
    shape = ntuple(N) do i
        current = map.shape[i]
        if key[i] <= current
            return current
        end
        extra = Base.checked_mul(cld(key[i] - current, map.increment[i]), map.increment[i])
        return Base.checked_add(current, extra)
    end
    expand_to!(map, shape)
    return @inbounds memoryref(map.reference, slot_address(map, key))
end

# A nested generator may have changed both the allocation and its strides.
@inline function reserve!(map::ExpandingArrayMap, key, routing, slot, hint)
    if slot !== nothing && slot.mem === map.reference.mem
        return slot
    end
    return map_reserve!(map, key, routing)
end

# A mutable snapshot is a single pointer, so its atomic publication also publishes
# a coherent reference/shape/strides tuple.
mutable struct ExpandingAllocation{T,N}
    const reference::GenericMemoryRef{:atomic,T,Core.CPU}
    const shape::NTuple{N,Int}
    const strides::NTuple{N,Int}
    @atomic successor::ExpandingAllocation{T,N}

    function ExpandingAllocation(reference::GenericMemoryRef{:atomic,T,Core.CPU},
        shape::NTuple{N,Int}, strides::NTuple{N,Int}) where {T,N}
        allocation = new{T,N}(reference, shape, strides)
        @atomic :monotonic allocation.successor = allocation
        return allocation
    end
end

struct Relocated end
const RELOCATED = Relocated()

mutable struct ConcurrentExpandingArrayMap{T,N}
    @atomic allocation::ExpandingAllocation{T,N}
    const lower::NTuple{N,Int}
    const increment::NTuple{N,Int}
    const empty::Empty
    const growth::ReentrantLock
end

function ConcurrentExpandingArrayMap(::Type{T}, shape::NTuple{N,Int}, lower::NTuple{N,Int}, increment) where {T,N}
    steps = expansion_increments(increment, Val(N))
    slots = empty_map(T, shape, EMPTY, Val(true))
    allocation = ExpandingAllocation(memoryref(slots), shape, Base.size_to_strides(1, shape...))
    return ConcurrentExpandingArrayMap(allocation, lower, steps, EMPTY, ReentrantLock())
end

@inline current_allocation(map::ConcurrentExpandingArrayMap) = @atomic :acquire map.allocation
@inline function slot_address(allocation::ExpandingAllocation{T,N}, key::NTuple{N,Int}) where {T,N}
    address = 1
    @inbounds for i in 1:N
        address += (key[i] - 1) * allocation.strides[i]
    end
    return address
end
@inline allocation_contains(allocation, key::NTuple{N,Int}) where N =
    all(i -> reinterpret(UInt, key[i] - 1) < UInt(allocation.shape[i]), 1:N)
@inline allocation_slot(allocation, key) = @inbounds memoryref(allocation.reference, slot_address(allocation, key))

# Prepare every destination before exposing it. An old-coordinate destination
# cannot accept requests until its source slot has been frozen and copied.
function prepare_expansion!(previous::ExpandingAllocation{T,N}, shape::NTuple{N,Int}) where {T,N}
    slots = empty_map(T, shape, EMPTY, Val(true))
    allocation = ExpandingAllocation(memoryref(slots), shape, Base.size_to_strides(1, shape...))
    for coordinate in CartesianIndices(previous.shape)
        slot_publish!(allocation_slot(allocation, Tuple(coordinate)), RELOCATED)
    end
    @atomic :release previous.successor = allocation
    return allocation
end

@inline function relocate_slot!(previous::ExpandingAllocation{T}, allocation, key) where T
    oldref = allocation_slot(previous, key)
    stored = if typeintersect(T, SharedLRUEntry) !== Union{}
        # Every LRU slot can change, including a completed entry.
        slot_exchange!(oldref, RELOCATED)
    else
        value = slot_entry(oldref)
        while value === EMPTY || value === GENERATING
            frozen = claim_slot!(oldref, value, RELOCATED)
            if frozen.success
                break
            end
            value = frozen.old
        end
        value
    end
    slot_publish!(allocation_slot(allocation, key), stored)
    return nothing
end

function grow_allocation!(map::ConcurrentExpandingArrayMap{T,N}, key::NTuple{N,Int}) where {T,N}
    lock(map.growth) do
        previous = current_allocation(map)
        if allocation_contains(previous, key)
            return previous
        end
        shape = ntuple(N) do i
            current = previous.shape[i]
            if key[i] <= current
                return current
            end
            extra = Base.checked_mul(cld(key[i] - current, map.increment[i]), map.increment[i])
            count = Base.checked_add(current, extra)
            Base.checked_add(map.lower[i], count - 1)
            return count
        end
        # All allocations and validation precede freezing any old slots.
        allocation = prepare_expansion!(previous, shape)
        for coordinate in CartesianIndices(previous.shape)
            relocate_slot!(previous, allocation, Tuple(coordinate))
        end
        @atomic :release map.allocation = allocation
        # A long-lived lookup must not retain every intermediate allocation.
        @atomic :release previous.successor = previous
        return allocation
    end
end

# Keep the lookup's allocation and coordinates together. A relocated slot is
# immutable; LRU mutations follow it without taking the structural growth lock.
struct ExpandingSlot{T,N}
    map::ConcurrentExpandingArrayMap{T,N}
    allocation::ExpandingAllocation{T,N}
    reference::GenericMemoryRef{:atomic,T,Core.CPU}
    key::NTuple{N,Int}
end

@noinline function relocated_slot(slot::ExpandingSlot)
    allocation, reference = relocated_slot(slot.map, slot.allocation, slot.key)
    return ExpandingSlot(slot.map, allocation, reference, slot.key)
end

@inline function slot_entry(slot::ExpandingSlot)
    stored = slot_entry(slot.reference)
    while stored === RELOCATED
        slot = relocated_slot(slot)
        stored = slot_entry(slot.reference)
    end
    return stored
end

@inline function claim_slot!(slot::ExpandingSlot, expected, desired)
    claimed = claim_slot!(slot.reference, expected, desired)
    if claimed.old === RELOCATED
        return relocated_claim!(slot, expected, desired)
    end
    return claimed
end

@noinline function relocated_claim!(slot::ExpandingSlot, expected, desired)
    while true
        slot = relocated_slot(slot)
        claimed = claim_slot!(slot.reference, expected, desired)
        if claimed.old !== RELOCATED
            return claimed
        end
    end
end

@inline function slot_exchange!(slot::ExpandingSlot, stored, previous)
    claimed = claim_slot!(slot, previous, stored)
    if claimed.success
        return previous
    end
    return retry_exchange!(slot, stored, claimed.old)
end

@noinline function retry_exchange!(slot::ExpandingSlot, stored, previous)
    while true
        claimed = claim_slot!(slot, previous, stored)
        if claimed.success
            return previous
        end
        previous = claimed.old
    end
end

@inline function remove_mapping!(map::ConcurrentExpandingArrayMap, entry)
    key = @atomic :acquire entry.key
    allocation = current_allocation(map)
    slot = ExpandingSlot(map, allocation, allocation_slot(allocation, key), key)
    # Entry ownership prevents another request from replacing this mapping.
    claim_slot!(slot, entry, EMPTY)
    return nothing
end

queue_key(::ConcurrentExpandingArrayMap{T,N}) where {T,N} = ntuple(_ -> 0, Val(N))

# Copied slots are usable before the whole allocation is published. Only the
# freeze-to-copy interval for this coordinate needs waiting.
@noinline function relocated_slot(map::ConcurrentExpandingArrayMap, previous, key)
    reference = allocation_slot(previous, key)
    polls = 0
    while true
        allocation = @atomic :acquire previous.successor
        if allocation !== previous
            return allocation, allocation_slot(allocation, key)
        end
        allocation = current_allocation(map)
        # A successor still being filled must never redirect to its predecessor.
        # Shapes only grow, so a larger shape identifies a completed relocation
        # whose temporary successor reference has already been cleared.
        if any(allocation.shape .> previous.shape)
            return allocation, allocation_slot(allocation, key)
        end
        if slot_entry(reference) !== RELOCATED
            return previous, reference
        end
        GC.safepoint()
        polls += 1
        if polls == 1024
            yield()
            polls = 0
        end
    end
end

# Completion races with growth: either publish first and be copied, or follow
# the moved marker and publish once at the corresponding new address.
@inline function publish_generated!(map::ConcurrentExpandingArrayMap, key, allocation, ref, stored)
    published = claim_slot!(ref, GENERATING, stored)
    while !published.success
        allocation, ref = relocated_slot(map, allocation, key)
        published = claim_slot!(ref, GENERATING, stored)
    end
    return nothing
end

@inline map_routing(::ConcurrentExpandingArrayMap, key) = nothing
@inline function map_slot(map::ConcurrentExpandingArrayMap, key, routing)
    allocation = current_allocation(map)
    return allocation_contains(allocation, key) ? allocation_slot(allocation, key) : nothing
end

# Readiness inspection must follow the same relocation as indexed reads.
@inline function entry(cache::AbstractMemoizedArray{V,N,M}, key) where {V,N,M<:ConcurrentExpandingArrayMap}
    map = cache.map
    allocation = current_allocation(map)
    if !allocation_contains(allocation, key)
        return EMPTY
    end
    return slot_entry(ExpandingSlot(map, allocation, allocation_slot(allocation, key), key))
end

const ExpandingBacked{V,N} = AbstractMemoizedArray{V,N,M} where {M<:Union{ExpandingArrayMap,ConcurrentExpandingArrayMap}}
Base.size(cache::ExpandingBacked) = cache.map.shape
Base.size(cache::AbstractMemoizedArray{V,N,M}) where {V,N,M<:ConcurrentExpandingArrayMap} = current_allocation(cache.map).shape
Base.axes(cache::ExpandingBacked) = map(cache.map.lower, size(cache)) do lower, count
    lower == 1 ? Base.OneTo(count) : AxisExtent(lower, Base.checked_add(lower, count - 1))
end

@inline function cache_key(cache::ExpandingBacked{V,N}, indices::NTuple{N,Int}) where {V,N}
    return ntuple(N) do i
        if indices[i] < cache.map.lower[i]
            throw(BoundsError(cache, indices))
        end
        # Growth and coordinate-overflow checks must also run under @inbounds.
        # With index >= lower, an unrepresentable extent wraps to a nonpositive Int.
        key = indices[i] - cache.map.lower[i] + 1
        if key < 1
            throw(OverflowError("coordinate extent exceeds Int"))
        end
        return key
    end
end

@inline function map_lookup(cache::ExpandingBacked{V,N}, indices::NTuple{N,Int}) where {V,N}
    key = cache_key(cache, indices)
    slot, hint = prepare(cache.map, key, nothing)
    return key, nothing, slot, hint
end

@inline function map_lookup(cache::AbstractMemoizedArray{V,N,M}, indices::NTuple{N,Int}) where {V,N,M<:ConcurrentExpandingArrayMap}
    key = cache_key(cache, indices)
    allocation = current_allocation(cache.map)
    ref = allocation_contains(allocation, key) ? allocation_slot(allocation, key) : nothing
    return key, allocation, ref, nothing
end


function assignment_slot(map::ConcurrentExpandingArrayMap,key,allocation,reference,hint)
    if reference === nothing
        allocation = grow_allocation!(map,key)
        reference = allocation_slot(allocation,key)
    end
    return ExpandingSlot(map,allocation,reference,key)
end
