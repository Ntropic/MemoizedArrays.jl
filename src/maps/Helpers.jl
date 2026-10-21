# Allocate and initialize slots for maps and tree blocks.
empty_map(::Type{T}, shape, empty=EMPTY) where T = empty_map(T, shape, empty, Val(false))
function empty_map(::Type{T}, shape, empty, ::Val{Atomic}, ::Val{Check}=Val(:check)) where {T,Atomic,Check}
    count = 1
    for dimension in shape
        if Check !== :nocheck && dimension < 0
            throw(ArgumentError("dimensions must be nonnegative"))
        end
        count = Check === :nocheck ? count * dimension : Base.checked_mul(count, dimension)
    end
    map = Atomic ? AtomicMemory{T}(undef, count) : Memory{T}(undef, count)
    @inbounds for address in eachindex(map)
        if Atomic
            @atomic :monotonic map[address] = empty
        else
            map[address] = empty
        end
    end
    return map
end

# Slot references share read/write operations with dictionary records and tree nodes.
@inline slot_entry(slot::MemoryRef{T}) where T = @inbounds slot[]
@inline slot_publish!(slot::MemoryRef{T}, value) where T = (@inbounds slot[] = value)
@inline slot_entry(slot::GenericMemoryRef{:atomic}) = Core.memoryrefget(slot, :acquire, false)
@inline slot_publish!(slot::GenericMemoryRef{:atomic}, value) = Core.memoryrefset!(slot, value, :release, false)
@inline prepare(map, key, routing) = (map_slot(map, key, routing), nothing)
@inline prepared_entry(slot, hint) = slot_entry(slot)

@inline claim_slot!(ref, expected, desired) =
    Core.memoryrefreplace!(ref, expected, desired, :acquire_release, :acquire, false)

@inline slot_exchange!(ref, stored) = Core.memoryrefswap!(ref, stored, :acquire_release, false)
@inline slot_exchange!(ref, stored, expected) = slot_exchange!(ref, stored)

# These wrappers can outlive the physical slot they originally referenced.
@inline function slot_entry(slot::Union{ExpandingSlot,DictSlot})
    stored = slot_entry(slot_reference(slot))
    while stored === RELOCATED
        slot = relocated_slot(slot)
        stored = slot_entry(slot_reference(slot))
    end
    return stored
end

@inline function claim_slot!(slot::Union{ExpandingSlot,DictSlot}, expected, desired)
    claimed = claim_slot!(slot_reference(slot), expected, desired)
    if claimed.old === RELOCATED
        return relocated_claim!(slot, expected, desired)
    end
    return claimed
end

@noinline function relocated_claim!(slot::Union{ExpandingSlot,DictSlot}, expected, desired)
    while true
        slot = relocated_slot(slot)
        claimed = claim_slot!(slot_reference(slot), expected, desired)
        if claimed.old !== RELOCATED
            return claimed
        end
    end
end
