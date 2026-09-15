# Allocate and initialize an atomic reference map for either cache.
function _empty_map(::Type{T}, shape, empty=_EMPTY) where T
    count = 1
    for dimension in shape
        if dimension < 0
            throw(ArgumentError("dimensions must be nonnegative"))
        end
        count = Base.checked_mul(count, dimension)
    end
    map = AtomicMemory{T}(undef, count)
    for address in eachindex(map)
        @atomic :release map[address] = empty
    end
    return map
end

# Dense slots use the same publication operations as dictionary records and tree nodes.
@inline _slot_entry(slot::AtomicMemoryRef) = Core.memoryrefget(slot, :acquire, false)
@inline _slot_publish!(slot::AtomicMemoryRef, value) = Core.memoryrefset!(slot, value, :release, false)
@inline _slot_claim!(slot::AtomicMemoryRef, empty, entry) =
    Core.memoryrefreplace!(slot, empty, entry, :acquire_release, :acquire, false)
@inline _store_hash(::AtomicMemory, key) = nothing
@inline _store_lookup(map::AtomicMemory, key::Int, routing) = @inbounds memoryref(map, key)
@inline _store_reserve!(map::AtomicMemory, key, routing, mode=Val(true)) = _store_lookup(map, key, routing)
_store_retain!(::AtomicMemory, slot) = nothing
@inline _store_delete!(::AtomicMemory, key, slot) = _slot_publish!(slot, _EMPTY)
