# A serial entry's permanent ID is its local link position.
mutable struct SerialLRUEntry{V,K}
    const id::UInt32
    key::K
    value::V
    SerialLRUEntry{V,K}(id::UInt32) where {V,K} = new{V,K}(id)
end

mutable struct SerialLRUQueue{E}
    const capacity::Int
    count::Int
    const entries::Vector{E}
    const links::MemoryRef{UInt32}
    oldest::UInt32
    newest::UInt32
end

function SerialLRUQueue(::Type{E}, capacity) where E
    entries = [E(UInt32(i)) for i in 1:capacity]
    links = memoryref(Memory{UInt32}(undef, 2capacity))
    return SerialLRUQueue{E}(capacity, 0, entries, links, 0, 0)
end

function map_entry(::Type{SerialLRUEntry}, ::Type{V}, Key, _) where V
    E = SerialLRUEntry{V,Key}
    return E, E(UInt32(0))
end

# The caller completes generation and conversion first, so nested requests can
# finish changing the queue before we select its current insertion target.
@inline function serial_insert!(cache, q::SerialLRUQueue, key, value)
    if q.count == q.capacity
        index = q.oldest
        stored = @inbounds q.entries[index]
        delete_entry!(cache, stored.key)
        stored.key = key
        stored.value = value
        queue_touch!(q, index)
    else
        index = UInt32(q.count + 1)
        stored = @inbounds q.entries[index]
        stored.key = key
        stored.value = value
        @inbounds begin
            if q.count == 0
                q.oldest = index
                memoryref(q.links, 2Int(index) - 1)[] = index
            else
                memoryref(q.links, 2Int(q.newest))[] = index
                memoryref(q.links, 2Int(index) - 1)[] = q.newest
            end
            memoryref(q.links, 2Int(index))[] = index
        end
        q.newest = index
        q.count += 1
    end
    return stored
end
