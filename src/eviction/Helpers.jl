# Shared link operations for serial and privately owned concurrent queues.

@inline function queue_touch!(q,index)
    @inbounds if index != q.newest
        previous,next = memoryref(q.links,2Int(index)-1)[],memoryref(q.links,2Int(index))[]
        if index == q.oldest
            q.oldest = next
            memoryref(q.links,2Int(next)-1)[] = next
        else
            memoryref(q.links,2Int(previous))[] = next
            memoryref(q.links,2Int(next)-1)[] = previous
        end
        memoryref(q.links,2Int(q.newest))[] = index
        memoryref(q.links,2Int(index)-1)[] = q.newest
        memoryref(q.links,2Int(index))[] = index
        q.newest = index
    end
    return nothing
end

# Remove only local membership; the vacated link becomes the local free-index list.
@inline function queue_unlink!(q,index)
    @inbounds begin
        previous,next = memoryref(q.links,2Int(index)-1)[],memoryref(q.links,2Int(index))[]
        if q.count == 1
            q.oldest = q.newest = 0
        elseif index == q.oldest
            q.oldest = next
            memoryref(q.links,2Int(next)-1)[] = next
        elseif index == q.newest
            q.newest = previous
            memoryref(q.links,2Int(previous))[] = previous
        else
            memoryref(q.links,2Int(previous))[] = next
            memoryref(q.links,2Int(next)-1)[] = previous
        end
        memoryref(q.links,2Int(index)-1)[] = 0
        memoryref(q.links,2Int(index))[] = q.vacant
        q.vacant = index
        q.count -= 1
    end
    return nothing
end

@inline function queue_append!(q,index)
    @inbounds begin
        q.vacant = memoryref(q.links,2Int(index))[]
        if q.count == 0
            q.oldest = index
            memoryref(q.links,2Int(index)-1)[] = index
        else
            memoryref(q.links,2Int(q.newest))[] = index
            memoryref(q.links,2Int(index)-1)[] = q.newest
        end
        memoryref(q.links,2Int(index))[] = index
        q.newest = index
        q.count += 1
    end
    return nothing
end
