function read_workload(rng, keys)
    readpattern == "cycle" && return keys, readrepeats
    @assert readpattern == "varying"
    @assert readrepeats % 16 == 0
    return [rand(rng, keyset, 16length(keyset)) for keyset in keys], readrepeats ÷ 16
end

# Bring existing empty slots and preallocated LRU memory onto the requesting core.
# Dict records are allocated during timed generation; setup does not reserve keys.
function touch_empty!(reader,keys)
    cache = reader isa MemoizedArrays.QueueHandle ? reader.cache : reader
    JA = MemoizedArrays
    if !(cache.map isa Union{JA.DictMap,JA.LRUDictMap,JA.ConcurrentDictMap})
        for key in keys
            ref = JA.map_slot(cache.map,JA.cache_key(cache,(key,)),nothing)
            @assert JA.slot_entry(ref) === cache.map.empty
            JA.slot_publish!(ref,cache.map.empty)
        end
    end
    if reader isa MemoizedArrays.QueueHandle
        q = reader.queue
        resetkey = zero.(JA.cache_key(cache,(1,)))
        @assert q.count == 0
        fill!(q.positions.mem,0)
        fill!(q.incarnations.mem,0)
        for i in 1:q.capacity
            entry=q.pool[i]
            @atomic :monotonic entry.state=UInt64(2)
            @atomic :monotonic entry.last_access=UInt64(0)
            @atomic :monotonic entry.key=resetkey
            @atomic :monotonic entry.value=JA.EMPTY
            q.free[i]=entry
            q.pool[i]=entry
            q.entries[i]=entry
            q.links.mem[2i-1]=0
            q.links.mem[2i]=UInt32(i==q.capacity ? 0 : i+1)
        end
        q.count=0
        q.oldest=q.newest=UInt32(0)
        q.vacant=UInt32(1)
        q.free_count=q.capacity
    elseif reader isa LRUMemoizedArray
        q = reader.eviction
        resetkey = zero.(JA.cache_key(cache,(1,)))
        Key = fieldtype(eltype(q.entries),:key)
        if Key <: JA.DictLocation
            resetkey = Key(resetkey,hash(resetkey),0)
        end
        @assert q.count == 0
        for i in 1:q.capacity
            entry=q.entries[i]
            entry.key=resetkey
            entry.value=payload
            q.entries[i]=entry
        end
        fill!(q.links.mem,0)
        for i in 1:q.capacity
            q.links.mem[2i]=UInt32(i==q.capacity ? 0 : i+1)
        end
        q.vacant=UInt32(1)
        q.oldest=q.newest=UInt32(0)
        q.count=0
    end
    return nothing
end

function makecase(name, callers, capacity, domain)
    base = replace(name, r"-serial$" => "")
    base in ("memoized", "lru", "expanding", "expanding-lru", "dict", "dict-lru") ||
        name in treecases || throw(ArgumentError("unknown case $name"))
    config = startswith(name, "dict") ? DictMap{Matrix{Float64}}(domain) :
        startswith(name, "expanding") ? ExpandingArrayMap{Matrix{Float64}}(domain; increment=8) :
        startswith(name, "radix") ? RadixMap{Matrix{Float64}}(domain; block_shape=(8, 8)) :
        startswith(name, "subtree") ? SubtreeRadixMap{Matrix{Float64}}(domain; block_shape=(8, 8)) : ArrayMap{Matrix{Float64}}(domain)
    concurrent = !endswith(name, "serial")
    bounded = occursin("lru", name)
    cache = bounded ?
        LRUMemoizedArray(Returns(payload), config; capacity, concurrent, queues=callers) :
        MemoizedArray(Returns(payload), config; concurrent)
    readers = bounded && concurrent ? [queue_handle(cache, w) for w in 1:callers] : fill(cache, callers)
    return cache, readers
end

function sample(name, callers, operation, inboxes, outboxes)
    if name in treecases
        return tree_sample(name, callers, operation, inboxes, outboxes)
    end
    capacity = parse(Int,get(ENV,"QUEUE_CAPACITY",operation === :generate ? "32768" : "8192"))
    count = operation === :evict ? parse(Int,get(ENV,"QUEUE_EVICT_COUNT","32768")) : capacity
    extent = operation === :evict ? capacity + count : capacity
    cache, readers = makecase(name, callers, capacity, (extent,))
    @assert count % callers == 0
    width = count ÷ callers
    rng = Xoshiro(2028)
    offset = operation === :evict ? capacity : 0
    keys = [shuffle(rng, collect(offset+(w-1)*width+1:offset+w*width)) for w in 1:callers]
    if operation === :read
        for w in 1:callers
            consume(readers[w], (w-1)*width+1:w*width, 1)
        end
    end
    repeats = operation === :read ? readrepeats : 1
    if operation === :read
        keys, repeats = read_workload(rng, keys)
    end
    prepare = w -> if operation === :generate
        ()->touch_empty!(readers[w],keys[w])
    elseif operation === :evict
        ()->begin
            touch_empty!(readers[w],keys[w])
            consume(readers[w],(w-1)*(capacity÷callers)+1:w*(capacity÷callers),1)
        end
    else
        ()->nothing
    end
    result = run_sample(readers, keys, repeats, operation === :read, prepare, inboxes, outboxes)
    if operation in (:read,:generate)
        @assert all(key->isready(cache,key),1:count)
    end
    return result
end

# Central generation batches independent caches so 63 insertions per central leaf
# provide a measurable window. Every caller shares the caches, but owns disjoint keys.
struct TreeBenchmarkBatch{C}
    caches::Vector{C}
end
@inline Base.getindex(batch::TreeBenchmarkBatch, key::Tuple{Int,CartesianIndex{2}}) =
    @inbounds batch.caches[key[1]][key[2]]

function tree_touch_empty!(cache, keys)
    cache = cache isa MemoizedArrays.QueueHandle ? cache.cache : cache
    for key in keys
        lookup = MemoizedArrays.map_lookup(cache, Tuple(key))
        ref = cache.map isa MemoizedArrays.TreeMap ?
            MemoizedArrays.map_slot(cache.map, lookup[1], lookup[2]) : lookup[3]
        @assert MemoizedArrays.slot_entry(ref) === cache.map.empty
        MemoizedArrays.slot_publish!(ref, cache.map.empty)
    end
end

# The caller owns queue bookkeeping; warming never resets resident memberships.
function tree_touch_queue!(reader)
    if reader isa MemoizedArrays.QueueHandle
        for entry in reader.queue.pool
            state = @atomic :monotonic entry.state
            @atomic :monotonic entry.state = state
        end
    elseif reader isa LRUMemoizedArray
        for entry in reader.eviction.entries
            entry.value = payload
        end
    end
    return nothing
end

function tree_sample(name, callers, operation, inboxes, outboxes)
    domain = (-251:260, -251:260)
    central = vec(collect(CartesianIndices((-3:4, -3:4))))
    candidates = vec(collect(CartesianIndices(domain)))
    rng = Xoshiro(2028)
    if operation in (Symbol("generate-central"), Symbol("evict-central"))
        batches = parse(Int, get(ENV, "QUEUE_TREE_BATCHES", "64"))
        evict = operation === Symbol("evict-central")
        outside = shuffle(rng, filter(key -> !(key in central), candidates))
        residents = evict ? outside[1:480] : [central[1]; outside[1:448]]
        requests = evict ? central[33:64] : central[2:64]
        @assert !evict || 32 % callers == 0
        setups = [makecase(name, callers, 512, domain) for _ in 1:batches]
        caches = first.(setups)
        readers = [TreeBenchmarkBatch([setup[2][w] for setup in setups]) for w in 1:callers]
        partitions = [requests[1+fld((w-1)*length(requests),callers):fld(w*length(requests),callers)] for w in 1:callers]
        for (_, handles) in setups
            offset = 0
            for w in 1:callers
                if evict
                    consume(handles[w], central[1+(w-1)*(32÷callers):w*(32÷callers)], 1)
                    count = 480 ÷ callers
                else
                    # Leave precisely this queue's 7/8 (or63 serial) entry slots free.
                    count = 512 ÷ callers - length(partitions[w])
                end
                consume(handles[w], residents[offset+1:offset+count], 1)
                offset += count
            end
            @assert offset == length(residents)
        end
        keys = [[(i, key) for i in 1:batches for key in partitions[w]] for w in 1:callers]
        prepare = w -> () -> begin
            for i in 1:batches
                handle = setups[i][2][w]
                tree_touch_queue!(handle)
                tree_touch_empty!(handle, partitions[w])
            end
        end
        result = run_sample(readers, keys, 1, false, prepare, inboxes, outboxes)
        for cache in caches
            @assert all(key -> isready(cache, key), requests)
            @assert all(key -> isready(cache, key), residents)
            if evict
                @assert all(key -> !isready(cache, key), central[1:32])
            else
                @assert all(key -> isready(cache, key), central)
            end
            cache isa LRUMemoizedArray && @assert MemoizedArrays.resident_count(cache) == 512
        end
        return result
    end

    isread = operation in (Symbol("read-central"), Symbol("read-spread"))
    evict = operation === Symbol("evict-spread")
    count = parse(Int, get(ENV, evict ? "QUEUE_EVICT_COUNT" : "QUEUE_CAPACITY", isread ? "8192" : "32768"))
    @assert count % callers == 0
    if !isread && !evict
        filter!(key -> mod(key[1]+3,8) != 0 || mod(key[2]+3,8) != 0, candidates)
    end
    shuffled = shuffle(rng, candidates)
    capacity = isread ? 0 : evict ?
        parse(Int,get(ENV,"QUEUE_CAPACITY","8192")) :
        operation === Symbol("generate-existing") ? count+4096 : count
    requested = evict ? shuffled[capacity+1:capacity+count] : shuffled[1:count]
    seeds = if isread
        unique([requested; central])
    elseif evict
        shuffled[1:capacity]
    elseif operation === Symbol("generate-existing")
        unique(CartesianIndex(fld(key[1]+3,8)*8-3, fld(key[2]+3,8)*8-3) for key in requested)
    else
        @assert operation === Symbol("generate-structural")
        CartesianIndex{2}[]
    end
    isread && (capacity = 8cld(length(seeds),8))
    @assert capacity % callers == 0
    repeats = isread ? readrepeats : 1
    width = count ÷ callers
    if operation === Symbol("read-central")
        @assert 64 % callers == 0 && count % 64 == 0
        shuffle!(rng, central)
        keys = [repeat(central[(w-1)*(64÷callers)+1:w*(64÷callers)],count÷64) for w in 1:callers]
    else
        keys = [requested[(w-1)*width+1:w*width] for w in 1:callers]
    end
    cache, readers = makecase(name, callers, capacity, domain)
    if isread
        if operation === Symbol("read-central")
            extra = filter(key -> !(key in central),requested)
            owned = [central[(w-1)*(64÷callers)+1:w*(64÷callers)] for w in 1:callers]
        else
            extra = filter(key -> !(key in requested),central)
            owned = keys
        end
        for w in 1:callers
            residents = [owned[w]; extra[1+fld((w-1)*length(extra),callers):fld(w*length(extra),callers)]]
            @assert length(residents) <= capacity ÷ callers
            consume(readers[w],residents,1)
        end
    else
        for w in 1:callers
            consume(readers[w], seeds[1+fld((w-1)*length(seeds),callers):fld(w*length(seeds),callers)], 1)
        end
    end
    if isread
        keys, repeats = read_workload(rng, keys)
    end
    prepare = w -> !isread ? () -> begin
        tree_touch_queue!(readers[w])
        operation === Symbol("generate-existing") && tree_touch_empty!(readers[w],keys[w])
    end : () -> nothing
    result = run_sample(readers,keys,repeats,isread,prepare,inboxes,outboxes)
    if evict
        @assert all(key -> !isready(cache,key),seeds)
        quota = capacity ÷ callers
        @assert all(keyset -> all(key -> isready(cache,key),keyset[end-quota+1:end]),keys)
        @assert MemoizedArrays.resident_count(cache) == capacity
    else
        @assert all(key -> isready(cache,key),requested)
        @assert all(key -> isready(cache,key),seeds)
    end
    return result
end

