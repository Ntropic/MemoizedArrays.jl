# Public cache benchmarks; preconfigured matrices and scalar checksum.
# Run with julia --project=. -t 8 benchmark.jl; set QUEUE_READ_PATTERN=varying for random repeats.
# Optional Linux affinity: QUEUE_PIN=true QUEUE_CPU_OFFSET=8 QUEUE_PRODUCER_CPU=0.
# Optional expanding LRU cases: QUEUE_CASES=expanding-lru-serial,expanding-lru.
# Optional Dict cases: QUEUE_CASES=dict-serial,dict,dict-lru-serial,dict-lru.
# Tree Memoized cases: array2d-serial,array2d,radix-serial,radix,subtree-serial,subtree.
# Tree LRU cases: append -lru before the optional -serial suffix.
# Tree operations: read-central,read-spread,generate-central,generate-existing,generate-structural.
# Tree LRU also supports evict-central and evict-spread.
using MemoizedArrays, Random, Statistics
const payload = ones(4, 4)
const rounds = parse(Int, get(ENV, "QUEUE_ROUNDS", "15"))
const readrepeats = parse(Int, get(ENV, "QUEUE_READ_REPEATS", "512"))
const readpattern = get(ENV,"QUEUE_READ_PATTERN","cycle")
const operations = Tuple(Symbol.(split(get(ENV, "QUEUE_OPERATIONS", "read,generate,evict"), ',')))
const workers = Threads.nthreads()
const cpu_offset = parse(Int,get(ENV,"QUEUE_CPU_OFFSET","0"))
const treecases = ("array2d-serial", "array2d", "radix-serial", "radix", "subtree-serial", "subtree",
    "array2d-lru-serial", "array2d-lru", "radix-lru-serial", "radix-lru", "subtree-lru-serial", "subtree-lru")
const treeoperations = Tuple(Symbol.(("read-central", "read-spread", "generate-central", "generate-existing", "generate-structural", "evict-central", "evict-spread")))

@inline function consume(reader::R, keys::K, repeats) where {R,K}
    checksum = 0.0
    for _ in 1:repeats, key in keys
        checksum += @inbounds reader[key][1,1]
    end
    return checksum
end
struct Job{R,K,P}
    reader::R
    keys::K
    repeats::Int
    warm::Bool
    prepare::P
    deadline::Base.RefValue{UInt64}
    ready::Channel{Nothing}
    start::Base.Event
end
function run_job(job::Job)
    job.prepare()
    if job.warm
        consume(job.reader, job.keys, 1)
    end
    put!(job.ready, nothing)
    wait(job.start)
    while time_ns() < job.deadline[]
        GC.safepoint()
    end
    begin_ns = time_ns()
    checksum = consume(job.reader, job.keys, job.repeats)
    end_ns = time_ns()
    return (begin_ns, end_ns, checksum)
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
    width = count ÷ callers
    rng = Xoshiro(2028)
    offset = operation === :evict ? capacity : 0
    keys = [shuffle(rng, collect(offset+(w-1)*width+1:offset+w*width)) for w in 1:callers]
    if operation === :read
        prefill = readers
        populated_count=count
        for key in 1:populated_count
            prefill[mod1(cld(key, cld(populated_count, length(prefill))), length(prefill))][key]
        end
    end
    repeats = operation === :read ? readrepeats : 1
    if operation === :read && readpattern == "varying"
        @assert readrepeats % 16 == 0
        keys = [rand(rng, keyset, 16width) for keyset in keys]
        repeats = readrepeats ÷ 16
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

function run_sample(readers, keys, repeats, warm, prepare, inboxes, outboxes)
    callers = length(readers)
    requests = sum(length,keys) * repeats
    ready, start = Channel{Nothing}(callers), Base.Event()
    deadline = Ref(UInt64(0))
    for w in 1:callers
        put!(inboxes[w], Job(readers[w], keys[w], repeats, warm, prepare(w), deadline, ready, start))
    end
    for _ in 1:callers
        take!(ready)
    end
    deadline[]=time_ns()+UInt64(2_000_000)
    notify(start)
    result = [take!(outboxes[w]) for w in 1:callers]
    for outcome in result
        if outcome isa Exception
            throw(outcome)
        end
    end
    starts, finishes = first.(result), getindex.(result, 2)
    @assert sum(last, result) == requests
    # This window excludes task creation and master-thread collection overhead.
    # Report start skew and actual overlap rather than assuming simultaneous execution.
    elapsed = maximum(finishes) - minimum(starts)
    skew = maximum(starts) - minimum(starts)
    overlap = callers == 1 || maximum(starts) < minimum(finishes)
    return (elapsed / requests, skew / max(elapsed,1), overlap)
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
    count = parse(Int, get(ENV, "QUEUE_CAPACITY", isread ? "8192" : "32768"))
    @assert count % callers == 0
    if !isread && !evict
        filter!(key -> mod(key[1]+3,8) != 0 || mod(key[2]+3,8) != 0, candidates)
    end
    shuffled = shuffle(rng, candidates)
    capacity = isread ? 0 : evict ?
        parse(Int,get(ENV,"QUEUE_TREE_EVICT_CAPACITY","8192")) :
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
    if isread && readpattern == "varying"
        @assert readrepeats % 16 == 0
        keys = [rand(rng,keyset,16width) for keyset in keys]
        repeats = readrepeats ÷ 16
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

function produce(inboxes,outboxes)
    if haskey(ENV,"QUEUE_PRODUCER_CPU")
        cpu=parse(Int,ENV["QUEUE_PRODUCER_CPU"])
        mask=Ref{UInt64}(UInt64(1)<<cpu)
        @assert ccall(:sched_setaffinity,Cint,(Cint,Csize_t,Ptr{UInt64}),0,8,mask)==0
    end
    cases = ["memoized-serial","lru-serial","memoized","lru"]
    cases = split(get(ENV,"QUEUE_CASES",join(cases,',')),',')
    println("Julia ", VERSION, "; ", Sys.CPU_NAME, "; ",workers," persistent workers; ", rounds, " rotated rounds; worker CPUs ",cpu_offset,":",cpu_offset+workers-1,".")
    println("Producer thread=",Threads.threadid()," pool=",Threads.threadpool())
    println("Preconfigured matrices; setup outside timing; aggregate wall ns/request [Q1,Q3].")
    println("Read pattern: ", readpattern, "; workload definitions: docs/src/benchmarks.md")
    println("| Case | Operation | Callers | ns/op [Q1,Q3] | Start skew | Overlap |")
    println("|---|---|---:|---:|---:|---:|")
    flush(stdout)
    try
        for operation in operations, callers in parse.(Int,split(get(ENV,"QUEUE_CALLERS","1,8"),','))
            selected = filter(cases) do name
                !(callers > 1 && endswith(name, "serial")) &&
                    (name in treecases ? operation in treeoperations : operation in (:read, :generate, :evict)) &&
                    (!startswith(string(operation), "evict") || occursin("lru", name))
            end
            samples = [[] for _ in selected]
            for iteration in 0:rounds
                GC.gc(false)
                for index in circshift(collect(eachindex(selected)), iteration)
                    result = sample(selected[index], callers, operation, inboxes, outboxes)
                    if iteration > 0
                        push!(samples[index], result)
                    end
                end
            end
            for (name, rows) in zip(selected, samples)
                times = first.(rows)
                println("| ",name," | ",operation," | ",callers," | ",round(median(times);sigdigits=3),
                    " [",round(quantile(times,.25);sigdigits=3),",",round(quantile(times,.75);sigdigits=3),
                    "] | ",round(100median(getindex.(rows,2));digits=2),
                    "% | ",count(row->row[3],rows),"/",length(rows)," |")
            end
            flush(stdout)
        end
    finally
        foreach(channel -> put!(channel,nothing),inboxes)
    end
end
function main()
    counts = parse.(Int, split(get(ENV,"QUEUE_CALLERS","1,8"), ','))
    all(n -> 1 <= n <= workers, counts) ||
        throw(ArgumentError("QUEUE_CALLERS must fit the available Julia threads; use -t 8 for the default 1/8 callers"))
    inboxes = [Channel{Any}(1) for _ in 1:workers]
    outboxes = [Channel{Any}(1) for _ in 1:workers]
    producer = @async produce(inboxes,outboxes)
    # Sticky static tasks stay on distinct Julia threads. Pin once before jobs,
    # so affinity changes/migration cannot inflate the timed start window.
    Threads.@threads :static for w in 1:workers
        original=Ref{UInt64}(0)
        mask=Ref{UInt64}(UInt64(1)<<(cpu_offset+w-1))
        pinned=get(ENV,"QUEUE_PIN","false")=="true"
        if pinned
            @assert ccall(:sched_getaffinity,Cint,(Cint,Csize_t,Ptr{UInt64}),0,8,original)==0
            @assert ccall(:sched_setaffinity,Cint,(Cint,Csize_t,Ptr{UInt64}),0,8,mask)==0
        end
        try
            while true
                job=take!(inboxes[w])
                if job===nothing
                    break
                end
                put!(outboxes[w],try run_job(job) catch error; error end)
            end
        finally
            if pinned
                @assert ccall(:sched_setaffinity,Cint,(Cint,Csize_t,Ptr{UInt64}),0,8,original)==0
            end
        end
    end
    fetch(producer)
end
if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
