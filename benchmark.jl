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

include("benchmark/workloads.jl")

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
