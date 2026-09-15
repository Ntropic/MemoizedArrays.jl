# Production caches only. No package installation or environment changes.
# Sampled eviction: Psounis & Prabhakar (2002), https://doi.org/10.1109/TNET.2002.801414;
# Redis 8.2's five samples and sixteen-candidate pool: https://github.com/redis/redis/blob/8.2/src/evict.c.
using JITArrays

const Value = Matrix{Float64}
const CAPACITY = 512
const ROUNDS = 15
const GENERATION_CACHES = 8
const PRECOMPUTED_VALUE = fill(1.0, 4, 4)
@inline value(key::Int, rest::Int...) = PRECOMPUTED_VALUE
@inline value(key::Tuple) = PRECOMPUTED_VALUE

@inline readvalue(cache, key::Int) = @inbounds cache[key]
@inline readvalue(cache, key::Tuple) = @inbounds cache[key...]
function reads!(output, cache, keys)
    @inbounds for i in eachindex(keys)
        output[i] = readvalue(cache, keys[i])
    end
    return nothing
end
function parallelreads!(output, cache, keys)
    Threads.@threads :static for worker in 1:4
        @inbounds for i in (1 + (worker - 1) * div(length(keys), 4)):(worker * div(length(keys), 4))
            output[i] = readvalue(cache, keys[i])
        end
    end
    return nothing
end
function generations!(output, cache::Vector{Value}, keys)
    @inbounds for i in eachindex(keys)
        output[i] = cache[keys[i]] = value(keys[i])
    end
    return nothing
end
generations!(output, cache, keys) = reads!(output, cache, keys)

# Lengthen generation samples without changing each cache's capacity or key sequence.
function generations_across_caches!(output, caches, keys)
    for cache in caches
        generations!(output, cache, keys)
    end
    return nothing
end

function measured(run::F, output, cache, keys) where F
    started = time_ns()
    run(output, cache, keys)
    return (time_ns() - started) / length(keys)
end
allocated(run::F, output, cache, keys) where F = @allocated run(output, cache, keys)

function row(workload, name, times, bytes)
    samples = sort(times)
    stats = map(i -> round(samples[i]; digits=2), (cld(ROUNDS, 2), 2, ROUNDS - 1))
    println("| ", workload, " | ", name, " | ", stats[1], " | ", stats[2], "–", stats[3],
        " | ", isnan(bytes) ? "—" : round(bytes; digits=2), " |")
end

function tableheader(unit)
    println("\n| Workload | Implementation | Median ns/", unit, " | p10–p90 ns/", unit, " | Bytes/", unit, " |")
    println("|---|---|---:|---:|---:|")
end

# Function barriers keep the measured loop specialized; setup and assertions stay outside timing.
function compare(names, makers, warmkeys, requestkeys)
    output = Vector{Value}(undef, max(length(requestkeys), length(warmkeys)))
    caches = map(make -> make(value), makers)
    for cache in caches
        generations!(output, cache, warmkeys)
    end
    for (label, run) in (("Ready", reads!), ("Ready, four threads", parallelreads!))
        times = [Float64[] for _ in makers]
        for cache in caches
            measured(run, output, cache, requestkeys)
            @assert all(i -> output[i] === readvalue(cache, requestkeys[i]), eachindex(requestkeys))
            allocated(run, output, cache, requestkeys)
        end
        for round in 1:ROUNDS, i in circshift(collect(eachindex(makers)), round)
            push!(times[i], measured(run, output, caches[i], requestkeys))
        end
        for i in eachindex(makers)
            bytes = allocated(run, output, caches[i], requestkeys) / length(requestkeys)
            @assert all(k -> output[k] === PRECOMPUTED_VALUE, eachindex(requestkeys))
            row(label, names[i], times[i], bytes)
        end
    end
    for full in (false, true)
        selected = full ? findall(i -> caches[i] isa LRUArray, eachindex(makers)) : collect(eachindex(makers))
        missing = full ? map(k -> k isa Int ? k + CAPACITY : (k[1] + 128, Base.tail(k)...), warmkeys) : warmkeys
        setup(make) = begin
            cache = make(value)
            if full
                generations!(output, cache, warmkeys)
                @assert all(k -> isready(cache, k), warmkeys)
                @assert all(k -> !isready(cache, k), missing)
            end
            cache
        end
        samples(make) = [setup(make) for _ in 1:GENERATION_CACHES]
        times = [Float64[] for _ in makers]
        for i in selected
            measured(generations_across_caches!, output, samples(makers[i]), missing)
        end
        for round in 1:ROUNDS, i in circshift(selected, round)
            batch = samples(makers[i])
            push!(times[i], measured(generations_across_caches!, output, batch, missing) / GENERATION_CACHES)
            @assert all(k -> output[k] === PRECOMPUTED_VALUE, eachindex(missing))
        end
        for i in selected
            row(full ? "Full cache generation" : "Generation", names[i], times[i],
                allocated(generations_across_caches!, output, samples(makers[i]), missing) / (GENERATION_CACHES * length(missing)))
        end
    end
    # Count generator calls separately so atomics do not contaminate the timing.
    for i in eachindex(makers)
        if caches[i] isa AbstractJITArray
            calls = Ref(0)
            cache = makers[i]((keys...) -> (calls[] += 1; value(keys...)))
            reads!(output, cache, warmkeys)
            @assert calls[] == length(warmkeys)
            reads!(output, cache, requestkeys)
            @assert calls[] == length(warmkeys)
            if cache isa LRUArray
                missing = map(k -> k isa Int ? k + CAPACITY : (k[1] + 128, Base.tail(k)...), warmkeys)
                reads!(output, cache, missing)
                @assert calls[] == 2length(warmkeys)
                @assert count(k -> isready(cache, k), [warmkeys; missing]) == CAPACITY
            end
        end
    end
end

function waiter_round(make)
    started, release = Channel{Nothing}(1), Channel{Nothing}(1)
    calls = Threads.Atomic{Int}(0)
    cache = make(i -> begin
        Threads.atomic_add!(calls, 1)
        put!(started, nothing)
        take!(release)
        value(i)
    end)
    owner = Threads.@spawn cache[1]
    take!(started)
    entry = JITArrays.entry(cache, 1)
    waiters = [Threads.@spawn cache[1] for _ in 1:3]
    registered = timedwait(5) do
        state = @atomic :acquire entry.state
        if !(state isa Threads.Condition)
            return false
        end
        lock(state)
        try
            return length(state.waitq) == 3
        finally
            unlock(state)
        end
    end
    @assert registered == :ok
    before = time_ns()
    put!(release, nothing)
    result = fetch(owner)
    results = fetch.(waiters)
    elapsed = time_ns() - before
    @assert calls[] == 1 && all(v -> v === result, results)
    return Float64(elapsed)
end

function benchmark(; sparse=false)
    @assert Threads.nthreads() >= 4 "Run with --threads=4"
    println("Julia ", VERSION, "; ", ROUNDS, " interleaved rounds; median and approximate p10–p90.")
    println("Single-thread rows report latency; four-thread rows report elapsed time / total requests.")
    println("Generation returns one precomputed 4×4 matrix; no matrix allocation or filling is timed.")
    tableheader("op")
    makers = (g -> Vector{Value}(undef, 4096), g -> JITArray{Value}(g, 4096),
        g -> ExactLRUArray{Value}(g, 4096; capacity=CAPACITY), g -> SampledLRUArray{Value}(g, 4096; capacity=CAPACITY))
    names = ("Direct array", "JITArray", "ExactLRUArray", "SampledLRUArray")
    warmkeys = collect(1:CAPACITY)
    keys = [mod(i * 73, CAPACITY) + 1 for i in 1:32768]
    compare(names, makers, warmkeys, keys)
    tableheader("batch")
    for i in 2:4
        waiter_round(makers[i])
        row("Release + generation + four callers", names[i],
            [waiter_round(makers[i]) for _ in 1:ROUNDS], NaN)
    end
    if sparse
        tableheader("op")
        for storage in (:dict, :msbtree, :lsbtree), shape in ((4, 8), (8, 4))
            if storage === :dict && shape !== (4, 8)
                continue
            end
            options = storage === :dict ? (; storage) : (; storage, block_shape=shape)
            domain = (typemin(Int):typemax(Int), typemin(Int):typemax(Int))
            sparsemakers = (g -> JITArray{Value}(g, domain; options...),
                g -> LRUArray{Value}(g, domain; capacity=CAPACITY, policy=:exact, options...),
                g -> LRUArray{Value}(g, domain; capacity=CAPACITY, policy=:sampled, options...))
            warm = [(2mod(i, 32), 2div(i, 32)) for i in 1:CAPACITY]
            requests = [warm[mod(i * 73, CAPACITY) + 1] for i in 1:32768]
            layout = storage === :dict ? "$storage" : "$storage / $shape"
            labels = map(name -> "$name / $layout", names[2:4])
            compare(labels, sparsemakers, warm, requests)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark(; sparse="--sparse" in ARGS)
end
