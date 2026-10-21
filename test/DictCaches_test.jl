# Dictionary-specific publication, probe-chain and retained-reference regressions.
function dict_test_cache(C, V, generator, domain; capacity=192, queues=3)
    options = C === MemoizedArray ? (;) : (; capacity, queues)
    return C(generator, DictMap{V}(domain); concurrent=true, options...)
end

function dict_test_completed(task)
    completed = timedwait(() -> istaskdone(task), 5; pollint=0.001)
    @test completed === :ok
    return completed === :ok ? fetch(task) : nothing
end

dict_test_collisions() = filter(i -> hash((i,)) & UInt(63) == 0 &&
    (hash((i,)) >> (4sizeof(UInt))) & UInt(1) == 0, 1:10000)[1:4]

@testset "Dictionary LRU collision chains and exact order" begin
    keys = dict_test_collisions()
    for concurrent in (false, true), capacity in (1, 2)
        cache = LRUMemoizedArray(identity, DictMap{Int}((1:10000,)); capacity, concurrent, queues=1)
        concurrent && @test_throws ArgumentError cache[keys[1]]
        handle = concurrent ? queue_handle(cache, 1) : cache
        order = Int[]
        for index in (1, 2, 1, 3, 2, 4, 4, 1, 3)
            key = keys[index]
            filter!(!=(key), order)
            push!(order, key)
            length(order) > capacity && popfirst!(order)
            @test handle[key] == key
            @test JA.resident_count(cache) == length(order)
            @test all(k -> isready(cache, k) == (k in order), keys)
            if !concurrent
                for k in order
                    stored = JA.entry(cache, JA.cache_key(cache, (k,)))
                    @test stored isa JA.SerialLRUEntry
                    @test cache.eviction.entries[stored.id] === stored
                    @test cache.map.table[stored.key.address] === stored
                end
            end
        end
    end
end

@testset "Dictionary LRU uses preallocated entries directly" begin
    value = ones(2, 2)
    make() = LRUMemoizedArray(Returns(value), DictMap{typeof(value)}((1:100,)); capacity=2)
    warmup = make()
    for key in 1:4
        request_allocations(warmup, key)
    end
    cache = make()
    entries = copy(cache.eviction.entries)
    table = cache.map.table
    for key in 1:4
        @test request_allocations(cache, key) == 0
        stored = JA.entry(cache, JA.cache_key(cache, (key,)))
        @test any(entry -> stored === entry, entries)
        @test cache.map.table[stored.key.address] === stored
        @test cache[key] === value
    end
    @test cache.map.table === table # allocation checks exclude table rebuilds
    @test cache.map.count == JA.resident_count(cache) == 2
    readyallocations(cache)
    @test readyallocations(cache) == 0

    matrices = LRUMemoizedArray(i -> fill(i, 2, 2), DictMap{Matrix{Int}}((1:3,)); capacity=1)
    retained = matrices[1]
    stored = JA.entry(matrices, JA.cache_key(matrices, (1,)))
    @test matrices[2] == fill(2, 2, 2)
    @test matrices[3] == fill(3, 2, 2)
    @test JA.entry(matrices, JA.cache_key(matrices, (3,))) === stored
    @test retained == fill(1, 2, 2)
end

@testset "Serial dictionary reentrant publication" begin
    for conversion in (false, true)
        owner = Ref{Any}()
        calls = Ref(0)
        nested() = (generate!(owner[], 1:80); 0)
        cache = LRUMemoizedArray(i -> begin
            calls[] += 1
            i == 0 && return conversion ? TestConversion(nested) : nested()
            return i
        end, DictMap{Int}((0:80,)); capacity=81)
        owner[] = cache
        table = cache.map.table
        @test cache[0] == 0
        @test cache.map.table !== table
        @test all(i -> cache[i] == i, 0:80)
        @test calls[] == cache.map.count == JA.resident_count(cache) == 81
        for stored in cache.eviction.entries
            @test cache.map.table[stored.key.address] === stored
        end
    end

    # A finite recursive request can publish the same key before its outer
    # generator/conversion finishes. It still owns only one queue membership.
    for C in (MemoizedArray, LRUMemoizedArray), capacity in (1, 2), conversion in (false, true)
        owner = Ref{Any}()
        entered = Ref(false)
        calls = Ref(0)
        nested() = (owner[][0]; 11)
        generator = i -> begin
            i != 0 && return i
            calls[] += 1
            if !entered[]
                entered[] = true
                return conversion ? TestConversion(nested) : nested()
            end
            return 7
        end
        options = C === MemoizedArray ? (;) : (; capacity)
        cache = C(generator, DictMap{Int}((0:3,)); options...)
        owner[] = cache
        generate!(cache, 1:capacity)
        @test cache[0] == cache[0] == 11
        @test calls[] == 2
        @test cache.map.count == (C === MemoizedArray ? capacity + 1 : capacity)
        if C === LRUMemoizedArray
            @test JA.resident_count(cache) == capacity
            @test cache[3] == 3
            @test isready(cache, 0) == (capacity == 2)
        end
    end
end

@testset "Concurrent dictionary nested generation and conversion rehash" begin
    for C in (MemoizedArray, LRUMemoizedArray), conversion in (false, true)
        owner = Ref{Any}()
        calls = Ref(0)
        keys = Int[]
        nested() = (generate!(owner[], keys); 0)
        cache = dict_test_cache(C, Int, i -> begin
            calls[] += 1
            i == 0 && return conversion ? TestConversion(nested) : nested()
            return i
        end, (0:10000,); capacity=96, queues=1)
        handle = concurrent_test_handle(cache, 1)
        owner[] = handle
        partition = JA.hash_partition(cache.map, hash((0,)))
        append!(keys, Iterators.take((i for i in 1:10000 if
            JA.hash_partition(cache.map, hash((i,))) === partition), 80))
        table = JA.slot_entry(partition.table)
        missing = JA.map_lookup(cache, (keys[end],))
        @test handle[0] == 0
        @test JA.slot_entry(partition.table) !== table
        @test JA.cache_get(handle, missing, (keys[end],)) == keys[end]
        @test all(i -> handle[i] == i, [0; keys])
        @test calls[] == 81
        C === LRUMemoizedArray && @test JA.resident_count(cache) == 81
    end
end

@testset "Concurrent dictionary paused publication survives rehash" begin
    for C in (MemoizedArray, LRUMemoizedArray), outcome in (:value, :generator_failure, :conversion_failure)
        entered = Channel{Nothing}(2)
        release = Base.Event()
        calls = Threads.Atomic{Int}(0)
        failure = ErrorException("dictionary outcome survives rehash: $outcome")
        generator = i -> begin
            if i == 0
                Threads.atomic_add!(calls, 1)
                put!(entered, nothing)
                wait(release)
                outcome === :generator_failure && throw(failure)
                outcome === :conversion_failure && return TestConversion(() -> throw(failure))
            end
            return i
        end
        cache = dict_test_cache(C, Int, generator, (0:144,))
        a, b, c = (concurrent_test_handle(cache, i) for i in 1:3)
        owner = Threads.@spawn try
            test_result(() -> a[0])
        finally
            put!(entered, nothing) # an early failure must not strand synchronization
        end
        waiter = pending = lookup = nothing
        try
            started = timedwait(() -> isready(entered), 5; pollint=0.001)
            @test started === :ok
            if started === :ok
                take!(entered)
                lookup = JA.map_lookup(cache, (0,))
                partition = JA.hash_partition(cache.map, hash((0,)))
                table = JA.slot_entry(partition.table)
                captured = Channel{Nothing}(1)
                waiter = Threads.@spawn test_result() do
                    saved = JA.map_lookup(b, (0,))
                    put!(captured, nothing)
                    JA.cache_get(b, saved, (0,))
                end
                @test timedwait(() -> isready(captured), 5; pollint=0.001) === :ok
                if C === LRUMemoizedArray
                    registered = timedwait(() -> JA.slot_entry(lookup[3]) isa JA.GenerateWait,
                        5; pollint=0.001)
                    @test registered === :ok
                    registered === :ok && (pending = JA.slot_entry(lookup[3]))
                end
                independent = Threads.@spawn begin
                    generate!(c, 1:80)
                    lock(partition.structural) do
                        JA.hash_rebuild!(partition, 4length(JA.slot_entry(partition.table)))
                    end
                    c
                end
                @test dict_test_completed(independent) === c
                @test JA.slot_entry(partition.table) !== table
                @test !isready(cache, 0)
            end
        finally
            notify(release)
        end
        expected = outcome === :value ? 0 : failure
        @test dict_test_completed(owner) === expected
        # Queue 1 has quota 64. The last request evicts its successful key 0;
        # failed generation remains in its dictionary record instead.
        generate!(a, 81:144)
        waiter !== nothing && @test dict_test_completed(waiter) === expected
        if pending !== nothing
            @test test_result(() -> JA.wait_generated(Int, pending.reference)) === expected
        end
        @test calls[] == 1
        @test isready(cache, 0) == (C === MemoizedArray && outcome === :value)
        if C === MemoizedArray || outcome !== :value
            @test test_result(() -> a[0]) === expected
            @test calls[] == 1
        end
    end
end

@testset "Dictionary reads and independent partitions progress under a partition lock" begin
    for C in (MemoizedArray, LRUMemoizedArray)
        cache = dict_test_cache(C, Int, identity, (1:1000,); capacity=2, queues=1)
        handle = concurrent_test_handle(cache, 1)
        @test handle[1] == 1
        lookup = JA.map_lookup(cache, (1,))
        reader = nothing
        partition = JA.hash_partition(cache.map, hash((1,)))
        other = first(i for i in 2:1000 if JA.hash_partition(cache.map, hash((i,))) !== partition)
        lock(partition.structural) do
            table = JA.slot_entry(partition.table)
            JA.hash_rebuild!(partition, 4length(table))
            reader = Threads.@spawn begin
                (handle[1], JA.cache_get(handle, lookup, (1,)), isready(cache, 1), handle[other])
            end
            @test timedwait(() -> istaskdone(reader), 5; pollint=0.001) === :ok
        end
        @test dict_test_completed(reader) == (1, 1, true, other)
    end
end

@testset "Dictionary stale ready and missing references follow recycled entries" begin
    keys = dict_test_collisions()
    calls = zeros(Int, 10000)
    cache = LRUMemoizedArray(i -> (calls[i] += 1; fill(i, 2, 2)), DictMap{Matrix{Int}}((1:10000,)); capacity=2, concurrent=true, queues=1)
    handle = queue_handle(cache, 1)
    first, second, third, fourth = keys
    missing = JA.map_lookup(cache, (first,))
    retained = handle[first]
    ready = JA.map_lookup(cache, (first,))
    stored = JA.entry(cache, JA.cache_key(cache, (first,)))
    @test handle[second] == fill(second, 2, 2)
    partition = JA.hash_partition(cache.map, hash((first,)))
    lock(partition.structural) do
        table = JA.slot_entry(partition.table)
        JA.hash_rebuild!(partition, 4length(table))
    end
    @test handle[third] == fill(third, 2, 2)
    @test !isready(cache, first) && isready(cache, second)
    @test JA.entry(cache, JA.cache_key(cache, (third,))) === stored
    @test JA.slot_entry(ready[3]) === JA.RELOCATED
    regenerated = JA.cache_get(handle, missing, (first,))
    @test regenerated == retained == fill(first, 2, 2)
    @test regenerated !== retained
    @test JA.cache_get(handle, ready, (first,)) === regenerated
    @test calls[first] == 2
    @test handle[fourth] == fill(fourth, 2, 2)
    @test handle[first] === regenerated
    @test JA.resident_count(cache) == 2
end
