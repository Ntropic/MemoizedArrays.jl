@testset "Concurrent expanding LRU relocation and entry reuse" begin
    cache = LRUMemoizedArray((i,j) -> [i,j], ExpandingArrayMap{Vector{Int}}((-1:0,4:5); increment=2); capacity=2, concurrent=true, queues=1)
    handle = queue_handle(cache, 1)
    @test_throws ArgumentError cache[-1,4]
    @test_throws BoundsError handle[-2,4]
    retained = handle[-1,5]
    recycled = JA.entry(cache, JA.cache_key(cache, (-1,5)))
    ready_lookup = JA.map_lookup(cache, (-1,5))
    empty_lookup = JA.map_lookup(cache, (0,5))
    @test handle[0,4] == [0,4]
    @test handle[1,4] == [1,4] # first-axis growth changes every later column's stride
    @test axes(cache) == (-1:2,4:5)
    @test !isready(cache, -1,5)
    @test isready(cache, 0,4)
    @test JA.entry(cache, JA.cache_key(cache, (1,4))) === recycled
    @test (@atomic :acquire recycled.key) == (3,1)
    @test retained == [-1,5]

    # A saved ready slot must not return the value of its recycled entry; an
    # empty slot captured from the old allocation must publish into the new one.
    @test JA.cache_get(handle, ready_lookup, (-1,5)) == [-1,5]
    @test JA.cache_get(handle, empty_lookup, (0,5)) == [0,5]
    @test isready(cache, -1,5) && isready(cache, 0,5)
    @test !isready(cache, 1,4)
    @test JA.resident_count(cache) == 2
    @test !isready(cache, 100,100)
    @test size(cache) == (4,2)
    @test (@inferred handle[-1,5]) == [-1,5]
end

@testset "Nested concurrent LRU generation and conversion can grow" begin
    for conversion in (false, true)
        owner = Ref{Any}()
        calls = Ref(0)
        generator = (i,j) -> begin
            calls[] += 1
            if (i,j) == (1,1)
                if conversion
                    return ConcurrentExpandingConversion(owner[], 11)
                end
                owner[][6,3]
                return 11
            end
            return 63
        end
        cache = LRUMemoizedArray(generator, ExpandingArrayMap{Int}((1,1); increment=4); capacity=1, concurrent=true, queues=1)
        handle = queue_handle(cache, 1)
        owner[] = handle
        @test handle[1,1] == 11
        @test size(cache) == (9,5)
        @test handle[1,1] == 11
        @test calls[] == 2
        @test !isready(cache, 6,3)
        @test isready(cache, 1,1)
        @test JA.resident_count(cache) == 1
    end
end

@testset "Expanding LRU retains waiting outcomes through relocation and reuse" begin
    for outcome in (:value, :generator_failure, :conversion_failure)
        entered, release = Base.Event(), Base.Event()
        calls = Threads.Atomic{Int}(0)
        failure = ErrorException("LRU failure survives expansion: $outcome")
        generator = (i,j) -> begin
            if (i,j) == (1,2)
                Threads.atomic_add!(calls, 1)
                notify(entered)
                wait(release)
                if outcome === :generator_failure
                    throw(failure)
                elseif outcome === :conversion_failure
                    return ConcurrentExpandingFailure(failure)
                end
                return 12
            end
            return 100i + j
        end
        cache = LRUMemoizedArray(generator, ExpandingArrayMap{Int}((2,2); increment=2); capacity=3, concurrent=true, queues=3)
        a, b, c = (queue_handle(cache, i) for i in 1:3)
        owner = Threads.@spawn begin
            try
                test_result(() -> a[1,2])
            finally
                notify(entered)
            end
        end
        wait(entered)
        old_lookup = JA.map_lookup(cache, (1,2))
        captured = Channel{Nothing}(1)
        waiter = expanding_captured_waiter(b, (1,2), captured)
        take!(captured)
        pending = nothing
        try
            registered = timedwait(() -> JA.slot_entry(old_lookup[3]) isa JA.GenerateWait,
                5; pollint=0.001)
            @test registered === :ok
            pending = JA.slot_entry(old_lookup[3])
            @test c[3,1] == 301
            @test c[5,4] == 504
            @test size(cache) == (6,4)
            @test !isready(cache, 1,2)
        finally
            notify(release)
        end
        expected = outcome === :value ? 12 : failure
        @test fetch(owner) === expected
        @test a[7,6] == 706 # immediately recycle the owner's entry and grow again
        @test fetch(waiter) === expected
        if pending isa JA.GenerateWait
            @test test_result(() -> JA.wait_generated(Int, pending.reference)) === expected
        end
        @test calls[] == 1
        @test !isready(cache, 1,2)
        if outcome !== :value
            @test test_result(() -> a[1,2]) === failure
            @test calls[] == 1
        end
    end
end

@testset "Independent queues grow and replace concurrently" begin
    callers = 4
    entered = Channel{Nothing}(callers)
    release = Base.Event()
    calls = Threads.Atomic{Int}(0)
    cache = LRUMemoizedArray((i,j) -> begin
        Threads.atomic_add!(calls, 1)
        if j == 1 && i <= callers
            put!(entered, nothing)
            wait(release)
        end
        (i,j)
    end, ExpandingArrayMap{Tuple{Int,Int}}((1,1); increment=4); capacity=8, concurrent=true, queues=callers)
    handles = [queue_handle(cache, i) for i in 1:callers]
    tasks = [Threads.@spawn begin
        result = test_result(() -> handles[i][i,1])
        if result isa Exception
            put!(entered, nothing)
        end
        result
    end for i in 1:callers]
    try
        for _ in 1:callers
            take!(entered)
        end
    finally
        notify(release)
    end
    @test fetch.(tasks) == [(i,1) for i in 1:callers]
    tasks = [Threads.@spawn begin
        valid = true
        for step in 1:8
            key = (i + callers * step, mod1(i + step, 5))
            valid &= handles[i][key...] === key
        end
        valid
    end for i in 1:callers]
    @test all(fetch.(tasks))
    @test calls[] == callers * 9
    @test size(cache) == (37,5)
    @test JA.resident_count(cache) == 8
    @test count(key -> isready(cache, key), CartesianIndices(cache)) == 8
end

@testset "Concurrent expanding LRU matrix operations within the allocation" begin
    value = ones(4,4)
    cache = LRUMemoizedArray(Returns(value), ExpandingArrayMap{typeof(value)}((8,2); increment=4); capacity=2, concurrent=true, queues=1)
    handle = queue_handle(cache, 1)
    request_allocations(handle, (1,1))
    @test request_allocations(handle, (2,1)) == 0
    request_allocations(handle, (3,1))
    @test request_allocations(handle, (4,1)) == 0
    @test read_allocations(handle, (4,1)) == 0
    @test handle[4,1] === value
end

@testset "LRU requests progress before the remaining slots are copied" begin
    cache = LRUMemoizedArray((i,j) -> 10i+j, ExpandingArrayMap{Int}((2,2); increment=2); capacity=2, concurrent=true, queues=2)
    a, b = queue_handle(cache, 1), queue_handle(cache, 2)
    @test a[1,1] == 11
    @test b[1,2] == 12
    readers = Task[]
    replacement = gap_reader = nothing
    lock(cache.map.growth) do
        previous = JA.current_allocation(cache.map)
        allocation = JA.prepare_expansion!(previous, (4,2))
        gap_stored = nothing
        gap_frozen = gap_copied = false
        try
            JA.relocate_slot!(previous, allocation, (1,1))
            JA.relocate_slot!(previous, allocation, (2,1))
            readers = [Threads.@spawn(a[1,1]), Threads.@spawn(b[1,2])]
            ready = timedwait(() -> all(istaskdone, readers), 5; pollint=0.001)
            @test ready === :ok
            @test JA.current_allocation(cache.map) === previous
            if ready === :ok
                # Both the victim and insertion destination have already moved.
                replacement = Threads.@spawn a[2,1]
                written = timedwait(() -> istaskdone(replacement), 5; pollint=0.001)
                @test written === :ok
                if written === :ok
                    @test !isready(cache, 1,1)
                    published_ready = isready(cache, 2,1)
                    @test published_ready
                    handle_ready = isready(a, 2,1)
                    @test handle_ready
                    # Pause exactly between freezing this source and copying it.
                    gap_stored = JA.slot_exchange!(JA.allocation_slot(previous, (1,2)), JA.RELOCATED)
                    gap_frozen = true
                    gap_reader = Threads.@spawn b[1,2]
                    @test timedwait(() -> istaskdone(gap_reader), 0.02; pollint=0.001) === :timed_out
                    JA.slot_publish!(JA.allocation_slot(allocation, (1,2)), gap_stored)
                    gap_copied = true
                    @test timedwait(() -> istaskdone(gap_reader), 5; pollint=0.001) === :ok
                    @test JA.current_allocation(cache.map) === previous
                end
            end
        finally
            if gap_frozen
                if !gap_copied
                    JA.slot_publish!(JA.allocation_slot(allocation, (1,2)), gap_stored)
                end
            else
                JA.relocate_slot!(previous, allocation, (1,2))
            end
            JA.relocate_slot!(previous, allocation, (2,2))
            @atomic :release cache.map.allocation = allocation
            @atomic :release previous.successor = previous
        end
    end
    @test fetch.(readers) == [11,12]
    if replacement !== nothing
        @test fetch(replacement) == 21
    end
    if gap_reader !== nothing
        @test fetch(gap_reader) == 12
    end
    @test a[2,1] == 21
    @test !isready(cache, 1,1)
    @test b[1,2] == 12
    @test JA.resident_count(cache) == 2
end

@testset "Memoized completion progresses before the remaining slots are copied" begin
    entered, release = Base.Event(), Base.Event()
    cache = MemoizedArray((i,j) -> begin
        notify(entered)
        wait(release)
        10i+j
    end, ExpandingArrayMap{Int}((2,1); increment=2); concurrent=true)
    owner = Threads.@spawn cache[1,1]
    wait(entered)
    lock(cache.map.growth) do
        previous = JA.current_allocation(cache.map)
        allocation = JA.prepare_expansion!(previous, (4,1))
        try
            JA.relocate_slot!(previous, allocation, (1,1))
            notify(release)
            @test timedwait(() -> istaskdone(owner), 5; pollint=0.001) === :ok
            @test JA.current_allocation(cache.map) === previous
            @test JA.slot_entry(JA.allocation_slot(allocation, (1,1))) == 11
            completed_ready = isready(cache, 1,1)
            @test completed_ready
        finally
            notify(release)
            JA.relocate_slot!(previous, allocation, (2,1))
            @atomic :release cache.map.allocation = allocation
            @atomic :release previous.successor = previous
        end
    end
    @test fetch(owner) == 11
    @test cache[1,1] == 11
end
