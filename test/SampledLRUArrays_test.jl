function _finish_sampled_batch(task)
    completed = timedwait(() -> istaskdone(task), 5)
    @test completed == :ok
    if completed == :ok
        return fetch(task)
    end
    return nothing
end

@testset "Sampled LRU clock selection" begin
    for (symbol, clocktype) in ((:coarse, CoarseClock), (:lossy, LossyCounter), (:nanosecond, NanosecondClock))
        clock = clocktype()
        first = @inferred JITArrays._clock_tick!(clock)
        second = @inferred JITArrays._clock_tick!(clock)
        @test first isa UInt64
        @test second >= first
        generator = i -> fill(i, 2, 2)
        cache = SampledLRUArray{Matrix{Int}}((8,), 4, generator; clock=symbol)
        explicit = SampledLRUArray{Matrix{Int}}((8,), 4, generator; clock)
        @test typeof(cache) === typeof(explicit)
        @test explicit.clock === clock
        expected = [cache[i] for i in 1:4]
        @test _ready_cache_allocations(cache, expected) == 0
        @test _ready_get_allocations(cache, expected) == 0
        @test (@inferred getindex(cache, 1)) === expected[1]
        jobs = [Threads.@spawn begin
            for step in 1:200
                key = mod(step + worker, 8) + 1
                @assert cache[key] == fill(key, 2, 2)
            end
        end for worker in 1:4]
        @test timedwait(() -> all(istaskdone, jobs), 10) == :ok
        @test all(task -> istaskdone(task) && !istaskfailed(task), jobs)
        @test cache.pending == 0
        @test cache.ready == count(i -> isready(cache, i), 1:8) == 4
    end
    a = SampledLRUArray{Int}((1,), 1, identity; clock=:lossy)
    b = SampledLRUArray{Int}((1,), 1, identity; clock=:lossy)
    @test a.clock !== b.clock
    a.clock.value = UInt64(typemax(UInt32))
    @test a[1] == 1
    @test JITArrays._clock_now(a.clock) == UInt64(typemax(UInt32)) + 1
    @test JITArrays._clock_now(b.clock) == 0
    @test_throws ArgumentError SampledLRUArray{Int}((1,), 1, identity; clock=:unknown)
end

@testset "Sampled LRU indexing and values" begin
    for shape in ((), (4,), (2, 3), (2, 1, 3))
        N = length(shape)
        cache = SampledLRUArray{NTuple{N,Int}}(shape, max(1, prod(shape)), (indexes...) -> indexes)
        for (address, key) in enumerate(CartesianIndices(shape))
            @test !isready(cache, address)
            @test (@inferred getindex(cache, address)) === Tuple(key)
            @test cache[key] === cache[Tuple(key)...] === getindex(cache, Int[Tuple(key)...]...)
            @test generate!(cache, Tuple(key)) === Tuple(key)
            @test isready(cache, key) && isready(cache, Int[Tuple(key)...])
        end
        @test_throws BoundsError cache[0]
        @test_throws BoundsError isready(cache, prod(shape) + 1)
    end
    scalar = SampledLRUArray{Int}((), 1, () -> 7)
    @test (@inferred scalar[]) === 7
    @test generate!(scalar, ()) === 7
    @test _finish_sampled_batch(generate!(scalar, ((), ()))) === scalar
    cache = SampledLRUArray{Matrix{Int}}([2, 3], 6, (i, j) -> fill(i + 2(j - 1), 2, 2))
    @test (@inferred cache[2, 3]) == fill(6, 2, 2)
    @test (@inferred getindex(cache, 2, 3)) === cache[6]
    @test (@inferred generate!(cache, 1, 3)) == fill(5, 2, 2)
    @test_throws BoundsError cache[3, 1]
    @test getindex(cache, [1]) == [cache[1]]
    @test cache[CartesianIndex(1)] == cache[1]
    @test eltype(cache) === Matrix{Int}
    expected = [cache[i] for i in 1:6]
    @test _ready_cache_allocations(cache, expected) == 0
    @test _ready_get_allocations(cache, expected) == 0
    @test_throws ArgumentError SampledLRUArray{Int}((2,), 0, identity)
    @test_throws ArgumentError SampledLRUArray{Int}((-1,), 1, identity)
    for shape in ((0,), (2, 0))
        empty_cache = SampledLRUArray{Int}(shape, 1, (indexes...) -> 1)
        @test _finish_sampled_batch(generate!(empty_cache, Int[])) === empty_cache
        @test_throws BoundsError empty_cache[1]
    end
    for V in (Any, Union{Nothing,Int})
        values = SampledLRUArray{V}((2,), 1, i -> i == 1 ? nothing : i)
        @test values[1] === nothing
        @test isready(values, 1)
        @test values[2] === 2
        @test !isready(values, 1)
        @test values[1] === nothing
    end
end

@testset "Sampled LRU capacity waits and retained results" begin
    started, release = Channel{Nothing}(1), Channel{Nothing}(1)
    calls = Threads.Atomic{Int}(0)
    cache = SampledLRUArray{Matrix{Int}}((3,), 1, i -> begin
        attempt = Threads.atomic_add!(calls, 1)
        if attempt == 0
            put!(started, nothing)
            take!(release)
        end
        return fill(i, 2, 2)
    end)
    owner = Threads.@spawn cache[1]
    tasks = Task[owner]
    entered = timedwait(() -> isready(started), 5)
    @test entered == :ok
    try
        if entered == :ok
            pending = JITArrays._entry(cache, 1)
            followers = [@async cache[1] for _ in 1:4]
            append!(tasks, followers)
            other = @async cache[2]
            push!(tasks, other)
            @test timedwait(() -> all(istaskstarted, tasks), 5) == :ok
            @test all(task -> !istaskdone(task), tasks)
            @test calls[] == 1
            put!(release, nothing)
            completed = timedwait(() -> all(istaskdone, tasks), 5)
            @test completed == :ok
            if completed == :ok
                value = fetch(owner)
                @test all(result -> result === value, fetch.(followers))
                @test fetch(other) == fill(2, 2, 2)
                @test cache[3] == fill(3, 2, 2)
                @test value == fill(1, 2, 2)
                JITArrays._wait_entry(pending)
                @test pending.value === value
                @test count(i -> isready(cache, i), 1:3) == 1
            end
        end
    finally
        if !isready(release) && !istaskdone(owner)
            put!(release, nothing)
        end
    end
end

@testset "Sampled LRU failure releases capacity and allows retry" begin
    started, release = Channel{Nothing}(1), Channel{Nothing}(1)
    failure = ErrorException("first attempt")
    calls = Threads.Atomic{Int}(0)
    cache = SampledLRUArray{Int}((2,), 1, i -> begin
        if Threads.atomic_add!(calls, 1) == 0
            put!(started, nothing)
            take!(release)
            throw(failure)
        end
        return i
    end)
    owner = Threads.@spawn try cache[1] catch exception; exception end
    entered = timedwait(() -> isready(started), 5)
    @test entered == :ok
    try
        if entered == :ok
            pending = JITArrays._entry(cache, 1)
            follower = @async try cache[1] catch exception; exception end
            other = @async cache[2]
            @test timedwait(() -> all(istaskstarted, (follower, other)), 5) == :ok
            @test !istaskdone(follower) && !istaskdone(other)
            put!(release, nothing)
            completed = timedwait(() -> all(istaskdone, (owner, follower, other)), 5)
            @test completed == :ok
            if completed == :ok
                @test fetch(owner) === failure
                @test fetch(follower) isa ErrorException
                @test fetch(follower) !== failure
                @test fetch(other) == 2
                @test cache[1] == 1
                @test_throws ErrorException JITArrays._wait_entry(pending)
                @test count(i -> isready(cache, i), 1:2) == 1
            end
        end
    finally
        if !isready(release) && !istaskdone(owner)
            put!(release, nothing)
        end
    end
    invalid = SampledLRUArray{Int}((1,), 1, _ -> 0.5)
    @test_throws InexactError invalid[1]
    @test !isready(invalid, 1)
    @test_throws InexactError invalid[1]
end

@testset "Sampled LRU duplicate batches and parallel churn" begin
    calls = [Threads.Atomic{Int}(0) for _ in 1:8]
    cache = SampledLRUArray{Int}((2, 4), 8, (i, j) -> begin
        key = i + 2(j - 1)
        Threads.atomic_add!(calls[key], 1)
        yield()
        return key
    end)
    batch = generate!(cache, ((1, 1), (1, 1), (2, 2), (2, 2)))
    @test batch isa Task
    @test _finish_sampled_batch(batch) === cache
    @test calls[1][] == calls[4][] == 1
    @test _finish_sampled_batch(generate!(cache, [CartesianIndex(2, 1), CartesianIndex(1, 3)])) === cache
    @test _finish_sampled_batch(generate!(cache, 1:8)) === cache
    @test all(call -> call[] == 1, calls)
    active = [Threads.Atomic{Int}(0) for _ in 1:8]
    duplicates = Threads.Atomic{Int}(0)
    bounded = SampledLRUArray{Matrix{Int}}((2, 4), 3, (i, j) -> begin
        key = i + 2(j - 1)
        if Threads.atomic_add!(active[key], 1) != 0
            Threads.atomic_add!(duplicates, 1)
        end
        yield()
        value = fill(key, 2, 2)
        Threads.atomic_sub!(active[key], 1)
        return value
    end)
    tasks = [Threads.@spawn begin
        for step in 1:100
            key = mod(step + worker, 8) + 1
            if bounded[key] != fill(key, 2, 2)
                error("incorrect value")
            end
        end
    end for worker in 1:8]
    completed = timedwait(() -> all(istaskdone, tasks), 10)
    @test completed == :ok
    if completed == :ok
        @test all(task -> !istaskfailed(task), tasks)
        @test duplicates[] == 0
        @test count(i -> isready(bounded, i), 1:8) == 3
    end
end
