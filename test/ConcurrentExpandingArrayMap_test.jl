function expanding_test_result(f)
    try
        return f()
    catch error
        return error
    end
end

# Capturing the lookup before growth forces the caller to retain an old slot,
# independently of when its task next gets scheduled.
function expanding_captured_waiter(cache, indices, captured)
    return Threads.@spawn begin
        expanding_test_result() do
            lookup = try
                JA.map_lookup(cache, indices)
            finally
                put!(captured, nothing)
            end
            JA.cache_get(cache, lookup, indices)
        end
    end
end

struct ConcurrentExpandingConversion
    cache
    value::Int
end
function Base.convert(::Type{Int}, source::ConcurrentExpandingConversion)
    source.cache[6, 3]
    return source.value
end

struct ConcurrentExpandingFailure
    error::ErrorException
end
Base.convert(::Type{Int}, source::ConcurrentExpandingFailure) = throw(source.error)

function expanding_read_allocations(cache, indices)
    cache[indices...]
    return @allocated cache[indices...]
end

expanding_request_allocations(cache, indices) = @allocated cache[indices...]

@testset "Concurrent expanding interface" begin
    map_spec = ((V, domain) -> ExpandingArrayMap{V}(domain; increment=8))
    cache = MemoizedArray(tuple, map_spec(Tuple{Int,Int}, (2,3)); concurrent=true)
    @test size(cache) == (2,3)
    @test !isready(cache, 1, 1)
    @test !isready(cache, 100, 100)
    @test size(cache) == (2,3)
    @test (@inferred cache[1,3]) === (1,3)
    @test (@inferred cache[3,2]) === (3,2)
    @test size(cache) == (10,3)
    @test cache[1,3] === (1,3)
    @test (@inbounds cache[22,12]) === (22,12)
    @test size(cache) == (26,19)
    @test isready(cache, 1, 3)
    @test_throws BoundsError cache[0,1]
    @test_throws ArgumentError cache[1]

    offset = MemoizedArray(tuple, map_spec(Tuple{Int,Int}, (-2:-1,4:5)); concurrent=true)
    selected = view(offset, -2:-1, 4:5)
    @test offset[-2,5] === (-2,5)
    @test offset[0,6] === (0,6)
    @test axes(offset) == (-2:7,4:13)
    @test copy(selected) == [(i,j) for i in -2:-1, j in 4:5]
    @test_throws BoundsError offset[-3,4]
    @test_throws BoundsError offset[-2,3]

    zero = MemoizedArray(+, map_spec(Int, (0,2)); concurrent=true)
    @test isempty(zero)
    @test zero[1,2] == 3
    @test size(zero) == (8,2)
    scalar = MemoizedArray(() -> 7, map_spec(Int, ()); concurrent=true)
    @test scalar[] == 7
    @test size(scalar) == ()

    matrix = ones(4,4)
    matrices = MemoizedArray(Returns(matrix), map_spec(typeof(matrix), (1,1)); concurrent=true)
    retained = matrices[1,1]
    @test matrices[4,5] === retained === matrices[1,1]
    expanding_read_allocations(matrices, (1,1))
    @test expanding_read_allocations(matrices, (1,1)) == 0
    expanding_request_allocations(matrices, (2,1))
    @test expanding_request_allocations(matrices, (3,1)) == 0
    @test_throws ArgumentError MemoizedArray(identity, map_spec(Int, unbound(1)); concurrent=true)
end

@testset "Saved empty slots and retained out-of-domain failures" begin
    calls = Ref(0)
    cache = MemoizedArray((i,j) -> (calls[] += 1; 10i+j), ExpandingArrayMap{Int}((2,2); increment=2); concurrent=true)
    empty_lookup = JA.map_lookup(cache, (1,2))
    @test cache[1,1] == 11
    ready_lookup = JA.map_lookup(cache, (1,1))
    @test cache[3,3] == 33
    @test JA.cache_get(cache, empty_lookup, (1,2)) == 12
    @test JA.cache_get(cache, ready_lookup, (1,1)) == 11
    @test cache[1,2] == 12 && calls[] == 3

    failure = ErrorException("out-of-domain generation failed")
    attempts = Ref(0)
    failing = MemoizedArray(i -> (attempts[] += 1; throw(failure)), ExpandingArrayMap{Int}((1,); increment=2); concurrent=true)
    @test expanding_test_result(() -> failing[2]) === failure
    @test size(failing) == (3,)
    @test expanding_test_result(() -> failing[2]) === failure
    @test attempts[] == 1 && !isready(failing,2)
end

@testset "Generation and old waiters survive relocation" begin
    for outcome in (:value, :generator_failure, :conversion_failure)
        entered, release = Base.Event(), Base.Event()
        calls = Threads.Atomic{Int}(0)
        failure = ErrorException("failure survives expansion: $outcome")
        generator = (i,j) -> begin
            if (i,j) == (1,2)
                Threads.atomic_add!(calls, 1)
                notify(entered)
                wait(release)
                outcome === :generator_failure && throw(failure)
                outcome === :conversion_failure && return ConcurrentExpandingFailure(failure)
                return 12
            end
            return 100i + j
        end
        cache = MemoizedArray(generator, ExpandingArrayMap{Int}((2,2); increment=2); concurrent=true)
        @test cache[2,1] == 201
        owner = Threads.@spawn begin
            try
                expanding_test_result(() -> cache[1,2])
            finally
                # Constructor/access failures must not leave the test waiting forever.
                notify(entered)
            end
        end
        wait(entered)
        captured = Channel{Nothing}(3)
        old_waiters = [expanding_captured_waiter(cache, (1,2), captured) for _ in 1:3]
        foreach(_ -> take!(captured), old_waiters)
        new_waiters = Task[]
        try
            @test !isready(cache, 1, 2)
            @test cache[3,1] == 301
            @test size(cache) == (4,2)
            @test cache[2,1] == 201
            # A second relocation moves the pending slot through three allocations.
            @test cache[5,4] == 504
            @test size(cache) == (6,4)
            new_waiters = [Threads.@spawn expanding_test_result(() -> cache[1,2]) for _ in 1:3]
            GC.gc(false)
        finally
            notify(release)
        end
        expected = outcome === :value ? 12 : failure
        @test all(result -> result === expected, fetch.([owner; old_waiters; new_waiters]))
        @test calls[] == 1
        @test isready(cache, 1,2) == (outcome === :value)
        @test expanding_test_result(() -> cache[1,2]) === expected
        @test calls[] == 1
        @test cache[9,7] == 907
        @test expanding_test_result(() -> cache[1,2]) === expected
        @test calls[] == 1
    end
end

@testset "Nested generators and conversions expand safely" begin
    for requested in ((1,1), (2,2))
        owner = Ref{Any}()
        calls = Threads.Atomic{Int}(0)
        cache = MemoizedArray((i,j) -> begin
            Threads.atomic_add!(calls, 1)
            (i,j) == requested && owner[][6,3]
            (i,j)
        end, ExpandingArrayMap{Tuple{Int,Int}}((1,1); increment=4); concurrent=true)
        owner[] = cache
        @test cache[requested...] === requested
        @test size(cache) == (9,5)
        @test cache[requested...] === requested
        @test cache[6,3] === (6,3)
        @test calls[] == 2
    end
    owner = Ref{Any}()
    cache = MemoizedArray((i,j) -> (i,j) == (1,1) ?
        ConcurrentExpandingConversion(owner[], 11) : 63, ExpandingArrayMap{Int}((1,1); increment=4); concurrent=true)
    owner[] = cache
    @test cache[1,1] == 11
    @test size(cache) == (9,5)
    @test cache[1,1] == 11 && cache[6,3] == 63
end

@testset "Parallel independent growth and overlapping requests" begin
    callers = 8
    start = Base.Event()
    entered = Channel{Nothing}(callers)
    release = Base.Event()
    calls = Threads.Atomic{Int}(0)
    cache = MemoizedArray((i,j) -> begin
        Threads.atomic_add!(calls, 1)
        if j == 1 && i <= callers
            put!(entered, nothing)
            wait(release)
        end
        yield()
        (i,j)
    end, ExpandingArrayMap{Tuple{Int,Int}}((1,1); increment=2); concurrent=true)
    tasks = [Threads.@spawn begin
        wait(start)
        result = expanding_test_result(() -> cache[i,1])
        result isa Exception && put!(entered, nothing)
        result
    end for i in 1:callers]
    notify(start)
    try
        # All generators must be able to enter before any one of them finishes.
        for _ in 1:callers
            take!(entered)
        end
    finally
        notify(release)
    end
    @test fetch.(tasks) == [(i,1) for i in 1:callers]
    @test calls[] == callers
    start2 = Base.Event()
    readers = [Threads.@spawn begin
        wait(start2)
        valid = true
        for step in 1:256
            i = mod(7step + caller, 32) + 1
            j = mod(11step + 3caller, 17) + 1
            valid &= cache[i,j] === (i,j)
        end
        valid
    end for caller in 1:callers]
    notify(start2)
    @test all(fetch.(readers))
    # Include the initial gated requests when comparing unique coordinates.
    requested = Set((i,1) for i in 1:callers)
    union!(requested, ((mod(7step + caller,32)+1, mod(11step + 3caller,17)+1)
        for caller in 1:callers for step in 1:256))
    @test calls[] == length(requested)
    @test all(key -> cache[key...] === key, requested)
    @test calls[] == length(requested)
end

@testset "Overflow preserves pending generation" begin
    entered, release = Base.Event(), Base.Event()
    lower = typemax(Int) - 2
    cache = MemoizedArray(i -> begin
        notify(entered)
        wait(release)
        i
    end, ExpandingArrayMap{Int}((lower:lower+1,); increment=8); concurrent=true)
    owner = Threads.@spawn begin
        try
            expanding_test_result(() -> cache[lower])
        finally
            notify(entered)
        end
    end
    wait(entered)
    try
        @test_throws OverflowError cache[typemax(Int)]
        @test size(cache) == (2,)
        @test !isready(cache, lower)
    finally
        notify(release)
    end
    @test fetch(owner) == lower
    @test cache[lower] == lower
    wide = MemoizedArray(identity, ExpandingArrayMap{Int}((typemin(Int):typemin(Int)+1,); increment=8); concurrent=true)
    @test_throws OverflowError wide[0]
    @test size(wide) == (2,)
    @test wide[typemin(Int)] == typemin(Int)
end

@testset "Any and marker values survive growth" begin
    matrix = ones(2,2)
    failure = JA.GenerationFailure(ErrorException("this is a value"))
    values = (nothing, false, JA.EMPTY, JA.GENERATING, JA.RELOCATED,
        failure, matrix, view(matrix,:,1:1))
    cache = MemoizedArray(i -> values[i], ExpandingArrayMap{Any}((1,); increment=2); concurrent=true)
    for i in eachindex(values)
        @test cache[i] === values[i]
        @test isready(cache, i)
    end
    @test all(i -> cache[i] === values[i], eachindex(values))
    for value in (JA.EMPTY, JA.GENERATING, JA.RELOCATED, failure)
        typed = MemoizedArray(Returns(value), ExpandingArrayMap{typeof(value)}((1,); increment=2); concurrent=true)
        @test typed[1] === value
        @test typed[3] === value
        @test typed[1] === value
        @test isready(typed, 1)
    end
end
