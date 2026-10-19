# Public constructors; each concurrent LRU handle has one active caller.
function concurrent_test_error(f)
    try
        return f()
    catch error
        return error
    end
end

function concurrent_test_cache(C, V, generator, domain; capacity=8, queues=8, map_factory=((V, domain) -> ArrayMap{V}(domain)))
    if C === MemoizedArray
        return C(generator, map_factory(V, domain); concurrent=true)
    end
    return C(generator, map_factory(V, domain); capacity, queues, concurrent=true)
end

concurrent_test_handle(cache::MemoizedArray, _) = cache
concurrent_test_handle(cache::LRUMemoizedArray, i) = queue_handle(cache, i)

function concurrent_test_readallocations(cache)
    cache[1]
    return @allocated cache[1]
end

@testset "Concurrent array and dictionary constructors and interface" begin
    for map_factory in (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain))), C in (MemoizedArray, LRUMemoizedArray)
        options = C === MemoizedArray ? (;) : (; capacity=4)
        ordinary = C(identity, map_factory(Int, (4,)); options...)
        explicit = C(identity, map_factory(Int, (4,)); options..., concurrent=false)
        @test typeof(ordinary) === typeof(explicit)
        cache = concurrent_test_cache(C, Int, (i, j) -> i + 10j, (-2:1, 4:6); map_factory)
        handle = concurrent_test_handle(cache, 1)
        @test size(cache) == size(handle) == (4, 3)
        @test !isready(cache, -2, 4)
        @test handle[-2, 4] == handle[CartesianIndex(-2, 4)] == 38
        @test isready(cache, (-2, 4))
        @test copy(view(handle, -1:0, 5:6)) == [i + 10j for i in -1:0, j in 5:6]
        @test generate!(handle, ((-2, 4), (1, 6))) === handle
        @test handle[1, 6] == 61
        @test_throws BoundsError handle[-3, 4]
        @test_throws ArgumentError handle[13]
        @test_throws Base.CanonicalIndexError (handle[-2, 4] = 0)
        scalar = concurrent_test_cache(C, Int, () -> 9, (); map_factory)
        @test concurrent_test_handle(scalar, 1)[] == 9
        @test isempty(concurrent_test_cache(C, Int, identity, (0,); map_factory))
        @test_throws ArgumentError C(identity, ArrayMap{Int}(unbound(1)); options..., concurrent=true)
    end
    ready = MemoizedArray(identity, ArrayMap{Int}((3,)); concurrent=true)
    ready[2]
    @test getready(ready, 2) == 2
    @test_throws BoundsError getready(ready, 4)
end

@testset "One generation for concurrent callers" begin
    for map_factory in (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain))), C in (MemoizedArray, LRUMemoizedArray), fail in (false, true)
        calls = Threads.Atomic{Int}(0)
        entered, release = Base.Event(), Base.Event()
        value = [11, 12]
        failure = ErrorException("concurrent generation failed")
        generator = _ -> begin
            Threads.atomic_add!(calls, 1)
            notify(entered)
            wait(release)
            if fail
                throw(failure)
            end
            value
        end
        cache = concurrent_test_cache(C, Vector{Int}, generator, (8,); map_factory)
        handles = [concurrent_test_handle(cache, i) for i in 1:4]
        owner = Threads.@spawn concurrent_test_error(() -> handles[1][1])
        wait(entered)
        started = Channel{Nothing}(3)
        waiters = [Threads.@spawn begin
            put!(started, nothing)
            concurrent_test_error(() -> handles[i][1])
        end for i in 2:4]
        for _ in waiters
            take!(started)
        end
        for _ in 1:8
            yield()
        end
        @test !isready(cache, 1)
        GC.gc(false)
        notify(release)
        results = fetch.([owner; waiters])
        @test all(result -> result === (fail ? failure : value), results)
        @test calls[] == 1
        @test isready(cache, 1) == !fail
        @test concurrent_test_error(() -> handles[1][1]) === (fail ? failure : value)
        @test calls[] == 1
    end
end

@testset "Independent generators and nested requests" begin
    for map_factory in (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain))), C in (MemoizedArray, LRUMemoizedArray)
        entered, release = Base.Event(), Base.Event()
        cache = concurrent_test_cache(C, Int, i -> begin
            if i == 1
                notify(entered)
                wait(release)
            end
            i
        end, (8,); map_factory)
        first = concurrent_test_handle(cache, 1)
        second = concurrent_test_handle(cache, 2)
        slow = Threads.@spawn first[1]
        wait(entered)
        fast = Threads.@spawn second[2]
        completed = timedwait(() -> istaskdone(fast), 5; pollint=0.001)
        notify(release)
        @test completed === :ok
        @test fetch(fast) == 2 && fetch(slow) == 1

        dependent = concurrent_test_cache(C, Int, i -> i == 1 ? handle[2] + 10 : i, (4,); map_factory)
        handle = concurrent_test_handle(dependent, 1)
        @test handle[1] == 12
        @test handle[2] == 2
    end
end

@testset "Concurrent generation and pooled entry replacement" begin
    for map_factory in (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain))), C in (MemoizedArray, LRUMemoizedArray)
        calls = Threads.Atomic{Int}(0)
        cache = concurrent_test_cache(C, Tuple{Int,Int}, i -> begin
            Threads.atomic_add!(calls, 1)
            yield()
            (i, i * i)
        end, (32,); map_factory)
        handles = [concurrent_test_handle(cache, i) for i in 1:8]
        start = Base.Event()
        tasks = [Threads.@spawn begin
            wait(start)
            valid = true
            for step in 1:128
                key = mod(7step + 5i, 32) + 1
                valid &= handles[i][key] === (key, key * key)
                if step % 16 == 0
                    yield()
                end
            end
            valid
        end for i in 1:8]
        notify(start)
        @test all(fetch.(tasks))
        populated = count(key -> isready(cache, key), 1:32)
        if C === MemoizedArray
            @test populated == calls[] == 32
        else
            @test 0 < populated == JA.resident_count(cache) <= 8
        end
        @test all(key -> handles[1][key] == (key, key * key), 1:32)
    end
end

@testset "One concurrent queue preserves serial eviction order" begin
    trace = (1, 2, 3, 4, 1, 3, 5, 6, 3, 7, 8, 2, 2, 9, 10, 4, 1, 11, 12, 3)
    for map_factory in (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain)))
        serial = LRUMemoizedArray(identity, map_factory(Int, (12,)); capacity=4)
        concurrent = LRUMemoizedArray(identity, map_factory(Int, (12,)); capacity=4, concurrent=true, queues=1)
        handle = queue_handle(concurrent, 1)
        for key in trace
            @test serial[key] == handle[key] == key
            @test serial.eviction.count == JA.resident_count(concurrent)
            @test all(i -> isready(serial, i) == isready(concurrent, i), 1:12)
        end
    end
end

@testset "Concurrent values remain distinct from slot states" begin
    matrix = ones(2, 2)
    selected = view(matrix, :, 1:1)
    values = (nothing, false, JA.EMPTY, JA.GENERATING, matrix, selected)
    for map_factory in (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain))), C in (MemoizedArray, LRUMemoizedArray)
        broad = concurrent_test_cache(C, Any, i -> values[i], (length(values),); map_factory)
        handle = concurrent_test_handle(broad, 1)
        @test all(i -> handle[i] === values[i] && isready(broad, i), eachindex(values))
        @test all(i -> handle[i] === values[i], reverse(eachindex(values)))
        for value in (matrix, selected)
            cache = concurrent_test_cache(C, typeof(value), Returns(value), (8,); map_factory)
            handle = concurrent_test_handle(cache, 1)
            @test handle[1] === value
            concurrent_test_readallocations(handle)
            @test concurrent_test_readallocations(handle) == 0
            retained = handle[1]
            generate!(handle, 2:8)
            @test retained === value
        end
        distinct = concurrent_test_cache(C, Matrix{Int}, i -> fill(i, 2, 2), (16,); map_factory)
        handle = concurrent_test_handle(distinct, 1)
        retained = handle[1]
        generate!(handle, 2:16)
        GC.gc(false)
        @test retained == fill(1, 2, 2)
        @test handle[16] == fill(16, 2, 2)
        calls = Ref(0)
        converted = concurrent_test_cache(C, Int, _ -> (calls[] += 1; 0.5), (1,); map_factory)
        handle = concurrent_test_handle(converted, 1)
        @test_throws InexactError handle[1]
        @test_throws InexactError handle[1]
        @test calls[] == 1 && !isready(converted, 1)
    end
end
