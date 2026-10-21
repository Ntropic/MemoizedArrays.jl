# Manual values have separate entries and consume no eviction capacity.
pinned_configs(T) = (ArrayMap{T}((0:128,)), ExpandingArrayMap{T}((0:3,); increment=8),
    RadixMap{T}((0:128,); block_shape=(8,)), SubtreeRadixMap{T}((0:128,); block_shape=(8,)),
    DictMap{T}((0:128,)))

@testset "Fresh pinned entries across maps" begin
    for config in pinned_configs(Vector{Int}), concurrent in (false,true)
        cache = LRUMemoizedArray(i -> [i], config; capacity=2, concurrent, queues=1)
        handle = concurrent ? queue_handle(cache,1) : cache
        handle[0] = [100]
        firstpin = JA.entry(cache,JA.cache_key(cache,(0,)))
        @test firstpin.explicit
        @test JA.resident_count(cache) == 0
        retained = handle[1]
        old = JA.entry(cache,JA.cache_key(cache,(1,)))
        handle[CartesianIndex(1)] = [101]
        pin = JA.entry(cache,JA.cache_key(cache,(1,)))
        @test pin !== old && pin.explicit && !old.explicit
        @test JA.resident_count(cache) == 0
        handle[1] = [102]
        @test JA.entry(cache,JA.cache_key(cache,(1,))) !== pin
        generated = [handle[i] for i in 2:128]
        @test generated == [[i] for i in 2:128]
        @test retained == [1]
        @test pin.value == [101]
        @test handle[0] == [100] && handle[1] == [102]
        @test JA.resident_count(cache) == 2
        @test isready(handle,127) && isready(handle,128)
        @test !isready(handle,2)
        if concurrent
            @test all(!e.explicit for q in cache.eviction.queues for e in q.pool)
        else
            @test all(!e.explicit for e in cache.eviction.entries)
        end
    end
end

@testset "Serial pool holes preserve order" begin
    for capacity in (1,3), removed in 1:capacity
        cache = LRUMemoizedArray(identity,ArrayMap{Int}((12,));capacity)
        foreach(i -> cache[i],1:capacity)
        old = JA.entry(cache,removed)
        cache[removed] = -removed
        @test cache.eviction.count == capacity-1
        cache[capacity+1]
        @test JA.entry(cache,capacity+1) === old
        @test cache.eviction.count == capacity
        if capacity > 1
            oldest = first(filter(!=(removed),collect(1:capacity)))
            cache[capacity+2]
            @test !isready(cache,oldest)
        end
        @test cache[removed] == -removed
    end
end

@testset "Pinned seeds and conversion failure" begin
    for concurrent in (false,true)
        calls = zeros(Int, 12)
        cache = LRUMemoizedArray(ArrayMap{Int}((12,)); capacity=1, queues=1, concurrent) do i
            calls[i] += 1
            handle[i-1] + 1
        end
        handle = concurrent ? queue_handle(cache,1) : cache
        handle[1] = 1
        @test handle[12] == 12
        @test calls == [0; ones(Int, 11)]
        @test isready(handle, 1) && !isready(handle, 2)
        @test JA.resident_count(cache) == 1
        @test handle[1] == 1
        @test_throws InexactError setindex!(handle,0.5,1)
        @test handle[1] == 1
        @test_throws BoundsError setindex!(handle,1,13)
    end
    # Nested generation or conversion must admit a key only once.
    for config in pinned_configs(Int), conversion in (false,true)
        calls = Ref(0)
        cache = LRUMemoizedArray(config; capacity=2) do i
            complete = () -> begin
                calls[] += 1
                i == 1 && calls[] == 1 ? cache[i] + 1 : i
            end
            conversion ? TestConversion(complete) : complete()
        end
        @test cache[1] == 2 && calls[] == 2
        @test JA.resident_count(cache) == 1
        cache[1] = 42
        @test JA.resident_count(cache) == 0
        @test [cache[i] for i in 2:4] == collect(2:4)
        @test isready(cache,1) && cache[1] == 42
        @test JA.resident_count(cache) == 2
    end
    # A nested generator can supply the requested value explicitly.
    for config in pinned_configs(Int)
        cache = LRUMemoizedArray(i -> (cache[i]=42; 7),config;capacity=1)
        @test cache[1] == 42
        @test cache.eviction.count == 0
    end
end

@testset "Other queues keep stale references" begin
    for config in pinned_configs(Int)
        cache = LRUMemoizedArray(identity,config;capacity=4,queues=2,concurrent=true)
        a,b = queue_handles(cache)
        @test a[1] == b[1] == 1
        old = JA.entry(cache,JA.cache_key(cache,(1,)))
        oldcount,oldest,newest = b.queue.count,b.queue.oldest,b.queue.newest
        a[1] = 100
        @test (b.queue.count,b.queue.oldest,b.queue.newest) == (oldcount,oldest,newest)
        @test b.queue.entries[oldest] === old
        @test b[1] == 100
        @test b.queue.count == oldcount
        for i in 2:64
            a[2i-1]
            b[2i]
        end
        @test a[1] == b[1] == 100
    end
end

@testset "Assignment waits for generation" begin
    entered,release = Base.Event(),Base.Event()
    cache = LRUMemoizedArray(i -> (notify(entered); wait(release); [i]),
        ArrayMap{Vector{Int}}((2,));capacity=2,queues=2,concurrent=true)
    a,b = queue_handles(cache)
    owner = Threads.@spawn a[1]
    wait(entered)
    setter = Threads.@spawn (b[1] = [99])
    ref = JA.map_slot(cache.map,1,nothing)
    while !(JA.slot_entry(ref) isa JA.GenerateWait)
        yield()
    end
    notify(release)
    @test fetch(owner) == [1]
    @test fetch(setter) == [99]
    @test a[1] == b[1] == [99]
end

@testset "Concurrent pinning and eviction" begin
    for config in pinned_configs(Int)
        cache = LRUMemoizedArray(identity,config;capacity=8,queues=4,concurrent=true)
        handles = queue_handles(cache)
        foreach(h -> h[0], handles)
        tasks = map(1:4) do q
            Threads.@spawn begin
                h, valid = handles[q], true
                for i in 1:24
                    h[q] = -q
                    h[0] = -q
                    h[4i+q]
                    valid &= h[q] == -q
                    valid &= h[0] in -4:-1
                end
                valid
            end
        end
        @test all(fetch.(tasks))
        @test all(q -> handles[q][q] == -q,1:4)
        @test JA.entry(cache, JA.cache_key(cache, (0,))).explicit
    end
end

@testset "Pinned broad values and multidimensional coordinates" begin
    for concurrent in (false,true)
        cache = LRUMemoizedArray((i,j) -> (i,j),ArrayMap{Any}((2,2));capacity=1,concurrent,queues=1)
        h = concurrent ? queue_handle(cache,1) : cache
        h[CartesianIndex(1,2)] = nothing
        h[2,1] = false
        @test h[1,2] === nothing && h[2,1] === false
        @test h[2,2] == (2,2)
        @test h[1,1] == (1,1)
        @test isready(h,1,2) && isready(h,2,1)
    end
end

@testset "Pinned matrix reads and pooled reuse do not allocate" begin
    payload = ones(4,4)
    for concurrent in (false,true)
        cache = LRUMemoizedArray(Returns(payload),ArrayMap{typeof(payload)}((8,));capacity=2,concurrent,queues=1)
        h = concurrent ? queue_handle(cache,1) : cache
        h[1] = payload
        request_allocations(h,1)
        @test request_allocations(h,1) == 0
        request_allocations(h,2)
        @test request_allocations(h,3) == 0
        @test request_allocations(h,4) == 0
        h[4] = payload
        @test request_allocations(h,5) == 0
        @test h[1] === h[4] === payload
    end
end
