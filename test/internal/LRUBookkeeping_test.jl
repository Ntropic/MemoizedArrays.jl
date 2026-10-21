@testset "Serial tree queue reentrant admission and entry reuse" begin
    for Config in (RadixMap, SubtreeRadixMap), conversion in (false, true)
        owner, entered = Ref{Any}(), Ref(false)
        nested() = (owner[][0]; 11)
        cache = LRUMemoizedArray(i -> begin
            i != 0 && return i
            entered[] && return 7
            entered[] = true
            return conversion ? TestConversion(nested) : nested()
        end, Config{Int}(unbound(1); block_shape=(8,)); capacity=1)
        owner[] = cache
        @test cache.eviction isa JA.SerialLRUQueue
        @test cache[0] == cache[0] == 11
        @test JA.resident_count(cache) == 1
        original = cache.eviction.entries[1]
        @test cache[64] == 64
        @test JA.entry(cache, (64,)) === original
        @test !isready(cache, 0)
    end
    value = ones(4, 4)
    for Config in (RadixMap, SubtreeRadixMap)
        cache = LRUMemoizedArray(Returns(value), Config{typeof(value)}(unbound(1); block_shape=(8,)); capacity=1)
        cache[0]
        request_allocations(cache, 1)
        @test request_allocations(cache, 2) == 0
        @test request_allocations(cache, 2) == 0
        @test cache[3] === value
    end
end

@testset "Serial LRU ordering and permanent node IDs" begin
    for capacity in (1, 2, 3), map_spec in
        (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain)), ((V, domain) -> RadixMap{V}(domain; block_shape=(8,))), ((V, domain) -> SubtreeRadixMap{V}(domain; block_shape=(8,))))
        cache = LRUMemoizedArray(identity, map_spec(Int, (0:6,)); capacity)
        @test cache.eviction.count == 0
        order = Int[]
        for key in (1, 2, 3, 2, 1, 4, 4, 3, 5, 6, 1)
            filter!(!=(key), order)
            push!(order, key)
            if length(order) > capacity
                popfirst!(order)
            end
            @test cache[key] == key
            @test cache.eviction.count == length(order)
            @test findall(i -> isready(cache, i), 1:6) == sort(order)
            for resident in order
                entry = JA.entry(cache, JA.cache_key(cache, (resident,)))
                @test cache.eviction.entries[entry.id] === entry
            end
            actual = Int[]
            position = cache.eviction.oldest
            for _ in order
                push!(actual, cache.eviction.entries[position].value)
                position = cache.eviction.links.mem[2Int(position)]
            end
            @test actual == order
        end
    end
end

@testset "Serial generator errors preserve residents" begin
    for map_spec in (:array, :dict, :radix, :subtree)
        configuration = test_map_config(map_spec)
        calls = Ref(0)
        cache = LRUMemoizedArray(i -> (calls[] += 1; i == 2 ? error("generator failed") : i), configuration(Int, (0:8,)); capacity=2)
        generate!(cache, [0, 1])
        @test_throws ErrorException cache[2]
        @test_throws ErrorException cache[2]
        @test calls[] == 4 && cache.eviction.count == 2
        @test isready(cache, 0) && isready(cache, 1) && !isready(cache, 2)
        @test cache[3] == 3
        @test !isready(cache, 0) && isready(cache, 1)
        @test count(key -> isready(cache, key), 0:8) == cache.eviction.count == 2
        if cache.map isa JA.TreeMap
            treecounts(cache.map)
        end
        converted = LRUMemoizedArray(i -> i == 2 ? 0.5 : i, configuration(Int, (0:8,)); capacity=2)
        generate!(converted, [0, 1])
        @test_throws InexactError converted[2]
        @test converted.eviction.count == 2 && !isready(converted, 2)
        @test isready(converted, 0) && isready(converted, 1)
        @test converted[3] == 3 && !isready(converted, 0)
        nested = LRUMemoizedArray(i -> i == 0 ? (nested[1]; error("outer generator failed")) : i, configuration(Int, (0:8,)); capacity=1)
        @test_throws ErrorException nested[0]
        @test nested[1] == 1 && !isready(nested, 0) && nested.eviction.count == 1
    end
end

@testset "Preallocated serial matrix entries" begin
    value = ones(4, 4)
    make() = LRUMemoizedArray(Returns(value), ArrayMap{typeof(value)}((8,)); capacity=3)
    warmup = make()
    for key in 1:4
        request_allocations(warmup, key)
    end
    cache = make()
    entries = copy(cache.eviction.entries)
    @test cache.eviction.count == 0 && !isready(cache, 1)
    for key in 1:3
        @test request_allocations(cache, key) == 0
        @test any(entry -> JA.entry(cache, key) === entry, entries)
    end
    @test request_allocations(cache, 4) == 0
    @test request_allocations(cache, 4) == 0
    @test cache.eviction.count == 3 && !isready(cache, 1)
end

@testset "Private LRU last-access eviction and reusable incarnations" begin
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((32,)); concurrent=true, capacity=4, queues=2)
    a,b = queue_handle(cache,1),queue_handle(cache,2)
    @test (@inferred a[1]) == 1
    @test a[2] == 2 && b[1] == 1
    retained = JA.entry(cache,1)
    victim = JA.entry(cache,2)
    version = JA.private_state(victim)
    @test a[3] == 3
    @test JA.entry(cache,1) === retained # queue 2 most recently read it
    @test !isready(cache,2) && JA.entry(cache,3) === victim
    @test JA.private_state(victim) != version
    @test b[3] == 3 && a[4] == 4 && a[5] == 5
    @test isready(cache,3) # queue 1 dropped its reference; queue 2 protects it
    check_private_queues(cache)

    # A read can evict without generating: retain the freed entry for reuse.
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); concurrent=true, capacity=2, queues=2)
    a,b = queue_handle(cache,1),queue_handle(cache,2)
    a[1]; b[2]
    old = JA.entry(cache,1)
    oldversion = JA.private_state(old)
    @test a[2] == 2 && !isready(cache,1)
    @test JA.private_state(old) & 3 == 2
    @test a[3] == 3 && JA.entry(cache,3) === old
    @test !JA.private_reclaim!(cache,b.cache.eviction.queues[2],old,oldversion)
    @test !first(JA.private_snapshot(old,1)) # retained pointer cannot return key 3 for key 1
    @test isready(cache,3)
    check_private_queues(cache)
    # Delayed recency writes from an old incarnation are harmless hints. They
    # must not pin the replacement on behalf of a queue that never read it.
    @atomic :monotonic old.last_access = oldversion | UInt64(2)
    @test a[4] == 4 && !isready(cache,3)
    @test JA.private_state(old) & 3 == 2 # another free entry served key 4
    check_private_queues(cache)

    # A full queue replaces membership while another queue protects the value.
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); concurrent=true, capacity=6, queues=3)
    a,b,c = (queue_handle(cache,i) for i in 1:3)
    a[1]; a[2]; b[1]; c[3]; c[4]
    @test a[3] == 3 && isready(cache,1)
    check_private_queues(cache)

    # Dropping stale membership must leave the object's newer incarnation live.
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); concurrent=true, capacity=4, queues=2)
    a,b = queue_handle(cache,1),queue_handle(cache,2)
    a[1]; a[2]; b[1]; b[3]; a[1]; a[4]; a[5]
    replacement = JA.entry(cache,5)
    @test b[4] == 4 && JA.entry(cache,5) === replacement
    check_private_queues(cache)

    # Same object and same map key can recur; a stale incarnation must still
    # fail before the unconditional map-slot deletion after an entry claim.
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((4,)); concurrent=true, capacity=1, queues=1)
    a = queue_handle(cache,1)
    @test a[1] == 1
    old = JA.entry(cache,1)
    oldversion = JA.private_state(old)
    q = cache.eviction.queues[1]
    position = q.positions.mem[old.id]
    @test a[2] == 2 && a[1] == 1
    @test JA.entry(cache,1) === old && q.positions.mem[old.id] == position
    @test JA.private_state(old) != oldversion
    @test !JA.private_reclaim!(cache,q,old,oldversion)
    @test JA.entry(cache,1) === old && a[1] == 1
    check_private_queues(cache)
end

@testset "Private LRU remembered reads across entry reuse" begin
    for V in (Int,Vector{Int}), replacement_key in (1,4)
        values = V === Int ? collect(101:108) : [[i,10i] for i in 1:8]
        calls = zeros(Int,8)
        generator = i -> begin
            calls[i] += 1
            values[i]
        end
        cache = LRUMemoizedArray(generator, ArrayMap{V}((8,)); concurrent=true, capacity=2, queues=2)
        a,b = queue_handle(cache,1),queue_handle(cache,2)
        qa,qb = a.queue,b.queue
        @test a[1] === values[1]
        @test b[2] === values[2]
        entry = JA.entry(cache,1)
        position = qa.positions.mem[entry.id]
        version = qa.incarnations.mem[position]
        @test a[1] === values[1] # already remembered by this queue
        @test calls[1] == 1
        @test qb.positions.mem[entry.id] == 0
        @test b[1] === values[1] # present in the map, absent from this queue
        @test calls[1] == 1 && qb.positions.mem[entry.id] != 0

        # Queue b frees and reuses this exact object; queue a retains its old
        # position and incarnation, for either the old key or a different key.
        @test b[3] === values[3]
        @test b[replacement_key] === values[replacement_key]
        @test JA.entry(cache,replacement_key) === entry
        @test qa.positions.mem[entry.id] == position
        @test qa.incarnations.mem[position] == version
        replacement_version = JA.private_state(entry)
        @test replacement_version != version
        generated = sum(calls)
        @test a[replacement_key] === values[replacement_key]
        @test qa.incarnations.mem[position] == replacement_version
        @test a[replacement_key] === values[replacement_key]
        @test sum(calls) == generated
        if replacement_key != 1
            @test !isready(cache,1)
            @test a[1] === values[1] # never return the replacement key's value
            @test calls[1] == 2
        end
        check_private_queues(cache)
    end
end

@testset "Private LRU generation into a read-filled queue" begin
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity=4, concurrent=true, queues=2)
    a, b = queue_handle(cache, 1), queue_handle(cache, 2)
    b[1]; b[2]; a[1]; a[2]
    @test a.queue.count == a.queue.capacity
    @test a.queue.free_count == a.queue.capacity
    @test a[3] == 3
    @test !isready(cache, 1)
    check_private_queues(cache)
end

@testset "Private LRU free entry retains local membership" begin
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity=4, concurrent=true, queues=2)
    a, b = queue_handle(cache, 1), queue_handle(cache, 2)
    a[1]; b[2]; b[3]; a[2]; a[3]; b[4]; a[4]; b[4]; b[3]; a[6]; b[6]
    q = a.queue
    entry = q.free[q.free_count]
    version = JA.private_state(entry)
    position = q.positions.mem[entry.id]
    @test version & 3 == 2
    @test position != 0
    @test a[7] == 7
    @test JA.entry(cache, 7) === entry
    @test q.positions.mem[entry.id] == position
    @test q.incarnations.mem[position] == (version & ~(q.step - 1)) + q.step
    check_private_queues(cache)
end

@testset "Private LRU matrix entries are reused without allocation" begin
    value = ones(4,4)
    cache = LRUMemoizedArray(Returns(value), ArrayMap{typeof(value)}((16,)); concurrent=true, capacity=2, queues=1)
    handle = queue_handle(cache,1)
    request_allocations(handle,1) # compile measurement and read paths
    @test request_allocations(handle,2) == 0
    request_allocations(handle,3) # compile eviction path
    @test request_allocations(handle,4) == 0
    @test request_allocations(handle,4) == 0
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
    read_allocations(cache)
    @test read_allocations(cache) == 0

    matrices = LRUMemoizedArray(i -> fill(i, 2, 2), DictMap{Matrix{Int}}((1:3,)); capacity=1)
    retained = matrices[1]
    stored = JA.entry(matrices, JA.cache_key(matrices, (1,)))
    @test matrices[2] == fill(2, 2, 2)
    @test matrices[3] == fill(3, 2, 2)
    @test JA.entry(matrices, JA.cache_key(matrices, (3,))) === stored
    @test retained == fill(1, 2, 2)
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
