@testset "Dictionary stable records and collisions across growth" begin
    JA = MemoizedArrays
    map = JA.DictMap(Union{JA.Empty,MapTestEntry{1}}, (-100000:100000,), JA.EMPTY)
    # Short and long keys distinguish full-hash collisions and bucket collisions.
    # Signed probe arithmetic also wraps at the table edge.
    for N in 1:3
        probe = JA.DictMap(Union{JA.Empty,MapTestEntry{N}}, ntuple(_ -> 0:4, N), JA.EMPTY)
        for (i, hashed) in enumerate((typemax(UInt), typemax(UInt),
            typemax(UInt) - UInt(64), UInt(1) << (8sizeof(UInt)-1)))
            key = ntuple(d -> d == N ? i : 0, N)
            slot = JA.map_reserve!(probe, key, hashed)
            @test slot.key == key
            @test JA.map_slot(probe, key, hashed) === slot
        end
    end
    keys = filter(i -> hash((i,)) & UInt(63) == 0, -100000:100000)[1:60]
    slots = [JA.map_reserve!(map, (i,), hash((i,))) for i in keys[1:48]]
    old = map.table
    @test length(old) == 64
    # Reserving an existing record must not rebuild the table or increment count.
    @test JA.map_reserve!(map, (keys[1],), hash((keys[1],))) === slots[1]
    @test map.table === old
    @test map.count == 48
    for i in keys[49:60]
        push!(slots, JA.map_reserve!(map, (i,), hash((i,))))
    end
    @test length((map.table)) == 256
    @test all(i -> mapslot(map, (keys[i],)) === slots[i], eachindex(keys))
    @test first(JA.hash_find(old, (keys[1],), hash((keys[1],)))) === slots[1]
    @test map.count == length(keys)
end

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

@testset "Nested generation preserves saved locations" begin
    # Nested generation can resize the dictionary before the outer value is stored.
    nested = MemoizedArray(i -> i == 0 ? sum(nested[k] for k in 1:60) : i, DictMap{Int}((0:60,)))
    @test nested[0] == sum(1:60)
    @test all(k -> isready(nested, k), 0:60)
    # Nested insertion can occupy a saved dictionary position without resizing.
    keys = filter(i -> hash((i,)) & UInt(63) == 0, 1:1000)[1:2]
    colliding = MemoizedArray(i -> i == keys[1] ? colliding[keys[2]] + 1 : i, DictMap{Int}((1:1000,)))
    @test colliding[keys[1]] == keys[2] + 1
    @test colliding[keys[2]] == keys[2]
    @test colliding.map.count == 2
    # Nested eviction can detach the branch saved for an outer tree miss.
    for map_spec in (:radix, :subtree)
        pruning = LRUMemoizedArray(i -> i == 40 ? pruning[512] + i : i, (test_map_config(map_spec, (8,)))(Int, (0:1024,)); capacity=2)
        pruning[32], pruning[36]
        @test pruning[40] == 552
        @test isready(pruning, 40) && isready(pruning, 512)
        @test !isready(pruning, 32) && !isready(pruning, 36)
        @test pruning.eviction.count == 2
    end
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
