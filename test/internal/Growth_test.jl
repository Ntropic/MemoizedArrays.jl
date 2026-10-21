@testset "Subtree insertion across missing shortcuts" begin
    for center in (0.5, -0.5)
        cache = MemoizedArray(identity, SubtreeRadixMap{Int}(unbound(1); block_shape=(8,), center))
        function checkshortcuts()
            map = cache.map
            for key in (0, 8, 64, 72, 512, -512, 4096, 8192, typemin(Int), typemax(Int))
                encoded = JA.map_routing(map, (key,))
                direct = JA.tree_find(map, encoded, JA.root_link(map), length(map.zero_path)-1, Val(false))
                shortcut = mapslot(map, (key,))
                @test (direct === nothing && shortcut === nothing) ||
                    (direct !== nothing && shortcut !== nothing && direct.leaf === shortcut.leaf)
            end
        end
        # Start off the zero path, split compressed edges, then fill missing lower levels.
        keys = (64, 72, 4096, 0, 8, 512, -512)
        for key in keys
            @test !isready(cache, key)
            @test cache[key] == key
            treecounts(cache.map)
            checkshortcuts()
        end
        for key in (0, 8, 4096)
            JA.delete_key!(cache.map, (key,))
            checkshortcuts()
        end
        for key in (1, 9, 8192)
            @test !isready(cache, key)
            @test cache[key] == key
            treecounts(cache.map)
            checkshortcuts()
        end
        @test all(key -> isready(cache, key) && cache[key] == key, (64, 72, 512, -512, 1, 9, 8192))
        @test all(key -> !isready(cache, key), (0, 8, 4096))
    end
end

@testset "Per-axis expansion increments" begin
    for increments in ((2, 4), [2, 4]), Cache in (MemoizedArray, LRUMemoizedArray), concurrent in (false, true)
        options = Cache === LRUMemoizedArray ? (; capacity=8, queues=1) : (;)
        cache = Cache(tuple, ExpandingArrayMap{Tuple{Int,Int}}((-2:0, 4:6); increment=increments); concurrent, options...)
        reader = concurrent && Cache === LRUMemoizedArray ? queue_handle(cache, 1) : cache
        @test reader[-2, 4] == (-2, 4)
        @test reader[0, 8] == (0, 8)
        @test size(cache) == (3, 7)
        @test reader[2, 8] == (2, 8)
        @test size(cache) == (5, 7)
        @test reader[7, 14] == (7, 14)
        @test size(cache) == (11, 11)
        @test reader[-2, 4] == (-2, 4)
        @test isready(cache, 0, 8)
        @test !isready(cache, 1, 7)
        @test_throws BoundsError reader[-3, 4]
    end
    for concurrent in (false, true), increments in ((2,), [2, 4, 6])
        @test_throws DimensionMismatch MemoizedArray(+, ExpandingArrayMap{Int}((3, 3); increment=increments); concurrent)
    end
    for increments in ((2, 0), [2, -1])
        @test_throws ArgumentError ExpandingArrayMap{Int}((3, 3); increment=increments)
    end
    increments = [2, 4]
    config = ExpandingArrayMap{Int}((3, 3); increment=increments)
    increments[1] = 0
    @test config.increment == (2, 4)
end

@testset "Contiguous expansion" begin
    empty = Ref(-1)
    map = JA.ExpandingArrayMap(typeof(empty), (2, 3), (1, 1), empty, 8)
    values = [Ref(i) for i in 1:6]
    map.reference.mem .= values
    old = map.reference.mem
    JA.expand_to!(map, (4, 5))
    for coordinate in CartesianIndices((2, 3))
        @test map.reference.mem[LinearIndices(map.shape)[coordinate]] ===
            values[LinearIndices((2, 3))[coordinate]]
    end
    @test count(value -> value !== empty, map.reference.mem) == 6
    @test map.reference.mem !== old
    current = map.reference.mem
    @test JA.expand_to!(map, map.shape) === map
    @test map.reference.mem === current
    @test_throws ArgumentError JA.expand_to!(map, (1, 5))
    @test_throws OverflowError JA.expand_to!(map, (typemax(Int), 5))
    @test map.reference.mem === current
    @test map.shape == (4, 5)

    zero = JA.ExpandingArrayMap(Int, (0, 2), (1, 1), -1, 8)
    JA.expand_to!(zero, (3, 2))
    @test all(==(-1), zero.reference.mem)

    # Growing any axis must preserve coordinates, including the bulk-copy path
    # when only the last axis grows.
    for axis in 1:3
        shape = (2,3,2)
        map = JA.ExpandingArrayMap(Int, shape, (1,1,1), -1, 8)
        map.reference.mem .= 1:12
        expanded = ntuple(i -> shape[i] + (i == axis ? 8 : 0), 3)
        JA.expand_to!(map, expanded)
        for coordinate in CartesianIndices(shape)
            @test map.reference.mem[LinearIndices(expanded)[coordinate]] == LinearIndices(shape)[coordinate]
        end
        @test count(!=(-1), map.reference.mem) == 12
    end
end

@testset "Radix read path and generation" begin
    for map_spec in (:radix, :subtree), seed in (false, true)
        calls = Dict{Tuple{Int,Int},Int}()
        cache = MemoizedArray(test_map_config(map_spec, (4, 8))(Int, (-32:1023, -32:1023))) do i, j
            key = (i,j)
            slot = mapslot(cache.map, key)
            @test slot !== nothing && JA.slot_entry(slot) === cache.map.empty
            calls[key] = get(calls, key, 0) + 1
            if key == (7,7)
                error("generation failed")
            elseif key == (0,0)
                # Generation can split branches before the outer entry is inserted.
                for nested in ((1,0), (128,128), (900,900))
                    cache[nested...]
                end
            end
            return i + 2000j
        end
        if seed
            @test cache[0,1] == 2000
        end
        @test cache[0,0] == cache[0,0] == 0
        @test all(key -> isready(cache, key...), ((0,0), (1,0), (128,128), (900,900)))
        @test all(==(1), values(calls))
        treecounts(cache.map)
        @test_throws ErrorException cache[7,7]
    end
    scalar = MemoizedArray(() -> 17, RadixMap{Int}((); block_shape=()))
    @test scalar[] == scalar[] == 17
end

@testset "Expanding LRU nested generation and conversion" begin
    for capacity in (1,2), conversion in (false,true)
        (; generator, owner) = nested_generator(Returns(63), (1, 1), cache -> (cache[6, 3]; 11); conversion)
        cache = LRUMemoizedArray(generator, ExpandingArrayMap{Int}((2,2); increment=2); capacity)
        owner[] = cache
        for i in 1:capacity
            cache[2,i]
        end
        @test cache[1,1] == 11
        @test size(cache) == (6,4)
        @test isready(cache,1,1) && isready(cache,6,3) == (capacity == 2)
        @test !isready(cache,2,1) && !isready(cache,2,2)
        @test cache.eviction.count == capacity
        @test count(entry -> entry !== cache.map.empty, cache.map.reference.mem) == capacity
    end
end

@testset "Expanding LRU failures leave residents and order intact" begin
    for conversion in (false,true)
        generator = i -> i == 3 ? (conversion ? 0.5 : error("generation failed")) : i
        cache = LRUMemoizedArray(generator, ExpandingArrayMap{Int}((2,); increment=2); capacity=2)
        cache[1], cache[2]
        @test_throws (conversion ? InexactError : ErrorException) cache[3]
        @test size(cache) == (2,) && cache.eviction.count == 2
        @test isready(cache,1) && isready(cache,2) && !isready(cache,3)
        @test cache[4] == 4
        @test !isready(cache,1) && isready(cache,2) && isready(cache,4)
    end
    lower = typemax(Int)-2
    cache = LRUMemoizedArray(identity, ExpandingArrayMap{Int}((lower:lower+1,); increment=8); capacity=1)
    @test cache[lower] == lower
    entry = JA.entry(cache,(1,))
    @test_throws OverflowError cache[typemax(Int)]
    @test size(cache) == (2,) && cache.eviction.count == 1
    @test isready(cache,lower) && JA.entry(cache,(1,)) === entry
end

@testset "Expanding LRU matrix requests allocate only on growth" begin
    value = ones(4,4)
    make() = LRUMemoizedArray(Returns(value), ExpandingArrayMap{typeof(value)}((4,2); increment=2); capacity=2)
    warmup = make()
    keys = ((1,2), (2,1), (3,2), (3,2))
    for key in keys
        request_allocations(warmup,key)
    end
    cache = make()
    entries = copy(cache.eviction.entries)
    for key in keys
        @test request_allocations(cache,key) == 0
        @test any(entry -> JA.entry(cache,key) === entry, entries)
    end
    @test cache.eviction.count == 2 && !isready(cache,1,2)
end

@testset "Expansion during generation and failures" begin
    # Both an initially present empty slot and an out-of-range request can be
    # invalidated by a nested generator expanding an earlier dimension.
    for requested in ((1,1),(2,2))
        owner = Ref{Any}()
        calls = Ref(0)
        cache = MemoizedArray((i,j) -> begin
            calls[] += 1
            if (i,j) == requested
                owner[][6,3]
            end
            (i,j)
        end, ExpandingArrayMap{Tuple{Int,Int}}((1,1); increment=4))
        owner[] = cache
        @test cache[requested...] === requested
        @test size(cache) == (9,5)
        @test cache[requested...] === requested
        @test cache[6,3] === (6,3)
        @test calls[] == 2
    end

    map_spec = ((V, domain) -> ExpandingArrayMap{V}(domain; increment=8))
    failure = ErrorException("generator failed")
    throwing = MemoizedArray(i -> i == 1 ? 1 : throw(failure), map_spec(Int, (2,)))
    @test throwing[1] == 1
    @test_throws ErrorException throwing[3]
    @test size(throwing) == (2,)
    @test throwing[1] == 1
    converting = MemoizedArray(i -> i == 1 ? 1 : "bad", map_spec(Int, (2,)))
    @test converting[1] == 1
    @test_throws MethodError converting[3]
    @test size(converting) == (2,)
    @test converting[1] == 1

    limit = MemoizedArray(identity, map_spec(Int, (typemax(Int)-2:typemax(Int)-1,)))
    @test_throws OverflowError limit[typemax(Int)]
    @test size(limit) == (2,)
    @test limit[typemax(Int)-1] == typemax(Int)-1
    wide = MemoizedArray(identity, map_spec(Int, (typemin(Int):typemin(Int)+1,)))
    @test_throws OverflowError wide[0]
    @test size(wide) == (2,)

    # Conversion, as well as the generator itself, can invalidate a saved slot.
    owner = Ref{Any}()
    converting_nested = MemoizedArray((i,j) -> (i,j) == (1,1) ?
        TestConversion(() -> (owner[][6, 3]; 11)) : 63, ExpandingArrayMap{Int}((1,1); increment=4))
    owner[] = converting_nested
    @test converting_nested[1,1] == 11
    @test size(converting_nested) == (9,5)
    @test converting_nested[1,1] == 11 && converting_nested[6,3] == 63

    for lower in (typemin(Int), -2, 0, typemax(Int)-2), index in
        (typemin(Int), -1, 0, 1, typemax(Int)-1, typemax(Int))
        cache = MemoizedArray(identity, map_spec(Int, (lower:lower,)))
        extent = big(index) - lower + 1
        if index < lower
            @test_throws BoundsError JA.cache_key(cache, (index,))
        elseif extent > typemax(Int)
            @test_throws OverflowError JA.cache_key(cache, (index,))
        else
            @test JA.cache_key(cache, (index,)) == (Int(extent),)
        end
    end
end
