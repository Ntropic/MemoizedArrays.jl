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

struct ExpandingConversion
    cache
    value::Int
end
function Base.convert(::Type{Int}, source::ExpandingConversion)
    source.cache[6,3]
    return source.value
end

@testset "Expanding MemoizedArray" begin
    map_spec = ((V, domain) -> ExpandingArrayMap{V}(domain; increment=8))
    calls = Ref(0)
    cache = MemoizedArray((i,j) -> (calls[] += 1; (i,j)), map_spec(Tuple{Int,Int}, (2,3)))
    for coordinate in CartesianIndices((2,3))
        @test cache[coordinate] == Tuple(coordinate)
    end
    @test (@inferred cache[3,2]) === (3,2)
    @test size(cache) == (10,3)
    @test (@inbounds cache[22,12]) === (22,12)
    @test size(cache) == (26,19)
    for coordinate in CartesianIndices((2,3))
        @test cache[coordinate] == Tuple(coordinate)
    end
    @test calls[] == 8
    @test !isready(cache,100,100)
    @test size(cache) == (26,19)
    @test_throws BoundsError cache[0,1]
    @test_throws ArgumentError cache[1]

    one = MemoizedArray(identity, map_spec(Int, [10]))
    @test one[11] == 11
    @test size(one) == (18,)
    @test one[35] == 35
    @test size(one) == (42,)

    offset = MemoizedArray(tuple, map_spec(Tuple{Int,Int}, (-2:-1,4:5)))
    selected = view(offset, -2:-1, 4:5)
    @test offset[-2,5] === (-2,5)
    @test offset[0,6] === (0,6)
    @test axes(offset) == (-2:7,4:13)
    @test copy(selected) == [(i,j) for i in -2:-1,j in 4:5]
    @test_throws BoundsError offset[-3,4]
    @test_throws BoundsError offset[-2,3]

    zero = MemoizedArray(+, map_spec(Int, (0,2)))
    @test isempty(zero)
    @test zero[1,2] == 3
    @test size(zero) == (8,2)
    scalar = MemoizedArray(() -> 7, map_spec(Int, ()))
    @test scalar[] == 7
    @test size(scalar) == ()

    value = ones(4,4)
    matrix = MemoizedArray(Returns(value), map_spec(typeof(value), (1,)))
    retained = matrix[1]
    @test matrix[2] === retained === matrix[1]
    @test readyallocations(matrix) == 0

    @test_throws UndefKeywordError ExpandingArrayMap{Int}((3,))
    @test_throws ArgumentError ExpandingArrayMap{Int}((3,); increment=0)
    @test_throws ArgumentError ExpandingArrayMap{Int}((3,); increment=-1)
    @test_throws ArgumentError MemoizedArray(identity, map_spec(Int, unbound(1)))
    @test_throws ArgumentError MemoizedArray(identity, map_spec(Int, (-1,)))
    @test LRUMemoizedArray(identity, map_spec(Int, (2,)); capacity=1)[1] == 1
end

@testset "Expanding serial LRU preserves coordinate order" begin
    trace = ((1,2), (2,1), (1,2), (3,2), (1,4), (3,2), (4,4))
    for capacity in (1,2)
        cache = LRUMemoizedArray(tuple, ExpandingArrayMap{Tuple{Int,Int}}((2,2); increment=2); capacity)
        order = Tuple{Int,Int}[]
        for key in trace
            filter!(!=(key), order)
            push!(order, key)
            if length(order) > capacity
                popfirst!(order)
            end
            @test cache[key...] === key
            @test cache.eviction.count == length(order)
            @test all(candidate -> isready(cache, candidate...) == (candidate in order), trace)
            @test count(entry -> entry !== cache.map.empty, cache.map.reference.mem) == length(order)
        end
        @test size(cache) == (4,4)
    end

    offset = LRUMemoizedArray(tuple, ExpandingArrayMap{Tuple{Int,Int}}((-2:-1,4:5); increment=2); capacity=1)
    @test offset[-2,5] === (-2,5)
    @test offset[0,4] === (0,4)
    @test !isready(offset,-2,5)
    @test offset[-1,6] === (-1,6)
    @test !isready(offset,0,4) && isready(offset,-1,6)
    @test axes(offset) == (-2:1,4:7)
    zero = LRUMemoizedArray(+, ExpandingArrayMap{Int}((0,2); increment=2); capacity=1)
    @test zero[1,2] == 3 && size(zero) == (2,2)
    @test zero[2,1] == 3 && !isready(zero,1,2)
    scalar = LRUMemoizedArray(() -> 7, ExpandingArrayMap{Int}((); increment=2); capacity=1)
    @test scalar[] == 7 && scalar.eviction.count == 1
end

@testset "Expanding LRU reuses entries without changing returned values" begin
    values = (fill(1,2,2), fill(2,2,2), fill(3,2,2))
    cache = LRUMemoizedArray((i,j) -> values[i], ExpandingArrayMap{Matrix{Int}}((1,2); increment=2); capacity=1)
    first = cache[1,2]
    entry = JA.entry(cache, (1,2))
    second = cache[2,1]
    @test JA.entry(cache,(2,1)) === entry
    @test !isready(cache,1,2)
    @test cache[3,4] === values[3]
    @test JA.entry(cache,(3,4)) === entry
    @test !isready(cache,2,1)
    @test first === values[1] && all(==(1),first)
    @test second === values[2] && all(==(2),second)
end

@testset "Expanding LRU nested generation and conversion" begin
    for capacity in (1,2), conversion in (false,true)
        owner = Ref{Any}()
        generator = (i,j) -> begin
            if (i,j) == (1,1)
                if conversion
                    return ExpandingConversion(owner[],11)
                end
                owner[][6,3]
                return 11
            end
            return 63
        end
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

expanding_lru_allocations(cache,key) = @allocated cache[key...]

@testset "Expanding LRU matrix requests allocate only on growth" begin
    value = ones(4,4)
    make() = LRUMemoizedArray(Returns(value), ExpandingArrayMap{typeof(value)}((4,2); increment=2); capacity=2)
    warmup = make()
    keys = ((1,2), (2,1), (3,2), (3,2))
    for key in keys
        expanding_lru_allocations(warmup,key)
    end
    cache = make()
    entries = copy(cache.eviction.entries)
    for key in keys
        @test expanding_lru_allocations(cache,key) == 0
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
        ExpandingConversion(owner[], 11) : 63, ExpandingArrayMap{Int}((1,1); increment=4))
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
