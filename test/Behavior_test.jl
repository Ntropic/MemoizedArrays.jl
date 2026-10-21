const behavior_maps = (ArrayMap, ExpandingArrayMap, DictMap, RadixMap, SubtreeRadixMap)

@testset "Generation, storage and reads: $Map / $C / concurrent=$concurrent" for
    Map in behavior_maps, C in (MemoizedArray, LRUMemoizedArray), concurrent in (false, true)
    calls = Tuple{Int,Int}[]
    cache, reader = behavior_cache(C, Map, Tuple{Int,Int},
        (i, j) -> (push!(calls, (i, j)); (i, j)), (-2:2, -2:2); concurrent)
    @test isempty(calls) && !isready(cache, -1, 2)
    @test size(cache) == (5, 5) && axes(cache) == (-2:2, -2:2)
    @test reader[-1, 2] === reader[CartesianIndex(-1, 2)] === (-1, 2)
    @test calls == [(-1, 2)]
    @test generate!(reader, [(-1, 2), (2, -1), (2, -1)]) === reader
    @test calls == [(-1, 2), (2, -1)]
    @test reader[2, -1] === (2, -1) && isready(cache, 2, -1)
    @test read_allocations(reader, (2, -1)) == 0
    @test_throws BoundsError reader[-3, 0]
    @test_throws ArgumentError reader[1]
    @test copy(view(reader, -1:1, 0:1)) == [(i, j) for i in -1:1, j in 0:1]
    if C === MemoizedArray && Map === ArrayMap
        @test (@inferred getready(cache, 2, -1)) === (2, -1)
    elseif C === LRUMemoizedArray && concurrent
        @test_throws ArgumentError cache[0, 0]
    end
end

@testset "Eviction and retained values: $Map / concurrent=$concurrent" for
    Map in behavior_maps, concurrent in (false, true)
    calls = zeros(Int, 5)
    cache, reader = behavior_cache(LRUMemoizedArray, Map, Vector{Int},
        i -> (calls[i] += 1; [i]), (0:5,); concurrent)
    retained = reader[1]
    reader[2]; reader[1]; reader[3]
    @test isready(cache, 1) && isready(cache, 3) && !isready(cache, 2)
    @test reader[2] == [2] && calls[2] == 2
    @test !isready(cache, 1) && retained == [1]
    @test JA.resident_count(cache) == 2
    reader[0] = [99]
    generate!(reader, 1:5)
    @test reader[0] == [99] && JA.resident_count(cache) == 2
end

@testset "Values and failures: $Map / $C / concurrent=$concurrent" for
    Map in behavior_maps, C in (MemoizedArray, LRUMemoizedArray), concurrent in (false, true)
    matrix = ones(2, 2)
    values = (nothing, false, JA.EMPTY, JA.GENERATING, JA.RELOCATED,
        JA.HASH_EMPTY, JA.HASH_DELETED, JA.GenerationFailure(ErrorException("value")), matrix, view(matrix, :, 1))
    cache, reader = behavior_cache(C, Map, Any, i -> values[i], (0:length(values),);
        concurrent, capacity=length(values))
    @test all(reader[i] === values[i] && isready(cache, i) for i in eachindex(values))
    @test all(reader[i] === values[i] for i in reverse(eachindex(values)))
    for conversion in (false, true)
        calls = Ref(0)
        cache, reader = behavior_cache(C, Map, Int,
            i -> (calls[] += 1; i == 2 ? (conversion ? 0.5 : error("failed")) : i),
            (0:3,); concurrent)
        @test reader[1] == 1
        for _ in 1:2
            @test_throws (conversion ? InexactError : ErrorException) reader[2]
        end
        @test isready(cache, 1) && !isready(cache, 2)
        @test calls[] == (concurrent ? 2 : 3)
    end
end

@testset "Growth preserves coordinates and values" begin
    for C in (MemoizedArray, LRUMemoizedArray), concurrent in (false, true)
        cache, reader = behavior_cache(C, ExpandingArrayMap, Tuple{Int,Int}, tuple,
            (-2:-1, 4:5); concurrent, capacity=8)
        keys = ((-2, 5), (0, 4), (-1, 6), (4, 8))
        for key in keys
            @test reader[key...] === key
        end
        @test all(reader[key...] === key && isready(cache, key) for key in keys)
        @test axes(cache) == (-2:5, 4:9)
        @test !isready(cache, 100, 100)
        @test axes(cache) == (-2:5, 4:9)
    end
end

@testset "Scalar, empty and unbounded domains" begin
    for C in (MemoizedArray, LRUMemoizedArray), Map in behavior_maps, concurrent in (false, true)
        _, scalar = behavior_cache(C, Map, Int, () -> 7, (); concurrent)
        @test scalar[] == 7
    end
    for Map in (DictMap, RadixMap, SubtreeRadixMap), C in (MemoizedArray, LRUMemoizedArray), concurrent in (false, true)
        _, reader = behavior_cache(C, Map, Tuple{Int,Int}, tuple, unbound(2); concurrent)
        @test reader[typemin(Int), typemax(Int)] === (typemin(Int), typemax(Int))
    end
    empty = MemoizedArray(identity, ArrayMap{Int}((0,)))
    @test isempty(empty)
    @test_throws BoundsError empty[1]
    @test isempty(detect_ambiguities(MemoizedArrays))
end

@testset "Invalid map domains and growth increments" begin
    for domain in ((-1,), (1.5,), ("axis",), unbound(1))
        @test_throws ArgumentError MemoizedArray(identity, ArrayMap{Int}(domain))
    end
    for increment in (0, -1)
        @test_throws ArgumentError ExpandingArrayMap{Int}((3,); increment)
    end
    @test_throws UndefKeywordError ExpandingArrayMap{Int}((3,))
    @test_throws ArgumentError MemoizedArray(identity, ExpandingArrayMap{Int}(unbound(1); increment=2))
    @test_throws DimensionMismatch MemoizedArray(identity, RadixMap{Int}((-2:2,); block_shape=(4, 8)))
    @test_throws ArgumentError unbound(-1)
    cache = MemoizedArray(+, ExpandingArrayMap{Int}((0, 2); increment=2))
    @test isempty(cache) && cache[1, 2] == 3 && size(cache) == (2, 2)
end

@testset "Constructors and bounds" begin
    cache = MemoizedArray(ArrayMap{Int}((2, 3))) do i, j
        i + j
    end
    @test cache[2, 3] == 5
    @test_throws BoundsError MemoizedArray(identity, ArrayMap{Int}((0,)))[1]
    @test_throws ArgumentError MemoizedArray(identity, ArrayMap{Int}((-1,)))
    @test_throws ArgumentError LRUMemoizedArray(identity, ArrayMap{Int}((4,)); capacity=0)
    @test_throws ArgumentError LRUMemoizedArray(identity, ArrayMap{Int}((4,)); capacity=2, queues=2)
    for map_spec in (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> ExpandingArrayMap{V}(domain; increment=8)), ((V, domain) -> DictMap{V}(domain)),
        ((V, domain) -> RadixMap{V}(domain; block_shape=(8,))), ((V, domain) -> SubtreeRadixMap{V}(domain; block_shape=(8,)))), concurrent in (false, true)
        @test_throws ArgumentError LRUMemoizedArray(identity, map_spec(Int, (0:3,)); capacity=Int(typemax(UInt32)) + 1, concurrent, queues=1)
    end
    @test_throws UndefKeywordError LRUMemoizedArray(identity, ArrayMap{Int}((4,)))
    @test JA.as_tuple([2, 3], Val(2)) === (2, 3)
    @test_throws DimensionMismatch JA.as_tuple([2], Val(2))
    @test isempty(detect_ambiguities(MemoizedArrays))
end

@testset "Array interface" begin
    calls = Ref(0)
    cache = MemoizedArray((i, j) -> (calls[] += 1; i + 3(j - 1)), ArrayMap{Int}((3, 4)))
    reference = reshape(collect(1:12), 3, 4)
    selected = view(cache, 1:2, 2:3)
    @test calls[] == 0
    @test selected[1, 1] == 4
    @test calls[] == 1
    @test size(cache) == size(reference) && axes(cache) == axes(reference)
    @test cache[2, 3] == cache[CartesianIndex(2, 3)] == 8
    @test generate!(cache, ((2, 3), (2, 3))) === cache
    @test copy(selected) == reference[1:2, 2:3]
    for indices in ((:, :), (2, :), ([3, 1, 3], :), (1:0, :))
        @test cache[indices...] == reference[indices...]
    end
    @test collect(cache) == reference
    @test cache .+ 1 == reference .+ 1
    @test copyto!(zeros(Int, 3, 4), cache) == reference
    @test copyto!(PermutedDimsArray(zeros(Int, 4, 3), (2, 1)), cache) == reference
    @test get(cache, (4, 1), -1) == -1
    @test_throws BoundsError cache[4, 1]
    @test_throws BoundsError view(cache, :, 5)
    @test_throws Base.CanonicalIndexError (cache[1] = 0)
end

@testset "Ready-only getter" begin
    calls = Ref(0)
    cache = MemoizedArray((i, j, k) -> (calls[] += 1; [i, j, k]), ArrayMap{Vector{Int}}((2, 3, 4)))
    value = cache[2, 3, 4]
    @test (@inferred getready(cache, 2, 3, 4)) === value
    @test calls[] == 1
    @test_throws MethodError getready(cache, 2, 3, 4, 1)
    @test_throws MethodError getready(cache, 25)
    @test_throws BoundsError getready(cache, 3, 1, 1)
end

@testset "Batch generator coordinates" begin
    coordinates = [(2, 1, 4), (1, 3, 2), (2, 1, 4)]
    for T in (MemoizedArray, LRUMemoizedArray), requests in (coordinates, CartesianIndex.(coordinates))
        calls = NTuple{3,Int}[]
        options = T === MemoizedArray ? (;) : (; capacity=2)
        cache = T((key...) -> (push!(calls, key); key), ArrayMap{NTuple{3,Int}}((2, 3, 4)); options...)
        @test generate!(cache, requests) === cache
        @test calls == coordinates[1:2]
        @test cache[2, 1, 4] == (2, 1, 4) && cache[1, 3, 2] == (1, 3, 2)
    end
end

@testset "Subtree reads retain generated view parents" begin
    parents = WeakRef[]
    makeview = (i, j) -> begin
        matrix = [i j; j i]
        push!(parents, WeakRef(matrix))
        return view(matrix, 1:1, 1:1)
    end
    cache = MemoizedArray(makeview, SubtreeRadixMap{typeof(view(zeros(Int, 2, 2), 1:1, 1:1))}(unbound(2); block_shape=(4,8)))
    keys = ((0,0), (1,0), (128,-128))
    for key in keys
        @test cache[key...][1,1] == key[1]
    end
    GC.gc(true)
    for (i, key) in enumerate(keys)
        @test parent(cache[key...]) === parents[i].value !== nothing
    end
end

@testset "Public queue selection and capacity constraints" begin
    defaults = Threads.nthreads(:default)
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity=2defaults, concurrent=true)
    @test length(cache.eviction.queues) == defaults
    @test queue_handle(cache, 1)[1] == 1
    handles = queue_handles(cache)
    @test handles isa Tuple
    @test length(handles) == defaults
    @test all(i -> handles[i].queue === cache.eviction.queues[i], eachindex(handles))
    @test with_queue(cache) do caller
        caller[2]
    end == 2
    @test_throws ErrorException with_queue(cache) do _
        error("release this queue")
    end
    @test with_queue(cache) do caller
        caller[3]
    end == 3
    @test_throws ArgumentError cache[1]
    @test_throws BoundsError queue_handle(cache, 0)
    @test_throws BoundsError queue_handle(cache, defaults + 1)
    @test_throws BoundsError queue_handle(cache, 1)[17]
    for (capacity, queues) in ((0, 1), (4, 0), (4, -1), (3, 2), (1, 2))
        @test_throws ArgumentError LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity, queues, concurrent=true)
    end
    if defaults > 1
        @test_throws ArgumentError LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity=1, concurrent=true)
        @test length(LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity=1, concurrent=true, queues=1).eviction.queues) == 1
    end
    single = LRUMemoizedArray(identity, ArrayMap{Int}((8,)); capacity=2, concurrent=true, queues=1)
    @test_throws ArgumentError single[1]
    @test queue_handle(single, 1)[1] == 1
end
