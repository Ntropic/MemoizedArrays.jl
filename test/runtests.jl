# Shared array behavior and public cache constructors.
using Test, MemoizedArrays
const JA = MemoizedArrays

test_map_config(kind, shape=(8,); center=0.5) =
    kind === :array ? ((V, domain) -> ArrayMap{V}(domain)) : kind === :dict ? ((V, domain) -> DictMap{V}(domain)) :
    kind === :radix ? ((V, domain) -> RadixMap{V}(domain; block_shape=shape, center)) : ((V, domain) -> SubtreeRadixMap{V}(domain; block_shape=shape, center))

function readyallocations(cache)
    cache[ntuple(_ -> 1, ndims(cache))...]
    return @allocated cache[ntuple(_ -> 1, ndims(cache))...]
end

replacementallocation(cache, key) = @allocated cache[key]

@testset "Concurrency is a cache type parameter" begin
    for concurrent in (false, true)
        config = ArrayMap{Int}((4,))
        memoized = MemoizedArray(identity, config; concurrent)
        lru = LRUMemoizedArray(identity, config; capacity=2, queues=1, concurrent)
        @test typeof(memoized).parameters[end] === concurrent
        @test typeof(lru).parameters[end] === concurrent
        @test memoized[2] == 2
        @test (concurrent ? queue_handle(lru, 1) : lru)[2] == 2
        if concurrent
            @test_throws ArgumentError lru[2]
        else
            @test_throws MethodError queue_handle(lru, 1)
        end
    end
end

@testset "Map/cache integration" begin
    for map_spec in (:array, :dict, :radix, :subtree), T in (MemoizedArray, LRUMemoizedArray)
        calls = Ref(0)
        generator = i -> (calls[] += 1; i)
        domain = map_spec === :array ? (6,) : (-1:6,)
        configuration = test_map_config(map_spec)
        capacity = T === MemoizedArray ? (;) : (; capacity=2)
        cache = T(generator, configuration(Int, domain); capacity...)
        @test size(cache) == (map_spec === :array ? (6,) : (8,))
        @test !isready(cache, 1)
        @test (@inferred cache[1]) == cache[1] == 1
        @test readyallocations(cache) == 0
        @test calls[] == 1
        @test generate!(cache, [2, 3]) === cache
        @test calls[] == 3
        @test count(i -> isready(cache, i), 1:6) == (T === MemoizedArray ? 3 : 2)
        @test cache[1] == 1
        @test_throws BoundsError cache[7]
    end
end

@testset "Constructors and bounds" begin
    @test LRUMemoizedArray(identity, ArrayMap{Int}((4,)); capacity=2) isa LRUMemoizedArray
    @test LRUMemoizedArray(identity, ArrayMap{Int}([4]); capacity=2, queues=1) isa LRUMemoizedArray
    @test MemoizedArray(identity, ArrayMap{Int}((4,)))[2] == 2
    @test MemoizedArray(identity, ArrayMap{Int}([4]))[2] == 2
    cache = MemoizedArray(ArrayMap{Int}((2, 3))) do i, j
        i + j
    end
    @test cache[2, 3] == 5
    @test MemoizedArray(() -> 7, ArrayMap{Int}(()))[] == 7
    @test isempty(MemoizedArray(identity, ArrayMap{Int}((0,))))
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

@testset "Explicit ArrayMap" begin
    for T in (MemoizedArray, LRUMemoizedArray)
        options = T === MemoizedArray ? (;) : (; capacity=2)
        cache = T(+, ArrayMap{Int}((2, 3)); options...)
        @test cache.map isa JA.ArrayMap
        @test size(cache) == (2, 3)
        @test cache[2, 3] == 5
        scalar = T(() -> 7, ArrayMap{Int}(()); options...)
        @test scalar.map isa JA.ArrayMap
        @test scalar[] == 7
        if T === MemoizedArray
            @test cache.map.empty isa JA.ArrayEntry{Int}
        end
        for axis in ((Colon(),), (typemin(Int):typemax(Int),))
            @test_throws ArgumentError T(identity, ArrayMap{Int}(axis); options...)
        end
        @test_throws MethodError T(identity, ArrayMap{Int}((4,)); block_shape=(8,), options...)
    end
    @test_throws ArgumentError MemoizedArray(identity, ArrayMap{Int}(unbound(1)))
    @test_throws ArgumentError LRUMemoizedArray(identity, ArrayMap{Int}(unbound(1)); capacity=2)
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

@testset "Shaped ArrayMap" begin
    for T in (MemoizedArray, LRUMemoizedArray), shape in ((4,), (2, 3), (2, 3, 4))
        options = T === MemoizedArray ? (;) : (; capacity=1)
        cache = T((key...) -> LinearIndices(shape)[key...], ArrayMap{Int}(shape); options...)
        @test size(cache) == cache.map.shape == shape
        @test cache.map.reference.mem isa Memory
        @test cache.map.strides == Base.size_to_strides(1, shape...)
        @test JA.cache_key(cache, shape) === prod(shape)
        @test all(JA.slot_address(cache.map, Tuple(k)) == LinearIndices(shape)[k] for k in CartesianIndices(shape))
        @test cache[shape...] == prod(shape)
        @test cache[ntuple(_ -> 1, length(shape))...] == 1
        @test cache[shape...] == prod(shape)
        invalid = (shape[1] + 1, Base.tail(shape)...)
        @test_throws BoundsError cache[invalid...]
    end
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

@testset "ArrayMap identity sentinel" begin
    matrix = ones(4, 4)
    for value in (0, false, nothing, matrix, view(matrix, 1:2, 1:2))
        calls = Ref(0)
        cache = MemoizedArray((i, j) -> (calls[] += 1; value), ArrayMap{typeof(value)}((2, 3)))
        @test all(slot -> slot === cache.map.empty, cache.map.reference.mem)
        @test isconcretetype(eltype(cache.map.reference.mem))
        @test !isready(cache, 2, 3)
        @test cache[2, 3] === getready(cache, 2, 3) === value
        @test calls[] == 1 && isready(cache, 2, 3) && !isready(cache, 1, 1)
        @test readyallocations(cache) == 0
    end
    failing = MemoizedArray(_ -> 0.5, ArrayMap{Int}((1,)))
    @test_throws InexactError failing[1]
    @test !isready(failing, 1)
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

@testset "Values and direct generation" begin
    for T in (MemoizedArray, LRUMemoizedArray)
        options = T === MemoizedArray ? (;) : (; capacity=1)
        value = ones(2, 2)
        cache = T(_ -> value, ArrayMap{Matrix{Float64}}((2,)); options...)
        @test (@inferred cache[1]) === value
        readyallocations(cache)
        @test readyallocations(cache) == 0
        retained = cache[1]
        @test cache[2] === value
        @test retained === value
        broad = T(i -> i == 1 ? nothing : false, ArrayMap{Any}((2,)); options...)
        @test broad[1] === nothing && isready(broad, 1)
        @test broad[2] === false
        failing = T(_ -> error("generator failed"), ArrayMap{Int}((2,)); options...)
        @test_throws ErrorException failing[1]
        converted = T(_ -> 0.5, ArrayMap{Int}((2,)); options...)
        @test_throws InexactError converted[1]
        dependent = T(i -> i == 1 ? dependent[2] + 10 : i, ArrayMap{Int}((2,)); options...)
        @test dependent[1] == 12
        @test dependent[1] == 12
    end
    # Reuse the cache entry across map families without changing returned value objects.
    for map_spec in (:array, :dict, :radix, :subtree)
        domain = map_spec === :array ? (1024,) : (0:1024,)
        configuration = test_map_config(map_spec)
        cache = LRUMemoizedArray(i -> [i], configuration(Vector{Int}, domain); capacity=1)
        retained = cache[1]
        stored = JA.entry(cache, JA.cache_key(cache, (1,)))
        @test cache[512] == [512]
        @test JA.entry(cache, JA.cache_key(cache, (512,))) === stored
        @test retained == [1] && !isready(cache, 1)
    end
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((4,)); capacity=1)
    cache[1]
    replacementallocation(cache, 2)
    @test replacementallocation(cache, 3) == 0
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

include("SparseIndexing_test.jl")
include("Maps_test.jl")

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
        replacementallocation(warmup, key)
    end
    cache = make()
    entries = copy(cache.eviction.entries)
    @test cache.eviction.count == 0 && !isready(cache, 1)
    for key in 1:3
        @test replacementallocation(cache, key) == 0
        @test any(entry -> JA.entry(cache, key) === entry, entries)
    end
    @test replacementallocation(cache, 4) == 0
    @test replacementallocation(cache, 4) == 0
    @test cache.eviction.count == 3 && !isready(cache, 1)
end

include("ConstructorDomains_test.jl")
include("ExpandingArrayMap_test.jl")
include("ConcurrentExpandingArrayMap_test.jl")
include("Concurrency_test.jl")
include("ConcurrentTreeMap_test.jl")
include("ConcurrentTreeLRU_test.jl")
include("MultiQueue_test.jl")
include("ConcurrentExpandingLRU_test.jl")
include("DictCaches_test.jl")

include("PinnedEntries_test.jl")
