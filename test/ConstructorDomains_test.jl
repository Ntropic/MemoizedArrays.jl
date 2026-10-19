@testset "Map value types and construction" begin
    domain = (-2:2, -3:3)
    configs = (ArrayMap{Vector{Int}}(domain), ExpandingArrayMap{Vector{Int}}(domain; increment=[2, 4]),
        DictMap{Vector{Int}}(domain), RadixMap{Vector{Int}}(domain; block_shape=(4, 4)),
        SubtreeRadixMap{Vector{Int}}(domain; block_shape=(4, 4)))
    for config in configs, C in (MemoizedArray, LRUMemoizedArray), concurrent in (false, true)
        calls = Tuple{Int,Int}[]
        generator = (i, j) -> (push!(calls, (i, j)); [length(calls)])
        options = C === LRUMemoizedArray ? (; capacity=4, queues=1) : (;)
        cache = C(generator, config; concurrent, options...)
        @test eltype(cache) === Vector{Int}
        @test isempty(calls)
        @test !isready(cache, -2, -3)
        reader = concurrent && C === LRUMemoizedArray ? queue_handle(cache, 1) : cache
        @test reader[-2, -3] == [1]
        @test calls == [(-2, -3)]
    end
    for config in (DictMap{Int}((-4:-2, :)), RadixMap{Int}((-4:-2, :); block_shape=(4, 4)))
        calls = Tuple{Int,Int}[]
        cache = MemoizedArray((i, j) -> (push!(calls, (i, j)); i+j), config)
        @test isempty(calls)
        @test eltype(cache) === Int
        @test !isready(cache, -4, 0)
    end
    for C in (MemoizedArray, LRUMemoizedArray)
        options = C === LRUMemoizedArray ? (; capacity=2) : (;)
        count = Ref(0)
        generator = i -> (count[] += 1; Float64(i))
        config = ArrayMap{Float64}((3,))
        first = C(generator, config; options...)
        second = C(generator, config; options...)
        @test count[] == 0
        @test first.map !== second.map
        @test first[1] == 1.0
        @test !isready(second, 1)
        @test_throws MethodError ArrayMap((0,))
        @test count[] == 1
        @test isempty(C(generator, ArrayMap{Float64}((0,)); options...))
        @test_throws MethodError C{Int}(identity, (3,); options...)
    end
    scalar_calls = Ref(0)
    scalar = MemoizedArray(() -> (scalar_calls[] += 1), ArrayMap{Int}(()))
    @test scalar_calls[] == 0 && !isready(scalar)
    @test scalar[] == 1
    recursive = MemoizedArray(ArrayMap{Int}((8,))) do i
        recursive[i - 1] + 1
    end
    recursive[1] = 1
    @test recursive[8] == 8
    domain_vector = [3, 4]
    config = ArrayMap{Int}(domain_vector)
    domain_vector[1] = 9
    @test config.domain == (3, 4)
end

@testset "Shared domain and map construction" begin
    for C in (MemoizedArray, LRUMemoizedArray)
        options = C === MemoizedArray ? (;) : (; capacity=2)
        explicit = C(identity, ArrayMap{Int}((4,)); options...)
        @test explicit.map isa JA.ArrayMap
        @test_throws ArgumentError C(identity, ArrayMap{Int}(unbound(1)); options...)
    end
    @test_throws UndefKeywordError RadixMap{Int}((-2:2,))
    @test_throws UndefKeywordError SubtreeRadixMap{Int}((-2:2,))
    for S in (RadixMap, SubtreeRadixMap)
        @test_throws DimensionMismatch MemoizedArray(identity, S{Int}((-2:2,); block_shape=(4, 8)))
    end
    shape = [4, 8]
    configured = SubtreeRadixMap{Tuple{Int,Int}}(unbound(2); block_shape=shape, center=-0.5)
    shape[1] = 16
    @test configured.block_shape === (4, 8)
    for (C, options) in ((MemoizedArray, (;)), (LRUMemoizedArray, (; capacity=2)))
        cache = C(tuple, configured; options...)
        @test JA.block_shape(cache.map) === (4, 8)
        @test cache[-1, 0] == (-1, 0)
    end
    @test unbound(3) === (:, :, :)
    @test unbound(0) === ()
    @test_throws ArgumentError unbound(-1)
    @test isunbound(:)
    @test !isunbound(3) && !isunbound(-2:2)
    for (C, options) in ((MemoizedArray, (;)), (LRUMemoizedArray, (; capacity=2)))
        @test_throws MethodError C(identity; options...)
        for domain in ((3, 4), (1:3, 1:4), [3, 4], [1:3, 1:4], (-1:1, 4), Any[-1:1, 4])
            cache = C(tuple, ArrayMap{Tuple{Int,Int}}(domain); options...)
            @test cache.map isa JA.ArrayMap
            @test cache[1, 2] == (1, 2)
        end
        for map_spec in (:array, :dict, :radix, :subtree)
            configuration = test_map_config(map_spec, (4, 8))
            if map_spec === :subtree
                @test_throws ArgumentError C(tuple, configuration(Tuple{Int,Int}, (-2:2, 4:7)); options...)
                @test_throws ArgumentError C(+, configuration(Int, (3, 4)); options...)
                continue
            end
            cache = C(tuple, configuration(Tuple{Int,Int}, (-2:2, 4:7)); options...)
            @test cache.map isa getproperty(JA, Dict(:array=>:ArrayMap, :dict=>(C === MemoizedArray ? :DictMap : :LRUDictMap),
                :radix=>:RadixMap, :subtree=>:SubtreeRadixMap)[map_spec])
            @test cache[-2, 4] == (-2, 4)
            @test cache[2, 7] == (2, 7)
            @test_throws ArgumentError cache[20]
            @test cache[0, 5] == (0, 5)
            @test cache[-2, 4] == (-2, 4)
            @test copy(view(cache, -1:1, 5:6)) == [(i, j) for i in -1:1, j in 5:6]
            @test_throws BoundsError cache[-3, 4]
            lengths = C(+, configuration(Int, (3, 4)); options...)
            @test lengths[2, 3] == 5
            if C !== MemoizedArray
                @test cache.eviction.count == 2
            end
        end
        one = C(identity, ArrayMap{Int}((-2:2,)); options...)
        @test one[-2] == -2 && one[2] == 2
        @test isready(one, -2)
        @test_throws BoundsError one[3]
        if C === MemoizedArray
            @test getready(one, -2) == -2
        end
        for domain in (unbound(2), [:, :], (:, -2:2), Any[:, -2:2])
            @test_throws ArgumentError C(tuple, ArrayMap{Tuple{Int,Int}}(domain); options...)
            cache = C(tuple, DictMap{Tuple{Int,Int}}(domain); options...)
            @test cache.map isa (C === MemoizedArray ? JA.DictMap : JA.LRUDictMap)
            @test cache[typemin(Int), 0] == (typemin(Int), 0)
            @test cache[typemax(Int), 1] == (typemax(Int), 1)
            @test_throws ArgumentError C(+, ArrayMap{Int}(domain); options...)
        end
        for map_spec in (:radix, :subtree)
            cache = C(tuple, (test_map_config(map_spec, (4, 8)))(Tuple{Int,Int}, unbound(2)); options...)
            @test cache[typemin(Int), typemax(Int)] == (typemin(Int), typemax(Int))
        end
        for domain in ((-1, 2), (1.5,), ("axis",))
            @test_throws ArgumentError C(identity, ArrayMap{Int}(domain); options...)
        end
        @test_throws MethodError C(identity, :unknown; options...)
    end
end

function tree_read_allocations(cache, key)
    cache[key...]
    return @allocated cache[key...]
end

@testset "Cache-side tree offsets" begin
    for (C, options) in ((MemoizedArray, (;)), (LRUMemoizedArray, (; capacity=2))),
        map_spec in (:radix, :subtree)
        plain = C(tuple, (test_map_config(map_spec, (4, 8)))(NTuple{2,Int}, (0:100, 0:100)); options...)
        shifted = C(tuple, (test_map_config(map_spec, (4, 8)))(NTuple{2,Int}, (-19:100, -11:100)); options...)
        @test JA.cache_offset(plain) === Val(JA.tree_offset(plain.map))
        @test JA.cache_offset(shifted) === JA.cache_offset(plain)
        @test typeof(plain.map) === typeof(shifted.map)
        @test typeof(plain) === typeof(shifted)
        @test !hasfield(typeof(plain), :offset)
        for domain in ((0:100, 0:100), (-19:100, -11:100), (0:100, -11:100),
                       (typemin(Int):typemax(Int), typemin(Int):typemax(Int)))
            cache = C(tuple, (test_map_config(map_spec, (4, 8)))(NTuple{2,Int}, domain); options...)
            for key in ((0,0), (first(domain[1]),first(domain[2])), (100,100), (0,0))
                encoded = ntuple(i -> UInt(mod(big(key[i]) + JA.tree_offset(cache.map)[i], big(1) << (8sizeof(UInt)))), 2)
                @test (@inferred JA.encode_coordinates(key, JA.cache_offset(cache))) === encoded
                route = @inferred JA.cache_routing(cache, key)
                if map_spec === :subtree
                    @test route === (encoded, JA.root_index(cache.map, encoded))
                else
                    @test route === encoded
                end
                @test cache[key...] === key
                @test isready(cache, key...)
                @test tree_read_allocations(cache, key) == 0
            end
            @test JA.entry(cache, (0,0)).value === (0,0)
            @test (@inferred cache[0,0]) === (0,0)
        end
    end
end

@testset "Centered tree grids" begin
    for center in (0.5, -0.5), map_spec in (:radix, :subtree),
        (C, options) in ((MemoizedArray, (;)), (LRUMemoizedArray, (; capacity=2)))
        cache = C(tuple, (test_map_config(map_spec, (8, 4); center))(Tuple{Int,Int}, unbound(2)); options...)
        lo = center == 0.5 ? (-3,-1) : (-4,-2)
        hi = (lo[1]+7, lo[2]+3)
        prefix(key) = JA.tree_prefix(cache.map, JA.map_routing(cache.map, key), 0)
        @test prefix(lo) == prefix(hi) == prefix((0,0))
        @test prefix((lo[1]-1,0)) != prefix((0,0))
        @test prefix((hi[1]+1,0)) != prefix((0,0))
        for key in (lo, hi, (0,0), (typemin(Int),typemax(Int)), (typemax(Int),typemin(Int)), lo, hi)
            @test (@inferred cache[key...]) == key
            @test isready(cache, key)
            @test JA.cache_routing(cache, key) == (map_spec === :subtree ?
                (JA.map_routing(cache.map, key), JA.root_index(cache.map, JA.map_routing(cache.map, key))) :
                JA.map_routing(cache.map, key))
        end
        @test tree_read_allocations(cache, hi) == 0
        @test_throws ArgumentError C(identity, (test_map_config(map_spec, (8,); center=0))(Int, unbound(1)); options...)
        @test_throws MethodError C(identity, ArrayMap{Int}((8,)); center, options...)
    end
    # Check geometry independently: each ancestor is the closest aligned interval
    # to the requested center, among the possible parents of the central child.
    for width in (2,4,8,16,32), center in (0.5,-0.5)
        cache = MemoizedArray(identity, SubtreeRadixMap{Int}(unbound(1); block_shape=(width,), center))
        bias = big(JA.tree_offset(cache.map)[1])
        lower, span = big(0), big(1)
        for _ in 1:div(8sizeof(UInt)-1,trailing_zeros(width))
            parent_span = span*width
            actual = -mod(bias,parent_span)
            candidates = [lower-i*span for i in 0:width-1]
            distance(start) = abs(2start+parent_span-1-Int(2center))
            @test actual in candidates
            @test distance(actual) == minimum(distance.(candidates))
            lower, span = actual, parent_span
        end
    end
end
@testset "Single constructor form" begin
    for C in (MemoizedArray, LRUMemoizedArray)
        options = C === MemoizedArray ? (;) : (; capacity=2)
        @test C(identity, ArrayMap{Int}((4,)); options...)[2] == 2
        @test C(identity, ArrayMap{Int}([4]); options...)[2] == 2
        @test_throws MethodError C(identity, ArrayMap{Int}(4); options...)
        @test_throws MethodError C(identity, (2, 3); options...)
        @test_throws MethodError C(identity, DictMap{Int}(4); options...)
        if C === MemoizedArray
            @test_throws MethodError C((4,), ArrayMap{Int}(identity))
        else
            @test_throws MethodError C((4,), identity; capacity=2)
        end
    end
    @test_throws MethodError MemoizedArray(identity, DictMap{Int}(unbound(1)); ndims=1)
    @test_throws MethodError LRUMemoizedArray(identity, DictMap{Int}(unbound(1)); ndims=1, capacity=2)
end
@testset "Coordinate-only indexing and views" begin
    for C in (MemoizedArray, LRUMemoizedArray), map_spec in
        (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain)), ((V, domain) -> RadixMap{V}(domain; block_shape=(4,4))), ((V, domain) -> SubtreeRadixMap{V}(domain; block_shape=(4,4))))
        options = C === MemoizedArray ? (;) : (; capacity=4)
        cache = C((i,j) -> 10i+j, map_spec(Int, (-2:2,-2:2)); options...)
        @test cache[-1,2] == cache[CartesianIndex(-1,2)] == -8
        @test_throws ArgumentError cache[1]
        @test_throws ArgumentError cache[:]
        @test_throws ArgumentError cache[1,1,1]
        @test_throws ArgumentError view(cache, :)
        @test_throws ArgumentError generate!(cache, [1,2])
        @test_throws MethodError isready(cache, 1)
        selected = view(cache, -1:1, 0:1)
        @test copy(selected) == [10i+j for i in -1:1,j in 0:1]
        @test copy(view(selected, 1:2, :)) == [10i+j for i in -1:0,j in 0:1]
        @test cache[[1,-1], [2,0]] == [12 10; -8 -10]
    end
    cache = MemoizedArray((i,j) -> i+10j, ArrayMap{Int}((3,2)))
    indices = [3,1,2]
    copyto!(indices, view(cache, indices, 1))
    @test indices == [13,11,12]
    @test_throws BoundsError copyto!(zeros(Int,1), view(cache,:,1))
    @test_throws MethodError getready(cache, 1)
end
function inferred_tree_memoized(config, ::Val{C}) where C
    return MemoizedArray(Returns(1.0), config; concurrent=C)
end

function inferred_tree_lru(config, ::Val{C}) where C
    return LRUMemoizedArray(Returns(1.0), config; capacity=4, queues=1, concurrent=C)
end

@testset "Tree constructor consistency" begin
    for Map in (RadixMap, SubtreeRadixMap), concurrent in (false, true),
        domain in ((-4:4, -4:4), unbound(2)), center in (-0.5, 0.5)
        config = Map{Float64}(domain; block_shape=(8, 4), center)
        memoized = @inferred inferred_tree_memoized(config, Val(concurrent))
        lru = @inferred inferred_tree_lru(config, Val(concurrent))
        @test memoized[0, 0] == 1.0
        @test (concurrent ? queue_handle(lru, 1) : lru)[0, 0] == 1.0
        @test config.block_shape === (8, 4)
        @test config.center === center
    end
end
