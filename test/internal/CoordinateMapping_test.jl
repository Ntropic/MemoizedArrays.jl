@testset "Radix bit extraction" begin
    for block in ((4,4), (8,8), (16,16), (4,8), (8,4), (2,32), (32,2))
        map = JA.RadixMap(Int, (0:1,0:1), block, 0)
        top = JA.tree_level(map, (typemax(UInt),typemax(UInt)))
        for key in ((UInt(0),UInt(0)), (typemax(UInt),typemax(UInt)),
                    (UInt(1) << (8sizeof(UInt)-1), UInt(1))), level in 0:top
            # BigInt division is independent of machine-word shift/wrap semantics.
            digits = ntuple(i -> Int((big(key[i]) ÷ big(block[i])^level) % block[i]), 2)
            @test JA.tree_address(map, key, level) == 1 + digits[1] + block[1]*digits[2]
            for candidate in ((UInt(0),UInt(0)), (typemax(UInt),typemax(UInt)), key)
                prefix = JA.tree_prefix(map, candidate, level)
                @test JA.tree_matches(map, key, prefix, level) == (JA.tree_prefix(map, key, level) == prefix)
            end
        end
    end
end

@testset "Radix aligned routing preserves domain coordinates" begin
    for domain in ((-19:44, 7:71), (typemin(Int):typemax(Int), -1:typemax(Int)))
        calls = Ref(0)
        cache = MemoizedArray((i,j) -> (calls[] += 1; (i,j)), RadixMap{Tuple{Int,Int}}(domain; block_shape=(8, 8)))
        for key in ((first(domain[1]),first(domain[2])), (last(domain[1]),last(domain[2])), (0,7))
            expected = ntuple(i -> UInt(mod(big(key[i]) + JA.tree_offset(cache.map)[i], big(1) << (8sizeof(UInt)))), 2)
            @test JA.map_routing(cache.map, key) == expected
            @test !isready(cache, key...)
            @test cache[key...] == cache[key...] == key
            @test isready(cache, key...)
        end
        @test calls[] == 3
        @test_throws BoundsError cache[0, first(domain[2])-1]
    end
end

@testset "Validated tree block allocation" begin
    empty = Ref(0)
    for shape in ((), (8,), (4, 8))
        checked = JA.ArrayMap(typeof(empty), shape, empty)
        unchecked = JA.ArrayMap(typeof(empty), shape, empty, Val(:nocheck))
        @test unchecked.shape == checked.shape
        @test unchecked.strides == checked.strides
        @test length(unchecked.reference.mem) == length(checked.reference.mem)
        @test all(value -> value === empty, unchecked.reference.mem)
    end
    @test_throws ArgumentError JA.ArrayMap(typeof(empty), (-1,), empty)
    width = 1 << (4sizeof(Int))
    @test_throws OverflowError JA.ArrayMap(typeof(empty), (width, width), empty)
    for map_spec in (RadixMap, SubtreeRadixMap)
        @test_throws ArgumentError MemoizedArray(identity, map_spec{Int}(unbound(1); block_shape=(3,)))
        @test_throws OverflowError MemoizedArray(+, map_spec{Int}(unbound(2); block_shape=(width, width)))
    end
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
        @test read_allocations(cache, hi) == 0
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
