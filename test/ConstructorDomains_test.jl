@testset "Shared domain and map construction" begin
    @test unbound(3) === (:, :, :)
    @test unbound(0) === ()
    @test_throws ArgumentError unbound(-1)
    @test isunbound(:)
    @test !isunbound(3) && !isunbound(-2:2)
    for (C, options) in ((JITArray, (;)), (ExactLRUArray, (; capacity=2)),
                         (SampledLRUArray, (; capacity=2)))
        @test_throws MethodError C{Int}(identity; options...)
        for domain in ((3, 4), (1:3, 1:4), [3, 4], [1:3, 1:4], (-1:1, 4), Any[-1:1, 4])
            cache = C{Tuple{Int,Int}}(tuple, domain; options...)
            @test cache.map isa JA.ArrayMap
            @test cache[1, 2] == (1, 2)
        end
        for storage in (:array, :dict, :radixtree, :subtree)
            extra = storage in (:radixtree, :subtree) ? (; block_shape=(4, 8)) : (;)
            if storage === :subtree
                @test_throws ArgumentError C{Tuple{Int,Int}}(tuple, (-2:2, 4:7); storage, extra..., options...)
                @test_throws ArgumentError C{Int}(+, (3, 4); storage, extra..., options...)
                continue
            end
            cache = C{Tuple{Int,Int}}(tuple, (-2:2, 4:7); storage, extra..., options...)
            @test cache.map isa getproperty(JA, Dict(:array=>:ArrayMap, :dict=>:DictMap,
                :radixtree=>:RadixMap, :subtree=>:SubtreeRadixMap)[storage])
            @test cache[-2, 4] == (-2, 4)
            @test cache[20] == (2, 7)
            @test cache[0, 5] == (0, 5)
            @test cache[-2, 4] == (-2, 4)
            @test copy(view(cache, -1:1, 5:6)) == [(i, j) for i in -1:1, j in 5:6]
            @test_throws BoundsError cache[-3, 4]
            lengths = C{Int}(+, (3, 4); storage, extra..., options...)
            @test lengths[2, 3] == 5
            if C !== JITArray
                @test cache.count == 2
            end
        end
        one = C{Int}(identity, (-2:2,); options...)
        @test one[-2] == -2 && one[2] == 2
        @test isready(one, -2)
        @test_throws BoundsError one[3]
        if C === JITArray
            @test getready(one, -2) == -2
        end
        for domain in (unbound(2), [:, :], (:, -2:2), Any[:, -2:2])
            cache = C{Tuple{Int,Int}}(tuple, domain; options...)
            @test cache.map isa JA.DictMap
            @test cache[typemin(Int), 0] == (typemin(Int), 0)
            @test cache[typemax(Int), 1] == (typemax(Int), 1)
            @test_throws ArgumentError C{Int}(+, domain; storage=:array, options...)
        end
        for storage in (:radixtree, :subtree)
            cache = C{Tuple{Int,Int}}(tuple, unbound(2); storage, block_shape=(4, 8), options...)
            @test cache[typemin(Int), typemax(Int)] == (typemin(Int), typemax(Int))
        end
        for domain in ((-1, 2), (1.5,), ("axis",))
            @test_throws ArgumentError C{Int}(identity, domain; options...)
        end
        @test_throws ArgumentError C{Int}(identity, (4,); storage=:unknown, options...)
        @test_throws ArgumentError C{Int}(identity, (4,); block_shape=(4,), options...)
    end
end

function tree_read_allocations(cache, key)
    cache[key...]
    return @allocated cache[key...]
end

@testset "Cache-side tree offsets" begin
    for (C, options) in ((JITArray, (;)), (ExactLRUArray, (; capacity=2)),
                         (SampledLRUArray, (; capacity=2))),
        storage in (:radixtree, :subtree)
        plain = C{NTuple{2,Int}}(tuple, (0:100, 0:100); storage, block_shape=(4,8), options...)
        shifted = C{NTuple{2,Int}}(tuple, (-19:100, -11:100); storage, block_shape=(4,8), options...)
        @test JA.cache_offset(plain) === Val(JA.tree_offset(plain.map))
        @test JA.cache_offset(shifted) === JA.cache_offset(plain)
        @test typeof(plain.map) === typeof(shifted.map)
        @test typeof(plain) === typeof(shifted)
        @test !hasfield(typeof(plain), :offset)
        for domain in ((0:100, 0:100), (-19:100, -11:100), (0:100, -11:100),
                       (typemin(Int):typemax(Int), typemin(Int):typemax(Int)))
            cache = C{NTuple{2,Int}}(tuple, domain; storage, block_shape=(4,8), options...)
            for key in ((0,0), (first(domain[1]),first(domain[2])), (100,100), (0,0))
                encoded = ntuple(i -> UInt(mod(big(key[i]) + JA.tree_offset(cache.map)[i], big(1) << (8sizeof(UInt)))), 2)
                @test (@inferred JA.encode_coordinates(cache, key, JA.cache_offset(cache))) === encoded
                route = @inferred JA.cache_routing(cache, key)
                if storage === :subtree
                    @test route === (encoded, JA.rootindex(cache.map, encoded))
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
    for center in (0.5, -0.5), storage in (:radixtree, :subtree),
        (C, options) in ((JITArray, (;)), (ExactLRUArray, (; capacity=2)),
                         (SampledLRUArray, (; capacity=2)))
        cache = C{Tuple{Int,Int}}(tuple, unbound(2); storage, block_shape=(8,4), center, options...)
        lo = center == 0.5 ? (-3,-1) : (-4,-2)
        hi = (lo[1]+7, lo[2]+3)
        prefix(key) = JA.tree_prefix(cache.map, JA.store_hash(cache.map, key), 0)
        @test prefix(lo) == prefix(hi) == prefix((0,0))
        @test prefix((lo[1]-1,0)) != prefix((0,0))
        @test prefix((hi[1]+1,0)) != prefix((0,0))
        for key in (lo, hi, (0,0), (typemin(Int),typemax(Int)), (typemax(Int),typemin(Int)), lo, hi)
            @test (@inferred cache[key...]) == key
            @test isready(cache, key)
            @test JA.cache_routing(cache, key) == (storage === :subtree ?
                (JA.store_hash(cache.map, key), JA.rootindex(cache.map, JA.store_hash(cache.map, key))) :
                JA.store_hash(cache.map, key))
        end
        @test tree_read_allocations(cache, hi) == 0
        @test_throws ArgumentError C{Int}(identity, unbound(1); storage, block_shape=(8,), center=0, options...)
        @test_throws ArgumentError C{Int}(identity, (8,); center, options...)
    end
    # Check geometry independently: each ancestor is the closest aligned interval
    # to the requested center, among the possible parents of the central child.
    for width in (2,4,8,16,32), center in (0.5,-0.5)
        cache = JITArray{Int}(identity, unbound(1); storage=:subtree, block_shape=(width,), center)
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
