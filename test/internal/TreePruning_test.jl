@testset "Radix map routing and bounded reclamation" begin
    JA = MemoizedArrays
    domain = (typemin(Int):typemax(Int), typemin(Int):typemax(Int))
    keys = unique(vcat([(i,j) for i in -1:1 for j in -1:1],
        [(0,typemin(Int)), (typemax(Int),0), (typemin(Int),typemax(Int)), (typemax(Int),typemin(Int))],
        [(reinterpret(Int, UInt(k) * 0x9e3779b97f4a7c15), k % 3 == 0 ? 0 : k) for k in 1:8]))
    for Map in (JA.RadixMap, JA.SubtreeRadixMap),
        block in ((8,8), (4,8))
        map = Map(Union{JA.Empty,MapTestEntry{2}}, domain, block, JA.EMPTY)
        slots = [JA.map_reserve!(map, key, JA.map_routing(map, key)) for key in keys]
        for (key, slot) in zip(keys, slots)
            JA.slot_publish!(slot, MapTestEntry(key))
            JA.map_retain!(map, slot)
            @test JA.slot_entry(slot).key == key
        end
        for (key, slot) in zip(keys, slots)
            found = mapslot(map, key)
            @test found.leaf === slot.leaf
            @test found.reference === slot.reference
            @test JA.slot_entry(found).key == key
        end
        treecounts(map)
        for (key, slot) in zip(keys, slots)
            captured = JA.slot_entry(slot)
            JA.map_delete!(map, key, slot)
            @test JA.slot_entry(slot) === JA.EMPTY
            @test captured.key == key
        end
        @test all(node -> node === map.roots.empty, map.roots.reference.mem)
    end
    for Map in (JA.RadixMap, JA.SubtreeRadixMap)
        map = Map(Union{JA.Empty,MapTestEntry{0}}, (), (), JA.EMPTY)
        slot = JA.map_reserve!(map, (), ())
        @test slot.reference === memoryref(slot.leaf.values.reference.mem)
        JA.slot_publish!(slot, MapTestEntry(()))
        JA.map_retain!(map, slot)
        @test JA.slot_entry(mapslot(map, ())).key == ()
        JA.map_delete!(map, (), slot)
        @test JA.slot_entry(slot) === JA.EMPTY
    end
end

@testset "Radix adjacent levels and compressed gaps" begin
    keys = ((0,0), (8,0), (64,0), (512,0), (4096,0), (8,8), (8192,2048), (8192,2056), (8192,2112), (1,1))
    for map_spec in (:radix, :subtree), block_shape in ((8,8), (4,8))
        cache = MemoizedArray(tuple, (test_map_config(map_spec, block_shape))(Tuple{Int,Int}, (0:8192, 0:8192)))
        for key in keys
            # A compressed edge must not mistake a different leaf's same local slot for this key.
            @test !isready(cache, key...)
            @test cache[key...] === key
        end
        @test all(key -> isready(cache, key...) && cache[key...] === key, keys)
        treecounts(cache.map)
        for key in reverse(keys)
            JA.delete_key!(cache.map, key)
            @test !isready(cache, key...)
            @test all(probe -> !isready(cache, probe...) || cache[probe...] === probe, keys)
            treecounts(cache.map)
        end
    end
end

@testset "Radix root prefix compression" begin
    JA = MemoizedArrays
    map = JA.RadixMap(Union{JA.Empty,MapTestEntry{2}},
        (typemin(Int):typemax(Int), typemin(Int):typemax(Int)), (4,8), JA.EMPTY)
    keys = ((0,0), (typemin(Int),typemax(Int)))
    central = JA.map_reserve!(map, first(keys), JA.map_routing(map, first(keys)))
    JA.slot_publish!(central, MapTestEntry(first(keys)))
    JA.map_retain!(map, central)
    @test (map.roots.reference.mem[1]) === central.leaf
    distant = JA.map_reserve!(map, last(keys), JA.map_routing(map, last(keys)))
    JA.slot_publish!(distant, MapTestEntry(last(keys)))
    JA.map_retain!(map, distant)
    @test mapslot(map, first(keys)).leaf === central.leaf
    middle = JA.map_reserve!(map, (4,0), JA.map_routing(map, (4,0)))
    JA.slot_publish!(middle, MapTestEntry((4,0)))
    JA.map_retain!(map, middle)
    @test treecounts(map) == (2, 3)
    JA.map_delete!(map, (4,0), middle)
    @test treecounts(map) == (1, 2)
    @test mapslot(map, last(keys)).leaf === distant.leaf
    JA.map_delete!(map, last(keys), distant)
    @test (map.roots.reference.mem[1]) === central.leaf
    @test JA.slot_entry(mapslot(map, first(keys))).key == first(keys)
    treecounts(map)
end

@testset "Depth shortcuts share the radix tree" begin
    JA = MemoizedArrays
    map = JA.SubtreeRadixMap(Union{JA.Empty,MapTestEntry{2}},
        (-19:typemax(Int), -11:typemax(Int)), (8,8), JA.EMPTY)
    keys = ((0,0), (-1,0), (5,0), (44,0), (45,0), (0,53))
    @test !hasfield(typeof(map), :zero)
    @test map.zero_path[1] != 1
    @test all(level -> map.zero_path[level + 1] == JA.tree_address(map, JA.tree_offset(map), level), 0:length(map.zero_path)-1)
    @test JA.root_index(map, JA.map_routing(map, (0,0))) == 1
    for key in keys
        slot = JA.map_reserve!(map, key, JA.map_routing(map, key))
        JA.slot_publish!(slot, MapTestEntry(key))
        JA.map_retain!(map, slot)
    end
    @test length(map.roots.reference.mem) == cld(8sizeof(UInt), 3)
    @test map.roots.reference.mem[1] === mapslot(map, (0,0)).leaf === mapslot(map, (-1,0)).leaf
    @test map.roots.reference.mem[2] isa JA.TreeBranch
    @test map.roots.reference.mem[3] isa JA.TreeBranch
    smallroots = Tuple(map.roots.reference.mem[1:3])
    distantkey = (typemax(Int), typemax(Int))
    distant = JA.map_reserve!(map, distantkey, JA.map_routing(map, distantkey))
    JA.slot_publish!(distant, MapTestEntry(distantkey))
    JA.map_retain!(map, distant)
    @test map.roots.reference.mem[end] isa JA.TreeBranch
    @test JA.node_level(map.roots.reference.mem[end]) == length(map.roots.reference.mem)-1
    @test all(i -> map.roots.reference.mem[i] === smallroots[i], 1:3)
    for key in ((5,0), (44,0), distantkey)
        JA.delete_key!(map, key)
    end
    @test map.roots.reference.mem[2] === map.roots.reference.mem[3]
    @test map.roots.reference.mem[end] === map.roots.reference.mem[3]
    @test all(key -> JA.slot_entry(mapslot(map, key)).key == key, ((0,0), (-1,0), (45,0), (0,53)))
    @test mapslot(map, (5,0)) === nothing || JA.slot_entry(mapslot(map, (5,0))) isa JA.Empty
    @test treecounts(map)[2] == 3
    treecounts(map)
end

@testset "Skipped subtree depths select an existing ancestor" begin
    cache = MemoizedArray(identity, SubtreeRadixMap{Int}(unbound(1); block_shape=(8,)))
    @test cache[0] == 0
    @test cache[4096] == 4096
    map = cache.map
    root = map.roots.reference.mem[end]
    @test root.level > 1
    @test all(i -> map.roots.reference.mem[i] === root, 2:root.level+1)
    @test map.roots.reference.mem[1] === mapslot(map, (0,)).leaf
    @test cache[8] == 8
    JA.delete_key!(map, (0,))
    @test map.roots.reference.mem[1] === root
    @test !isready(cache, 0)
    @test cache[0] == 0
    @test all(i -> isready(cache, i) && cache[i] == i, (0, 8, 4096))
    treecounts(map)
end

@testset "Subtree zero-path routing and pruning" begin
    for byslot in (false, true)
        map = JA.SubtreeRadixMap(Any, (typemin(Int):typemax(Int),), (8,), JA.EMPTY)
        for key in ((0,), (typemin(Int),))
            slot = JA.map_reserve!(map, key, JA.map_routing(map, key))
            JA.slot_publish!(slot, key)
            JA.map_retain!(map, slot)
        end
        if byslot
            JA.map_delete!(map, (0,), mapslot(map, (0,)))
        else
            JA.delete_key!(map, (0,))
        end
        @test all(node -> node === map.roots.empty, map.roots.reference.mem[1:end-1])
        @test JA.slot_entry(mapslot(map, (typemin(Int),))) == (typemin(Int),)
    end
    # Above an off-zero compressed child, retain the ancestor; a root may stand alone.
    for distant in ((), (512,))
        map = JA.SubtreeRadixMap(Any, (0:1024,), (8,), JA.EMPTY)
        for i in (64, 72, distant...)
            slot = JA.map_reserve!(map, (i,), JA.map_routing(map, (i,)))
            JA.slot_publish!(slot, i)
            JA.map_retain!(map, slot)
        end
        @test map.roots.reference.mem[3] isa JA.TreeBranch
        JA.delete_key!(map, (64,))
        @test map.roots.reference.mem[3] === (isempty(distant) ? mapslot(map, (72,)).leaf : map.roots.reference.mem[4])
        JA.delete_key!(map, (72,))
        @test map.roots.reference.mem[3] === map.roots.empty
    end
    for domain in ((1:8,), (-8:-1,), (0:8, 1:8), (1:0,))
        @test_throws ArgumentError JA.SubtreeRadixMap(Any, domain, ntuple(_ -> 8, length(domain)), JA.EMPTY)
    end
    for domain in ((-19:100, -11:100), (typemin(Int):typemax(Int), typemin(Int):typemax(Int))),
        block in ((8,8), (4,8))
        cache = LRUMemoizedArray(tuple, SubtreeRadixMap{Tuple{Int,Int}}(domain; block_shape=block); capacity=3)
        normal = JA.RadixMap(Any, domain, block, JA.EMPTY)
        keys = [(mod(37i, 120)-19, mod(53i, 112)-11) for i in 0:39]
        append!(keys, [(0,0), (first(domain[1]),first(domain[2])), (last(domain[1]),last(domain[2])), (0,0)])
        for key in keys
            @test cache[key...] == key
            @test JA.map_routing(cache.map, key) == JA.map_routing(normal, key)
            for probe in (key, (0,0), (-1,-1), (1,1))
                encoded = JA.map_routing(cache.map, probe)
                direct = JA.tree_find(cache.map, encoded, JA.root_link(cache.map),
                    length(cache.map.zero_path)-1, Val(false))
                shortcut = mapslot(cache.map, probe)
                @test (direct === nothing && shortcut === nothing) ||
                    (direct !== nothing && shortcut !== nothing && direct.leaf === shortcut.leaf &&
                     direct.reference === shortcut.reference)
            end
            treecounts(cache.map)
        end
    end
end

@testset "Tree Memoized same-key nested publication" begin
    for map_spec in (:radix, :subtree), conversion in (false, true)
        owner = Ref{Any}()
        entered = Ref(false)
        calls = Ref(0)
        nested() = (owner[][0]; 11)
        generator = i -> begin
            calls[] += 1
            if !entered[]
                entered[] = true
                return conversion ? TestConversion(nested) : nested()
            end
            return 7
        end
        cache = MemoizedArray(generator, (test_map_config(map_spec))(Int, (-1:1,)))
        owner[] = cache
        @test cache[0] == cache[0] == 11
        @test calls[] == 2
        @test mapslot(cache.map, (0,)).leaf.used == 1
        @test treecounts(cache.map) == (0, 1)
    end
end

@testset "Slice leaf reuse across generation and eviction" begin
    for Map in (RadixMap, SubtreeRadixMap), C in (MemoizedArray, LRUMemoizedArray), width in (2, 4)
        config = Map{Int}((-32:32, -8:8); block_shape=(width, width))
        make(log) = C === MemoizedArray ? C((i, j) -> (push!(log, (i, j)); i + 10j), config) :
            C((i, j) -> (push!(log, (i, j)); i + 10j), config; capacity=2)
        calls, scalarcalls = Tuple{Int,Int}[], Tuple{Int,Int}[]
        cache, scalar = make(calls), make(scalarcalls)
        for rows in (-7:9, -32:16:32, [0, 1, 16, 0, -16, 1], 1:0)
            empty!(calls); empty!(scalarcalls)
            expected = [scalar[i, j] for i in rows, j in -2:2]
            @test cache[rows, -2:2] == expected
            @test calls == scalarcalls
            @test [isready(cache, i, j) for i in -32:32, j in -8:8] ==
                  [isready(scalar, i, j) for i in -32:32, j in -8:8]
            @test copy(view(cache, rows, -2:2)) == expected
        end
    end
    for Map in (RadixMap, SubtreeRadixMap), width in (2, 4)
        # Generating coordinate 1 evicts and prunes the leaf remembered at coordinate 0.
        cache = LRUMemoizedArray(Map{Int}(unbound(1); block_shape=(width,)); capacity=1) do i
            i == 1 && cache[100]
            i
        end
        @test cache[0:2] == [0, 1, 2]
        @test cache[0:2] == [0, 1, 2]
        @test isready(cache, 2)
        @test !isready(cache, 1)
    end
end

@testset "Bounded slice traversal preserves ordered cache effects" begin
    for Map in (RadixMap, SubtreeRadixMap), C in (MemoizedArray, LRUMemoizedArray)
        config = Map{Int}(unbound(2); block_shape=(8,8))
        make(log) = C === MemoizedArray ? C((i,j)->(push!(log,(i,j)); i+j), config) :
            C((i,j)->(push!(log,(i,j)); i+j), config;capacity=3)
        calls, scalarcalls = Tuple{Int,Int}[], Tuple{Int,Int}[]
        cache, scalar = make(calls), make(scalarcalls)
        cache[0,0]; scalar[0,0]
        for _ in 1:2
            empty!(calls); empty!(scalarcalls)
            keys = CartesianIndices((-16:15,-16:15))
            expected = [scalar[key] for key in keys]
            @test copy(view(cache,-16:15,-16:15)) == expected
            @test calls == scalarcalls
            @test [isready(cache,key) for key in keys] == [isready(scalar,key) for key in keys]
        end
    end
end

@testset "Warm slice copies preserve pins and eviction order" begin
    for Map in (RadixMap, SubtreeRadixMap)
        cache = LRUMemoizedArray(+,Map{Int}(unbound(2);block_shape=(8,8));capacity=1024)
        for key in reverse(CartesianIndices((1:32,1:32)))
            cache[key]
        end
        cache[1,1] = 777
        cache[1000,1000]
        expected = [i+j for i in 1:32,j in 1:32]
        expected[1,1] = 777
        @test copy(view(cache,1:32,1:32)) == expected
        cache[1001,1000]
        @test !isready(cache,1000,1000) && isready(cache,2,1)
        cache[1002,1000]
        @test !isready(cache,2,1) && cache[1,1] == 777
    end
end

@testset "Slice destination conversion can replace collected leaves" begin
    for Map in (RadixMap, SubtreeRadixMap)
        owner = Ref{Any}()
        generator = (i,j) -> TestConversion(() -> begin
            if (i,j) == (1,1)
                for k in 10_000:11_023
                    owner[][k,10_000]
                end
                owner[][2,1] = TestConversion(() -> 999)
            end
            i+j
        end)
        cache = LRUMemoizedArray(generator,Map{TestConversion}(unbound(2);block_shape=(8,8));capacity=1024)
        owner[] = cache
        for key in CartesianIndices((1:32,1:32))
            cache[key]
        end
        expected = [i+j for i in 1:32,j in 1:32]
        expected[2,1] = 999
        @test copyto!(zeros(Int,32,32),view(cache,1:32,1:32)) == expected
    end
end

@testset "Slice copies preserve boundary and multidimensional coordinates" begin
    for Map in (RadixMap, SubtreeRadixMap), (ranges,block) in (
            ((typemax(Int)-1023:typemax(Int),),(8,)),
            ((1:16,-3:4,5:12),(4,8,2)),
            ((1:32,4:4,-16:15),(4,8,2)))
        cache = MemoizedArray((key...) -> sum(key),Map{Int}(unbound(length(ranges));block_shape=block))
        keys = CartesianIndices(ranges)
        expected = [cache[key] for key in keys]
        @test cache[ranges...] == expected
    end
end

@testset "Bounded tree views remain live across pruning" begin
    for Map in (RadixMap, SubtreeRadixMap), C in (MemoizedArray, LRUMemoizedArray)
        calls = Ref(0)
        config = Map{Int}((-128:128,-128:128); block_shape=(8,8))
        generator = (i,j) -> (calls[] += 1; i + 10j)
        cache = C === MemoizedArray ? C(generator,config) : C(generator,config;capacity=2)
        selected = view(cache,-2:2,-3:3)
        @test selected isa JA.TreeView
        @test calls[] == 0
        @test parent(selected) === cache
        @test parentindices(selected) == (-2:2,-3:3)
        @test size(selected) == (5,7)
        @test selected[1,1] == -32
        @test selected[CartesianIndex(1,1)] == -32
        @test_throws BoundsError selected[0,1]
        @test_throws BoundsError selected[6,1]
        for _ in 1:3
            cache[100,100]; cache[-100,-100]
            @test selected[1,1] == -32
            @test isready(cache,-2,-3)
        end
        cache[-2,-3] = 42
        @test selected[1,1] == 42
        selected[2,2] = 43
        @test cache[-1,-2] == 43
        nested = view(selected,1:2,1:2)
        @test parent(nested) === cache
        @test copy(nested) == [42 -22; -31 43]
        full = view(cache,:,:)
        @test axes(full) == axes(view(full.selection,:,:))
        @test full[-2,-3] == 42
        @test sum(nested) == sum(copy(nested))
        column = view(selected,:,1)
        @test column isa JA.TreeView
        @test column[1] == 42
        @test copy(column) == copy(selected)[:,1]
        @test view(selected,1,1)[] == 42
        @test view(cache,Int32(-2):Int32(0),Int32(-3):Int32(0))[1,1] == 42
    end
    for Map in (RadixMap, SubtreeRadixMap)
        cache = LRUMemoizedArray(Map{Int}(unbound(1);block_shape=(8,));capacity=1) do i
            i == 1 && cache[100]
            i
        end
        selected = view(cache,0:3)
        @test collect(selected) == [0,1,2,3]
        @test collect(selected) == [0,1,2,3]
    end
end

@testset "Tree view iteration preserves live values and access order" begin
    values(view) = reshape([identity(value) for value in view], size(view))
    for Map in (RadixMap, SubtreeRadixMap), C in (MemoizedArray, LRUMemoizedArray)
        config = Map{Int}(unbound(2); block_shape=(8,8))
        calls, expectedcalls = Tuple{Int,Int}[], Tuple{Int,Int}[]
        make(log) = C === MemoizedArray ? C((i,j) -> (push!(log,(i,j)); i+10j),config) :
            C((i,j) -> (push!(log,(i,j)); i+10j),config;capacity=3)
        cache, scalar = make(calls), make(expectedcalls)
        selected = view(cache,10:73,-16:-1)
        @test isempty(calls)
        for _ in 1:2
            empty!(calls); empty!(expectedcalls)
            expected = [scalar[k] for k in CartesianIndices((10:73,-16:-1))]
            @test values(selected) == expected
            @test calls == expectedcalls
            @test [isready(cache,k) for k in CartesianIndices((10:73,-16:-1))] ==
                [isready(scalar,k) for k in CartesianIndices((10:73,-16:-1))]
        end
        # Pause two independent traversals, change the next value, and evict other leaves.
        _, firststate = iterate(selected)
        _, secondstate = iterate(selected)
        cache[11,-16] = 123
        cache[1000,1000]
        @test iterate(selected,firststate)[1] == 123
        @test iterate(selected,secondstate)[1] == 123
        @test values(view(cache,1:0,1:2)) == zeros(Int,0,2)
        @test values(view(cache,11,-16))[] == 123
        @test values(view(cache,1:1024,-16)) == [cache[i,-16] for i in 1:1024]
        @test values(view(cache,11,-1024:-1)) == [cache[11,j] for j in -1024:-1]
        hot = C === MemoizedArray ? C(+,config) : C(+,config;capacity=1024)
        hot[10:73,-16:-1]
        selected = view(hot,10:73,-16:-1)
        @test values(selected) == [i+j for i in 10:73,j in -16:-1]
        _, state = iterate(selected)
        @test state[1].leaves !== nothing
        hot[11,-16] = 123
        value, state = iterate(selected,state)
        @test value == 123
        for i in 1000:3048
            hot[i,1000]
        end
        @test iterate(selected,state)[1] == -4
        # Relative leaf coordinates also cover unequal leaf widths and dropped dimensions.
        for indices in ((typemin(Int)+13,-8:23,-16:15),
                        (typemax(Int)-31:typemax(Int),9,-16:15))
            config3 = Map{Int}(unbound(3);block_shape=(4,8,2))
            cache3 = C === MemoizedArray ? C(+,config3) : C(+,config3;capacity=1024)
            expected = cache3[indices...]
            selected3 = view(cache3,indices...)
            @test map(identity,selected3) == expected
            visited = Int[]
            foreach(value -> push!(visited,value),selected3)
            @test visited == vec(expected)
        end
    end
    for Map in (RadixMap, SubtreeRadixMap)
        cache = LRUMemoizedArray(Map{Int}(unbound(2);block_shape=(8,8));capacity=2) do i,j
            isodd(i) && cache[10000,10000]
            i+j
        end
        @test values(view(cache,1:64,1:16)) == [i+j for i in 1:64,j in 1:16]
        cache3 = MemoizedArray(+,Map{Int}(unbound(3);block_shape=(8,8,8)))
        @test values(view(cache3,1:16,1:8,1:8)) == [i+j+k for i in 1:16,j in 1:8,k in 1:8]
        # A huge finite view must not allocate a table proportional to its extent.
        @test iterate(view(cache3,1:10^9,1:2,1:2))[1] == 3
    end
end

@testset "Tree view mapping preserves destination and live-source behavior" begin
    for Map in (RadixMap, SubtreeRadixMap), C in (MemoizedArray, LRUMemoizedArray)
        config = Map{Int}(unbound(3);block_shape=(4,8,2))
        cache = C === MemoizedArray ? C(+,config) : C(+,config;capacity=1024)
        expected = cache[1:32,4,1:32]
        selected = view(cache,1:32,4,1:32)
        dest = fill(-7,1026)
        @test map!(abs2,dest,selected) === dest
        @test dest == vcat(vec(abs2.(expected)),[-7,-7])
        @test map!(-,zeros(Int,3),selected) == -vec(expected)[1:3]
        @test isempty(map!(error,Int[],selected))
        checked = @inline value -> begin
            @boundscheck throw(BoundsError())
            value
        end
        @test_throws BoundsError map!(checked,dest,selected)

        for conversion in (false,true)
            config = Map{Int}(unbound(2);block_shape=(8,8))
            cache = C === MemoizedArray ? C(+,config) : C(+,config;capacity=1024)
            cache[1:32,1:32]
            first = Ref(true)
            mutate(value) = begin
                if first[]
                    first[] = false
                    for k in 10_000:11_023
                        cache[k,10_000]
                    end
                    cache[2,1] = 999
                end
                2value
            end
            f = conversion ? (value -> TestConversion(() -> mutate(value))) : mutate
            expected = [2(i+j) for i in 1:32,j in 1:32]
            expected[2,1] = 1998
            @test map!(f,zeros(Int,32,32),view(cache,1:32,1:32)) == expected
        end
    end
end

@testset "Tree view reductions preserve promotion and evaluation order" begin
    for Map in (RadixMap, SubtreeRadixMap), V in (Bool,Int8,UInt8,Float64)
        cache = MemoizedArray((i,j) -> V(isodd(i+j)),Map{V}(unbound(2);block_shape=(8,8)))
        expected = cache[1:32,1:32]
        selected = view(cache,1:32,1:32)
        @test sum(selected) === mapfoldl(identity,Base.add_sum,expected)
        @test sum(x for x in selected if false) == 0
        @test sum(x for x in selected if x > 0) === sum(selected)
        @test sum(abs2,selected;init=2.0) === mapfoldl(abs2,Base.add_sum,expected;init=2.0)
        ordered(a,b) = mod(3a+b,257)
        @test mapfoldl(Int,ordered,selected;init=7) == mapfoldl(Int,ordered,expected;init=7)
        @test sum(selected;dims=1) == sum(expected;dims=1)
    end
    for Map in (RadixMap, SubtreeRadixMap)
        cache = MemoizedArray((i,j) -> (1e16,1.0,-1e16,1.0)[mod1(i,4)],
            Map{Float64}(unbound(2);block_shape=(8,8)))
        cache[1:32,1:32]
        @test sum(view(cache,1:32,1:32)) === 1.0
    end
    for Map in (RadixMap, SubtreeRadixMap), C in (MemoizedArray, LRUMemoizedArray)
        function scenario(optimized)
            events = Tuple{Symbol,Int,Int}[]
            generator = (i,j) -> (push!(events,(:generate,i,j));i+10j)
            config = Map{Int}(unbound(2);block_shape=(8,8))
            cache = C === MemoizedArray ? C(generator,config) : C(generator,config;capacity=1024)
            cache[1:32,1:32]
            empty!(events)
            first = Ref(true)
            f = value -> begin
                push!(events,(:map,value,0))
                if first[]
                    first[] = false
                    for k in 10_000:11_023
                        cache[k,10_000]
                    end
                    cache[2,1] = 999
                end
                value
            end
            op = Base.MappingRF(f,Base.BottomRF(+))
            selected = view(cache,1:32,1:32)
            result = optimized ? mapfoldl(f,+,selected;init=7) :
                invoke(Base._foldl_impl,Tuple{Any,Any,Any},op,7,selected)
            return result,events
        end
        @test scenario(true) == scenario(false)
    end
end

@testset "Tree view entry points cover the entire selection" begin
    for Map in (RadixMap, SubtreeRadixMap), C in (MemoizedArray, LRUMemoizedArray), center in (-0.5,0.5)
        config = Map{Int}(unbound(2);block_shape=(8,8),center)
        cache = C === MemoizedArray ? C(+,config) : C(+,config;capacity=8192)
        cache[-10^9,10^9]; cache[10^9,-10^9]
        for (rows,cols) in ((-16:15,-16:15),(1:32,1:32),(10001:10032,-32:-1))
            cache[rows,cols]
            selected = view(cache,rows,cols)
            node = JA.view_root(cache.map,selected.low,selected.high)
            @test JA.tree_matches(cache.map,selected.low,JA.node_prefix(node),JA.node_level(node))
            @test JA.tree_matches(cache.map,selected.high,JA.node_prefix(node),JA.node_level(node))
            step = iterate(selected)
            values = Int[]
            used_table = true
            for key in CartesianIndices((rows,cols))
                value, state = step
                push!(values,value)
                # A warm rectangle should stay on the leaf table throughout the pass.
                used_table &= state[1].leaves !== nothing
                step = iterate(selected,state)
            end
            @test values == vec([i+j for i in rows,j in cols])
            @test used_table
            @test step === nothing
        end
    end
end
