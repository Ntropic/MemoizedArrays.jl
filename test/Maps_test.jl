# Map-specific routing, collisions, growth and pruning regressions.
function treecounts(node, empty)
    if node === empty
        return (0, 0, true)
    elseif node isa JA.TreeLeaf
        entries = count(i -> node.values.reference.mem[i] !== node.values.empty, eachindex(node.values.reference.mem))
        return (0, 1, node.used == entries)
    end
    branches, leaves, used, occupied = 1, 0, 0, 0
    valid = true
    for i in eachindex(node.children.reference.mem)
        child = node.children.reference.mem[i]
        if child !== node.children.empty
            used += 1
            occupied ⊻= i
            b, l, child_valid = treecounts(child, empty)
            branches += b
            leaves += l
            valid &= child_valid
        end
    end
    valid &= node.used == used
    valid &= node.occupied == occupied
    return branches, leaves, valid
end
function treecounts(map::JA.TreeMap)
    branches, leaves, valid = 0, 0, true
    for root in (map.roots.reference.mem[end],)
        b, l, rootvalid = treecounts(root, map.roots.empty)
        branches += b
        leaves += l
        valid &= rootvalid
    end
    @test valid
    return branches, leaves
end

mutable struct MapTestEntry{N}
    key::NTuple{N,Int}
end

mapslot(map, key) = MemoizedArrays.map_slot(map, key, MemoizedArrays.map_routing(map, key))

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

@testset "Tree ArrayMap sentinels" begin
    matrix = ones(4, 4)
    values = (nothing, false, 0, matrix, view(matrix, 1:2, 1:2))
    for map_spec in (:radix, :subtree)
        calls = Ref(0)
        cache = MemoizedArray(i -> (calls[] += 1; values[i ÷ 64 + 1]), (test_map_config(map_spec, (8,)))(Any, (0:256,)))
        @test cache.map.roots.empty isa JA.TreeLeaf
        @test fieldtype(typeof(cache.map.roots), :empty) === eltype(cache.map.roots.reference.mem)
        @test all(node -> node === cache.map.roots.empty, cache.map.roots.reference.mem)
        for (i, value) in enumerate(values)
            key = 64(i - 1)
            @test !isready(cache, key)
            @test cache[key] === cache[key] === value
            @test isready(cache, key) && !isready(cache, key + (key == 256 ? -1 : 1))
        end
        @test calls[] == length(values)
        leaf = mapslot(cache.map, (0,)).leaf
        branch = cache.map.roots.reference.mem[end]
        # The blocks must remain inline: no extra object between a node and its slots.
        @test Base.allocatedinline(typeof(leaf.values))
        @test Base.allocatedinline(typeof(branch.children))
        @test eltype(leaf.values.reference.mem) === JA.entrytype(cache)
        @test treecounts(cache.map)[2] == length(values)
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
