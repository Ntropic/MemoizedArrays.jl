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

mapslot(map, key) = JITArrays.store_lookup(map, key, JITArrays.store_hash(map, key))

@testset "Radix map routing and bounded reclamation" begin
    JA = JITArrays
    domain = (typemin(Int):typemax(Int), typemin(Int):typemax(Int))
    keys = unique(vcat([(i,j) for i in -1:1 for j in -1:1],
        [(0,typemin(Int)), (typemax(Int),0), (typemin(Int),typemax(Int)), (typemax(Int),typemin(Int))],
        [(reinterpret(Int, UInt(k) * 0x9e3779b97f4a7c15), k % 3 == 0 ? 0 : k) for k in 1:8]))
    for Map in (JA.RadixMap, JA.SubtreeRadixMap),
        block in ((8,8), (4,8))
        map = Map(Union{JA.Empty,MapTestEntry{2}}, domain, block, JA.EMPTY)
        slots = [JA.store_reserve!(map, key, JA.store_hash(map, key)) for key in keys]
        for (key, slot) in zip(keys, slots)
            JA.slot_publish!(slot, MapTestEntry(key))
            JA.store_retain!(map, slot)
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
            JA.store_delete!(map, key, slot)
            @test JA.slot_entry(slot) === JA.EMPTY
            @test captured.key == key
        end
        @test all(node -> node === map.roots.empty, map.roots.reference.mem)
    end
    for Map in (JA.RadixMap, JA.SubtreeRadixMap)
        map = Map(Union{JA.Empty,MapTestEntry{0}}, (), (), JA.EMPTY)
        slot = JA.store_reserve!(map, (), ())
        @test slot.reference === memoryref(slot.leaf.values.reference.mem)
        JA.slot_publish!(slot, MapTestEntry(()))
        JA.store_retain!(map, slot)
        @test JA.slot_entry(mapslot(map, ())).key == ()
        JA.store_delete!(map, (), slot)
        @test JA.slot_entry(slot) === JA.EMPTY
    end
end

@testset "Radix adjacent levels and compressed gaps" begin
    cache = JITArray{Tuple{Int,Int}}((i,j) -> (i,j), (0:8192,0:8192); storage=:radixtree, block_shape=(8,8))
    keys = ((0,0), (8,0), (64,0), (512,0), (4096,0), (8,8), (8192,2048), (8192,2056), (8192,2112), (1,1))
    for key in keys
        @test cache[key...] === key
    end
    @test all(key -> isready(cache, key...) && cache[key...] === key, keys)
    treecounts(cache.map)
    for key in reverse(keys)
        JA.delete_key!(cache.map, key)
        @test !isready(cache, key...)
        treecounts(cache.map)
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

@testset "Radix domain-relative routing" begin
    for domain in ((-19:44, 7:71), (typemin(Int):typemax(Int), -1:typemax(Int)))
        calls = Ref(0)
        cache = JITArray{Tuple{Int,Int}}((i,j) -> (calls[] += 1; (i,j)), domain; storage=:radixtree, block_shape=(8,8))
        for key in ((first(domain[1]),first(domain[2])), (last(domain[1]),last(domain[2])), (0,7))
            expected = ntuple(i -> UInt(big(key[i]) - first(domain[i])), 2)
            @test JA.store_hash(cache.map, key) == expected
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
    for storage in (:radixtree, :subtreetree)
        calls = Ref(0)
        cache = JITArray{Any}(i -> (calls[] += 1; values[i ÷ 64 + 1]), 0:256; storage, block_shape=(8,))
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
    for seed in (false, true)
        calls = Dict{Tuple{Int,Int},Int}()
        cache = JITArray{Int}((-32:1023, -32:1023); storage=:radixtree, block_shape=(4,8)) do i, j
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
    scalar = JITArray{Int}(() -> 17, (); storage=:radixtree, block_shape=())
    @test scalar[] == scalar[] == 17
end

@testset "Radix root prefix compression" begin
    JA = JITArrays
    map = JA.RadixMap(Union{JA.Empty,MapTestEntry{2}},
        (typemin(Int):typemax(Int), typemin(Int):typemax(Int)), (4,8), JA.EMPTY)
    keys = ((0,0), (typemin(Int),typemax(Int)))
    central = JA.store_reserve!(map, first(keys), JA.store_hash(map, first(keys)))
    JA.slot_publish!(central, MapTestEntry(first(keys)))
    JA.store_retain!(map, central)
    @test (map.roots.reference.mem[1]) === central.leaf
    distant = JA.store_reserve!(map, last(keys), JA.store_hash(map, last(keys)))
    JA.slot_publish!(distant, MapTestEntry(last(keys)))
    JA.store_retain!(map, distant)
    @test mapslot(map, first(keys)).leaf === central.leaf
    middle = JA.store_reserve!(map, (4,0), JA.store_hash(map, (4,0)))
    JA.slot_publish!(middle, MapTestEntry((4,0)))
    JA.store_retain!(map, middle)
    @test treecounts(map) == (2, 3)
    JA.store_delete!(map, (4,0), middle)
    @test treecounts(map) == (1, 2)
    @test mapslot(map, last(keys)).leaf === distant.leaf
    JA.store_delete!(map, last(keys), distant)
    @test (map.roots.reference.mem[1]) === central.leaf
    @test JA.slot_entry(mapslot(map, first(keys))).key == first(keys)
    treecounts(map)
end

@testset "Depth shortcuts share the radix tree" begin
    JA = JITArrays
    map = JA.SubtreeRadixMap(Union{JA.Empty,MapTestEntry{2}},
        (typemin(Int):typemax(Int), typemin(Int):typemax(Int)), (8,8), JA.EMPTY)
    keys = ((3,0), (-4,0), (4,0), (31,0), (32,0), (0,32))
    for key in keys
        slot = JA.store_reserve!(map, key, JA.tree_key(key))
        JA.slot_publish!(slot, MapTestEntry(key))
        JA.store_retain!(map, slot)
    end
    @test length(map.roots.reference.mem) == cld(8sizeof(UInt), 3)
    @test map.roots.reference.mem[1] === mapslot(map, (3,0)).leaf === mapslot(map, (-4,0)).leaf
    @test map.roots.reference.mem[2] isa JA.TreeBranch
    @test map.roots.reference.mem[3] isa JA.TreeBranch
    smallroots = Tuple(map.roots.reference.mem[1:3])
    distantkey = (typemin(Int), typemax(Int))
    distant = JA.store_reserve!(map, distantkey, JA.tree_key(distantkey))
    JA.slot_publish!(distant, MapTestEntry(distantkey))
    JA.store_retain!(map, distant)
    @test map.roots.reference.mem[end] isa JA.TreeBranch
    @test JA.node_level(map.roots.reference.mem[end]) == length(map.roots.reference.mem)-1
    @test all(i -> map.roots.reference.mem[i] === smallroots[i], 1:3)
    for key in ((4,0), (31,0), distantkey)
        JA.delete_key!(map, key)
    end
    @test map.roots.reference.mem[2] === map.roots.reference.mem[1]
    @test map.roots.reference.mem[end] === map.roots.reference.mem[3]
    @test all(key -> JA.slot_entry(mapslot(map, key)).key == key, ((3,0), (-4,0), (32,0), (0,32)))
    @test mapslot(map, (4,0)) === nothing || JA.slot_entry(mapslot(map, (4,0))) isa JA.Empty
    @test all(node -> node === map.roots.empty || JA.node_level(node) <= findfirst(==(node), map.roots.reference.mem)-1, map.roots.reference.mem)
    @test treecounts(map)[2] == 3
    treecounts(map)
end

@testset "Dictionary stable records, collisions, and tombstones" begin
    JA = JITArrays
    map = JA.DictMap(Union{JA.Empty,MapTestEntry{1}}, Val(1), JA.EMPTY)
    # Signed probe arithmetic preserves high hash bits and wraps at the table edge.
    probe = JA.DictMap(Union{JA.Empty,MapTestEntry{1}}, Val(1), JA.EMPTY)
    for (key, hashed) in (((1,), typemax(UInt)), ((2,), typemax(UInt)), ((3,), UInt(1) << (8sizeof(UInt)-1)))
        slot = JA.store_reserve!(probe, key, hashed)
        @test JA.store_lookup(probe, key, hashed) === slot
    end
    keys = filter(i -> hash((i,)) & UInt(63) == 0, -100000:100000)[1:60]
    slots = [JA.store_reserve!(map, (i,), hash((i,))) for i in keys[1:48]]
    old = map.table
    @test length(old) == 64
    # At the load limit, replacing a deleted slot must not rebuild the table.
    JA.store_delete!(map, (keys[1],), slots[1])
    slots[1] = JA.store_reserve!(map, (keys[1],), hash((keys[1],)))
    @test map.table === old
    @test map.count == map.used == 48
    for i in keys[49:60]
        push!(slots, JA.store_reserve!(map, (i,), hash((i,))))
    end
    @test length((map.table)) == 256
    @test all(i -> mapslot(map, (keys[i],)) === slots[i], eachindex(keys))
    @test first(JA.hash_find(old, (keys[1],), hash((keys[1],)))) === slots[1]
    for (i, slot) in zip(keys, slots)
        JA.slot_publish!(slot, MapTestEntry((i,)))
        captured = JA.slot_entry(slot)
        JA.store_delete!(map, (i,), slot)
        @test JA.slot_entry(slot) === JA.EMPTY
        @test captured.key == (i,)
        @test mapslot(map, (i,)) === nothing
    end
    @test map.count == 0
    current = map.table
    for i in 1:512
        slot = JA.store_reserve!(map, (i,), hash((i,)))
        JA.store_delete!(map, (i,), slot)
    end
    @test (map.table) !== current
    @test length((map.table)) == 256
end
