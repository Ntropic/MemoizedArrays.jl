# Map-specific routing, collisions, growth and pruning regressions.
function treecounts(node)
    if node === nothing
        return (0, 0, true)
    elseif node isa JA.TreeLeaf
        entries = count(i -> !((node.values[i]) isa JA.Empty), eachindex(node.values))
        return (0, 1, node.used == entries)
    end
    branches, leaves, used, occupied = 1, 0, 0, 0
    valid = true
    for i in eachindex(node.children)
        child = node.children[i]
        if child !== nothing
            used += 1
            occupied ⊻= i
            b, l, child_valid = treecounts(child)
            branches += b
            leaves += l
            valid &= child_valid
        end
    end
    valid &= node.used == used
    valid &= node.occupied == occupied
    return branches, leaves, valid
end
function treecounts(map::JA.RadixMap)
    branches, leaves, valid = 0, 0, true
    for root in map.roots
        b, l, rootvalid = treecounts(root)
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
    for Map in (JA.MSBRadixMap, JA.LSBRadixMap),
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
            @test found.address == slot.address
            @test JA.slot_entry(found).key == key
        end
        treecounts(map)
        for (key, slot) in zip(keys, slots)
            captured = JA.slot_entry(slot)
            JA.store_delete!(map, key, slot)
            @test JA.slot_entry(slot) === JA.EMPTY
            @test captured.key == key
        end
        @test all(isnothing, map.roots)
    end
    for Map in (JA.MSBRadixMap, JA.LSBRadixMap)
        map = Map(Union{JA.Empty,MapTestEntry{0}}, (), (), JA.EMPTY)
        slot = JA.store_reserve!(map, (), ())
        @test slot.address == 1
        JA.slot_publish!(slot, MapTestEntry(()))
        JA.store_retain!(map, slot)
        @test JA.slot_entry(mapslot(map, ())).key == ()
        JA.store_delete!(map, (), slot)
        @test JA.slot_entry(slot) === JA.EMPTY
    end
end

@testset "MSB root prefix compression" begin
    JA = JITArrays
    map = JA.MSBRadixMap(Union{JA.Empty,MapTestEntry{2}},
        (typemin(Int):typemax(Int), typemin(Int):typemax(Int)), (4,8), JA.EMPTY)
    keys = ((0,0), (typemin(Int),typemax(Int)))
    central = JA.store_reserve!(map, first(keys), JA.tree_key(first(keys)))
    JA.slot_publish!(central, MapTestEntry(first(keys)))
    JA.store_retain!(map, central)
    @test (map.roots[1]) === central.leaf
    distant = JA.store_reserve!(map, last(keys), JA.tree_key(last(keys)))
    JA.slot_publish!(distant, MapTestEntry(last(keys)))
    JA.store_retain!(map, distant)
    @test mapslot(map, first(keys)).leaf === central.leaf
    middle = JA.store_reserve!(map, (4,0), JA.tree_key((4,0)))
    JA.slot_publish!(middle, MapTestEntry((4,0)))
    JA.store_retain!(map, middle)
    @test treecounts(map) == (2, 3)
    JA.store_delete!(map, (4,0), middle)
    @test treecounts(map) == (1, 2)
    @test mapslot(map, last(keys)).leaf === distant.leaf
    JA.store_delete!(map, last(keys), distant)
    @test (map.roots[1]) === central.leaf
    @test JA.slot_entry(mapslot(map, first(keys))).key == first(keys)
    treecounts(map)
end

@testset "Depth roots preserve small-coordinate trees" begin
    JA = JITArrays
    map = JA.LSBRadixMap(Union{JA.Empty,MapTestEntry{2}},
        (typemin(Int):typemax(Int), typemin(Int):typemax(Int)), (8,8), JA.EMPTY)
    keys = ((3,0), (-4,0), (4,0), (31,0), (32,0), (0,32))
    for key in keys
        slot = JA.store_reserve!(map, key, JA.tree_key(key))
        JA.slot_publish!(slot, MapTestEntry(key))
        JA.store_retain!(map, slot)
    end
    @test length(map.roots) == cld(8sizeof(UInt), 3)
    @test map.roots[1] === mapslot(map, (3,0)).leaf === mapslot(map, (-4,0)).leaf
    @test map.roots[2] isa JA.TreeBranch
    @test map.roots[3] isa JA.TreeBranch
    smallroots = Tuple(map.roots[1:3])
    distantkey = (typemin(Int), typemax(Int))
    distant = JA.store_reserve!(map, distantkey, JA.tree_key(distantkey))
    JA.slot_publish!(distant, MapTestEntry(distantkey))
    JA.store_retain!(map, distant)
    @test map.roots[end] === distant.leaf
    @test all(i -> map.roots[i] === smallroots[i], 1:3)
    for key in ((4,0), (31,0), distantkey)
        JA.delete_key!(map, key)
    end
    @test map.roots[2] === map.roots[end] === nothing
    @test all(key -> JA.slot_entry(mapslot(map, key)).key == key, ((3,0), (-4,0), (32,0), (0,32)))
    treecounts(map)
end

@testset "Dictionary stable records, collisions, and tombstones" begin
    JA = JITArrays
    map = JA.DictMap(Union{JA.Empty,MapTestEntry{1}}, Val(1), JA.EMPTY)
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
