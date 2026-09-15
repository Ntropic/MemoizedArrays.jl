mutable struct MapTestEntry{N}
    key::NTuple{N,Int}
end

function map_test_slot(map, key)
    return JITArrays.store_lookup(map, key, JITArrays.store_hash(map, key))
end

@testset "Radix map routing and bounded reclamation" begin
    JA = JITArrays
    domain = (typemin(Int):typemax(Int), typemin(Int):typemax(Int))
    keys = unique(vcat([(i,j) for i in -4:4 for j in -4:4],
        [(0,typemin(Int)), (typemax(Int),0), (typemin(Int),typemax(Int)), (typemax(Int),typemin(Int))],
        [(reinterpret(Int, UInt(k) * 0x9e3779b97f4a7c15), k % 3 == 0 ? 0 : k) for k in 1:60]))
    for Map in (JA.MSBRadixMap, JA.LSBRadixMap), bounded in (true, false),
        block in ((2,8), (8,2), (4,8))
        map = Map(Union{JA.Empty,MapTestEntry{2}}, domain, block, JA._EMPTY)
        slots = [JA.store_reserve!(map, key, JA.store_hash(map, key), Val(bounded)) for key in keys]
        for (key, slot) in zip(keys, slots)
            @test JA.slot_claim!(slot, JA._EMPTY, MapTestEntry(key)).success
            if bounded
                JA.store_retain!(map, slot)
            end
        end
        for (key, slot) in zip(keys, slots)
            found = map_test_slot(map, key)
            @test found.leaf === slot.leaf
            @test found.address == slot.address
            @test JA.slot_entry(found).key == key
        end
        root = map isa JA.MSBRadixMap ? (@atomic :acquire map.roots[1]) : map.root
        sparse_tree_counts(map; bounded)
        if bounded
            for (key, slot) in zip(keys, slots)
                captured = JA.slot_entry(slot)
                JA.store_delete!(map, key, slot)
                @test JA.slot_entry(slot) === JA._EMPTY
                @test captured.key == key
            end
            if map isa JA.MSBRadixMap
                @test (@atomic :acquire map.roots[1]) === nothing
            else
                @test map.root === root
                @test map.root.used == map.root.children_used == 0
                @test all(i -> (@atomic :acquire root.children[i]) === nothing, eachindex(root.children))
            end
        end
    end
    for Map in (JA.MSBRadixMap, JA.LSBRadixMap)
        map = Map(Union{JA.Empty,MapTestEntry{0}}, (), (), JA._EMPTY)
        slot = JA.store_reserve!(map, (), ())
        @test slot.address == 1
        JA.slot_publish!(slot, MapTestEntry(()))
        JA.store_retain!(map, slot)
        @test JA.slot_entry(map_test_slot(map, ())).key == ()
        JA.store_delete!(map, (), slot)
        @test JA.slot_entry(slot) === JA._EMPTY
    end
end

@testset "MSB root prefix compression" begin
    JA = JITArrays
    map = JA.MSBRadixMap(Union{JA.Empty,MapTestEntry{2}},
        (typemin(Int):typemax(Int), typemin(Int):typemax(Int)), (4,8), JA._EMPTY)
    keys = ((0,0), (typemin(Int),typemax(Int)))
    central = JA.store_reserve!(map, first(keys), JA.tree_key(first(keys)))
    JA.slot_publish!(central, MapTestEntry(first(keys)))
    JA.store_retain!(map, central)
    @test (@atomic :acquire map.roots[1]) === central.leaf
    distant = JA.store_reserve!(map, last(keys), JA.tree_key(last(keys)))
    JA.slot_publish!(distant, MapTestEntry(last(keys)))
    JA.store_retain!(map, distant)
    @test map_test_slot(map, first(keys)).leaf === central.leaf
    JA.store_delete!(map, last(keys), distant)
    @test (@atomic :acquire map.roots[1]) === central.leaf
    @test JA.slot_entry(map_test_slot(map, first(keys))).key == first(keys)
    sparse_tree_counts(map; bounded=true)
end

@testset "Least-significant routing preserves central paths" begin
    JA = JITArrays
    map = JA.LSBRadixMap(Union{JA.Empty,MapTestEntry{2}},
        (typemin(Int):typemax(Int), typemin(Int):typemax(Int)), (4,8), JA._EMPTY)
    central = JA.store_reserve!(map, (0,0), JA.tree_key((0,0)), Val(false))
    nearby = JA.store_reserve!(map, (1,4), JA.tree_key((1,4)), Val(false))
    @test central.leaf === map.root
    @test nearby.leaf === (@atomic :acquire map.root.children[3])
    JA.store_reserve!(map, (typemin(Int),typemax(Int)), JA.tree_key((typemin(Int),typemax(Int))), Val(false))
    @test map_test_slot(map, (0,0)).leaf === central.leaf
    @test map_test_slot(map, (1,4)).leaf === nearby.leaf
end

@testset "Concurrent child publication and entry claims" begin
    JA = JITArrays
    domain = (typemin(Int):typemax(Int), typemin(Int):typemax(Int))
    keys = [(i <= 24 ? i : reinterpret(Int, UInt(i) * 0x9e3779b97f4a7c15), i % 5 == 0 ? 0 : i)
        for i in 1:48]
    for Map in (JA.MSBRadixMap, JA.LSBRadixMap), round in 1:8
        map = Map(Union{JA.Empty,MapTestEntry{2}}, domain, (2,8), JA._EMPTY)
        claimed = [Threads.Atomic{Int}(0) for _ in keys]
        tasks = [Threads.@spawn begin
            for k in eachindex(keys)
                i = mod1(k + offset, length(keys))
                key = keys[i]
                slot = JA.store_reserve!(map, key, JA.tree_key(key), Val(false))
                if JA.slot_claim!(slot, JA._EMPTY, MapTestEntry(key)).success
                    Threads.atomic_add!(claimed[i], 1)
                end
            end
        end for offset in (0,0,1,7,24,47,0,24)]
        foreach(fetch, tasks)
        @test all(counter -> counter[] == 1, claimed)
        @test all(key -> JA.slot_entry(map_test_slot(map, key)).key == key, keys)
        sparse_tree_counts(map)
    end
end

@testset "Dictionary stable records, collisions, and tombstones" begin
    JA = JITArrays
    map = JA.DictMap(Union{JA.Empty,MapTestEntry{1}}, Val(1), JA._EMPTY)
    keys = filter(i -> hash((i,)) & UInt(63) == 0, -10000:10000)[1:60]
    slots = [JA.store_reserve!(map, (i,), hash((i,))) for i in keys[1:48]]
    old = @atomic :acquire map.table
    @test length(old.slots) == 64
    for i in keys[49:60]
        push!(slots, JA.store_reserve!(map, (i,), hash((i,))))
    end
    @test length((@atomic :acquire map.table).slots) == 128
    @test all(i -> map_test_slot(map, (keys[i],)) === slots[i], eachindex(keys))
    @test first(JA.hash_find(old, (keys[1],), hash((keys[1],)))) === slots[1]
    for (i, slot) in zip(keys, slots)
        JA.slot_publish!(slot, MapTestEntry((i,)))
        captured = JA.slot_entry(slot)
        JA.store_delete!(map, (i,), slot)
        @test JA.slot_entry(slot) === JA._EMPTY
        @test captured.key == (i,)
        @test map_test_slot(map, (i,)) === nothing
    end
    @test map.count == 0
    current = @atomic :acquire map.table
    for i in 1:1000
        slot = JA.store_reserve!(map, (i,), hash((i,)))
        JA.store_delete!(map, (i,), slot)
    end
    @test (@atomic :acquire map.table) !== current
    @test length((@atomic :acquire map.table).slots) == 128
end
