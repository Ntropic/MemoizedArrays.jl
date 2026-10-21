test_map_config(kind, shape=(8,); center=0.5) =
    kind === :array ? ((V, domain) -> ArrayMap{V}(domain)) : kind === :dict ? ((V, domain) -> DictMap{V}(domain)) :
    kind === :radix ? ((V, domain) -> RadixMap{V}(domain; block_shape=shape, center)) : ((V, domain) -> SubtreeRadixMap{V}(domain; block_shape=shape, center))

function behavior_cache(C, Map, T, f, domain; concurrent=false, capacity=2)
    config = Map === ExpandingArrayMap ? Map{T}(domain; increment=2) :
        Map in (RadixMap, SubtreeRadixMap) ? Map{T}(domain; block_shape=ntuple(_ -> 4, length(domain))) : Map{T}(domain)
    options = C === LRUMemoizedArray ? (; capacity, queues=1) : (;)
    cache = C(f, config; concurrent, options...)
    return cache, concurrent && C === LRUMemoizedArray ? queue_handle(cache, 1) : cache
end

pinned_configs(T) = (ArrayMap{T}((0:128,)), ExpandingArrayMap{T}((0:3,); increment=8),
    RadixMap{T}((0:128,); block_shape=(8,)), SubtreeRadixMap{T}((0:128,); block_shape=(8,)),
    DictMap{T}((0:128,)))

function test_result(f)
    try
        return f()
    catch error
        return error
    end
end

function concurrent_test_cache(C, V, generator, domain; capacity=8, queues=8, map_factory=((V, domain) -> ArrayMap{V}(domain)))
    if C === MemoizedArray
        return C(generator, map_factory(V, domain); concurrent=true)
    end
    return C(generator, map_factory(V, domain); capacity, queues, concurrent=true)
end

concurrent_test_handle(cache::MemoizedArray, _) = cache
concurrent_test_handle(cache::LRUMemoizedArray, i) = queue_handle(cache, i)

# Warm ready reads before measuring; request probes also cover misses and eviction.
request_allocations(cache, key) = @allocated cache[key]
request_allocations(cache, key::Tuple) = @allocated cache[key...]
function read_allocations(cache, key::Tuple=ntuple(_ -> 1, ndims(cache)))
    cache[key...]
    return @allocated cache[key...]
end

struct TestConversion{F}
    convert::F
end
Base.convert(::Type{Int}, value::TestConversion) = value.convert()

# Pause only the selected key; other requests can grow or evict around it.
function paused_generator(f, key; outcome=:value, value=f(key...))
    entered, release = Channel{Nothing}(2), Base.Event()
    calls = Threads.Atomic{Int}(0)
    failure = ErrorException("paused generation: $outcome")
    generator = (indices...) -> begin
        indices == key || return f(indices...)
        Threads.atomic_add!(calls, 1)
        put!(entered, nothing)
        wait(release)
        outcome === :generator_failure && throw(failure)
        outcome === :conversion_failure && return TestConversion(() -> throw(failure))
        return value
    end
    return (; generator, entered, release, calls, failure)
end

# Bind owner[] to the cache or queue handle after constructing it.
function nested_generator(f, key, nested; conversion=false)
    owner, calls = Ref{Any}(), Ref(0)
    generate_nested() = nested(owner[])
    generator = (indices...) -> begin
        calls[] += 1
        indices == key || return f(indices...)
        return conversion ? TestConversion(generate_nested) : generate_nested()
    end
    return (; generator, owner, calls)
end

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


function check_private_queues(cache)
    resident = 0
    for q in cache.eviction.queues
        positions = zeros(UInt32,length(q.positions.mem))
        index,previous = q.oldest,q.oldest
        for _ in 1:q.count
            @test q.links.mem[2Int(index)-1] == previous
            entry = q.entries[index]
            @test positions[entry.id] == 0
            positions[entry.id] = index
            previous,index = index,q.links.mem[2Int(index)]
        end
        @test previous == q.newest
        @test positions == collect(q.positions.mem)
        @test q.count <= q.capacity
        for entry in q.pool
            version = JA.private_state(entry)
            if JA.private_live(version)
                resident += 1
                key = @atomic :acquire entry.key
                @test JA.entry(cache,key) === entry
            else
                @test version & 3 == 2 # no abandoned edit claims
            end
        end
    end
    @test resident == JA.resident_count(cache)
    @test resident == count(key->isready(cache,key),1:length(cache))
    @test resident <= sum(q.capacity for q in cache.eviction.queues)
end


dict_test_collisions() = filter(i -> hash((i,)) & UInt(63) == 0 &&
    (hash((i,)) >> (4sizeof(UInt))) & UInt(1) == 0, 1:10000)[1:4]
