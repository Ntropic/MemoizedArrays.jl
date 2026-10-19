struct TreeLRUConversion{F}
    convert::F
end
Base.convert(::Type{Int}, value::TreeLRUConversion) = value.convert()

tree_lru_result(f) = try
    f()
catch failure
    failure
end
tree_lru_allocations(handle, key) = @allocated handle[key...]
tree_lru_wait(f) = timedwait(f, 5; pollint=0.001)

@testset "Concurrent tree LRU coordinates, ordering and reuse" begin
    for Config in (RadixMap, SubtreeRadixMap), capacity in (1, 2, 3)
        cache = LRUMemoizedArray((i, j) -> [i, j], Config{Vector{Int}}((-16:16, -16:16); block_shape=(8, 8)); capacity, concurrent=true, queues=1)
        handle = queue_handle(cache, 1)
        @test_throws ArgumentError cache[0, 0]
        @test_throws BoundsError handle[-17, 0]
        @test_throws ArgumentError handle[1]
        saved_value = handle[0, 0]
        saved_entry = JA.entry(cache, JA.cache_key(cache, (0, 0)))
        for i in 1:capacity-1
            @test handle[i, 0] == [i, 0]
        end
        @test handle[0, 0] === saved_value # zero becomes newest
        @test handle[capacity, 0] == [capacity, 0]
        victim = capacity == 1 ? 0 : 1
        @test !isready(cache, victim, 0)
        @test JA.resident_count(cache) == capacity
        @test saved_value == [0, 0] # recycling does not mutate returned values
        if capacity == 1
            @test JA.entry(cache, JA.cache_key(cache, (1, 0))) === saved_entry
        else
            @test isready(cache, 0, 0)
        end
        @test (@inferred handle[-1, -1]) == [-1, -1]
        @test handle[CartesianIndex(-1, -1)] == [-1, -1]
    end
end

@testset "Different generators overlap within one tree leaf" begin
    for Config in (RadixMap, SubtreeRadixMap)
        entered, release = Channel{Nothing}(2), Base.Event()
        cache = LRUMemoizedArray((i, j) -> begin
            put!(entered, nothing)
            wait(release)
            i + j
        end, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=2, concurrent=true, queues=2)
        handles = [queue_handle(cache, i) for i in 1:2]
        requests = [Threads.@spawn tree_lru_result(() -> handles[i][i-1, 0]) for i in 1:2]
        try
            for _ in 1:2
                arrived = tree_lru_wait(() -> isready(entered))
                @test arrived === :ok
                arrived === :ok || break
                take!(entered)
            end
            @test !isready(cache, 0, 0) && !isready(cache, 1, 0)
        finally
            notify(release)
        end
        @test fetch.(requests) == [0, 1]
        @test JA.resident_count(cache) == 2
    end
end

@testset "Pending generation keeps an otherwise empty leaf attached" begin
    for Config in (RadixMap, SubtreeRadixMap)
        entered, release = Channel{Nothing}(1), Base.Event()
        cache = LRUMemoizedArray((i, j) -> begin
            if (i, j) == (9, 0)
                put!(entered, nothing)
                wait(release)
            end
            i + j
        end, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=2, concurrent=true, queues=2)
        a, b = queue_handle(cache, 1), queue_handle(cache, 2)
        @test a[8, 0] == 8
        stale = JA.map_lookup(cache, (8, 1))
        leaf = stale[4]
        request = Threads.@spawn tree_lru_result(() -> b[9, 0])
        try
            started = tree_lru_wait(() -> isready(entered))
            @test started === :ok
            if started === :ok
                @test a[64, 0] == 64 # evicts the last completed value in the pending leaf
                @test !isready(cache, 8, 0)
                @test (@atomic :acquire leaf.retention.count) == 1
                pending = JA.map_lookup(cache, (9, 0))[3]
                @test pending !== nothing
                if pending !== nothing
                    @test JA.slot_entry(pending) === JA.GENERATING
                end
            end
        finally
            notify(release)
        end
        @test fetch(request) === 9
        @test isready(cache, 9, 0)
        @test b[128, 0] == 128 # no values or pending requests remain in the leaf
        @test (@atomic :acquire leaf.retention.count) == typemin(Int)
        reservations = [Threads.@spawn begin
            valid = true
            for _ in 1:64
                valid &= !JA.reserve_leaf!(leaf)
                valid &= (@atomic :acquire leaf.retention.count) < 0
            end
            valid
        end for _ in 1:Threads.nthreads(:default)]
        @test all(fetch.(reservations))
        @test (@atomic :acquire leaf.retention.count) == typemin(Int)
        @test JA.map_lookup(cache, (9, 0))[3] === nothing
        @test JA.cache_get(a, stale, (8, 1)) == 9 # redirect the detached destination
        @test isready(cache, 8, 1)
        @test JA.resident_count(cache) == 2
    end
end

@testset "Reserving an empty tree leaf prevents retirement" begin
    for Config in (RadixMap, SubtreeRadixMap)
        cache = LRUMemoizedArray(+, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=1, concurrent=true, queues=1)
        key, routing, _, _ = JA.map_lookup(cache, (8, 0))
        leaf = JA.map_reserve!(cache.map, key, routing, Val(true)).leaf
        @test (@atomic :acquire leaf.retention.count) == 0
        @test JA.reserve_leaf!(leaf)
        fetch(Threads.@spawn JA.retire_leaf!(cache.map, leaf))
        @test (@atomic :acquire leaf.retention.count) == 1
        @test JA.map_lookup(cache, (8, 0))[4] === leaf
        JA.release_leaf!(cache.map, leaf)
        @test (@atomic :acquire leaf.retention.count) == typemin(Int)
        @test JA.map_lookup(cache, (8, 0))[4] === nothing
    end
end

@testset "Subtree central leaf survives growth and value eviction" begin
    for center in (-0.5, 0.5), kind in (:memoized, :lru)
        map_spec = ((V, domain) -> SubtreeRadixMap{V}(domain; block_shape=(4, 8), center))
        calls = Ref(0)
        generator = (i, j) -> (calls[] += 1; (i, j))
        cache = kind === :memoized ?
            MemoizedArray(generator, map_spec(Tuple{Int,Int}, unbound(2)); concurrent=true) :
            LRUMemoizedArray(generator, map_spec(Tuple{Int,Int}, unbound(2)); capacity=1, concurrent=true, queues=1)
        handle = kind === :memoized ? cache : queue_handle(cache, 1)
        central = JA.map_lookup(cache, (0, 0))[4]
        if kind === :memoized
            central = JA.slot_entry(memoryref(cache.map.roots, 1))
        end
        @test calls[] == 0
        @test central !== cache.map.empty_node
        @test all(i -> JA.slot_entry(memoryref(cache.map.roots, i)) === central,
            eachindex(cache.map.roots.mem))
        for key in ((0, 0), (64, 0), (-64, 128), (typemin(Int), typemax(Int)), (0, 0))
            @test handle[key...] === key
            @test JA.slot_entry(memoryref(cache.map.roots, 1)) === central
        end
        JA.retire_leaf!(cache.map, central)
        @test JA.slot_entry(memoryref(cache.map.roots, 1)) === central
        @test kind === :memoized ? central.retention === nothing :
            (@atomic :acquire central.retention.count) == 0
        @test calls[] == (kind === :memoized ? 4 : 5)
    end
    cache = LRUMemoizedArray(+, SubtreeRadixMap{Int}((0:8, 0:8); block_shape=(4, 8)); capacity=1, concurrent=true, queues=1)
    handle = queue_handle(cache, 1)
    central = JA.map_lookup(cache, (0, 0))[4]
    @test handle[8, 8] == 16
    @test handle[0, 0] == 0
    @test JA.map_lookup(cache, (0, 0))[4] === central
    @test_throws BoundsError handle[-1, 0]
end

@testset "Subtree retirement refreshes shortcuts below surviving branches" begin
    cache = LRUMemoizedArray(+, SubtreeRadixMap{Int}(unbound(2); block_shape=(8, 8)); capacity=4, concurrent=true, queues=4)
    a, b, c, d = (queue_handle(cache, i) for i in 1:4)
    central = JA.map_lookup(cache, (0, 0))[4]
    @test a[8, 0] == 8
    old_leaf = JA.map_lookup(cache, (8, 0))[4]
    @test b[16, 0] == 16
    @test c[64, 0] == 64
    @test d[512, 0] == 512
    @test a[512, 1] == 513 # existing destination; retire only the leaf containing eight
    @test (@atomic :acquire old_leaf.retention.count) == typemin(Int)
    @test JA.slot_entry(memoryref(cache.map.roots, 1)) !== old_leaf
    @test a[8, 0] == 8
    new_leaf = JA.map_lookup(cache, (8, 0))[4]
    @test new_leaf !== old_leaf
    @test JA.slot_entry(memoryref(cache.map.roots, 1)) === central
end

@testset "Existing tree LRU leaf bypasses structural lock" begin
    for Config in (RadixMap, SubtreeRadixMap), cross_leaf in (false, true)
        cache = LRUMemoizedArray(+, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=4, concurrent=true, queues=2)
        a, b = queue_handle(cache, 1), queue_handle(cache, 2)
        @test a[0, 0] == 0
        @test a[1, 0] == 1
        @test b[64, 0] == 64
        @test b[65, 0] == 65
        old_entry = JA.entry(cache, JA.cache_key(cache, (0, 0)))
        central = JA.map_lookup(cache, (0, 0))[4]
        other = JA.map_lookup(cache, (64, 0))[4]
        destination = cross_leaf ? 66 : 2
        lock(cache.map.structural)
        request = Threads.@spawn (a[1, 0], a[destination, 0])
        completed = try
            tree_lru_wait(() -> istaskdone(request))
        finally
            unlock(cache.map.structural)
        end
        @test completed === :ok
        @test fetch(request) === (1, destination)
        @test !isready(cache, 0, 0) && isready(cache, destination, 0)
        @test JA.entry(cache, JA.cache_key(cache, (destination, 0))) === old_entry
        @test JA.map_lookup(cache, (1, 0))[4] === central
        @test JA.map_lookup(cache, (64, 0))[4] === other
        @test (@atomic :acquire central.retention.count) == (Config === SubtreeRadixMap ? 0 : cross_leaf ? 1 : 2)
        @test (@atomic :acquire other.retention.count) == (cross_leaf ? 3 : 2)
        @test JA.resident_count(cache) == 4
    end
end

@testset "Existing tree leaf retains generation and conversion failures" begin
    for Config in (RadixMap, SubtreeRadixMap), conversion in (false, true)
        failure, calls = ErrorException("existing leaf failure"), Ref(0)
        cache = LRUMemoizedArray((i, j) -> begin
            if i == 0
                calls[] += 1
                if conversion
                    return TreeLRUConversion(() -> throw(failure))
                end
                throw(failure)
            end
            i + j
        end, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=1, concurrent=true, queues=1)
        handle = queue_handle(cache, 1)
        @test handle[1, 0] == 1
        @test handle[2, 0] == 2
        @test tree_lru_result(() -> handle[0, 0]) === failure
        @test tree_lru_result(() -> handle[0, 0]) === failure
        @test calls[] == 1
        @test isready(cache, 2, 0) && !isready(cache, 0, 0)
        @test JA.resident_count(cache) == 1
    end
end

@testset "Tree LRU waiters and retained failures survive entry recycling" begin
    for Config in (RadixMap, SubtreeRadixMap), outcome in (:value, :generator_failure, :conversion_failure)
        entered, release = Channel{Nothing}(1), Base.Event()
        calls = Threads.Atomic{Int}(0)
        failure = ErrorException("tree LRU outcome: $outcome")
        cache = LRUMemoizedArray((i, j) -> begin
            if (i, j) == (8, 0)
                Threads.atomic_add!(calls, 1)
                put!(entered, nothing)
                wait(release)
                outcome === :generator_failure && throw(failure)
                outcome === :conversion_failure && return TreeLRUConversion(() -> throw(failure))
            end
            i + j
        end, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=3, concurrent=true, queues=3)
        a, b, c = (queue_handle(cache, i) for i in 1:3)
        # Compile structural and existing-leaf misses before timing coordination.
        a[9, 0]
        a[10, 0]
        # Warm redirection too, then retain an EMPTY slot from a detached leaf.
        stale = JA.map_lookup(cache, (11, 0))
        a[64, 0]
        JA.cache_get(a, stale, (11, 0))
        stale = JA.map_lookup(cache, (8, 0))
        a[64, 0]
        @test (@atomic :acquire stale[4].retention.count) == typemin(Int)
        owner = Threads.@spawn tree_lru_result(() -> JA.cache_get(a, stale, (8, 0)))
        waiter, pending = nothing, nothing
        try
            started = tree_lru_wait(() -> isready(entered))
            @test started === :ok
            if started === :ok
                lookup = JA.map_lookup(cache, (8, 0))
                slot, leaf = lookup[3], lookup[4]
                @test leaf !== stale[4]
                @test (@atomic :acquire leaf.retention.count) == 1
                @test JA.slot_entry(stale[3]) === JA.EMPTY
                waiter = Threads.@spawn tree_lru_result(() -> b[8, 0])
                @test tree_lru_wait(() -> JA.slot_entry(slot) isa JA.GenerateWait) === :ok
                pending = JA.slot_entry(slot)
                @test c[-64, 128] == 64
            end
        finally
            notify(release)
        end
        expected = outcome === :value ? 8 : failure
        @test fetch(owner) === expected
        @test a[9, 0] == 9 # recycle the successful owner's entry immediately
        if waiter !== nothing
            @test fetch(waiter) === expected
        end
        if pending isa JA.GenerateWait
            @test tree_lru_result(() -> JA.wait_generated(Int, pending.reference)) === expected
        end
        @test calls[] == 1
        if outcome !== :value
            @test tree_lru_result(() -> a[8, 0]) === failure
            @test !isready(cache, 8, 0)
            @test calls[] == 1
            @test a[64, 0] == 64 # remove the last ready value from the failure's leaf
            leaf = JA.map_lookup(cache, (8, 0))[4]
            @test leaf !== nothing
            if leaf !== nothing
                @test (@atomic :acquire leaf.retention.count) == 1
            end
        end
    end
end

@testset "Tree LRU nested generation and conversion with capacity one" begin
    for Config in (RadixMap, SubtreeRadixMap), conversion in (false, true), nested in ((1, 0), (64, -128))
        owner, calls = Ref{Any}(), Ref(0)
        inner() = (owner[][nested...]; 17)
        cache = LRUMemoizedArray((i, j) -> begin
            calls[] += 1
            (i, j) == (0, 0) && return conversion ? TreeLRUConversion(inner) : inner()
            i + j
        end, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=1, concurrent=true, queues=1)
        handle = queue_handle(cache, 1)
        owner[] = handle
        @test handle[0, 0] == 17
        @test handle[0, 0] == 17
        @test calls[] == 2
        @test isready(cache, 0, 0) && !isready(cache, nested)
        @test JA.resident_count(cache) == 1
    end
end

@testset "Concurrent tree LRU matrices do not allocate within an existing leaf" begin
    value = ones(4, 4)
    for Config in (RadixMap, SubtreeRadixMap)
        cache = LRUMemoizedArray(Returns(value), Config{typeof(value)}(unbound(2); block_shape=(8, 8)); capacity=3, concurrent=true, queues=1)
        handle = queue_handle(cache, 1)
        tree_lru_allocations(handle, (0, 0))
        tree_lru_allocations(handle, (1, 0)) # warm the existing-leaf miss separately
        @test tree_lru_allocations(handle, (2, 0)) == 0
        tree_lru_allocations(handle, (3, 0))
        @test tree_lru_allocations(handle, (-1, 0)) == 0
        @test tree_lru_allocations(handle, (-1, 0)) == 0
        @test handle[-1, 0] === value
    end
end

@testset "Concurrent tree LRU signed requests and structural eviction" begin
    for Config in (RadixMap, SubtreeRadixMap), center in (-0.5, 0.5)
        cache = LRUMemoizedArray(tuple, Config{Tuple{Int,Int}}(unbound(2); block_shape=(4, 8), center); capacity=16, concurrent=true, queues=4)
        handles = [queue_handle(cache, i) for i in 1:4]
        requests = [Threads.@spawn begin
            valid = true
            for step in 1:64
                key = (32i + 256step, -64i - 512step)
                valid &= handles[i][key...] === key
                valid &= handles[i][key...] === key
            end
            valid
        end for i in 1:4]
        @test all(fetch.(requests))
        @test JA.resident_count(cache) == 16
        @test all(i -> isready(cache, 32i + 256*64, -64i - 512*64), 1:4)
        nodes = Any[JA.slot_entry(JA.root_link(cache.map))]
        counted = 0
        valid_branches = true
        valid_routes = true
        while !isempty(nodes)
            node = pop!(nodes)
            node === cache.map.empty_node && continue
            if node isa JA.ConcurrentTreeLeaf
                present = count(index -> JA.slot_entry(memoryref(node.values, index)) !== JA.EMPTY,
                    1:prod(JA.block_shape(cache.map)))
                if JA.permanent_leaf(cache.map, node)
                    @test (@atomic :acquire node.retention.count) == 0
                else
                    @test (@atomic :acquire node.retention.count) == present > 0
                end
                counted += present
                for index in 1:prod(JA.block_shape(cache.map))
                    stored = JA.slot_entry(memoryref(node.values, index))
                    if stored isa JA.SharedLRUEntry
                        valid_routes &= JA.entry(cache, (@atomic :acquire stored.key)) === stored
                    end
                end
            else
                positions = Int[]
                for index in eachindex(node.children.mem)
                    child = JA.slot_entry(memoryref(node.children, index))
                    if child !== cache.map.empty_node
                        push!(positions, index)
                        push!(nodes, child)
                    end
                end
                valid_branches &= (@atomic :acquire node.used) == length(positions) >= 2
                valid_branches &= (@atomic :acquire node.occupied) == foldl(xor, positions; init=0)
            end
        end
        @test counted == 16
        @test valid_branches
        @test valid_routes
    end
end

# Constructor boundary regression
@testset "Concurrent tree LRU scalar, extreme and encoded values" begin
    for Config in (RadixMap, SubtreeRadixMap)
        scalar = LRUMemoizedArray(() -> 7, Config{Int}((); block_shape=()); capacity=1, concurrent=true, queues=1)
        handle = queue_handle(scalar, 1)
        @test (@inferred handle[]) == 7
        @test handle[] == 7 && isready(scalar)
        @test size(scalar) == () && JA.resident_count(scalar) == 1
        @test_throws ArgumentError scalar[]

        extremes = LRUMemoizedArray(identity, Config{Int}(unbound(1); block_shape=(4,)); capacity=1, concurrent=true, queues=1)
        handle = queue_handle(extremes, 1)
        for key in (typemin(Int), typemax(Int), 0, typemin(Int))
            @test (@inferred handle[key]) == key
            @test isready(extremes, key) && JA.resident_count(extremes) == 1
        end
        @test !isready(extremes, 0) && !isready(extremes, typemax(Int))

        values = (nothing, false, JA.EMPTY, JA.GENERATING, JA.GenerationFailure(ErrorException("stored value")))
        broad = LRUMemoizedArray(i -> values[mod1(i, length(values))], Config{Any}(unbound(1); block_shape=(4,)); capacity=2, concurrent=true, queues=1)
        handle = queue_handle(broad, 1)
        for key in 1:2length(values)
            @test handle[key] === values[mod1(key, length(values))]
            @test isready(broad, key)
        end
        @test JA.resident_count(broad) == 2
        @test !isready(broad, 1)
    end
end
