# Each concurrent LRU handle has one active caller.
@testset "One generation for concurrent callers" begin
    for map_factory in (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain))), C in (MemoizedArray, LRUMemoizedArray), fail in (false, true)
        value = [11, 12]
        (; generator, entered, release, calls, failure) =
            paused_generator(Returns(value), (1,); outcome=fail ? :generator_failure : :value)
        cache = concurrent_test_cache(C, Vector{Int}, generator, (8,); map_factory)
        handles = [concurrent_test_handle(cache, i) for i in 1:4]
        owner = Threads.@spawn test_result(() -> handles[1][1])
        take!(entered)
        started = Channel{Nothing}(3)
        waiters = [Threads.@spawn begin
            put!(started, nothing)
            test_result(() -> handles[i][1])
        end for i in 2:4]
        for _ in waiters
            take!(started)
        end
        for _ in 1:8
            yield()
        end
        @test !isready(cache, 1)
        GC.gc(false)
        notify(release)
        results = fetch.([owner; waiters])
        @test all(result -> result === (fail ? failure : value), results)
        @test calls[] == 1
        @test isready(cache, 1) == !fail
        @test test_result(() -> handles[1][1]) === (fail ? failure : value)
        @test calls[] == 1
    end
end

@testset "Independent generators and nested requests" begin
    for map_factory in (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain))), C in (MemoizedArray, LRUMemoizedArray)
        entered, release = Base.Event(), Base.Event()
        cache = concurrent_test_cache(C, Int, i -> begin
            if i == 1
                notify(entered)
                wait(release)
            end
            i
        end, (8,); map_factory)
        first = concurrent_test_handle(cache, 1)
        second = concurrent_test_handle(cache, 2)
        slow = Threads.@spawn first[1]
        wait(entered)
        fast = Threads.@spawn second[2]
        completed = timedwait(() -> istaskdone(fast), 5; pollint=0.001)
        notify(release)
        @test completed === :ok
        @test fetch(fast) == 2 && fetch(slow) == 1

        dependent = concurrent_test_cache(C, Int, i -> i == 1 ? handle[2] + 10 : i, (4,); map_factory)
        handle = concurrent_test_handle(dependent, 1)
        @test handle[1] == 12
        @test handle[2] == 2
    end
end

@testset "Concurrent generation and pooled entry replacement" begin
    for map_factory in (((V, domain) -> ArrayMap{V}(domain)), ((V, domain) -> DictMap{V}(domain))), C in (MemoizedArray, LRUMemoizedArray)
        calls = Threads.Atomic{Int}(0)
        cache = concurrent_test_cache(C, Tuple{Int,Int}, i -> begin
            Threads.atomic_add!(calls, 1)
            yield()
            (i, i * i)
        end, (32,); map_factory)
        handles = [concurrent_test_handle(cache, i) for i in 1:8]
        start = Base.Event()
        tasks = [Threads.@spawn begin
            wait(start)
            valid = true
            for step in 1:128
                key = mod(7step + 5i, 32) + 1
                valid &= handles[i][key] === (key, key * key)
                if step % 16 == 0
                    yield()
                end
            end
            valid
        end for i in 1:8]
        notify(start)
        @test all(fetch.(tasks))
        populated = count(key -> isready(cache, key), 1:32)
        if C === MemoizedArray
            @test populated == calls[] == 32
        else
            @test 0 < populated == JA.resident_count(cache) <= 8
        end
        @test all(key -> handles[1][key] == (key, key * key), 1:32)
    end
end

# Capturing the lookup before growth forces the caller to retain an old slot,
# independently of when its task next gets scheduled.
function expanding_captured_waiter(cache, indices, captured)
    return Threads.@spawn begin
        test_result() do
            lookup = try
                JA.map_lookup(cache, indices)
            finally
                put!(captured, nothing)
            end
            JA.cache_get(cache, lookup, indices)
        end
    end
end

@testset "Saved empty slots and retained out-of-domain failures" begin
    calls = Ref(0)
    cache = MemoizedArray((i,j) -> (calls[] += 1; 10i+j), ExpandingArrayMap{Int}((2,2); increment=2); concurrent=true)
    empty_lookup = JA.map_lookup(cache, (1,2))
    @test cache[1,1] == 11
    ready_lookup = JA.map_lookup(cache, (1,1))
    @test cache[3,3] == 33
    @test JA.cache_get(cache, empty_lookup, (1,2)) == 12
    @test JA.cache_get(cache, ready_lookup, (1,1)) == 11
    @test cache[1,2] == 12 && calls[] == 3

    failure = ErrorException("out-of-domain generation failed")
    attempts = Ref(0)
    failing = MemoizedArray(i -> (attempts[] += 1; throw(failure)), ExpandingArrayMap{Int}((1,); increment=2); concurrent=true)
    @test test_result(() -> failing[2]) === failure
    @test size(failing) == (3,)
    @test test_result(() -> failing[2]) === failure
    @test attempts[] == 1 && !isready(failing,2)
end

@testset "Generation and old waiters survive relocation" begin
    for outcome in (:value, :generator_failure, :conversion_failure)
        (; generator, entered, release, calls, failure) =
            paused_generator((i, j) -> 100i + j, (1, 2); outcome, value=12)
        cache = MemoizedArray(generator, ExpandingArrayMap{Int}((2,2); increment=2); concurrent=true)
        @test cache[2,1] == 201
        owner = Threads.@spawn begin
            try
                test_result(() -> cache[1,2])
            finally
                # Constructor/access failures must not leave the test waiting forever.
                put!(entered, nothing)
            end
        end
        take!(entered)
        captured = Channel{Nothing}(3)
        old_waiters = [expanding_captured_waiter(cache, (1,2), captured) for _ in 1:3]
        foreach(_ -> take!(captured), old_waiters)
        new_waiters = Task[]
        try
            @test !isready(cache, 1, 2)
            @test cache[3,1] == 301
            @test size(cache) == (4,2)
            @test cache[2,1] == 201
            # A second relocation moves the pending slot through three allocations.
            @test cache[5,4] == 504
            @test size(cache) == (6,4)
            new_waiters = [Threads.@spawn test_result(() -> cache[1,2]) for _ in 1:3]
            GC.gc(false)
        finally
            notify(release)
        end
        expected = outcome === :value ? 12 : failure
        @test all(result -> result === expected, fetch.([owner; old_waiters; new_waiters]))
        @test calls[] == 1
        @test isready(cache, 1,2) == (outcome === :value)
        @test test_result(() -> cache[1,2]) === expected
        @test calls[] == 1
        @test cache[9,7] == 907
        @test test_result(() -> cache[1,2]) === expected
        @test calls[] == 1
    end
end

@testset "Nested generators and conversions expand safely" begin
    for requested in ((1,1), (2,2))
        owner = Ref{Any}()
        calls = Threads.Atomic{Int}(0)
        cache = MemoizedArray((i,j) -> begin
            Threads.atomic_add!(calls, 1)
            (i,j) == requested && owner[][6,3]
            (i,j)
        end, ExpandingArrayMap{Tuple{Int,Int}}((1,1); increment=4); concurrent=true)
        owner[] = cache
        @test cache[requested...] === requested
        @test size(cache) == (9,5)
        @test cache[requested...] === requested
        @test cache[6,3] === (6,3)
        @test calls[] == 2
    end
    owner = Ref{Any}()
    cache = MemoizedArray((i,j) -> (i,j) == (1,1) ?
        TestConversion(() -> (owner[][6, 3]; 11)) : 63, ExpandingArrayMap{Int}((1,1); increment=4); concurrent=true)
    owner[] = cache
    @test cache[1,1] == 11
    @test size(cache) == (9,5)
    @test cache[1,1] == 11 && cache[6,3] == 63
end

@testset "Parallel independent growth and overlapping requests" begin
    callers = 8
    start = Base.Event()
    entered = Channel{Nothing}(callers)
    release = Base.Event()
    calls = Threads.Atomic{Int}(0)
    cache = MemoizedArray((i,j) -> begin
        Threads.atomic_add!(calls, 1)
        if j == 1 && i <= callers
            put!(entered, nothing)
            wait(release)
        end
        yield()
        (i,j)
    end, ExpandingArrayMap{Tuple{Int,Int}}((1,1); increment=2); concurrent=true)
    tasks = [Threads.@spawn begin
        wait(start)
        result = test_result(() -> cache[i,1])
        result isa Exception && put!(entered, nothing)
        result
    end for i in 1:callers]
    notify(start)
    try
        # All generators must be able to enter before any one of them finishes.
        for _ in 1:callers
            take!(entered)
        end
    finally
        notify(release)
    end
    @test fetch.(tasks) == [(i,1) for i in 1:callers]
    @test calls[] == callers
    start2 = Base.Event()
    readers = [Threads.@spawn begin
        wait(start2)
        valid = true
        for step in 1:256
            i = mod(7step + caller, 32) + 1
            j = mod(11step + 3caller, 17) + 1
            valid &= cache[i,j] === (i,j)
        end
        valid
    end for caller in 1:callers]
    notify(start2)
    @test all(fetch.(readers))
    # Include the initial gated requests when comparing unique coordinates.
    requested = Set((i,1) for i in 1:callers)
    union!(requested, ((mod(7step + caller,32)+1, mod(11step + 3caller,17)+1)
        for caller in 1:callers for step in 1:256))
    @test calls[] == length(requested)
    @test all(key -> cache[key...] === key, requested)
    @test calls[] == length(requested)
end

@testset "Overflow preserves pending generation" begin
    entered, release = Base.Event(), Base.Event()
    lower = typemax(Int) - 2
    cache = MemoizedArray(i -> begin
        notify(entered)
        wait(release)
        i
    end, ExpandingArrayMap{Int}((lower:lower+1,); increment=8); concurrent=true)
    owner = Threads.@spawn begin
        try
            test_result(() -> cache[lower])
        finally
            notify(entered)
        end
    end
    wait(entered)
    try
        @test_throws OverflowError cache[typemax(Int)]
        @test size(cache) == (2,)
        @test !isready(cache, lower)
    finally
        notify(release)
    end
    @test fetch(owner) == lower
    @test cache[lower] == lower
    wide = MemoizedArray(identity, ExpandingArrayMap{Int}((typemin(Int):typemin(Int)+1,); increment=8); concurrent=true)
    @test_throws OverflowError wide[0]
    @test size(wide) == (2,)
    @test wide[typemin(Int)] == typemin(Int)
end

# Concurrent Memoized leaves remain valid when insertion creates new ancestors.

@testset "Tree structure permits insertion peers and excludes pruning" begin
    gate = JA.TreeStructure()
    entered, release = Channel{Nothing}(1), Base.Event()
    inserting = @async JA.tree_inserting(gate) do
        put!(entered, nothing)
        wait(release)
    end
    take!(entered)
    @test JA.tree_inserting(() -> 7, gate) == 7
    pruning = @async lock(gate) do
        @test (@atomic :acquire gate.inserting) == 0
        lock(gate) do
            @test gate.depth == 2
        end
    end
    try
        @test timedwait(() -> (@atomic :acquire gate.pruning), 5) === :ok
        @test !istaskdone(pruning)
    finally
        notify(release)
    end
    wait(inserting)
    wait(pruning)
    @test_throws ErrorException JA.tree_inserting(() -> error("insertion failed"), gate)
    @test (@atomic :acquire gate.inserting) == 0
    @test !(@atomic :acquire gate.pruning)
end

@testset "Concurrent tree completion survives ancestor insertion" begin
    for Config in (RadixMap, SubtreeRadixMap), outcome in (:value, :generator_failure, :conversion_failure)
        (; generator, entered, release, calls, failure) = paused_generator(xor, (0, 0); outcome)
        cache = MemoizedArray(generator, Config{Int}(unbound(2); block_shape=(4, 8)); concurrent=true)
        owner = Threads.@spawn try
            test_result(() -> cache[0, 0])
        finally
            put!(entered, nothing)
        end
        waiter = nothing
        try
            started = timedwait(() -> isready(entered), 5; pollint=0.001)
            @test started === :ok
            if started === :ok
                take!(entered)
                saved = JA.map_lookup(cache, (0, 0))
                @test JA.slot_entry(saved[3]) === JA.GENERATING
                captured = Channel{Nothing}(1)
                waiter = Threads.@spawn begin
                    lookup = JA.map_lookup(cache, (0, 0))
                    put!(captured, nothing)
                    test_result(() -> JA.cache_get(cache, lookup, (0, 0)))
                end
                @test timedwait(() -> isready(captured), 5; pollint=0.001) === :ok
                for key in ((8, 8), (-32, 64), (1024, -8192), (typemin(Int), typemax(Int)))
                    @test cache[key...] == xor(key...)
                end
                @test !isready(cache, 0, 0)
                @test JA.slot_entry(saved[3]) === JA.GENERATING
                GC.gc(false)
            end
        finally
            notify(release)
        end
        expected = outcome === :value ? 0 : failure
        @test fetch(owner) === expected
        if waiter !== nothing
            @test fetch(waiter) === expected
        end
        @test test_result(() -> cache[0, 0]) === expected
        @test isready(cache, 0, 0) == (outcome === :value)
        @test calls[] == 1
    end
end

@testset "Concurrent tree nested generation and conversion" begin
    for Config in (RadixMap, SubtreeRadixMap), conversion in (false, true)
        keys = ((32, -64), (-4096, 8192), (typemin(Int), typemax(Int)))
        (; generator, owner) = nested_generator(xor, (0, 0), cache -> (generate!(cache, keys); 17); conversion)
        cache = MemoizedArray(generator, Config{Int}(unbound(2); block_shape=(4, 8)); concurrent=true)
        owner[] = cache
        @test cache[0, 0] == 17
        @test all(key -> cache[key...] == xor(key...), keys)
        @test cache[1, 1] == 0
    end
end

@testset "Concurrent tree insertion and shortcut refresh" begin
    for Config in (RadixMap, SubtreeRadixMap), center in (-0.5, 0.5)
        calls = Threads.Atomic{Int}(0)
        cache = MemoizedArray((key...) -> begin
            Threads.atomic_add!(calls, 1)
            yield()
            key
        end, Config{Tuple{Int,Int}}(unbound(2); block_shape=(4, 8), center); concurrent=true)
        # Every task first creates a separate branch, then all contend for zero.
        keys = [(w * (1 << shift), -w * (1 << (shift + 1))) for w in 1:8, shift in 3:3:18]
        start = Base.Event()
        tasks = [Threads.@spawn begin
            wait(start)
            valid = true
            for k in axes(keys, 2)
                key = keys[w, k]
                valid &= cache[key...] === key
                valid &= cache[0, 0] === (0, 0)
            end
            valid
        end for w in 1:8]
        notify(start)
        @test all(fetch.(tasks))
        @test all(key -> cache[key...] === key && isready(cache, key), keys)
        @test calls[] == length(unique(vec(keys))) + 1
        for key in ((typemin(Int), typemax(Int)), (typemax(Int), typemin(Int)), (-1, 1), (1, -1))
            @test cache[key...] === key
        end
        @test cache[0, 0] === (0, 0)
    end
end

@testset "Existing tree leaf bypasses structural lock" begin
    for Config in (RadixMap, SubtreeRadixMap)
        cache = MemoizedArray(+, Config{Int}(unbound(2); block_shape=(4, 8)); concurrent=true)
        @test cache[0, 0] == 0
        lock(cache.map.structural)
        request = Threads.@spawn (cache[0, 0], cache[1, 0])
        completed = try
            timedwait(() -> istaskdone(request), 5; pollint=0.001)
        finally
            unlock(cache.map.structural)
        end
        @test completed === :ok
        @test fetch(request) === (0, 1)
    end
end

tree_lru_wait(f) = timedwait(f, 5; pollint=0.001)

@testset "Different generators overlap within one tree leaf" begin
    for Config in (RadixMap, SubtreeRadixMap)
        entered, release = Channel{Nothing}(2), Base.Event()
        cache = LRUMemoizedArray((i, j) -> begin
            put!(entered, nothing)
            wait(release)
            i + j
        end, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=2, concurrent=true, queues=2)
        handles = [queue_handle(cache, i) for i in 1:2]
        requests = [Threads.@spawn test_result(() -> handles[i][i-1, 0]) for i in 1:2]
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
        request = Threads.@spawn test_result(() -> b[9, 0])
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
                    return TestConversion(() -> throw(failure))
                end
                throw(failure)
            end
            i + j
        end, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=1, concurrent=true, queues=1)
        handle = queue_handle(cache, 1)
        @test handle[1, 0] == 1
        @test handle[2, 0] == 2
        @test test_result(() -> handle[0, 0]) === failure
        @test test_result(() -> handle[0, 0]) === failure
        @test calls[] == 1
        @test isready(cache, 2, 0) && !isready(cache, 0, 0)
        @test JA.resident_count(cache) == 1
    end
end

@testset "Tree LRU waiters and retained failures survive entry recycling" begin
    for Config in (RadixMap, SubtreeRadixMap), outcome in (:value, :generator_failure, :conversion_failure)
        (; generator, entered, release, calls, failure) = paused_generator(+, (8, 0); outcome)
        cache = LRUMemoizedArray(generator, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=3, concurrent=true, queues=3)
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
        owner = Threads.@spawn test_result(() -> JA.cache_get(a, stale, (8, 0)))
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
                waiter = Threads.@spawn test_result(() -> b[8, 0])
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
            @test test_result(() -> JA.wait_generated(Int, pending.reference)) === expected
        end
        @test calls[] == 1
        if outcome !== :value
            @test test_result(() -> a[8, 0]) === failure
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
        (; generator, owner, calls) = nested_generator(+, (0, 0), cache -> (cache[nested...]; 17); conversion)
        cache = LRUMemoizedArray(generator, Config{Int}(unbound(2); block_shape=(8, 8)); capacity=1, concurrent=true, queues=1)
        handle = queue_handle(cache, 1)
        owner[] = handle
        @test handle[0, 0] == 17
        @test handle[0, 0] == 17
        @test calls[] == 2
        @test isready(cache, 0, 0) && !isready(cache, nested)
        @test JA.resident_count(cache) == 1
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


@testset "Concurrent expanding LRU relocation and entry reuse" begin
    cache = LRUMemoizedArray((i,j) -> [i,j], ExpandingArrayMap{Vector{Int}}((-1:0,4:5); increment=2); capacity=2, concurrent=true, queues=1)
    handle = queue_handle(cache, 1)
    @test_throws ArgumentError cache[-1,4]
    @test_throws BoundsError handle[-2,4]
    retained = handle[-1,5]
    recycled = JA.entry(cache, JA.cache_key(cache, (-1,5)))
    ready_lookup = JA.map_lookup(cache, (-1,5))
    empty_lookup = JA.map_lookup(cache, (0,5))
    @test handle[0,4] == [0,4]
    @test handle[1,4] == [1,4] # first-axis growth changes every later column's stride
    @test axes(cache) == (-1:2,4:5)
    @test !isready(cache, -1,5)
    @test isready(cache, 0,4)
    @test JA.entry(cache, JA.cache_key(cache, (1,4))) === recycled
    @test (@atomic :acquire recycled.key) == (3,1)
    @test retained == [-1,5]

    # A saved ready slot must not return the value of its recycled entry; an
    # empty slot captured from the old allocation must publish into the new one.
    @test JA.cache_get(handle, ready_lookup, (-1,5)) == [-1,5]
    @test JA.cache_get(handle, empty_lookup, (0,5)) == [0,5]
    @test isready(cache, -1,5) && isready(cache, 0,5)
    @test !isready(cache, 1,4)
    @test JA.resident_count(cache) == 2
    @test !isready(cache, 100,100)
    @test size(cache) == (4,2)
    @test (@inferred handle[-1,5]) == [-1,5]
end

@testset "Nested concurrent LRU generation and conversion can grow" begin
    for conversion in (false, true)
        (; generator, owner, calls) = nested_generator(Returns(63), (1, 1), cache -> (cache[6, 3]; 11); conversion)
        cache = LRUMemoizedArray(generator, ExpandingArrayMap{Int}((1,1); increment=4); capacity=1, concurrent=true, queues=1)
        handle = queue_handle(cache, 1)
        owner[] = handle
        @test handle[1,1] == 11
        @test size(cache) == (9,5)
        @test handle[1,1] == 11
        @test calls[] == 2
        @test !isready(cache, 6,3)
        @test isready(cache, 1,1)
        @test JA.resident_count(cache) == 1
    end
end

@testset "Expanding LRU retains waiting outcomes through relocation and reuse" begin
    for outcome in (:value, :generator_failure, :conversion_failure)
        (; generator, entered, release, calls, failure) =
            paused_generator((i, j) -> 100i + j, (1, 2); outcome, value=12)
        cache = LRUMemoizedArray(generator, ExpandingArrayMap{Int}((2,2); increment=2); capacity=3, concurrent=true, queues=3)
        a, b, c = (queue_handle(cache, i) for i in 1:3)
        owner = Threads.@spawn begin
            try
                test_result(() -> a[1,2])
            finally
                put!(entered, nothing)
            end
        end
        take!(entered)
        old_lookup = JA.map_lookup(cache, (1,2))
        captured = Channel{Nothing}(1)
        waiter = expanding_captured_waiter(b, (1,2), captured)
        take!(captured)
        pending = nothing
        try
            registered = timedwait(() -> JA.slot_entry(old_lookup[3]) isa JA.GenerateWait,
                5; pollint=0.001)
            @test registered === :ok
            pending = JA.slot_entry(old_lookup[3])
            @test c[3,1] == 301
            @test c[5,4] == 504
            @test size(cache) == (6,4)
            @test !isready(cache, 1,2)
        finally
            notify(release)
        end
        expected = outcome === :value ? 12 : failure
        @test fetch(owner) === expected
        @test a[7,6] == 706 # immediately recycle the owner's entry and grow again
        @test fetch(waiter) === expected
        if pending isa JA.GenerateWait
            @test test_result(() -> JA.wait_generated(Int, pending.reference)) === expected
        end
        @test calls[] == 1
        @test !isready(cache, 1,2)
        if outcome !== :value
            @test test_result(() -> a[1,2]) === failure
            @test calls[] == 1
        end
    end
end

@testset "Independent queues grow and replace concurrently" begin
    callers = 4
    entered = Channel{Nothing}(callers)
    release = Base.Event()
    calls = Threads.Atomic{Int}(0)
    cache = LRUMemoizedArray((i,j) -> begin
        Threads.atomic_add!(calls, 1)
        if j == 1 && i <= callers
            put!(entered, nothing)
            wait(release)
        end
        (i,j)
    end, ExpandingArrayMap{Tuple{Int,Int}}((1,1); increment=4); capacity=8, concurrent=true, queues=callers)
    handles = [queue_handle(cache, i) for i in 1:callers]
    tasks = [Threads.@spawn begin
        result = test_result(() -> handles[i][i,1])
        if result isa Exception
            put!(entered, nothing)
        end
        result
    end for i in 1:callers]
    try
        for _ in 1:callers
            take!(entered)
        end
    finally
        notify(release)
    end
    @test fetch.(tasks) == [(i,1) for i in 1:callers]
    tasks = [Threads.@spawn begin
        valid = true
        for step in 1:8
            key = (i + callers * step, mod1(i + step, 5))
            valid &= handles[i][key...] === key
        end
        valid
    end for i in 1:callers]
    @test all(fetch.(tasks))
    @test calls[] == callers * 9
    @test size(cache) == (37,5)
    @test JA.resident_count(cache) == 8
    @test count(key -> isready(cache, key), CartesianIndices(cache)) == 8
end

@testset "LRU requests progress before the remaining slots are copied" begin
    cache = LRUMemoizedArray((i,j) -> 10i+j, ExpandingArrayMap{Int}((2,2); increment=2); capacity=2, concurrent=true, queues=2)
    a, b = queue_handle(cache, 1), queue_handle(cache, 2)
    @test a[1,1] == 11
    @test b[1,2] == 12
    readers = Task[]
    replacement = gap_reader = nothing
    lock(cache.map.growth) do
        previous = JA.current_allocation(cache.map)
        allocation = JA.prepare_expansion!(previous, (4,2))
        gap_stored = nothing
        gap_frozen = gap_copied = false
        try
            JA.relocate_slot!(previous, allocation, (1,1))
            JA.relocate_slot!(previous, allocation, (2,1))
            readers = [Threads.@spawn(a[1,1]), Threads.@spawn(b[1,2])]
            ready = timedwait(() -> all(istaskdone, readers), 5; pollint=0.001)
            @test ready === :ok
            @test JA.current_allocation(cache.map) === previous
            if ready === :ok
                # Both the victim and insertion destination have already moved.
                replacement = Threads.@spawn a[2,1]
                written = timedwait(() -> istaskdone(replacement), 5; pollint=0.001)
                @test written === :ok
                if written === :ok
                    @test !isready(cache, 1,1)
                    published_ready = isready(cache, 2,1)
                    @test published_ready
                    handle_ready = isready(a, 2,1)
                    @test handle_ready
                    # Pause exactly between freezing this source and copying it.
                    gap_stored = JA.slot_exchange!(JA.allocation_slot(previous, (1,2)), JA.RELOCATED)
                    gap_frozen = true
                    gap_reader = Threads.@spawn b[1,2]
                    @test timedwait(() -> istaskdone(gap_reader), 0.02; pollint=0.001) === :timed_out
                    JA.slot_publish!(JA.allocation_slot(allocation, (1,2)), gap_stored)
                    gap_copied = true
                    @test timedwait(() -> istaskdone(gap_reader), 5; pollint=0.001) === :ok
                    @test JA.current_allocation(cache.map) === previous
                end
            end
        finally
            if gap_frozen
                if !gap_copied
                    JA.slot_publish!(JA.allocation_slot(allocation, (1,2)), gap_stored)
                end
            else
                JA.relocate_slot!(previous, allocation, (1,2))
            end
            JA.relocate_slot!(previous, allocation, (2,2))
            @atomic :release cache.map.allocation = allocation
            @atomic :release previous.successor = previous
        end
    end
    @test fetch.(readers) == [11,12]
    if replacement !== nothing
        @test fetch(replacement) == 21
    end
    if gap_reader !== nothing
        @test fetch(gap_reader) == 12
    end
    @test a[2,1] == 21
    @test !isready(cache, 1,1)
    @test b[1,2] == 12
    @test JA.resident_count(cache) == 2
end

@testset "Memoized completion progresses before the remaining slots are copied" begin
    entered, release = Base.Event(), Base.Event()
    cache = MemoizedArray((i,j) -> begin
        notify(entered)
        wait(release)
        10i+j
    end, ExpandingArrayMap{Int}((2,1); increment=2); concurrent=true)
    owner = Threads.@spawn cache[1,1]
    wait(entered)
    lock(cache.map.growth) do
        previous = JA.current_allocation(cache.map)
        allocation = JA.prepare_expansion!(previous, (4,1))
        try
            JA.relocate_slot!(previous, allocation, (1,1))
            notify(release)
            @test timedwait(() -> istaskdone(owner), 5; pollint=0.001) === :ok
            @test JA.current_allocation(cache.map) === previous
            @test JA.slot_entry(JA.allocation_slot(allocation, (1,1))) == 11
            completed_ready = isready(cache, 1,1)
            @test completed_ready
        finally
            notify(release)
            JA.relocate_slot!(previous, allocation, (2,1))
            @atomic :release cache.map.allocation = allocation
            @atomic :release previous.successor = previous
        end
    end
    @test fetch(owner) == 11
    @test cache[1,1] == 11
end

@testset "Queue generation ownership and retained outcomes" begin
    for fail in (false,true)
        calls=Threads.Atomic{Int}(0)
        entered,release=Base.Event(),Base.Event()
        value=[7]
        failure=ErrorException("queue generation failed")
        generator=i->begin
            Threads.atomic_add!(calls,1)
            if i==1
                notify(entered)
                wait(release)
                if fail
                    throw(failure)
                end
            end
            value
        end
        cache=LRUMemoizedArray(generator, ArrayMap{Vector{Int}}((16,)); concurrent=true, capacity=3, queues=3)
        a,b,c=queue_handle(cache,1),queue_handle(cache,2),queue_handle(cache,3)
        owner=Threads.@spawn test_result(()->a[1])
        wait(entered)
        waiter=Threads.@spawn test_result(()->b[1])
        ref=JA.map_slot(cache.map,1,nothing)
        while !(JA.slot_entry(ref) isa JA.GenerateWait)
            yield()
        end
        pending=JA.slot_entry(ref)
        # A third exclusive handle can generate while the other two wait.
        @test c[2]===value
        notify(release)
        expected=fail ? failure : value
        @test fetch(owner)===expected
        @test a[3]===value # eviction cannot erase the registered completion
        @test fetch(waiter)===expected
        @test test_result(()->JA.wait_generated(Vector{Int},pending.reference))===expected
        @test calls[]==3
        if fail
            @test test_result(()->a[1])===failure
            @test calls[]==3
        end
    end
end

@testset "Private queues: simultaneous reads, generation and reuse" begin
    for V in (Int,Vector{Int})
        generator = V === Int ? identity : i->[i]
        cache = LRUMemoizedArray(generator, ArrayMap{V}((512,)); concurrent=true, capacity=64, queues=8)
        handles = [queue_handle(cache,i) for i in 1:8]
        @sync for i in 1:8
            Threads.@spawn for request in 1:1024
                key = mod1(17request+7i,512)
                expected = V === Int ? key : [key]
                @test handles[i][key] == expected
            end
        end
        check_private_queues(cache)
    end
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((8,)); concurrent=true, capacity=2, queues=1)
    a = queue_handle(cache,1)
    for key in (1,2,1,3,4,3,5)
        @test a[key] == key
    end
    @test isready(cache,3) && isready(cache,5)
    check_private_queues(cache)
    nested = LRUMemoizedArray(i->i==1 ? handle[2]+1 : i, ArrayMap{Int}((8,)); concurrent=true, capacity=1, queues=1)
    handle = queue_handle(nested,1)
    @test handle[1] == 3
    check_private_queues(nested)
    offset = LRUMemoizedArray(+, ArrayMap{Int}((-1:1,4:5)); concurrent=true, capacity=8, queues=2)
    @test copy(view(queue_handle(offset,1),-1:0,4:5)) == [i+j for i in -1:0,j in 4:5]
end

# Dictionary-specific publication, probe-chain and retained-reference regressions.
function dict_test_cache(C, V, generator, domain; capacity=192, queues=3)
    options = C === MemoizedArray ? (;) : (; capacity, queues)
    return C(generator, DictMap{V}(domain); concurrent=true, options...)
end

function dict_test_completed(task)
    completed = timedwait(() -> istaskdone(task), 5; pollint=0.001)
    @test completed === :ok
    return completed === :ok ? fetch(task) : nothing
end

@testset "Concurrent dictionary nested generation and conversion rehash" begin
    for C in (MemoizedArray, LRUMemoizedArray), conversion in (false, true)
        keys = Int[]
        (; generator, owner, calls) = nested_generator(identity, (0,), cache -> (generate!(cache, keys); 0); conversion)
        cache = dict_test_cache(C, Int, generator, (0:10000,); capacity=96, queues=1)
        handle = concurrent_test_handle(cache, 1)
        owner[] = handle
        partition = JA.hash_partition(cache.map, hash((0,)))
        append!(keys, Iterators.take((i for i in 1:10000 if
            JA.hash_partition(cache.map, hash((i,))) === partition), 80))
        table = JA.slot_entry(partition.table)
        missing = JA.map_lookup(cache, (keys[end],))
        @test handle[0] == 0
        @test JA.slot_entry(partition.table) !== table
        @test JA.cache_get(handle, missing, (keys[end],)) == keys[end]
        @test all(i -> handle[i] == i, [0; keys])
        @test calls[] == 81
        C === LRUMemoizedArray && @test JA.resident_count(cache) == 81
    end
end

@testset "Concurrent dictionary paused publication survives rehash" begin
    for C in (MemoizedArray, LRUMemoizedArray), outcome in (:value, :generator_failure, :conversion_failure)
        (; generator, entered, release, calls, failure) = paused_generator(identity, (0,); outcome)
        cache = dict_test_cache(C, Int, generator, (0:144,))
        a, b, c = (concurrent_test_handle(cache, i) for i in 1:3)
        owner = Threads.@spawn try
            test_result(() -> a[0])
        finally
            put!(entered, nothing) # an early failure must not strand synchronization
        end
        waiter = pending = lookup = nothing
        try
            started = timedwait(() -> isready(entered), 5; pollint=0.001)
            @test started === :ok
            if started === :ok
                take!(entered)
                lookup = JA.map_lookup(cache, (0,))
                partition = JA.hash_partition(cache.map, hash((0,)))
                table = JA.slot_entry(partition.table)
                captured = Channel{Nothing}(1)
                waiter = Threads.@spawn test_result() do
                    saved = JA.map_lookup(b, (0,))
                    put!(captured, nothing)
                    JA.cache_get(b, saved, (0,))
                end
                @test timedwait(() -> isready(captured), 5; pollint=0.001) === :ok
                if C === LRUMemoizedArray
                    registered = timedwait(() -> JA.slot_entry(lookup[3]) isa JA.GenerateWait,
                        5; pollint=0.001)
                    @test registered === :ok
                    registered === :ok && (pending = JA.slot_entry(lookup[3]))
                end
                independent = Threads.@spawn begin
                    generate!(c, 1:80)
                    lock(partition.structural) do
                        JA.hash_rebuild!(partition, 4length(JA.slot_entry(partition.table)))
                    end
                    c
                end
                @test dict_test_completed(independent) === c
                @test JA.slot_entry(partition.table) !== table
                @test !isready(cache, 0)
            end
        finally
            notify(release)
        end
        expected = outcome === :value ? 0 : failure
        @test dict_test_completed(owner) === expected
        # Queue 1 has quota 64. The last request evicts its successful key 0;
        # failed generation remains in its dictionary record instead.
        generate!(a, 81:144)
        waiter !== nothing && @test dict_test_completed(waiter) === expected
        if pending !== nothing
            @test test_result(() -> JA.wait_generated(Int, pending.reference)) === expected
        end
        @test calls[] == 1
        @test isready(cache, 0) == (C === MemoizedArray && outcome === :value)
        if C === MemoizedArray || outcome !== :value
            @test test_result(() -> a[0]) === expected
            @test calls[] == 1
        end
    end
end

@testset "Dictionary reads and independent partitions progress under a partition lock" begin
    for C in (MemoizedArray, LRUMemoizedArray)
        cache = dict_test_cache(C, Int, identity, (1:1000,); capacity=2, queues=1)
        handle = concurrent_test_handle(cache, 1)
        @test handle[1] == 1
        lookup = JA.map_lookup(cache, (1,))
        reader = nothing
        partition = JA.hash_partition(cache.map, hash((1,)))
        other = first(i for i in 2:1000 if JA.hash_partition(cache.map, hash((i,))) !== partition)
        lock(partition.structural) do
            table = JA.slot_entry(partition.table)
            JA.hash_rebuild!(partition, 4length(table))
            reader = Threads.@spawn begin
                (handle[1], JA.cache_get(handle, lookup, (1,)), isready(cache, 1), handle[other])
            end
            @test timedwait(() -> istaskdone(reader), 5; pollint=0.001) === :ok
        end
        @test dict_test_completed(reader) == (1, 1, true, other)
    end
end

@testset "Dictionary stale ready and missing references follow recycled entries" begin
    keys = dict_test_collisions()
    calls = zeros(Int, 10000)
    cache = LRUMemoizedArray(i -> (calls[i] += 1; fill(i, 2, 2)), DictMap{Matrix{Int}}((1:10000,)); capacity=2, concurrent=true, queues=1)
    handle = queue_handle(cache, 1)
    first, second, third, fourth = keys
    missing = JA.map_lookup(cache, (first,))
    retained = handle[first]
    ready = JA.map_lookup(cache, (first,))
    stored = JA.entry(cache, JA.cache_key(cache, (first,)))
    @test handle[second] == fill(second, 2, 2)
    partition = JA.hash_partition(cache.map, hash((first,)))
    lock(partition.structural) do
        table = JA.slot_entry(partition.table)
        JA.hash_rebuild!(partition, 4length(table))
    end
    @test handle[third] == fill(third, 2, 2)
    @test !isready(cache, first) && isready(cache, second)
    @test JA.entry(cache, JA.cache_key(cache, (third,))) === stored
    @test JA.slot_entry(ready[3]) === JA.RELOCATED
    regenerated = JA.cache_get(handle, missing, (first,))
    @test regenerated == retained == fill(first, 2, 2)
    @test regenerated !== retained
    @test JA.cache_get(handle, ready, (first,)) === regenerated
    @test calls[first] == 2
    @test handle[fourth] == fill(fourth, 2, 2)
    @test handle[first] === regenerated
    @test JA.resident_count(cache) == 2
end

# Manual values have separate entries and consume no eviction capacity.
@testset "Other queues keep stale references" begin
    for config in pinned_configs(Int)
        cache = LRUMemoizedArray(identity,config;capacity=4,queues=2,concurrent=true)
        a,b = queue_handles(cache)
        @test a[1] == b[1] == 1
        old = JA.entry(cache,JA.cache_key(cache,(1,)))
        oldcount,oldest,newest = b.queue.count,b.queue.oldest,b.queue.newest
        a[1] = 100
        @test (b.queue.count,b.queue.oldest,b.queue.newest) == (oldcount,oldest,newest)
        @test b.queue.entries[oldest] === old
        @test b[1] == 100
        @test b.queue.count == oldcount
        for i in 2:64
            a[2i-1]
            b[2i]
        end
        @test a[1] == b[1] == 100
    end
end

@testset "Assignment waits for generation" begin
    entered,release = Base.Event(),Base.Event()
    cache = LRUMemoizedArray(i -> (notify(entered); wait(release); [i]),
        ArrayMap{Vector{Int}}((2,));capacity=2,queues=2,concurrent=true)
    a,b = queue_handles(cache)
    owner = Threads.@spawn a[1]
    wait(entered)
    setter = Threads.@spawn (b[1] = [99])
    ref = JA.map_slot(cache.map,1,nothing)
    while !(JA.slot_entry(ref) isa JA.GenerateWait)
        yield()
    end
    notify(release)
    @test fetch(owner) == [1]
    @test fetch(setter) == [99]
    @test a[1] == b[1] == [99]
end

@testset "Concurrent pinning and eviction" begin
    for config in pinned_configs(Int)
        cache = LRUMemoizedArray(identity,config;capacity=8,queues=4,concurrent=true)
        handles = queue_handles(cache)
        foreach(h -> h[0], handles)
        tasks = map(1:4) do q
            Threads.@spawn begin
                h, valid = handles[q], true
                for i in 1:24
                    h[q] = -q
                    h[0] = -q
                    h[4i+q]
                    valid &= h[q] == -q
                    valid &= h[0] in -4:-1
                end
                valid
            end
        end
        @test all(fetch.(tasks))
        @test all(q -> handles[q][q] == -q,1:4)
        @test JA.entry(cache, JA.cache_key(cache, (0,))).explicit
    end
end
