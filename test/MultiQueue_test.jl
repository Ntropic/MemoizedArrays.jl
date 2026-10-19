@testset "Public queue selection and capacity constraints" begin
    defaults = Threads.nthreads(:default)
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity=2defaults, concurrent=true)
    @test length(cache.eviction.queues) == defaults
    @test queue_handle(cache, 1)[1] == 1
    handles = queue_handles(cache)
    @test handles isa Tuple
    @test length(handles) == defaults
    @test all(i -> handles[i].queue === cache.eviction.queues[i], eachindex(handles))
    @test with_queue(cache) do caller
        caller[2]
    end == 2
    @test_throws ErrorException with_queue(cache) do _
        error("release this queue")
    end
    @test with_queue(cache) do caller
        caller[3]
    end == 3
    @test_throws ArgumentError cache[1]
    @test_throws BoundsError queue_handle(cache, 0)
    @test_throws BoundsError queue_handle(cache, defaults + 1)
    @test_throws BoundsError queue_handle(cache, 1)[17]
    for (capacity, queues) in ((0, 1), (4, 0), (4, -1), (3, 2), (1, 2))
        @test_throws ArgumentError LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity, queues, concurrent=true)
    end
    if defaults > 1
        @test_throws ArgumentError LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity=1, concurrent=true)
        @test length(LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity=1, concurrent=true, queues=1).eviction.queues) == 1
    end
    single = LRUMemoizedArray(identity, ArrayMap{Int}((8,)); capacity=2, concurrent=true, queues=1)
    @test_throws ArgumentError single[1]
    @test queue_handle(single, 1)[1] == 1
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
        owner=Threads.@spawn concurrent_test_error(()->a[1])
        wait(entered)
        waiter=Threads.@spawn concurrent_test_error(()->b[1])
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
        @test concurrent_test_error(()->JA.wait_generated(Vector{Int},pending.reference))===expected
        @test calls[]==3
        if fail
            @test concurrent_test_error(()->a[1])===failure
            @test calls[]==3
        end
    end
end

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

@testset "Private LRU last-access eviction and reusable incarnations" begin
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((32,)); concurrent=true, capacity=4, queues=2)
    a,b = queue_handle(cache,1),queue_handle(cache,2)
    @test (@inferred a[1]) == 1
    @test a[2] == 2 && b[1] == 1
    retained = JA.entry(cache,1)
    victim = JA.entry(cache,2)
    version = JA.private_state(victim)
    @test a[3] == 3
    @test JA.entry(cache,1) === retained # queue 2 most recently read it
    @test !isready(cache,2) && JA.entry(cache,3) === victim
    @test JA.private_state(victim) != version
    @test b[3] == 3 && a[4] == 4 && a[5] == 5
    @test isready(cache,3) # queue 1 dropped its reference; queue 2 protects it
    check_private_queues(cache)

    # A read can evict without generating: retain the freed entry for reuse.
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); concurrent=true, capacity=2, queues=2)
    a,b = queue_handle(cache,1),queue_handle(cache,2)
    a[1]; b[2]
    old = JA.entry(cache,1)
    oldversion = JA.private_state(old)
    @test a[2] == 2 && !isready(cache,1)
    @test JA.private_state(old) & 3 == 2
    @test a[3] == 3 && JA.entry(cache,3) === old
    @test !JA.private_reclaim!(cache,b.cache.eviction.queues[2],old,oldversion)
    @test !first(JA.private_snapshot(old,1)) # retained pointer cannot return key 3 for key 1
    @test isready(cache,3)
    check_private_queues(cache)
    # Delayed recency writes from an old incarnation are harmless hints. They
    # must not pin the replacement on behalf of a queue that never read it.
    @atomic :monotonic old.last_access = oldversion | UInt64(2)
    @test a[4] == 4 && !isready(cache,3)
    @test JA.private_state(old) & 3 == 2 # another free entry served key 4
    check_private_queues(cache)

    # A full queue replaces membership while another queue protects the value.
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); concurrent=true, capacity=6, queues=3)
    a,b,c = (queue_handle(cache,i) for i in 1:3)
    a[1]; a[2]; b[1]; c[3]; c[4]
    @test a[3] == 3 && isready(cache,1)
    check_private_queues(cache)

    # Dropping stale membership must leave the object's newer incarnation live.
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); concurrent=true, capacity=4, queues=2)
    a,b = queue_handle(cache,1),queue_handle(cache,2)
    a[1]; a[2]; b[1]; b[3]; a[1]; a[4]; a[5]
    replacement = JA.entry(cache,5)
    @test b[4] == 4 && JA.entry(cache,5) === replacement
    check_private_queues(cache)

    # Same object and same map key can recur; a stale incarnation must still
    # fail before the unconditional map-slot deletion after an entry claim.
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((4,)); concurrent=true, capacity=1, queues=1)
    a = queue_handle(cache,1)
    @test a[1] == 1
    old = JA.entry(cache,1)
    oldversion = JA.private_state(old)
    q = cache.eviction.queues[1]
    position = q.positions.mem[old.id]
    @test a[2] == 2 && a[1] == 1
    @test JA.entry(cache,1) === old && q.positions.mem[old.id] == position
    @test JA.private_state(old) != oldversion
    @test !JA.private_reclaim!(cache,q,old,oldversion)
    @test JA.entry(cache,1) === old && a[1] == 1
    check_private_queues(cache)
end

@testset "Private LRU remembered reads across entry reuse" begin
    for V in (Int,Vector{Int}), replacement_key in (1,4)
        values = V === Int ? collect(101:108) : [[i,10i] for i in 1:8]
        calls = zeros(Int,8)
        generator = i -> begin
            calls[i] += 1
            values[i]
        end
        cache = LRUMemoizedArray(generator, ArrayMap{V}((8,)); concurrent=true, capacity=2, queues=2)
        a,b = queue_handle(cache,1),queue_handle(cache,2)
        qa,qb = a.queue,b.queue
        @test a[1] === values[1]
        @test b[2] === values[2]
        entry = JA.entry(cache,1)
        position = qa.positions.mem[entry.id]
        version = qa.incarnations.mem[position]
        @test a[1] === values[1] # already remembered by this queue
        @test calls[1] == 1
        @test qb.positions.mem[entry.id] == 0
        @test b[1] === values[1] # present in the map, absent from this queue
        @test calls[1] == 1 && qb.positions.mem[entry.id] != 0

        # Queue b frees and reuses this exact object; queue a retains its old
        # position and incarnation, for either the old key or a different key.
        @test b[3] === values[3]
        @test b[replacement_key] === values[replacement_key]
        @test JA.entry(cache,replacement_key) === entry
        @test qa.positions.mem[entry.id] == position
        @test qa.incarnations.mem[position] == version
        replacement_version = JA.private_state(entry)
        @test replacement_version != version
        generated = sum(calls)
        @test a[replacement_key] === values[replacement_key]
        @test qa.incarnations.mem[position] == replacement_version
        @test a[replacement_key] === values[replacement_key]
        @test sum(calls) == generated
        if replacement_key != 1
            @test !isready(cache,1)
            @test a[1] === values[1] # never return the replacement key's value
            @test calls[1] == 2
        end
        check_private_queues(cache)
    end
end

@testset "Private LRU generation into a read-filled queue" begin
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity=4, concurrent=true, queues=2)
    a, b = queue_handle(cache, 1), queue_handle(cache, 2)
    b[1]; b[2]; a[1]; a[2]
    @test a.queue.count == a.queue.capacity
    @test a.queue.free_count == a.queue.capacity
    @test a[3] == 3
    @test !isready(cache, 1)
    check_private_queues(cache)
end

@testset "Private LRU free entry retains local membership" begin
    cache = LRUMemoizedArray(identity, ArrayMap{Int}((16,)); capacity=4, concurrent=true, queues=2)
    a, b = queue_handle(cache, 1), queue_handle(cache, 2)
    a[1]; b[2]; b[3]; a[2]; a[3]; b[4]; a[4]; b[4]; b[3]; a[6]; b[6]
    q = a.queue
    entry = q.free[q.free_count]
    version = JA.private_state(entry)
    position = q.positions.mem[entry.id]
    @test version & 3 == 2
    @test position != 0
    @test a[7] == 7
    @test JA.entry(cache, 7) === entry
    @test q.positions.mem[entry.id] == position
    @test q.incarnations.mem[position] == (version & ~(q.step - 1)) + q.step
    check_private_queues(cache)
end

private_request_allocations(handle,key) = @allocated handle[key]

@testset "Private LRU matrix entries are reused without allocation" begin
    value = ones(4,4)
    cache = LRUMemoizedArray(Returns(value), ArrayMap{typeof(value)}((16,)); concurrent=true, capacity=2, queues=1)
    handle = queue_handle(cache,1)
    private_request_allocations(handle,1) # compile measurement and read paths
    @test private_request_allocations(handle,2) == 0
    private_request_allocations(handle,3) # compile eviction path
    @test private_request_allocations(handle,4) == 0
    @test private_request_allocations(handle,4) == 0
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

@testset "Multi-queue broad values and conversion failures" begin
    for V in (Any,Int)
        value=V===Any ? JA.GENERATING : 7
        cache=LRUMemoizedArray(Returns(value), ArrayMap{V}((4,)); concurrent=true, capacity=2, queues=2)
        a=queue_handle(cache,1)
        @test a[1]===value && a[2]===value && a[1]===value
    end
    let calls=Ref(0)
        cache=LRUMemoizedArray(i->(calls[]+=1;0.5), ArrayMap{Int}((4,)); concurrent=true, capacity=2, queues=2)
        a=queue_handle(cache,1)
        @test_throws InexactError a[1]
        @test_throws InexactError a[1]
        @test calls[]==1
    end
end
