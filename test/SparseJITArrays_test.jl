const JA = JITArrays

sparse_options(storage, n; block=ntuple(_ -> 4, n)) =
    storage === :dict ? (; storage) : (; storage, block_shape=block)

function sparse_cache(::Type{V}, ranges, generator, storage, policy=:none; capacity=8, kwargs...) where V
    options = sparse_options(storage, length(ranges); kwargs...)
    if policy === :none
        return JITArray{V}(ranges, generator; options...)
    end
    return LRUArray{V}(ranges, capacity, generator; policy, options...)
end

sparse_slot(cache, key) = JA.store_lookup(cache.map, key, JA.store_hash(cache.map, key))
sparse_pending(cache, key) = JA.slot_entry(sparse_slot(cache, key))
sparse_signal(pending) = pending

function sparse_waiter_count(signal)
    state = @atomic :acquire signal.state
    if !(state isa Threads.Condition)
        return 0
    end
    lock(state)
    try
        return length(state.waitq)
    finally
        unlock(state)
    end
end

function sparse_tree_counts(node; bounded=false)
    if node === nothing
        return (0, 0, true)
    elseif node isa JA._TreeLeaf
        entries = count(i -> !((@atomic :acquire node.values[i]) isa JA.Empty), eachindex(node.values))
        return (0, 1, node.used == (bounded ? entries : 0))
    end
    branches, leaves, used, occupied = 1, 0, 0, 0
    valid = true
    if node isa JA._LSBNode
        entries = count(i -> !((@atomic :acquire node.values[i]) isa JA.Empty), eachindex(node.values))
        leaves = entries > 0 ? 1 : 0
        valid = node.used == (bounded ? entries : 0)
    end
    for i in eachindex(node.children)
        child = @atomic :acquire node.children[i]
        if child !== nothing
            used += 1
            occupied ⊻= i
            b, l, child_valid = sparse_tree_counts(child; bounded)
            branches += b
            leaves += l
            valid &= child_valid
        end
    end
    if node isa JA._LSBNode
        valid &= node.children_used == (bounded ? used : 0)
    else
        valid &= node.used == (bounded ? used : 0)
        valid &= node.occupied == (bounded ? occupied : 0)
    end
    return branches, leaves, valid
end
function sparse_tree_counts(map::JA._RadixMap; bounded=false)
    root = map isa JA.MSBRadixMap ? (@atomic :acquire map.roots[1]) : map.root
    branches, leaves, valid = sparse_tree_counts(root; bounded)
    @test valid
    return branches, leaves
end

@testset "Sparse constructors and signed axes" begin
    @test JITArray{Int}((3,), identity) isa JITArray
    @test JITArray{Int}((), () -> 1) isa JITArray
    @test LRUArray{Int}((), 1, () -> 1) isa SampledLRUArray
    @test JITArray{Int}([-3:3], identity) isa SparseJITArray{Int,1}
    @test LRUArray{Int}([-3:3], 2, identity) isa SparseLRUArray{Int,1}
    for storage in (:dict, :msbtree, :lsbtree), policy in (:none, :exact, :sampled)
        cache = sparse_cache(Int, (-5:6, -4:7), (i,j) -> i + 20j, storage, policy)
        @test size(cache) == (12, 12)
        @test size(cache, 3) == 1
        @test length(cache) == 144
        @test first(axes(cache, 1)) == -5
        @test last(axes(cache, 2)) == 7
        @test !isready(cache, -1, 1)
        @test (@inferred cache[-1, 1]) == 19
        @test (@inferred isready(cache, [-1, 1]))
        @test cache[CartesianIndex(-1, 1)] == 19
        @test cache[Int32(-1), Int32(1)] === 19
        @test cache[-1, 1, 1] === 19
        @test_throws BoundsError cache[-1, 1, 2]
        @test cache[1] == -85
        @test cache[144] == 146
        @test_throws BoundsError cache[-6, 0]
        @test_throws BoundsError cache[145]
        @test_throws BoundsError isready(cache, 0, 8)
        @test_throws BoundsError generate!(cache, (-6, 0))
        @test (@inferred generate!(cache, (-2, 2))) == 38
        @test fetch(generate!(cache, ((-3, 2), (-3, 2), (3, -2)))) === cache
        @test fetch(generate!(cache, CartesianIndex.([(-1, 1), (1, -1)]))) === cache
        @test fetch(generate!(cache, [1, 144])) === cache
        @test fetch(generate!(cache, NTuple{2,Int}[])) === cache
        one = sparse_cache(Int, (-4:4,), identity, storage, policy)
        @test one[-4] == -4
        @test one[[0, -4, 4]] == [0, -4, 4]
        @test one[Int32(-4)] === -4
        @test fetch(generate!(one, [-4, 0, 4])) === one
        empty = sparse_cache(Int, (1:0, -2:2), (i,j) -> error("unreachable"), storage, policy)
        @test isempty(empty)
        @test size(empty) == (0, 5)
        @test_throws BoundsError empty[1, 1]
        scalar = sparse_cache(Int, (), () -> 9, storage, policy)
        @test scalar[] == scalar[1] == 9
    end
    @test_throws ArgumentError JITArray{Int}((1:3,), identity; storage=:magnitude)
    @test_throws ArgumentError JITArray{Int}((1:3,), identity; storage=:unknown)
    @test_throws ArgumentError JITArray{Int}((1:3,), identity; storage=:msbtree)
    @test_throws ArgumentError JITArray{Int}((1:3,), identity; block_shape=(2,))
    @test_throws ArgumentError JITArray{Int}((1:3,), identity; storage=:msbtree, block_shape=(3,))
    @test_throws ArgumentError JITArray{Int}((1:3,), identity; storage=:msbtree, block_shape=(1,))
    @test_throws ArgumentError JITArray{Int}((1:3,), identity; storage=:msbtree, block_shape=(0,))
    @test_throws DimensionMismatch JITArray{Int}((1:3,), identity; storage=:msbtree, block_shape=[2,2])
    @test_throws OverflowError JITArray{Int}((1:3, 1:3), +; storage=:msbtree,
        block_shape=(1 << 40, 1 << 40))
    @test_throws ArgumentError LRUArray{Int}((1:3,), 0, identity)
    @test_throws ArgumentError LRUArray{Int}((1:3,), 1, identity; policy=:unknown)
end

@testset "Full signed domain and finite selections" begin
    domain = (typemin(Int):typemax(Int), typemin(Int):typemax(Int))
    for storage in (:dict, :msbtree, :lsbtree), policy in (:none, :exact, :sampled)
        cache = sparse_cache(NTuple{2,Int}, domain, tuple, storage, policy;
            capacity=3, block=(4,8))
        for key in ((0,0), (-1,1), (typemin(Int),typemax(Int)), (typemax(Int),typemin(Int)))
            @test (@inferred cache[key...]) === key
        end
        @test cache[-2:2,-1:1] == [(i,j) for i in -2:2, j in -1:1]
        @test copy(view(cache,-2:2,-1:1)) == [(i,j) for i in -2:2, j in -1:1]
        @test cache[typemax(Int)-1:typemax(Int), typemin(Int):typemin(Int)+1] ==
            [(i,j) for i in typemax(Int)-1:typemax(Int), j in typemin(Int):typemin(Int)+1]
        @test_throws OverflowError size(cache)
        @test_throws OverflowError length(cache)
        @test_throws OverflowError cache[1]
        @test collect(axes(cache)[1][typemax(Int)-1:typemax(Int)]) == [typemax(Int)-1,typemax(Int)]
        one = sparse_cache(Int, (typemin(Int):typemax(Int),), identity, storage, policy; capacity=2)
        for selection in (-2:2, typemin(Int):typemin(Int)+2, typemax(Int)-2:typemax(Int))
            @test one[selection] == collect(selection)
            @test copy(view(one, selection)) == collect(selection)
        end
    end
    large = JITArray{Int}((1:(1 << 40), 1:(1 << 40)), (i,j) -> i + j)
    @test size(large) == (1 << 40, 1 << 40)
    @test_throws OverflowError length(large)
    @test_throws OverflowError large[1]
    @test large[1,1] == 2
end

@testset "Sparse slices, views, and batch clocks" begin
    for storage in (:dict, :msbtree, :lsbtree), policy in (:none, :exact, :sampled)
        calls = Threads.Atomic{Int}(0)
        cache = sparse_cache(Int, (-4:4, -3:3), (i,j) -> (Threads.atomic_add!(calls, 1); i + 10j),
            storage, policy; capacity=2)
        v = view(cache, -2:2, -1:1)
        @test calls[] == 0
        @test size(v) == (5,3)
        @test copy(v) == [i + 10j for i in -2:2, j in -1:1]
        @test v[3,2] == 0
        @test cache[[-4, 0, 4], [-3, 3]] == [i + 10j for i in (-4,0,4), j in (-3,3)]
        @test cache[-1:1, :] == [i + 10j for i in -1:1, j in -3:3]
        @test cache[Int[], -1:1] == Array{Int}(undef, 0, 3)
        @test copy(view(v, 2:4, 2:3)) == [i + 10j for i in -1:1, j in 0:1]
        dest = zeros(Int, 5,3)
        @test copyto!(dest, v) === dest
        @test dest == [i + 10j for i in -2:2, j in -1:1]
        if policy === :sampled
            before = JA.clock_now(cache.clock)
            cache[-1:1, -1:1]
            @test JA.clock_now(cache.clock) == before + 9
            before = JA.clock_now(cache.clock)
            copy(v)
            @test JA.clock_now(cache.clock) == before + 15
            before = JA.clock_now(cache.clock)
            fetch(generate!(cache, ((0,0),(0,0),(1,1))))
            @test JA.clock_now(cache.clock) == before + 3
            @test !isready(cache, 4, 3)
            @test JA.clock_now(cache.clock) == before + 3
        end
    end
end

@testset "Dictionary collisions, growth, and tombstones" begin
    cache = JITArray{Int}((-10000:10000,), identity)
    keys = filter(i -> hash((i,)) & UInt(63) == 0, -10000:10000)[1:60]
    for key in keys[1:48]
        @test cache[key] == key
    end
    old = @atomic :acquire cache.map.table
    retained = sparse_slot(cache, (keys[1],))
    @test length(old.slots) == 64
    for key in keys[49:60]
        @test cache[key] == key
    end
    @test length((@atomic :acquire cache.map.table).slots) == 128
    @test sparse_slot(cache, (keys[1],)) === retained
    @test all(key -> cache[key] == key, keys)
    for policy in (:exact, :sampled)
        bounded = LRUArray{Int}((-10000:10000,), 2, identity; policy)
        initial = @atomic :acquire bounded.map.table
        for key in -500:500
            @test bounded[key] == key
            @test bounded.map.count <= 2
        end
        current = @atomic :acquire bounded.map.table
        @test length(current.slots) == 64
        @test current !== initial
        @test bounded.map.used <= 48
    end
end

@testset "Tree routing, compression, and reclamation" begin
    for storage in (:msbtree, :lsbtree)
        for n in (1,2,4,8)
            ranges = ntuple(_ -> typemin(Int):typemax(Int), n)
            cache = sparse_cache(NTuple{n,Int}, ranges, tuple, storage;
                block=ntuple(_ -> 2, n))
            for k in 1:80
                key = ntuple(i -> reinterpret(Int, UInt(k) * UInt(i) * 0x9e3779b97f4a7c15), n)
                @test cache[key...] === key
                @test cache[key...] === key
            end
            central = ntuple(_ -> 0, n)
            @test cache[central...] === central
            branches, leaves = sparse_tree_counts(cache.map)
            @test leaves > 0
            if cache.map isa JA.MSBRadixMap
                @test branches <= leaves + 1
            end
        end
        for policy in (:exact, :sampled)
            bounded = sparse_cache(Int, (typemin(Int):typemax(Int),), identity, storage, policy; capacity=1)
            for k in 1:120
                key = reinterpret(Int, UInt(k) * 0x9e3779b97f4a7c15)
                @test bounded[key] == key
                branches, leaves = sparse_tree_counts(bounded.map; bounded=true)
                @test leaves == 1
                @test branches <= (bounded.map isa JA.MSBRadixMap ? 1 : 33)
            end
        end
    end
end

@testset "Sparse exact order and retained values" begin
    for storage in (:dict, :msbtree, :lsbtree)
        cache = sparse_cache(Vector{Int}, (-10:10,), i -> [i], storage, :exact; capacity=3)
        retained = cache[-1]
        cache[0]; cache[1]; cache[-1]; cache[2]
        @test !isready(cache,0)
        @test all(i -> isready(cache,i), (-1,1,2))
        @test retained == [-1]
        order = [1,-1,2]
        for k in 1:200
            key = mod(k*k + 3k, 21) - 10
            position = findfirst(==(key), order)
            if position !== nothing
                deleteat!(order, position)
            elseif length(order) == 3
                popfirst!(order)
            end
            push!(order, key)
            @test cache[key] == [key]
            actual, entry, previous = Int[], cache.oldest, nothing
            while entry !== nothing && length(actual) <= cache.capacity
                @test entry.previous === previous
                @test JA.entry(cache, entry.key) === entry
                push!(actual, entry.key[1])
                previous, entry = entry, entry.next
            end
            @test cache.newest === previous
            @test actual == order
            @test cache.count == 3
        end
    end
end

@testset "Sparse parallel generation and late waiters" begin
    for storage in (:dict, :msbtree, :lsbtree), policy in (:none, :exact, :sampled)
        started, release = Channel{Nothing}(1), Channel{Nothing}(1)
        calls = Threads.Atomic{Int}(0)
        cache = sparse_cache(Vector{Int}, (-100:100,), i -> begin
            Threads.atomic_add!(calls, 1)
            if i == -1
                put!(started, nothing)
                take!(release)
            end
            [i]
        end, storage, policy; capacity=1)
        owner = Threads.@spawn cache[-1]
        take!(started)
        pending = sparse_pending(cache, (-1,))
        followers = [Threads.@spawn cache[-1] for _ in 1:4]
        signal = sparse_signal(pending)
        @test timedwait(() -> sparse_waiter_count(signal) == length(followers), 5) == :ok
        if policy !== :none
            other = Threads.@spawn cache[2]
            @test timedwait(() -> istaskdone(other), 0.02) == :timed_out
            @test calls[] == 1
        end
        put!(release, nothing)
        tasks = [owner; followers]
        if policy !== :none
            push!(tasks, other)
        end
        done = timedwait(() -> all(istaskdone, tasks), 5)
        @test done == :ok
        if done == :ok
            value = fetch(owner)
            @test all(task -> fetch(task) === value, followers)
            if policy === :none
                JA.wait_entry(pending)
                @test cache[-1] === value
            else
                @test fetch(other) == [2]
                @test JA.wait_entry(pending) === value
            end
        end
    end
end

@testset "Sparse failures retry and reclaim" begin
    for storage in (:dict, :msbtree, :lsbtree), policy in (:none, :exact, :sampled)
        started, release = Channel{Nothing}(1), Channel{Nothing}(1)
        failure = ErrorException("first attempt")
        calls = Threads.Atomic{Int}(0)
        cache = sparse_cache(Int, (-100:100,), i -> begin
            attempt = Threads.atomic_add!(calls, 1)
            if attempt == 0
                put!(started, nothing)
                take!(release)
                throw(failure)
            end
            i
        end, storage, policy; capacity=1)
        owner = Threads.@spawn try cache[-5] catch e; e end
        take!(started)
        pending = sparse_pending(cache, (-5,))
        follower = Threads.@spawn try cache[-5] catch e; e end
        signal = sparse_signal(pending)
        @test timedwait(() -> (@atomic :acquire signal.state) isa Threads.Condition, 5) == :ok
        put!(release, nothing)
        done = timedwait(() -> all(istaskdone, (owner, follower)), 5)
        @test done == :ok
        if done == :ok
            @test fetch(owner) === failure
            @test fetch(follower) isa ErrorException
            @test fetch(follower) !== failure
            @test !isready(cache,-5)
            if policy !== :none && storage !== :dict
                branches, leaves = sparse_tree_counts(cache.map; bounded=true)
                @test leaves == 0
                @test branches == (cache.map isa JA.LSBRadixMap ? 1 : 0)
            end
            @test cache[-5] == -5
            @test calls[] == 2
            @test_throws ErrorException JA.wait_entry(signal)
        end
    end
end

@testset "Concurrent resizing, pruning, and independent generators" begin
    for storage in (:dict, :msbtree, :lsbtree), policy in (:none, :exact, :sampled)
        counts = [Threads.Atomic{Int}(0) for _ in 1:128]
        active, peak = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
        entered = Threads.Atomic{Int}(0)
        started, release = Channel{Nothing}(2), Channel{Nothing}(2)
        cache = sparse_cache(Int, (-64:63,), i -> begin
            Threads.atomic_add!(counts[i + 65], 1)
            running = Threads.atomic_add!(active, 1) + 1
            Threads.atomic_max!(peak, running)
            if Threads.atomic_add!(entered, 1) < 2
                put!(started, nothing)
                take!(release)
            end
            yield()
            Threads.atomic_sub!(active, 1)
            i
        end, storage, policy; capacity=7)
        tasks = [Threads.@spawn begin
            for k in 1:600
                i = mod(29k + 17worker, 128) - 64
                @assert cache[i] == i
            end
        end for worker in 1:4]
        @test timedwait(() -> Base.n_avail(started) == 2, 5) == :ok
        put!(release, nothing)
        put!(release, nothing)
        done = timedwait(() -> all(istaskdone, tasks), 10)
        @test done == :ok
        if done == :ok
            @test all(task -> fetch(task) === nothing, tasks)
            if policy === :none
                @test all(count -> count[] == 1, counts)
            elseif policy === :exact
                @test cache.count == 7
            else
                @test cache.ready == 7 && cache.pending == 0
            end
        end
        if Threads.nthreads() > 1
            @test peak[] > 1
        end
    end
end

function sparse_read_allocations(cache)
    return @allocated for _ in 1:100
        cache[-1,1]
    end
end

@testset "Sparse value types and ready allocations" begin
    @testset "$storage/$policy" for storage in (:dict, :msbtree, :lsbtree), policy in (:none, :exact, :sampled)
        cache = sparse_cache(Matrix{Float64}, (-2:2, -2:2), (i,j) -> fill(Float64(i+j),4,4), storage, policy)
        @test (@inferred cache[-1,1]) == zeros(4,4)
        sparse_read_allocations(cache)
        @test sparse_read_allocations(cache) == 0
        for V in (Any, Union{Int,Nothing}, AbstractVector{Int})
            generator = V === AbstractVector{Int} ? (i -> [i]) : (i -> i == 0 ? nothing : i)
            mixed = sparse_cache(V, (-2:2,), generator, storage, policy)
            @test mixed[0] == generator(0)
            @test mixed[-1] == generator(-1)
        end
    end
end
