# Concurrent Memoized leaves remain valid when insertion creates new ancestors.
struct ConcurrentTreeConversion{F}
    convert::F
end
Base.convert(::Type{Int}, source::ConcurrentTreeConversion) = source.convert()

tree_request_allocations(cache, key) = @allocated cache[key...]

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

@testset "Concurrent tree dimensions" begin
    for Config in (RadixMap, SubtreeRadixMap)
        scalar = MemoizedArray(() -> 7, Config{Int}((); block_shape=()); concurrent=true)
        @test (@inferred scalar[]) == 7
        @test size(scalar) == () && isready(scalar)
        one = MemoizedArray(identity, Config{Int}((-8:8,); block_shape=(2,)); concurrent=true)
        @test one[-8] == -8 && one[8] == 8 && one[0] == 0
        three = MemoizedArray(tuple, Config{NTuple{3,Int}}(unbound(3); block_shape=(2, 4, 8)); concurrent=true)
        for key in ((0, 0, 0), (typemin(Int), 1, typemax(Int)))
            @test (@inferred three[key...]) === key
            @test isready(three, key)
        end
    end
    empty = MemoizedArray(identity, RadixMap{Int}((0,); block_shape=(4,)); concurrent=true)
    @test isempty(empty)
    @test_throws BoundsError empty[1]
    @test_throws ArgumentError MemoizedArray(identity, SubtreeRadixMap{Int}((0,); block_shape=(4,)); concurrent=true)
end

@testset "Concurrent tree coordinates and encoded values" begin
    for configuration in (((V, domain) -> RadixMap{V}(domain; block_shape=(4, 8))), ((V, domain) -> SubtreeRadixMap{V}(domain; block_shape=(4, 8))))
        cache = MemoizedArray(tuple, configuration(Tuple{Int,Int}, (-7:9, -11:12)); concurrent=true)
        @test size(cache) == (17, 24)
        @test axes(cache) == (-7:9, -11:12)
        @test !isready(cache, 0, 0)
        for key in ((0, 0), (-7, 12), (9, -11), (1, -1))
            @test (@inferred cache[key...]) === key
            @test cache[CartesianIndex(key)] === key
            @test isready(cache, key)
        end
        @test copy(view(cache, -1:1, -1:1)) == [(i, j) for i in -1:1, j in -1:1]
        @test_throws BoundsError cache[-8, 0]
        @test_throws BoundsError cache[0, 13]
        @test_throws ArgumentError cache[1]

        matrix = ones(2, 2)
        selected = view(matrix, :, 1:1)
        values = (nothing, false, JA.EMPTY, JA.GENERATING, JA.GenerationFailure(ErrorException("value")), matrix, selected)
        broad = MemoizedArray((i, _) -> values[i], configuration(Any, (0:8, 0:1)); concurrent=true)
        @test all(i -> broad[i, 0] === values[i] && isready(broad, i, 0), eachindex(values))
        @test all(i -> broad[i, 0] === values[i], reverse(eachindex(values)))
        for value in (matrix, selected)
            matrices = MemoizedArray(Returns(value), configuration(typeof(value), unbound(2)); concurrent=true)
            @test matrices[0, 0] === value
            tree_request_allocations(matrices, (0, 0))
            @test tree_request_allocations(matrices, (0, 0)) == 0
            # Existing-leaf generation allocates neither tree nodes nor values.
            tree_request_allocations(matrices, (1, 0))
            if value === matrix
                @test tree_request_allocations(matrices, (0, 1)) == 0
            end
            @test matrices[typemax(Int), typemin(Int)] === value
            GC.gc(false)
            @test matrices[0, 0] === value
        end
    end
end

@testset "Concurrent tree completion survives ancestor insertion" begin
    for Config in (RadixMap, SubtreeRadixMap), outcome in (:value, :generator_failure, :conversion_failure)
        entered = Channel{Nothing}(2)
        release = Base.Event()
        calls = Threads.Atomic{Int}(0)
        failure = ErrorException("tree generation: $outcome")
        generator = (i, j) -> begin
            if i == j == 0
                Threads.atomic_add!(calls, 1)
                put!(entered, nothing)
                wait(release)
                outcome === :generator_failure && throw(failure)
                outcome === :conversion_failure && return ConcurrentTreeConversion(() -> throw(failure))
            end
            return i ⊻ j
        end
        cache = MemoizedArray(generator, Config{Int}(unbound(2); block_shape=(4, 8)); concurrent=true)
        owner = Threads.@spawn try
            concurrent_test_error(() -> cache[0, 0])
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
                    concurrent_test_error(() -> JA.cache_get(cache, lookup, (0, 0)))
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
        @test concurrent_test_error(() -> cache[0, 0]) === expected
        @test isready(cache, 0, 0) == (outcome === :value)
        @test calls[] == 1
    end
end

@testset "Concurrent tree nested generation and conversion" begin
    for Config in (RadixMap, SubtreeRadixMap), conversion in (false, true)
        owner = Ref{Any}()
        keys = ((32, -64), (-4096, 8192), (typemin(Int), typemax(Int)))
        nested() = (foreach(key -> owner[][key...], keys); 17)
        cache = MemoizedArray((i, j) -> begin
            if i == j == 0
                return conversion ? ConcurrentTreeConversion(nested) : nested()
            end
            return i ⊻ j
        end, Config{Int}(unbound(2); block_shape=(4, 8)); concurrent=true)
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
