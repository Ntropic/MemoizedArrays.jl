using Test
using JITArrays

include("Constructors_test.jl")

@testset "LRU constructor selection" begin
    sampled = LRUArray{Int}((4,), 2, identity)
    exact = LRUArray{Int}([4], 2, identity; policy=:exact)
    @test sampled isa SampledLRUArray{Int}
    @test sampled.clock isa LossyCounter
    @test exact isa ExactLRUArray{Int}
    @test sampled isa LRUArray{Int} && exact isa LRUArray{Int}
    @test LRUArray{Int}((4,), 2, identity; policy=:sampled, clock=:coarse).clock isa CoarseClock
    for cache in (sampled, exact)
        @test cache[1] == 1
        @test fetch(generate!(cache, [2, 3])) === cache
        @test count(i -> isready(cache, i), 1:4) == 2
    end
    @test_throws ArgumentError LRUArray{Int}((4,), 2, identity; policy=:unknown)
    @test_throws MethodError LRUArray{Int}((4,), 2, identity; policy=:exact, clock=:coarse)
end

@testset "LRU order, retained values, and indexing" begin
    cache = ExactLRUArray{Matrix{Int}}([2, 3], 3, (i, j) -> fill(i + 2(j - 1), 2, 2))
    retained = @inferred cache[1]
    @test (@inferred getindex(cache, 2, 1)) == fill(2, 2, 2)
    @test (@inferred generate!(cache, (1, 2))) == fill(3, 2, 2)
    @test cache[CartesianIndex(1, 1)] === retained
    @test isready(cache, (2, 1))
    @test cache[4] == fill(4, 2, 2)
    @test !isready(cache, 2)
    @test cache.oldest.key == 3 && cache.newest.key == 4
    @test retained == fill(1, 2, 2)
    @test eltype(cache) === Matrix{Int}

    order = [3, 1, 4]
    for step in 1:150
        key = mod(step * step + 3step, 6) + 1
        position = findfirst(==(key), order)
        if position !== nothing
            deleteat!(order, position)
        elseif length(order) == cache.capacity
            popfirst!(order)
        end
        push!(order, key)
        @test cache[key] == fill(key, 2, 2)
        actual = Int[]
        entry, previous = cache.oldest, nothing
        while entry !== nothing && length(actual) <= cache.capacity
            @test entry.previous === previous
            @test JITArrays._entry(cache, entry.key) === entry
            push!(actual, entry.key)
            previous, entry = entry, entry.next
        end
        @test actual == order
        @test cache.newest.key == last(order)
        @test cache.count == length(order)
        @test findall(i -> isready(cache, i), 1:6) == sort(order)
    end
    @test_throws BoundsError cache[0]
    @test_throws BoundsError cache[3, 1]
    @test_throws BoundsError isready(cache, 7)
    @test getindex(cache, [1]) == [cache[1]]
    @test cache[CartesianIndex(1)] == cache[1]
    @test_throws ArgumentError ExactLRUArray{Int}((2,), 0, identity)
    @test_throws ArgumentError ExactLRUArray{Int}((-1,), 1, identity)
    empty_cache = ExactLRUArray{Int}((0,), 1, identity)
    @test fetch(generate!(empty_cache, Int[])) === empty_cache
    @test_throws BoundsError empty_cache[1]
    scalar = ExactLRUArray{Int}((), 1, () -> 7)
    @test (@inferred scalar[]) == 7
    @test scalar[1] == generate!(scalar, ()) == 7
    @test fetch(generate!(scalar, ((), ()))) === scalar
    for V in (Any, Union{Nothing,Int})
        c = ExactLRUArray{V}((2,), 1, i -> i == 1 ? nothing : i)
        @test c[1] === nothing
        @test c[2] === 2
        @test c[1] === nothing
    end
end

@testset "LRU capacity waits and eviction during wakeup" begin
    started, release = Channel{Nothing}(1), Channel{Nothing}(1)
    calls = Threads.Atomic{Int}(0)
    cache = ExactLRUArray{Matrix{Int}}((3,), 1, i -> begin
        Threads.atomic_add!(calls, 1)
        if i == 1
            put!(started, nothing)
            take!(release)
        end
        return fill(i, 2, 2)
    end)
    owner = Threads.@spawn cache[1]
    take!(started)
    pending = JITArrays._entry(cache, 1)
    followers = [@async cache[1] for _ in 1:4]
    @test timedwait(() -> all(istaskstarted, followers), 5) == :ok
    @test timedwait(() -> (@atomic :acquire pending.state) isa Threads.Condition, 5) == :ok
    other = Threads.@spawn cache[2]
    @test timedwait(() -> istaskdone(other), 0.05) == :timed_out
    @test calls[] == 1 && cache.count == 1
    put!(release, nothing)
    tasks = [owner; followers; other]
    completed = timedwait(() -> all(istaskdone, tasks), 5)
    @test completed == :ok
    if completed == :ok
        value = fetch(owner)
        @test all(result -> result == value, fetch.(followers))
        @test fetch(other) == fill(2, 2, 2)
        @test cache[3] == fill(3, 2, 2)
        @test JITArrays._wait_entry(pending) === value
        @test cache.count == 1
    end
end

@testset "LRU failures preserve attempts and release capacity" begin
    started, release = Channel{Nothing}(1), Channel{Nothing}(1)
    failure = ErrorException("first attempt")
    calls = Threads.Atomic{Int}(0)
    cache = ExactLRUArray{Int}((2,), 1, i -> begin
        attempt = Threads.atomic_add!(calls, 1)
        if attempt == 0
            put!(started, nothing)
            take!(release)
            throw(failure)
        end
        return i
    end)
    owner = Threads.@spawn try cache[1] catch exception; exception end
    take!(started)
    pending = JITArrays._entry(cache, 1)
    follower = Threads.@spawn JITArrays._wait_entry(pending)
    @test timedwait(() -> (@atomic :acquire pending.state) isa Threads.Condition, 5) == :ok
    other = Threads.@spawn cache[2]
    put!(release, nothing)
    completed = timedwait(() -> all(istaskdone, (owner, follower, other)), 5)
    @test completed == :ok
    if completed == :ok
        @test fetch(owner) === failure
        @test_throws TaskFailedException fetch(follower)
        @test fetch(other) == 2
        @test cache[1] == 1
        @test_throws ErrorException JITArrays._wait_entry(pending)
        @test cache.count == 1
    end
    invalid = ExactLRUArray{Int}((1,), 1, _ -> 0.5)
    @test_throws InexactError invalid[1]
    @test invalid.count == 0 && !isready(invalid, 1)
end

@testset "LRU parallel generation and bounded batches" begin
    active = [Threads.Atomic{Int}(0) for _ in 1:8]
    duplicates = Threads.Atomic{Int}(0)
    cache = ExactLRUArray{Matrix{Int}}((2, 4), 3, (i, j) -> begin
        key = i + 2(j - 1)
        if Threads.atomic_add!(active[key], 1) != 0
            Threads.atomic_add!(duplicates, 1)
        end
        yield()
        value = fill(key, 2, 2)
        Threads.atomic_sub!(active[key], 1)
        return value
    end)
    tasks = [Threads.@spawn begin
        for step in 1:100
            key = mod(step + worker, 8) + 1
            if cache[key] != fill(key, 2, 2)
                error("incorrect value")
            end
        end
    end for worker in 1:8]
    @test timedwait(() -> all(istaskdone, tasks), 10) == :ok
    @test all(task -> !istaskfailed(task), tasks)
    @test duplicates[] == 0
    @test fetch(generate!(cache, ((1, 1), (2, 2), (1, 4)))) === cache
    @test fetch(generate!(cache, [CartesianIndex(2, 1), CartesianIndex(1, 3)])) === cache
    @test fetch(generate!(cache, 1:8)) === cache
    @test cache.count == count(i -> isready(cache, i), 1:8) == 3
end

function _test_astuple(::Val{N}, controls, indexes) where {N}
    _as_tuple = nothing
    Val = nothing
    JITArrays.@astuple N controls indexes
    return controls, indexes
end

@testset "Tuple input helpers" begin
    @test (1.0, 2.0) isa JITArrays.VT{2,Float64}
    @test [1.0, 2.0] isa JITArrays.VT{2,Float64}
    @test !((1.0,) isa JITArrays.VT{2,Float64})
    @test (@inferred JITArrays._as_tuple((1.0, 2.0), Val(2))) === (1.0, 2.0)
    @test (@inferred JITArrays._as_tuple([1.0, 2.0], Val(2))) === (1.0, 2.0)
    @test JITArrays._as_tuple(view([1.0, 2.0, 3.0], 2:3), Val(2)) === (2.0, 3.0)
    @test_throws DimensionMismatch JITArrays._as_tuple([1.0], Val(2))
    @test_throws MethodError JITArrays._as_tuple((1.0,), Val(2))
    @test (@inferred _test_astuple(Val(2), [1.0, 2.0], [3, 4])) === ((1.0, 2.0), (3, 4))
    @test (@inferred _test_astuple(Val(2), (1.0, 2.0), (3, 4))) === ((1.0, 2.0), (3, 4))
    @test_throws DimensionMismatch _test_astuple(Val(2), [1.0], [3, 4])
    @test_throws MethodError _test_astuple(Val(2), (1.0,), (3, 4))
end


function _read_ready_cache_values(cache, expected)
    for _ in 1:64
        for index in eachindex(expected)
            if cache[index] !== expected[index]
                error("ready lookup changed the stored value")
            end
        end
    end
    return nothing
end

function _ready_cache_allocations(cache, expected)
    _read_ready_cache_values(cache, expected)
    return @allocated _read_ready_cache_values(cache, expected)
end

function _read_ready_get_values(cache, expected)
    for _ in 1:64
        for index in eachindex(expected)
            if get(cache, index, nothing) !== expected[index]
                error("ready get changed the stored value")
            end
        end
    end
    return nothing
end

function _ready_get_allocations(cache, expected)
    _read_ready_get_values(cache, expected)
    return @allocated _read_ready_get_values(cache, expected)
end

@testset "Indexed access and key forms" begin
    generated = Tuple{Int,Int}[]
    cache = JITArray{Int}((3, 4), (i, j) -> begin
        push!(generated, (i, j))
        return 10i + j
    end)
    @test (@inferred getindex(cache, 2, 3)) === 23
    @test (@inferred cache[2, 3]) === 23
    @test (@inferred generate!(cache, 2, 3)) === 23
    @test getindex(cache, 1, 4) == 14
    @test cache[1, 4] == 14
    @test generate!(cache, (1, 4)) == 14
    @test generate!(cache, CartesianIndex(3, 2)) == 32
    @test getindex(cache, CartesianIndex(3, 2)) == 32
    @test cache[CartesianIndex(3, 2)] == 32
    @test generated == [(2, 3), (1, 4), (3, 2)]

    converted = JITArray{Int}((1,), i -> Float64(i))
    @test (@inferred getindex(converted, 1)) === 1
    @test (@inferred generate!(converted, 1)) === 1
    scalar = JITArray{Int}((), () -> 7)
    @test scalar[] == 7
    @test getindex(scalar) == 7
    @test generate!(scalar, CartesianIndex()) == 7
    @test fetch(generate!(scalar, [()])) === scalar
end

@testset "Flattened indexes" begin
    for shape in ((), (4,), (2, 3), (2, 1, 3))
        calls = Tuple[]
        cache = JITArray{NTuple{length(shape),Int}}(shape, (indexes...) -> begin
            push!(calls, indexes)
            return indexes
        end)
        for (k, key) in enumerate(CartesianIndices(shape))
            @test !isready(cache, k)
            @test (@inferred getindex(cache, k)) === Tuple(key)
            @test cache[k] === cache[key] === getindex(cache, Tuple(key)...)
            @test isready(cache, k)
            @test generate!(cache, k) === Tuple(key)
        end
        @test calls == map(Tuple, vec(collect(CartesianIndices(shape))))
        for k in (0, prod(shape) + 1)
            @test_throws BoundsError cache[k]
            @test_throws BoundsError isready(cache, k)
            @test_throws BoundsError generate!(cache, k)
        end
    end
    empty_cache = JITArray{Int}((2, 0), (i, j) -> error("empty grid"))
    @test_throws BoundsError empty_cache[1]
    @test_throws BoundsError isready(empty_cache, 1)

    values = [fill(k, 2, 2) for k in 1:6]
    cache = JITArray{Matrix{Int}}((2, 3), (i, j) -> values[i + 2(j - 1)])
    @test fetch(generate!(cache, 1:6)) === cache
    @test _ready_cache_allocations(cache, values) == 0
    @test _ready_get_allocations(cache, values) == 0
end

@testset "Batch generation" begin
    calls = [Threads.Atomic{Int}(0) for _ in 1:5]
    cache = JITArray{Int}((5,), i -> begin
        Threads.atomic_add!(calls[i], 1)
        yield()
        return 10i
    end)
    requests = [5, 1, 3, 1, 5, 2, 4]
    @test fetch(generate!(cache, requests)) === cache
    @test [getindex(cache, key) for key in requests] == 10 .* requests
    @test [counter[] for counter in calls] == ones(Int, 5)
    @test fetch(generate!(cache, @view requests[1:3])) === cache
    @test [counter[] for counter in calls] == ones(Int, 5)

    empty_cache = JITArray{Int}((0,), _ -> error("empty request generated"))
    @test fetch(generate!(empty_cache, Int[])) === empty_cache

    calls_2d = Threads.Atomic{Int}(0)
    cache_2d = JITArray{Int}((3, 4), (i, j) -> begin
        Threads.atomic_add!(calls_2d, 1)
        return 10i + j
    end)
    tuple_requests = [(3, 1), (1, 4), (3, 1)]
    @test fetch(generate!(cache_2d, tuple_requests)) === cache_2d
    @test [cache_2d[key...] for key in tuple_requests] == [31, 14, 31]
    cartesian_requests = [CartesianIndex(2, 3), CartesianIndex(1, 4)]
    @test fetch(generate!(cache_2d, cartesian_requests)) === cache_2d
    @test [getindex(cache_2d, key) for key in cartesian_requests] == [23, 14]
    @test calls_2d[] == 3
    @test fetch(generate!(cache_2d, ((2, 2), (2, 2), (3, 4)))) === cache_2d
    @test cache_2d[2, 2] == 22
    @test cache_2d[12] == 34
    @test calls_2d[] == 5
    @test generate!(cache_2d, (2, 2)) === 22
    @test fetch(generate!(cache_2d, ())) === cache_2d
    @test fetch(generate!(cache_2d, NTuple{2,Int}[])) === cache_2d
    @test_throws MethodError generate!(cache_2d, ((1,), (2,)))

    scalar_cache = JITArray{Int}((), () -> 7)
    @test generate!(scalar_cache, ()) === 7
    @test fetch(generate!(scalar_cache, ((), ()))) === scalar_cache
    one_dimensional = JITArray{Int}((2,), identity)
    @test fetch(generate!(one_dimensional, ((1,), (2,)))) === one_dimensional

    fail = Ref(true)
    failing = JITArray{Int}((2,), i -> begin
        if i == 1 && fail[]
            error("batch failure")
        end
        return i
    end)
    @test_throws TaskFailedException fetch(generate!(failing, [1, 2]))
    fail[] = false
    @test fetch(generate!(failing, [1, 2])) === failing
    @test [failing[i] for i in 1:2] == [1, 2]
end

@testset "Concurrent getters and batch share work" begin
    started, release = Channel{Nothing}(1), Channel{Nothing}(1)
    calls = [Threads.Atomic{Int}(0) for _ in 1:2]
    expected = [fill(i, 2, 3) for i in 1:2]
    cache = JITArray{Matrix{Int}}((1, 2), (i, j) -> begin
        Threads.atomic_add!(calls[j], 1)
        if j == 1
            put!(started, nothing)
            take!(release)
        end
        return expected[j]
    end)
    readers = [Threads.@spawn getindex(cache, key) for key in (1, CartesianIndex(1, 1), CartesianIndex(1, 1), 1)]
    take!(started)
    @test all(task -> !istaskdone(task), readers)

    independent = Threads.@spawn generate!(cache, 2)
    independent_done = timedwait(() -> istaskdone(independent), 5)
    @test independent_done == :ok
    if independent_done == :ok
        @test fetch(independent) === expected[2]
        @test !isready(cache, 1) && isready(cache, 2)
    end

    batch = generate!(cache, ((1, 1), (1, 2), (1, 1)))
    @test batch isa Task
    @test timedwait(() -> istaskdone(batch), 0.05) == :timed_out
    put!(release, nothing)
    completed = timedwait(() -> all(istaskdone, [readers; independent; batch]), 5)
    @test completed == :ok
    if completed == :ok
        @test all(value -> value === expected[1], fetch.(readers))
        @test fetch(batch) === cache
        @test [counter[] for counter in calls] == [1, 1]
        @test isready(cache, 1)
    end
end

@testset "Array and abstract values" begin
    sources = [fill(i, i, i + 1) for i in 1:4]
    cache = JITArray{Matrix{Int}}((4,), i -> sources[i])
    retained = @inferred getindex(cache, 1)
    @test retained === sources[1]
    @test fetch(generate!(cache, 1:4)) === cache
    @test all(i -> cache[i] === sources[i], 1:4)
    retained[1, 1] = -7
    @test getindex(cache, 1)[1, 1] == -7
    @test eltype(cache) === Matrix{Int}

    for dims in ((), (4,), (2, 3), (2, 1, 3), (0, 2))
        source = fill(1, dims)
        c = JITArray{typeof(source)}((1,), _ -> source)
        @test (@inferred getindex(c, 1)) === source
    end
    converted = JITArray{Matrix{Float64}}((1,), _ -> ones(Int, 2, 3))
    @test (@inferred converted[1]) isa Matrix{Float64}
    @test converted[1] == ones(2, 3)
    @test getindex(converted, 1) === converted[1]

    nothing_calls = Threads.Atomic{Int}(0)
    nothing_cache = JITArray{Nothing}((2,), _ -> begin
        Threads.atomic_add!(nothing_calls, 1)
        return nothing
    end)
    @test !isready(nothing_cache, 1)
    @test (@inferred getindex(nothing_cache, 1)) === nothing
    @test isready(nothing_cache, 1)
    @test fetch(generate!(nothing_cache, [1, 2])) === nothing_cache
    @test nothing_cache[2] === nothing
    @test nothing_calls[] == 2

    expected = Any[nothing, false, 0, "value", ones(2, 3)]
    any_cache = JITArray{Any}((length(expected),), i -> expected[i])
    @test fetch(generate!(any_cache, [3, 1, 5, 2, 4])) === any_cache
    @test all(i -> getindex(any_cache, i) === expected[i], eachindex(expected))

    union_values = (nothing, 7)
    union_cache = JITArray{Union{Nothing,Int}}((2,), i -> union_values[i])
    @test union_cache[1] === nothing
    @test union_cache[2] === 7
    matrices = JITArray{AbstractMatrix{Int}}((1,), _ -> ones(Int, 2, 3))
    @test matrices[1] == ones(Int, 2, 3)
    @test eltype(matrices) === AbstractMatrix{Int}
end

@testset "Original conversion errors and retry" begin
    calls = Ref(0)
    cache = JITArray{Int}((1,), _ -> begin
        calls[] += 1
        if calls[] == 1
            return 0.5
        end
        return 5
    end)
    @test_throws InexactError getindex(cache, 1)
    @test generate!(cache, 1) == 5
    @test cache[1] == 5 && calls[] == 2
end

@testset "Waiting callers observe an empty attempt across retry" begin
    started, release = Channel{Nothing}(1), Channel{Nothing}(1)
    calls = Ref(0)
    failure = ErrorException("first attempt")
    result = ones(2, 3)
    cache = JITArray{Matrix{Float64}}((1,), _ -> begin
        calls[] += 1
        if calls[] == 1
            put!(started, nothing)
            take!(release)
            throw(failure)
        end
        return result
    end)
    owner = @async begin
        first = try
            getindex(cache, 1)
        catch exception
            exception
        end
        return first, getindex(cache, 1)
    end
    take!(started)
    first_attempt = @atomic :acquire cache.map[1]
    followers = [@async try
        getindex(cache, 1)
    catch exception
        exception
    end for _ in 1:2]
    @test timedwait(() -> all(istaskstarted, followers), 5) == :ok
    @test all(task -> !istaskdone(task), followers)
    put!(release, nothing)
    completed = timedwait(() -> all(istaskdone, [owner; followers]), 5)
    @test completed == :ok
    if completed == :ok
        first, retried = fetch(owner)
        @test first === failure && retried === result
        @test all(exception -> exception isa ErrorException && exception !== failure, fetch.(followers))
        @test calls[] == 2 && cache[1] === result
        late = try
            JITArrays._wait_entry(first_attempt)
        catch exception
            exception
        end
        @test late isa ErrorException && late !== failure
    end
end

@testset "Late waiter observes completed publication" begin
    for result in (ones(2, 3), true, false)
        started, release = Channel{Nothing}(1), Channel{Nothing}(1)
        cache = JITArray{typeof(result)}((1,), _ -> begin
            put!(started, nothing)
            take!(release)
            return result
        end)
        owner = @async getindex(cache, 1)
        take!(started)
        attempt = @atomic :acquire cache.map[1]
        put!(release, nothing)
        @test fetch(owner) === result
        @test JITArrays._wait_entry(attempt) === result
        @test JITArrays._entry(cache, 1) === attempt
        @test attempt.value === result
    end
end

@testset "Allocation-free ready lookups" begin
    for expected in ([10, 20, 30], [fill(i, 2, 3) for i in 1:3])
        ValueType = eltype(expected)
        generator = i -> expected[i]
        for cache in (JITArray{ValueType}((3,), generator), ExactLRUArray{ValueType}((3,), 3, generator))
            @test all(i -> cache[i] === expected[i], eachindex(expected))
            @test _ready_cache_allocations(cache, expected) == 0
            @test _ready_get_allocations(cache, expected) == 0
        end
    end
end

@testset "Minimal cache shape and bounds" begin
    @test isempty(detect_ambiguities(JITArrays))
    for shape in ((), (0,), (4,), (2, 3), (2, 1, 3))
        cache = JITArray{Int}(shape, (indexes...) -> sum(indexes))
        @test cache.map isa AtomicMemory && cache.domain == shape
        @test fieldnames(typeof(cache)) == (:map, :domain, :generator)
        @test !hasfield(typeof(cache), :available)
        @test all(name -> isconst(typeof(cache), name), (:map, :domain, :generator))
        @test JITArrays._entrytype(cache) === JITArrays.Entry{Int}
        @test length(cache.map) == prod(shape)
    end
    cache = JITArray{Int}([2, 3], (i, j) -> Float64(10i + j))
    @test cache isa AbstractJITArray{Int} && eltype(cache) === Int
    @test (@inferred getindex(cache, 2, 3)) === 23
    @test_throws BoundsError getindex(cache, 0, 1)
    @test_throws BoundsError generate!(cache, 3, 1)
    @test_throws BoundsError cache[CartesianIndex(2, 4)]
    @test (@inferred getindex(cache, 2, 3)) === 23
    @test (@inferred cache[[2, 3]]) == [21, 12]
    @test isready(cache, [2, 3])
    @test getindex(cache, [1]) == [cache[1]]
    @test getindex(cache, 1) == 11
    @test_throws MethodError generate!(cache, (1, 1, 1))
    @test cache[CartesianIndex(1)] == cache[1]
    @test JITArray{Int}(@view([2, 3][1:2]), (i, j) -> i + j).domain === (2, 3)
    @test_throws ArgumentError JITArray{Int}((-1,), identity)
    @test_throws MethodError JITArray{Int}((2.0,), identity)
end

@testset "Dependent cache generators" begin
    child = JITArray{Int}((4,), i -> (yield(); i))
    parent_cache = JITArray{Int}((4,), i -> 10child[i])
    @test fetch(generate!(parent_cache, 1:4)) === parent_cache
    @test [parent_cache[i] for i in 1:4] == [10, 20, 30, 40]

    leaf = JITArray{Int}((4,), i -> (yield(); i))
    middle = JITArray{Int}((4,), i -> (yield(); 10leaf[i]))
    outer_cache = JITArray{Int}((4,), i -> (yield(); middle[i] + 1))
    keys = [1, 2, 1, 4, 3, 2]
    requests = [Threads.@spawn outer_cache[i] for i in keys]
    @test fetch.(requests) == 10 .* keys .+ 1
    @test [leaf[i] for i in 1:4] == collect(1:4)
end

include("SampledLRUArrays_test.jl")

@testset "Standard array indexing and views" begin
    for maker in (gen -> JITArray{Int}((3, 4), gen),
        gen -> ExactLRUArray{Int}((3, 4), 3, gen),
        gen -> SampledLRUArray{Int}((3, 4), 3, gen))
        calls = Ref(0)
        cache = maker((i, j) -> (calls[] += 1; i + 3(j - 1)))
        reference = reshape(collect(1:12), 3, 4)
        @test cache isa AbstractMatrix{Int}
        @test size(cache) == size(reference)
        @test size(cache, 3) == 1
        @test axes(cache) == axes(reference)
        @test length(cache) == 12 && ndims(cache) == 2
        @test IndexStyle(typeof(cache)) == IndexLinear()
        selected = @view cache[1:2, 2:3]
        @test calls[] == 0
        @test size(selected) == (2, 2)
        @test selected[1, 1] == 4
        @test calls[] == 1
        @test collect(selected) == reference[1:2, 2:3]
        @test cache[end] == 12
        @test cache[end, 2] == 6
        @test cache[2, 3, 1] == reference[2, 3, 1]
        for indices in ((2:5,), (:,), (1:2, 2:4), (2, :), (:, 2), ([3, 1, 3],),
            (reshape([1, 4, 7, 10], 2, 2),), (view([1, 4, 7], 1:2),),
            (reference .% 2 .== 0,), ([true, false, true], :), (1:0, :), (Int[],),
            (CartesianIndex(2, 2),), ([CartesianIndex(1, 2), CartesianIndex(3, 4)],))
            @test cache[indices...] == reference[indices...]
            @test copy(view(cache, indices...)) == copy(view(reference, indices...))
        end
        @test cache[Int32(2), Int32(3)] == reference[2, 3]
        @test get(cache, 13, -1) == -1
        @test get(cache, (2, 3), -1) == reference[2, 3]
        @test collect(cache) == reference
        @test cache .+ 1 == reference .+ 1
        @test copyto!(zeros(Int, 3, 4), cache) == reference
        permuted = PermutedDimsArray(zeros(Int, 4, 3), (2, 1))
        @test copyto!(permuted, cache) == reference
        @test_throws BoundsError cache[4, 1]
        @test_throws BoundsError cache[[13]]
        @test_throws BoundsError view(cache, :, 5)
        @test_throws ArgumentError cache[(1, 2)]
        @test_throws Base.CanonicalIndexError (cache[1] = 0)
    end
    for shape in ((), (0,), (2, 0), (2, 1, 3)), maker in (JITArray, ExactLRUArray, SampledLRUArray)
        gen = (indices...) -> sum(indices; init=0)
        cache = maker === JITArray ? maker{Int}(shape, gen) : maker{Int}(shape, 2, gen)
        reference = map(key -> gen(Tuple(key)...), CartesianIndices(shape))
        @test collect(cache) == reference
        @test copy(cache) == reference
        @test cache[:] == reference[:]
    end
end

@testset "Bulk sampled timestamps and references" begin
    for option in (:lossy, :coarse, :nanosecond)
        values = [fill(i, 2, 2) for i in 1:12]
        cache = SampledLRUArray{Matrix{Int}}((3, 4), 12, (i, j) -> values[i + 3(j - 1)]; clock=option)
        selected = view(cache, 1:2, 2:3)
        @test cache.ready == 0
        output = copy(selected)
        keys = [4, 5, 7, 8]
        @test all(output[i] === values[keys[i]] for i in eachindex(output))
        stamps = [(@atomic :monotonic JITArrays._entry(cache, key).accessed) for key in keys]
        @test all(==(first(stamps)), stamps)
        if option === :lossy
            @test JITArrays._clock_now(cache.clock) == 4
        end
        flat = view(cache, 2:5)
        @test copy(flat) == values[2:5]
        if option === :lossy
            @test JITArrays._clock_now(cache.clock) == 8
        end
        dest = Matrix{Matrix{Int}}(undef, 3, 2)
        copyto!(view(dest, 1:2, :), selected)
        @test dest[1:2, :] == output
        stamps = [(@atomic :monotonic JITArrays._entry(cache, key).accessed) for key in keys]
        @test all(==(first(stamps)), stamps)
        before = JITArrays._clock_now(cache.clock)
        @test isempty(cache[Int[]])
        @test fetch(generate!(cache, Int[])) === cache
        @test_throws BoundsError copyto!(Matrix{Int}[], selected)
        if option === :lossy
            @test JITArrays._clock_now(cache.clock) == before
        end
        @test fetch(generate!(cache, ((1, 1), (2, 2), (1, 1)))) === cache
        @test (@atomic :monotonic JITArrays._entry(cache, 1).accessed) ==
            (@atomic :monotonic JITArrays._entry(cache, 5).accessed)
        if option === :lossy
            @test JITArrays._clock_now(cache.clock) == before + 3
        end
        @test cache[1] === values[1]
    end
    cache = SampledLRUArray{Int}((6,), 6, i -> 10i)
    indexes = [2, 3, 1]
    copyto!(indexes, view(cache, indexes))
    @test indexes == [20, 30, 10]
    small = SampledLRUArray{Matrix{Int}}((6,), 1, i -> fill(i, 2, 2))
    retained = small[:]
    @test [v[1] for v in retained] == collect(1:6)
    @test small.ready == 1
    @test JITArrays._clock_now(small.clock) == 6
    tasks = [Threads.@spawn small[1:6] for _ in 1:4]
    @test timedwait(() -> all(istaskdone, tasks), 10) == :ok
    @test all(task -> [v[1] for v in fetch(task)] == collect(1:6), tasks)
end

include("SparseJITArrays_test.jl")

@testset "Shared missing control" begin
    for constructor in (g -> JITArray{Matrix{Float64}}((3,), g),
                        g -> ExactLRUArray{Matrix{Float64}}((3,), 1, g),
                        g -> SampledLRUArray{Matrix{Float64}}((3,), 1, g))
        attempts = Ref(0)
        cache = constructor(i -> begin
            attempts[] += 1
            if attempts[] == 1
                error("retry")
            end
            fill(Float64(i), 2, 2)
        end)
        empty = JITArrays._entry(cache, 1)
        @test length(Base.uniontypes(eltype(cache.map))) == 2
        @test JITArrays._entry(cache, 2) === empty
        @test_throws ErrorException cache[1]
        @test JITArrays._entry(cache, 1) === empty
        @test cache[1] == fill(1.0, 2, 2)
        @test cache[2] == fill(2.0, 2, 2)
        @test JITArrays._entry(cache, 3) === empty
        @test empty isa JITArrays.Empty
        @test fieldcount(typeof(empty)) == 0
        if cache isa LRUArray
            @test JITArrays._entry(cache, 1) === empty
            @test cache[1] == fill(1.0, 2, 2)
        end
    end
end

function combination_cache(::Type{V}, storage, policy, generator; capacity=3) where V
    if storage === :dense
        if policy === :none
            return JITArray{V}((6,), generator)
        end
        return LRUArray{V}((6,), capacity, generator; policy)
    end
    return sparse_cache(V, (1:6,), generator, storage, policy; capacity)
end

@testset "Twelve combinations share typed entries and completion identity" begin
    containers = Set{Symbol}()
    for storage in (:dense, :dict, :msbtree, :lsbtree), policy in (:none, :exact, :sampled)
        started, release = Channel{Nothing}(1), Channel{Nothing}(1)
        calls = Threads.Atomic{Int}(0)
        expected = fill(7, 2, 2)
        cache = combination_cache(Matrix{Int}, storage, policy, i -> begin
            Threads.atomic_add!(calls, 1)
            if i == 1
                put!(started, nothing)
                take!(release)
            end
            expected
        end; capacity=1)
        push!(containers, nameof(typeof(cache)))
        key = storage === :dense ? 1 : (1,)
        E = JA._entrytype(cache)
        maptype = storage === :dense ? AtomicMemory : storage === :dict ? JA.DictMap :
            storage === :msbtree ? JA.MSBRadixMap : JA.LSBRadixMap
        @test cache.map isa maptype
        slot_type = cache.map isa AtomicMemory ? eltype(cache.map) : typeof(cache.map).parameters[1]
        @test slot_type === Union{JA.Empty,E}
        @test fieldtype(E, :value) === Matrix{Int}
        @test all(name -> isconcretetype(fieldtype(typeof(cache), name)), (:map, :domain, :generator))
        @test all(name -> isconst(typeof(cache), name), (:map, :domain, :generator))
        @test !hasfield(typeof(cache), :policy)
        @test JA._entry(cache, key) isa JA.Empty
        owner = Threads.@spawn cache[1]
        take!(started)
        entry = JA._entry(cache, key)
        @test entry isa E
        @test !isready(cache, 1)
        if policy === :exact
            @test cache.oldest === cache.newest === nothing
            @test cache.count == 1
        elseif policy === :sampled
            @test cache.ready == 0 && cache.pending == 1
            @test length(cache.active) == 1 && eltype(cache.active) === E
            @test entry.slot == 0 && !entry.live
        else
            @test fieldnames(typeof(cache)) == (:map, :domain, :generator)
        end
        followers = [Threads.@spawn cache[1] for _ in 1:2]
        @test timedwait(() -> sparse_waiter_count(entry) == 2, 5) == :ok
        put!(release, nothing)
        @test fetch(owner) === expected
        @test all(task -> fetch(task) === expected, followers)
        @test calls[] == 1
        @test JA._entry(cache, key) === entry
        @test entry.value === expected
        @test (@atomic :acquire entry.state) === true
        @test JA._wait_entry(entry) === expected
        if policy === :exact
            @test cache.oldest === cache.newest === entry
        elseif policy === :sampled
            @test cache.active[entry.slot] === entry
            @test entry.live && cache.ready == 1 && cache.pending == 0
        end
        cache[2]
        if policy !== :none
            @test JA._entry(cache, key) isa JA.Empty
            @test JA._wait_entry(entry) === expected
            if policy === :exact
                @test entry.previous === entry.next === nothing
            else
                @test !entry.live
            end
        end
        values = Any[nothing, true, false, expected, 42]
        broad = combination_cache(Any, storage, policy, i -> values[i])
        for i in eachindex(values)
            @test broad[i] === values[i]
            stored = JA._entry(broad, storage === :dense ? i : (i,))
            @test stored isa JA._entrytype(broad)
            @test stored.value === values[i]
            @test (@atomic :acquire stored.state) === true
        end
    end
    @test containers == Set((:JITArray, :ExactLRUArray, :SampledLRUArray))
end

include("Maps_test.jl")

include("Completion_test.jl")

@testset "Zero-dimensional range vectors preserve sparse storage" begin
    for policy in (:none, :exact, :sampled)
        generator = () -> 19
        make(domain; options...) = policy === :none ? JITArray{Int}(domain, generator; options...) :
            LRUArray{Int}(domain, 1, generator; policy, options...)
        for domain in ((), Int[])
            cache = make(domain)
            @test cache.map isa AtomicMemory
            @test cache[] == cache[1] == 19
            @test_throws ArgumentError make(domain; block_shape=(3,))
            @test_throws ArgumentError make(domain; block_shape=(0,))
        end
        for storage in (:dict, :msbtree, :lsbtree)
            options = sparse_options(storage, 0)
            cache = make(UnitRange{Int}[]; options...)
            expected = storage === :dict ? JA.DictMap : storage === :msbtree ? JA.MSBRadixMap : JA.LSBRadixMap
            @test cache.map isa expected
            @test cache[] == cache[1] == 19
            @test isready(cache, ())
            @test fetch(generate!(cache, ((), ()))) === cache
        end
        default = make(UnitRange{Int}[])
        @test default.map isa JA.DictMap
    end
    for cache in (SparseJITArray{Int}(UnitRange{Int}[], () -> 1),
        SparseLRUArray{Int}(UnitRange{Int}[], 1, () -> 1; policy=:exact),
        SparseLRUArray{Int}(UnitRange{Int}[], 1, () -> 1; policy=:sampled))
        @test cache.map isa JA.DictMap
        @test cache[] == 1
    end
end
