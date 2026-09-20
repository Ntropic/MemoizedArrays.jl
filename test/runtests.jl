# Small integration checks; shared array behavior is exercised once, policies separately.
using Test, JITArrays
const JA = JITArrays

function readyallocations(cache)
    cache[1]
    return @allocated cache[1]
end

replacementallocation(cache, key) = @allocated cache[key]

@testset "Map/cache integration" begin
    for storage in (:array, :dict, :radixtree, :subtree), T in (JITArray, ExactLRUArray, SampledLRUArray)
        calls = Ref(0)
        generator = i -> (calls[] += 1; i)
        domain = storage === :array ? (6,) : (-1:6,)
        options = storage === :array ? (;) : storage === :dict ? (; storage) : (; storage, block_shape=(8,))
        capacity = T === JITArray ? (;) : (; capacity=2)
        cache = T{Int}(generator, domain; options..., capacity...)
        if T !== JITArray
            @test hasfield(typeof(cache), :domain)
        end
        @test size(cache) == (storage === :array ? (6,) : (8,))
        @test !isready(cache, 1)
        @test (@inferred cache[1]) == cache[1] == 1
        @test readyallocations(cache) == 0
        @test calls[] == 1
        @test generate!(cache, [2, 3]) === cache
        @test calls[] == 3
        @test count(i -> isready(cache, i), 1:6) == (T === JITArray ? 3 : 2)
        @test cache[1] == 1
        @test_throws BoundsError cache[7]
    end
end

@testset "Constructors and bounds" begin
    @test LRUArray{Int}(identity, 4; capacity=2) isa SampledLRUArray
    @test LRUArray{Int}([4], 2, identity; policy=:exact) isa ExactLRUArray
    @test JITArray{Int}((4,), identity)[2] == 2
    @test JITArray{Int}(identity, [4])[2] == 2
    cache = JITArray{Int}(2, 3) do i, j
        i + j
    end
    @test cache[2, 3] == 5
    @test JITArray{Int}(() -> 7, ())[] == 7
    @test isempty(JITArray{Int}(identity, 0))
    @test_throws BoundsError JITArray{Int}(identity, 0)[1]
    @test_throws ArgumentError JITArray{Int}(identity, -1)
    @test_throws ArgumentError ExactLRUArray{Int}(identity, 4; capacity=0)
    @test_throws ArgumentError SampledLRUArray{Int}(identity, 4; capacity=0)
    @test_throws ArgumentError LRUArray{Int}(identity, 4; capacity=2, policy=:unknown)
    @test_throws UndefKeywordError LRUArray{Int}(identity, 4)
    @test JA.as_tuple([2, 3], Val(2)) === (2, 3)
    @test_throws DimensionMismatch JA.as_tuple([2], Val(2))
    @test isempty(detect_ambiguities(JITArrays))
end

@testset "Explicit array storage" begin
    for T in (JITArray, ExactLRUArray, SampledLRUArray, LRUArray)
        options = T === JITArray ? (;) : (; capacity=2)
        cache = T{Int}(+, (2, 3); storage=:array, options...)
        @test cache.map isa JA.ArrayMap
        @test size(cache) == (2, 3)
        @test cache[2, 3] == cache[6] == 5
        scalar = T{Int}(() -> 7, (); storage=:array, options...)
        @test scalar.map isa JA.ArrayMap
        @test scalar[] == 7
        if T === JITArray
            @test cache.map.empty isa JA.ArrayEntry{Int}
        end
        for axis in ((Colon(),), (typemin(Int):typemax(Int),))
            @test_throws ArgumentError T{Int}(identity, axis; storage=:array, options...)
        end
        @test_throws ArgumentError T{Int}(identity, 4; storage=:array, block_shape=(8,), options...)
    end
    @test_throws ArgumentError JITArray{Int}(identity, unbound(1); storage=:array)
    @test_throws ArgumentError LRUArray{Int}(identity, unbound(1); capacity=2, storage=:array)
end

@testset "Array interface" begin
    calls = Ref(0)
    cache = JITArray{Int}((i, j) -> (calls[] += 1; i + 3(j - 1)), 3, 4)
    reference = reshape(collect(1:12), 3, 4)
    selected = view(cache, 1:2, 2:3)
    @test calls[] == 0
    @test selected[1, 1] == 4
    @test calls[] == 1
    @test size(cache) == size(reference) && axes(cache) == axes(reference)
    @test cache[2, 3] == cache[8] == cache[CartesianIndex(2, 3)] == 8
    @test generate!(cache, ((2, 3), (2, 3))) === cache
    @test copy(selected) == reference[1:2, 2:3]
    for indices in ((2:5,), (:,), (2, :), ([3, 1, 3],), (reference .% 2 .== 0,), (1:0, :))
        @test cache[indices...] == reference[indices...]
    end
    @test collect(cache) == reference
    @test cache .+ 1 == reference .+ 1
    @test copyto!(zeros(Int, 3, 4), cache) == reference
    @test copyto!(PermutedDimsArray(zeros(Int, 4, 3), (2, 1)), cache) == reference
    @test get(cache, 13, -1) == -1
    @test_throws BoundsError cache[4, 1]
    @test_throws BoundsError view(cache, :, 5)
    @test_throws Base.CanonicalIndexError (cache[1] = 0)
end

@testset "Shaped ArrayMap" begin
    for T in (JITArray, ExactLRUArray, SampledLRUArray), shape in ((4,), (2, 3), (2, 3, 4))
        options = T === JITArray ? (;) : (; capacity=1)
        cache = T{Int}((key...) -> LinearIndices(shape)[key...], shape; options...)
        @test size(cache) == cache.map.shape == shape
        @test cache.map.reference.mem isa Memory
        @test cache.map.strides == Base.size_to_strides(1, shape...)
        @test JA.cache_key(cache, prod(shape)) === prod(shape)
        @test JA.cache_key(cache, shape) === prod(shape)
        @test all(JA.slotaddress(cache.map, Tuple(k)) == LinearIndices(shape)[k] for k in CartesianIndices(shape))
        @test cache[shape...] == cache[prod(shape)] == prod(shape)
        @test cache[1] == 1
        @test cache[shape...] == prod(shape)
        invalid = (shape[1] + 1, Base.tail(shape)...)
        @test_throws BoundsError cache[invalid...]
    end
end

@testset "Ready-only getter" begin
    calls = Ref(0)
    cache = JITArray{Vector{Int}}((i, j, k) -> (calls[] += 1; [i, j, k]), 2, 3, 4)
    value = cache[2, 3, 4]
    @test (@inferred getready(cache, 2, 3, 4)) === getready(cache, 24) === value
    @test calls[] == 1
    @test getready(cache, 2, 3, 4, 1) === value
    @test_throws BoundsError getready(cache, 25)
    @test_throws BoundsError getready(cache, 3, 1, 1)
end

@testset "ArrayMap identity sentinel" begin
    matrix = ones(4, 4)
    for value in (0, false, nothing, matrix, view(matrix, 1:2, 1:2))
        calls = Ref(0)
        cache = JITArray{typeof(value)}((i, j) -> (calls[] += 1; value), 2, 3)
        @test all(slot -> slot === cache.map.empty, cache.map.reference.mem)
        @test isconcretetype(eltype(cache.map.reference.mem))
        @test !isready(cache, 2, 3)
        @test cache[2, 3] === cache[6] === getready(cache, 2, 3) === value
        @test calls[] == 1 && isready(cache, 2, 3) && !isready(cache, 1)
        @test readyallocations(cache) == 0
    end
    failing = JITArray{Int}(_ -> 0.5, 1)
    @test_throws InexactError failing[1]
    @test !isready(failing, 1)
end

@testset "Batch generator coordinates" begin
    coordinates = [(2, 1, 4), (1, 3, 2), (2, 1, 4)]
    for T in (JITArray, ExactLRUArray, SampledLRUArray), requests in (coordinates, CartesianIndex.(coordinates))
        calls = NTuple{3,Int}[]
        options = T === JITArray ? (;) : (; capacity=2)
        cache = T{NTuple{3,Int}}((key...) -> (push!(calls, key); key), 2, 3, 4; options...)
        @test generate!(cache, requests) === cache
        @test calls == coordinates[1:2]
        @test cache[2, 1, 4] == (2, 1, 4) && cache[1, 3, 2] == (1, 3, 2)
    end
end

@testset "Values and direct generation" begin
    for T in (JITArray, ExactLRUArray, SampledLRUArray)
        options = T === JITArray ? (;) : (; capacity=1)
        value = ones(2, 2)
        cache = T{Matrix{Float64}}(_ -> value, 2; options...)
        @test (@inferred cache[1]) === value
        readyallocations(cache)
        @test readyallocations(cache) == 0
        retained = cache[1]
        @test cache[2] === value
        @test retained === value
        broad = T{Any}(i -> i == 1 ? nothing : false, 2; options...)
        @test broad[1] === nothing && isready(broad, 1)
        @test broad[2] === false
        failing = T{Int}(_ -> error("generator failed"), 2; options...)
        @test_throws ErrorException failing[1]
        converted = T{Int}(_ -> 0.5, 2; options...)
        @test_throws InexactError converted[1]
        dependent = T{Int}(i -> i == 1 ? dependent[2] + 10 : i, 2; options...)
        @test dependent[1] == 12
        @test dependent[1] == 12
    end
    # Reuse the cache entry across map families without changing returned value objects.
    for C in (ExactLRUArray, SampledLRUArray), storage in (:array, :dict, :radixtree, :subtree)
        domain = storage === :array ? (1024,) : (0:1024,)
        options = (; storage=storage === :array ? nothing : storage,
            block_shape=storage in (:radixtree, :subtree) ? (8,) : nothing)
        cache = C{Vector{Int}}(i -> [i], domain; capacity=1, options...)
        retained = cache[1]
        stored = JA.entry(cache, JA.cache_key(cache, 1))
        @test cache[512] == [512]
        @test JA.entry(cache, JA.cache_key(cache, 512)) === stored
        @test retained == [1] && !isready(cache, 1)
    end
    for C in (ExactLRUArray, SampledLRUArray)
        cache = C{Int}(identity, 4; capacity=1)
        cache[1]
        replacementallocation(cache, 2)
        @test replacementallocation(cache, 3) == 0
    end
    # Nested generation can resize the dictionary before the outer value is stored.
    nested = JITArray{Int}(i -> i == 0 ? sum(nested[k] for k in 1:60) : i, 0:60; storage=:dict)
    @test nested[0] == sum(1:60)
    @test all(k -> isready(nested, k), 0:60)
    # Nested insertion can occupy a saved dictionary position without resizing.
    keys = filter(i -> hash((i,)) & UInt(63) == 0, 1:1000)[1:2]
    colliding = JITArray{Int}(i -> i == keys[1] ? colliding[keys[2]] + 1 : i, 1:1000; storage=:dict)
    @test colliding[keys[1]] == keys[2] + 1
    @test colliding[keys[2]] == keys[2]
    @test colliding.map.count == 2
    # Nested eviction can detach the branch saved for an outer tree miss.
    for storage in (:radixtree, :subtree)
        pruning = ExactLRUArray{Int}(i -> i == 40 ? pruning[512] + i : i, 0:1024;
            capacity=2, storage, block_shape=(8,))
        pruning[32], pruning[36]
        @test pruning[40] == 552
        @test isready(pruning, 40) && isready(pruning, 512)
        @test !isready(pruning, 32) && !isready(pruning, 36)
        @test pruning.count == 2
    end
end

@testset "Exact LRU ordering" begin
    cache = ExactLRUArray{Int}(identity, 6; capacity=3)
    @test eltype(cache.map.reference.mem) === JA.entrytype(cache)
    @test cache.map.empty === cache.root
    @test fieldtype(JA.entrytype(cache), :previous) === JA.entrytype(cache)
    @test cache.root.next === cache.root.previous === cache.root
    order = Int[]
    for key in (1, 2, 3, 2, 1, 4, 4, 3, 5, 6, 1)
        filter!(!=(key), order)
        push!(order, key)
        length(order) > 3 && popfirst!(order)
        @test cache[key] == key
        actual, previous, entry = Int[], cache.root, cache.root.next
        while entry !== cache.root && length(actual) <= 3
            @test entry.previous === previous
            @test JA.entry(cache, entry.key) === entry
            push!(actual, entry.key)
            previous, entry = entry, entry.next
        end
        @test actual == order
        @test cache.root.previous === previous
        @test cache.count == length(order)
        @test findall(i -> isready(cache, i), 1:6) == sort(order)
    end
end

include("SparseJITArrays_test.jl")
include("Maps_test.jl")

@testset "Approximate LRU specializations" begin
    @test_throws ArgumentError SampledLRUArray{Int}(identity,4;capacity=1,policy=:sampled)
    @test_throws ArgumentError LRUArray{Int}(identity,4;capacity=1,policy=:sample5)
    for policy in (:sieve,:batch_reinsert,:batch_age)
        cache = SampledLRUArray{Int}(identity, [64]; capacity=4, policy)
        @test cache isa SampledLRUArray && typeof(cache).parameters[5] === policy
        @test hasfield(typeof(cache), :domain) && !hasfield(typeof(cache), :counter)
        valid = true
        for (request, k) in enumerate([1:32; 1:8; 33:64])
            valid &= cache[k] == k && cache.count == count(i -> isready(cache, i), 1:64)
            valid &= 1 <= cache.count <= min(request, 4)
        end
        @test valid
        @test SampledLRUArray{Int}(() -> 7, (); capacity=1, policy)[] == 7
        @test SampledLRUArray{Int}((4,), 1, identity; policy)[2] == 2
        @test_throws ArgumentError SampledLRUArray{Int}(identity,4;capacity=0,policy)
        sparse = SparseLRUArray{Int}(identity; ndims=1, capacity=2, policy)
        @test sparse[-1] == -1
        @test copy(view(cache,1:3)) == [1,2,3]
        for storage in (:array,:dict,:radixtree,:subtree)
            options = storage === :array ? (;) : storage === :dict ? (;storage) : (;storage,block_shape=(8,))
            domain = storage === :array ? (1024,) : (0:1024,)
            nested = SampledLRUArray{Int}(i -> i == 40 ? nested[512] + i : i, domain; capacity=2, policy, options...)
            nested[32]; nested[36]
            @test nested[40] == 552
            @test count(i -> isready(nested,i), (32,36,40,512)) == 2
            reused = SampledLRUArray{Vector{Int}}(i -> [i],domain;capacity=1,policy,options...)
            retained = reused[1]
            stored = JA.entry(reused,JA.cache_key(reused,1))
            @test reused[512] == [512] && retained == [1]
            @test JA.entry(reused,JA.cache_key(reused,512)) === stored && !isready(reused,1)
        end
        converted = SampledLRUArray{Float64}(identity,4;capacity=1,policy)
        @test (@inferred converted[1]) === 1.0
        readyallocations(converted)
        @test readyallocations(converted) == 0
        replacementallocation(converted,2)
        @test replacementallocation(converted,3) == 0
    end
    sieve = SampledLRUArray{Int}(identity,8;capacity=3,policy=:sieve)
    generate!(sieve,[1,2,3]); sieve[1]; sieve[4]
    @test isready(sieve,1) && !isready(sieve,2)
    sieve[5]
    @test !isready(sieve,3) && isready(sieve,5)
    default = SampledLRUArray{Int}(identity,8;capacity=3)
    @test typeof(default).parameters[5] === :sieve
    generate!(default,[1,2,3]); default[1]; default[4]
    @test isready(default,1) && !isready(default,2)
end

function checkbatch(cache, keys)
    state = cache.recency
    @test cache.count + state.nfree == cache.capacity
    queued = [state.queue[mod1(state.head + i, cache.capacity)] for i in 0:cache.count-1]
    free = cache.active[1:state.nfree]
    @test length(IdDict(entry => nothing for entry in [queued; free])) == cache.capacity
    @test all(entry -> JA.entry(cache, entry.key) === entry, queued)
    @test count(key -> isready(cache, key), keys) == cache.count
    if cache.map isa JA.TreeMap
        treecounts(cache.map)
    elseif cache.map isa JA.DictMap
        @test cache.map.count == cache.count
    end
end

@testset "Batch recency and pool reuse" begin
    for policy in (:batch_reinsert, :batch_age), storage in (:array, :dict, :radixtree, :subtree)
        options = storage in (:radixtree, :subtree) ? (; block_shape=(8,)) : (;)
        cache = LRUArray{Int}(i -> 3i + 1, (0:1024,);
            capacity=7, policy, storage, batch_size=3, options...)
        # An odd queue length and partial blocks exercise wrap and reuse across distant leaves.
        for requests in (0:6, (0,1,0), 512:525, 0:13)
            @test all(cache[key] == 3key + 1 for key in requests)
            checkbatch(cache, [0:13; 512:525])
        end
    end
    for policy in (:batch_reinsert, :batch_age)
        cache = SampledLRUArray{Int}(identity, 8; capacity=4, policy, batch_size=2)
        generate!(cache, 1:4)
        cache[1]
        @test cache[5] == 5
        @test isready(cache, 1) && !isready(cache, 2)
        @test cache.count == 4
        @test cache[6] == 6
        @test !isready(cache, 3) && !isready(cache, 4) && cache.count == 3
        @test cache[7] == 7 && cache.count == 4
        checkbatch(cache, 1:8)
        # An entirely reused pool must eventually age/reset enough entries to make progress.
        generate!(cache, filter(key -> isready(cache, key), 1:8))
        @test cache[8] == 8
        checkbatch(cache, 1:8)
        larger = SampledLRUArray{Int}(identity, 2; capacity=7, policy)
        @test larger[1] == 1 && larger[2] == 2 && larger.count == 2
    end
    for keep_epochs in (2, 4)
        cache = SampledLRUArray{Int}(identity, 8;
            capacity=4, policy=:batch_age, batch_size=2, keep_epochs)
        generate!(cache, 1:4)
        cache[1]
        generate!(cache, 5:8)
        @test isready(cache, 1) == (keep_epochs == 4)
    end
    for storage in (:radixtree, :subtree), policy in (:batch_reinsert, :batch_age)
        cache = SampledLRUArray{Int}(identity, unbound(1); storage, policy,
            capacity=2, batch_size=2, block_shape=(8,))
        generate!(cache, [0, 1])
        @test cache[2] == 2 && cache.count == 1
        @test !isready(cache, 0) && !isready(cache, 1) && isready(cache, 2)
        treecounts(cache.map)
        @test cache[512] == 512
        @test cache[1024] == 1024 && cache.count == 1
        @test !isready(cache, 2) && !isready(cache, 512)
        treecounts(cache.map)
    end
end

@testset "Batch options and generator errors" begin
    @test_throws ArgumentError SampledLRUArray{Int}(identity, 4; capacity=2, batch_size=1)
    for policy in (:sieve, :batch_reinsert)
        @test_throws ArgumentError SampledLRUArray{Int}(identity, 4; capacity=2, policy, keep_epochs=1)
    end
    for policy in (:batch_reinsert, :batch_age), invalid in (0, -1)
        @test_throws ArgumentError SampledLRUArray{Int}(identity, 4; capacity=2, policy, batch_size=invalid)
    end
    for invalid in (0, -1)
        @test_throws ArgumentError SampledLRUArray{Int}(identity, 4; capacity=2, policy=:batch_age, keep_epochs=invalid)
    end
    for (capacity, batch_size, expected) in ((1, nothing, 4), (65, nothing, 4), (65, 3, 22))
        age = SampledLRUArray{Int}(identity, 2; capacity, policy=:batch_age, batch_size)
        @test age.recency.blocksize == min(capacity, batch_size === nothing ? 64 : batch_size)
        @test age.recency.keep_epochs == expected
    end
    for policy in (:batch_reinsert, :batch_age), storage in (:array, :dict, :radixtree, :subtree)
        options = storage in (:radixtree, :subtree) ? (; block_shape=(8,)) : (;)
        calls = Ref(0)
        cache = SampledLRUArray{Int}(i -> (calls[] += 1; i == 2 ? error("generator failed") : i),
            (0:8,); capacity=2, policy, storage, options...)
        generate!(cache, [0, 1])
        @test_throws ErrorException cache[2]
        @test calls[] == 3 && cache.count == 2
        @test isready(cache, 0) && isready(cache, 1) && !isready(cache, 2)
        @test cache[3] == 3
        checkbatch(cache, 0:8)
        converted = SampledLRUArray{Int}(_ -> 0.5, (0:8,); capacity=2, policy, storage, options...)
        @test_throws InexactError converted[1]
        @test converted.count == 0 && !isready(converted, 1)
    end
end

@testset "Preallocated entries" begin
    for make in (()->ExactLRUArray{Int}(identity,8;capacity=3),
                 ()->SampledLRUArray{Int}(identity,8;capacity=3,policy=:sieve),
                 ()->SampledLRUArray{Int}(identity,8;capacity=3,policy=:batch_reinsert),
                 ()->SampledLRUArray{Int}(identity,8;capacity=3,policy=:batch_age))
        replacementallocation(make(),1)
        cache=make()
        entries=copy(cache isa ExactLRUArray ? cache.entries : cache.active)
        @test cache.count == 0 && !isready(cache,1)
        for k in 1:3
            @test replacementallocation(cache,k) == 0
            @test any(entry -> JITArrays.entry(cache,k) === entry, entries)
        end
        @test replacementallocation(cache,4) == 0 && 1 <= cache.count <= 3
    end
end

include("ConstructorDomains_test.jl")
