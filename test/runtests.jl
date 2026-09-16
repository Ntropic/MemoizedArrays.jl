# Small integration checks; shared array behavior is exercised once, policies separately.
using Test, JITArrays
const JA = JITArrays

function readyallocations(cache)
    cache[1]
    return @allocated cache[1]
end

replacementallocation(cache, key) = @allocated cache[key]

@testset "Map/cache integration" begin
    for storage in (:array, :dict, :msbtree, :lsbtree), T in (JITArray, ExactLRUArray, SampledLRUArray)
        calls = Ref(0)
        generator = i -> (calls[] += 1; i)
        domain = storage === :array ? (6,) : (-1:6,)
        options = storage === :array ? (;) : storage === :dict ? (; storage) : (; storage, block_shape=(8,))
        capacity = T === JITArray ? (;) : (; capacity=2)
        cache = T{Int}(generator, domain; options..., capacity...)
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
    @test JITArray{Int}(() -> 7)[] == 7
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
    for C in (ExactLRUArray, SampledLRUArray), storage in (:array, :dict, :msbtree, :lsbtree)
        domain = storage === :array ? (1024,) : (1:1024,)
        options = (; storage=storage === :array ? nothing : storage,
            block_shape=storage in (:msbtree, :lsbtree) ? (8,) : nothing)
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
    # A reused sampled entry must not revive duplicate candidates from its old residency.
    cache = SampledLRUArray{Int}(identity, 4; capacity=1)
    cache[1]
    stored = JA.entry(cache, 1)
    JA.sampled_pooladd!(cache, stored, UInt64(10))
    JA.sampled_pooladd!(cache, stored, UInt64(10))
    @test cache[2] == 2
    @test JA.entry(cache, 2) === stored
    @test all(i -> cache.versions[i] != stored.version, 1:cache.pooled)
    # Nested generation can resize the dictionary before the outer value is stored.
    nested = JITArray{Int}(i -> i == 0 ? sum(nested[k] for k in 1:60) : i, 0:60)
    @test nested[0] == sum(1:60)
    @test all(k -> isready(nested, k), 0:60)
    # Nested insertion can occupy a saved dictionary position without resizing.
    keys = filter(i -> hash((i,)) & UInt(63) == 0, 1:1000)[1:2]
    colliding = JITArray{Int}(i -> i == keys[1] ? colliding[keys[2]] + 1 : i, 1:1000)
    @test colliding[keys[1]] == keys[2] + 1
    @test colliding[keys[2]] == keys[2]
    @test colliding.map.count == 2
    # Nested eviction can detach the branch saved for an outer tree miss.
    for storage in (:msbtree, :lsbtree)
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
    order = Int[]
    for key in (1, 2, 3, 2, 1, 4, 4, 3, 5, 6, 1)
        filter!(!=(key), order)
        push!(order, key)
        length(order) > 3 && popfirst!(order)
        @test cache[key] == key
        actual, previous, entry = Int[], nothing, cache.oldest
        while entry !== nothing && length(actual) <= 3
            @test entry.previous === previous
            @test JA.entry(cache, entry.key) === entry
            push!(actual, entry.key)
            previous, entry = entry, entry.next
        end
        @test actual == order
        @test cache.newest === previous
        @test cache.count == length(order)
        @test findall(i -> isready(cache, i), 1:6) == sort(order)
    end
end

@testset "Sampled eviction and counter" begin
    cache = SampledLRUArray{Int}(identity, 64; capacity=4)
    for key in 1:64
        @test cache[key] == key
    end
    @test cache.counter == 64
    @test cache.ready == count(i -> isready(cache, i), 1:64) == 4
    cache = SampledLRUArray{Int}(identity, 6; capacity=6)
    @test generate!(cache, [1, 2, 1]) === cache
    @test cache.counter == 3
    @test JA.entry(cache, 1).accessed == JA.entry(cache, 2).accessed
    @test copy(view(cache, 2:4)) == [2, 3, 4]
    @test cache.counter == 6
    @test generate!(cache, Int[]) === cache
    @test cache.counter == 6
    cache.counter = UInt64(typemax(UInt32))
    @test cache[1] == 1
    @test cache.counter == UInt64(typemax(UInt32)) + 1
end

include("SparseJITArrays_test.jl")
include("Maps_test.jl")
