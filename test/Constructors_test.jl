using Test, JITArrays

@testset "Generator-first constructors" begin
    for T in (JITArray, LRUArray, ExactLRUArray, SampledLRUArray)
        options = T === JITArray ? (;) : (; capacity=4)
        for domain in ((2, 3), [2, 3])
            a = T{Tuple{Int,Int}}(tuple, domain; options...)
            @test a[2, 3] == (2, 3)
        end
        a = T{Tuple{Int,Int}}(2, 3; options...) do i, j
            (i, j)
        end
        @test a[2, 3] == (2, 3)
        @test T{Int}(() -> 7; options...)[] == 7
    end
    for T in (JITArray, LRUArray, ExactLRUArray, SampledLRUArray, SparseJITArray, SparseLRUArray)
        options = T in (JITArray, SparseJITArray) ? (;) : (; capacity=4)
        for storage in (:dict, :msbtree, :lsbtree)
            blocks = storage === :dict ? (;) : (; block_shape=(2, 4))
            a = T{Tuple{Int,Int}}(tuple, -10:10, -20:20; storage, options..., blocks...)
            @test a[-3, 2] == (-3, 2)
            b = T{Tuple{Int,Int}}(tuple, [-10:10, -20:20]; storage, options..., blocks...)
            @test b[-3, 2] == (-3, 2)
        end
    end
    @test_throws UndefKeywordError LRUArray{Int}(identity, 4)
    @test JITArray{Int}((4,), identity)[2] == 2
    @test LRUArray{Int}((4,), 2, identity)[2] == 2
end

@testset "Full Int coordinate domains" begin
    for storage in (:dict, :msbtree, :lsbtree), policy in (:none, :exact, :sampled)
        options = storage === :dict ? (; storage) : (; storage, block_shape=(4, 4))
        cache = policy === :none ? SparseJITArray{Tuple{Int,Int}}(tuple; ndims=2, options...) :
            SparseLRUArray{Tuple{Int,Int}}(tuple; ndims=2, capacity=2, policy, options...)
        @test ndims(cache) == 2
        @test all(axis -> axis isa JITArrays.AxisExtent, axes(cache))
        for key in ((typemin(Int), typemax(Int)), (0, -1), (typemax(Int), typemin(Int)))
            @test cache[key...] == key
        end
        @test_throws OverflowError size(cache)
        @test_throws OverflowError length(cache)
    end
    @test SparseJITArray{Int}(identity; ndims=1)[typemin(Int)] == typemin(Int)
    @test SparseJITArray{Int}(() -> 7; ndims=0)[] == 7
    @test SparseLRUArray{Int}(() -> 7; ndims=0, capacity=1)[] == 7
    @test_throws ArgumentError SparseJITArray{Int}(identity; ndims=-1)
    @test_throws ArgumentError SparseLRUArray{Int}(identity; ndims=-1, capacity=1)
    @test fieldnames(typeof(JITArray{Int}(identity, 2))) == (:map, :domain, :generator)
    @test !(ExactLRUArray{Int}(identity, 2; capacity=1) isa JITArray)
end
