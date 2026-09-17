# Shared signed indexing and sparse constructor boundaries.
@testset "Sparse axes and selections" begin
    cache = JITArray{Int}((i, j) -> i + 10j, -4:4, -3:3)
    @test size(cache) == (9, 7)
    @test first(axes(cache, 1)) == -4
    @test cache[-1, 1] == cache[CartesianIndex(-1, 1)] == 9
    @test cache[1] == -34
    @test cache[-2:2, -1:1] == [i + 10j for i in -2:2, j in -1:1]
    @test copy(view(cache, -2:2, -1:1)) == [i + 10j for i in -2:2, j in -1:1]
    @test_throws BoundsError cache[-5, 0]
    full = SparseJITArray{Tuple{Int,Int}}(tuple; ndims=2)
    low, high = first(axes(full, 1)), last(axes(full, 1))
    @test full[low, high] == (low, high)
    @test full[1] == (low, low)
    @test full[length(full)] == (high, high)
    @test_throws BoundsError full[low - 1, 0]
    @test full[-1:1, 0:1] == [(i, j) for i in -1:1, j in 0:1]
    for n in (0, 1, 2, 3, 8, 64)
        ranges = JA.int_axes(n)
        @test JA.domain_length(ranges) == prod(big(length(r)) for r in ranges; init=big(1)) <= typemax(Int)
        if n > 0
            width = length(first(ranges))
            @test (big(width) + 1)^n > typemax(Int)
        end
    end
    explicit = SparseJITArray{Tuple{Int,Int}}(tuple, typemin(Int):typemax(Int), typemin(Int):typemax(Int))
    @test explicit[typemin(Int), typemax(Int)] == (typemin(Int), typemax(Int))
    @test_throws OverflowError length(explicit)
    full = JA.AxisExtent(typemin(Int), typemax(Int))
    @test checkindex(Bool, full, typemin(Int))
    @test checkindex(Bool, full, typemax(Int))
    @test !checkindex(Bool, full, big(typemax(Int)) + 1)
    @test !checkindex(Bool, full, Inf)
    scalar = JITArray{Int}(() -> 7, UnitRange{Int}[])
    @test scalar.map isa JA.DictMap
    @test scalar[] == scalar[1] == 7
    @test_throws ArgumentError SparseJITArray{Int}(identity; ndims=-1)
    @test_throws ArgumentError JITArray{Int}(identity, 1:3; storage=:unknown)
    @test_throws ArgumentError JITArray{Int}(identity, 1:3; storage=:radixtree)
    @test_throws ArgumentError JITArray{Int}(identity, 1:3; storage=:radixtree, block_shape=(3,))
    @test_throws DimensionMismatch JITArray{Int}(identity, 1:3; storage=:radixtree, block_shape=[8, 8])
end
