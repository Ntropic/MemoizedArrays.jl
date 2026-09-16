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
    for key in ((typemin(Int), typemax(Int)), (0, -1), (typemax(Int), typemin(Int)))
        @test full[key...] == key
    end
    @test_throws OverflowError size(full)
    @test_throws OverflowError length(full)
    @test full[-1:1, 0:1] == [(i, j) for i in -1:1, j in 0:1]
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
    @test_throws ArgumentError JITArray{Int}(identity, 1:3; storage=:msbtree)
    @test_throws ArgumentError JITArray{Int}(identity, 1:3; storage=:msbtree, block_shape=(3,))
    @test_throws DimensionMismatch JITArray{Int}(identity, 1:3; storage=:msbtree, block_shape=[8, 8])
end
