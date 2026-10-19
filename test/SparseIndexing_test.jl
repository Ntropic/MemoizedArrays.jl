# Shared signed indexing and sparse constructor boundaries.
@testset "Sparse axes and selections" begin
    cache = MemoizedArray((i, j) -> i + 10j, ArrayMap{Int}((-4:4, -3:3)))
    @test size(cache) == (9, 7)
    @test first(axes(cache, 1)) == -4
    @test cache[-1, 1] == cache[CartesianIndex(-1, 1)] == 9
    @test_throws ArgumentError cache[1]
    @test cache[-2:2, -1:1] == [i + 10j for i in -2:2, j in -1:1]
    @test copy(view(cache, -2:2, -1:1)) == [i + 10j for i in -2:2, j in -1:1]
    @test_throws BoundsError cache[-5, 0]
    explicit = MemoizedArray(tuple, DictMap{Tuple{Int,Int}}((typemin(Int):typemax(Int), typemin(Int):typemax(Int))))
    @test explicit[typemin(Int), typemax(Int)] == (typemin(Int), typemax(Int))
    @test_throws OverflowError length(explicit)
    full = JA.AxisExtent(typemin(Int), typemax(Int))
    @test checkindex(Bool, full, typemin(Int))
    @test checkindex(Bool, full, typemax(Int))
    @test !checkindex(Bool, full, big(typemax(Int)) + 1)
    @test !checkindex(Bool, full, Inf)
    scalar = MemoizedArray(() -> 7, ArrayMap{Int}(UnitRange{Int}[]))
    @test scalar.map isa JA.ArrayMap
    @test scalar[] == 7
    @test_throws ArgumentError scalar[1]
    @test_throws ArgumentError MemoizedArray(identity, DictMap{Int}(unbound(-1)))
    @test_throws MethodError MemoizedArray(identity, :unknown)
    @test_throws MethodError MemoizedArray(identity, :radix)
    @test_throws ArgumentError MemoizedArray(identity, RadixMap{Int}((1:3,); block_shape=(3,)))
    @test_throws DimensionMismatch MemoizedArray(identity, RadixMap{Int}((1:3,); block_shape=[8, 8]))
end

struct DictMemoizedConversion{F}
    convert::F
end
Base.convert(::Type{Int}, value::DictMemoizedConversion) = value.convert()

@testset "Dictionary Memoized publication preserves values and conversion effects" begin
    values = (nothing, false, JA.EMPTY, JA.GENERATING, JA.RELOCATED,
        JA.HASH_EMPTY, JA.HASH_DELETED, JA.GenerationFailure(ErrorException("a cached value")))
    broad = MemoizedArray(i -> values[i], DictMap{Any}((length(values),)))
    @test all(i -> broad[i] === values[i], eachindex(values))
    @test all(i -> broad[i] === values[i], reverse(eachindex(values)))
    @test all(i -> isready(broad, i), eachindex(values))

    owner = Ref{Any}()
    calls = Ref(0)
    converted = DictMemoizedConversion(() -> begin
        generate!(owner[], 1:80) # conversion rehashes the table before the outer record is inserted
        0
    end)
    nested = MemoizedArray(i -> begin
        calls[] += 1
        if i == 0
            return converted
        end
        return i
    end, DictMap{Int}((0:80,)))
    owner[] = nested
    @test nested[0] == 0
    @test all(i -> nested[i] == i, 0:80)
    @test calls[] == 81

    for conversion in (false, true)
        failure = ErrorException("dictionary generation or conversion failed")
        failing = MemoizedArray(i -> begin
            if i == 2
                if conversion
                    return DictMemoizedConversion(() -> throw(failure))
                end
                throw(failure)
            end
            return i
        end, DictMap{Int}((1:2,)))
        @test failing[1] == 1
        @test_throws ErrorException failing[2]
        @test !isready(failing, 2) && isready(failing, 1)
        @test JA.map_lookup(failing, (2,))[3] === nothing
    end

    matrix = ones(2,2)
    matrices = MemoizedArray(Returns(matrix), DictMap{typeof(matrix)}((2,)))
    @test matrices[1] === matrix
    @test matrices[2] === matrix && matrices[1] === matrix
    readyallocations(matrices)
    @test readyallocations(matrices) == 0
end
