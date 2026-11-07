# Run with julia --project=. benchmark/slices.jl. Setup and compilation are excluded.
using MemoizedArrays, Statistics, Printf

function iterate_view(selected)
    total = 0
    for value in selected
        total += value
    end
    return total
end

function measure(f; repeats=100)
    f()
    times = Float64[]
    bytes = Int[]
    for _ in 1:9
        GC.gc()
        sample = @timed for _ in 1:repeats
            f()
        end
        push!(times, sample.time * 1e6 / repeats)
        push!(bytes, sample.bytes ÷ repeats)
    end
    return median(times), median(bytes)
end

function run_case(Map, C, pattern, dims)
    rows = pattern == :contiguous ? (10_001:10_000+dims[1]) : (10_001:32:10_001+32(dims[1]-1))
    cols = -dims[2]:-1
    config = Map{Int}(unbound(2); block_shape=(8, 8))
    makecache() = C === MemoizedArray ? C(+, config) : C(+, config; capacity=2prod(dims))
    cache = makecache()
    expected = [i + j for i in rows, j in cols]
    @assert cache[rows, cols] == expected
    selected = view(cache, rows, cols)
    dest = similar(expected)
    repeats = max(2, min(100, 1_000_000 ÷ prod(dims)))
    for (operation, f, result) in (("iterate view", () -> iterate_view(selected), sum(expected)),
                            ("slice", () -> cache[rows, cols], expected),
                            ("copy view", () -> copy(selected), expected),
                            ("copyto!", () -> copyto!(dest, selected), expected),
                            ("sum view", () -> sum(selected), sum(expected)),
                            ("sum(abs2) view", () -> sum(abs2, selected), sum(abs2, expected)),
                            ("map! view", () -> map!(abs2, dest, selected), abs2.(expected)))
        @assert f() == result
        us, bytes = measure(f; repeats)
        @printf("%s,%s,%dx%d,%s,%s,%.2f,%.0f\n", Map, C, dims..., pattern, operation, us, bytes)
    end
    # Each cold sample uses a fresh cache; construction is outside timing.
    for (operation, cold, result) in (
            ("cold slice", cache -> cache[rows,cols], expected),
            ("cold sum view", cache -> sum(view(cache,rows,cols)), sum(expected)),
            ("cold iterate view", cache -> iterate_view(view(cache,rows,cols)), sum(expected)))
        cold(makecache())
        times = Float64[]
        for _ in 1:9
            fresh = makecache()
            GC.gc()
            sample = @timed cold(fresh)
            @assert sample.value == result
            push!(times, sample.time * 1e6)
        end
        @printf("%s,%s,%dx%d,%s,%s,%.2f,-\n", Map, C, dims..., pattern, operation, median(times))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# Julia ", VERSION, "; CPU ", Sys.CPU_NAME, "; 8x8 leaves; nine samples; 2–100 warm calls per sample")
    println("map,cache,view_shape,selection,operation,microseconds,bytes")
    for Map in (RadixMap, SubtreeRadixMap), C in (MemoizedArray, LRUMemoizedArray)
        for dims in ((8,8), (128,64), (1024,256), (4096,1), (1,4096))
            run_case(Map, C, :contiguous, dims)
        end
        run_case(Map, C, :scattered, (128,64))
    end
end
