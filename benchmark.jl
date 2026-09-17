# Compare synchronous storage and cache operations with one timed loop.
# Setup, prefill, round-boundary GC and validation are outside timing.
using JITArrays
using Random
using LinearAlgebra
using Statistics

const Value = Matrix{Float64}
const CAPACITY = 512
const ROUNDS = 15
const GENERATION_CACHES = 8
const PRECOMPUTED_VALUE = fill(1.0, 4, 4)
@inline value(keys...) = PRECOMPUTED_VALUE
@inline readvalue(cache, key::Int) = @inbounds cache[key]
@inline readvalue(cache, key::Tuple) = @inbounds cache[key...]
@inline replacement(key::Int) = key + CAPACITY
@inline replacement(key::Tuple) = (key[1] + 128, Base.tail(key)...)

@inline mapread(map, key) = get(map, key, nothing)
@inline mapread(cache::AbstractJITArray, key) = readvalue(cache, key)
@inline function mapread(map::JITArrays.SparseMap, key)
    slot = JITArrays.store_lookup(map, key, JITArrays.store_hash(map, key))
    return slot === nothing ? nothing : JITArrays.slot_entry(slot)
end
@inline mapinsert!(map, key) = (map[key] = PRECOMPUTED_VALUE)
@inline mapinsert!(cache::AbstractJITArray, key) = readvalue(cache, key)
@inline function mapinsert!(map::JITArrays.SparseMap, key)
    slot = JITArrays.store_reserve!(map, key, JITArrays.store_hash(map, key))
    JITArrays.store_retain!(map, slot)
    return JITArrays.slot_publish!(slot, PRECOMPUTED_VALUE)
end

@inline mapdelete!(map, key) = delete!(map, key)
@inline mapdelete!(map::JITArrays.SparseMap, key) = JITArrays.delete_key!(map, key)
mapsizehint!(map::Dict) = sizehint!(map, CAPACITY)
mapsizehint!(map::JITArrays.DictMap) = JITArrays.hash_rebuild!(map, 2CAPACITY)

@inline operation!(map, key, ::Val{:read}) = mapread(map, key)
@inline operation!(map, key, ::Val{:insert}) = mapinsert!(map, key)
@inline function operation!(map, key, ::Val{:delete})
    mapdelete!(map, key)
    return PRECOMPUTED_VALUE
end
@inline function operation!(map, key, ::Val{:churn})
    mapdelete!(map, key)
    return mapinsert!(map, replacement(key))
end
@inline operation!(cache::AbstractJITArray, key, ::Val{:churn}) = readvalue(cache, replacement(key))
@inline function operation!(map, key, ::Val{:threshold})
    mapdelete!(map, key)
    return mapinsert!(map, key)
end
@inline operation!(map, key, ::Val{:reserved}) = mapinsert!(map, key)

# Consume one matrix element per request without writing an output array.
function mapoperations!(maps, keys, operation)
    checksum = 0.0
    for map in maps, key in keys
        result = operation!(map, key, operation)
        checksum += @inbounds result[1, 1]
    end
    return checksum
end

function measured(maps, keys, operation)
    sample = @timed mapoperations!(maps, keys, operation)
    requests = length(maps) * length(keys)
    @assert sample.value == requests
    return (sample.time * 1e9 / requests, sample.bytes / requests, sample.gctime * 1e9 / requests)
end

formattime(ns) = ns < 1000 ? string(round(ns; digits=1), " ns") : string(round(ns / 1000; digits=1), " µs")

function benchmarktable(results)
    println("\nMedian time per operation.")
    names = unique(first.(results))
    workloads = unique(getindex.(results, 2))
    println("\n| Implementation | ", join(workloads, " | "), " |")
    println("|---|", join(fill("---:", length(workloads)), "|"), "|")
    for name in names
        timings = Dict((r[2], r[3]) for r in results if r[1] == name)
        values = [haskey(timings, workload) ? formattime(timings[workload]) : "—" for workload in workloads]
        println("| ", name, " | ", join(values, " | "), " |")
    end
end

hasvalue(cache::AbstractJITArray, key) = isready(cache, key)
hasvalue(map, key) = mapread(map, key) === PRECOMPUTED_VALUE

function comparemaps(names, makers, warmkeys, requestkeys, operations, results=Tuple{String,String,Float64}[])
    for (label, operation) in operations
        residents = operation === :threshold ? 48 : CAPACITY
        keys = operation === :read ? requestkeys : operation === :threshold ?
            [warmkeys[mod(i - 1, residents) + 1] for i in 1:CAPACITY] : warmkeys
        checked = operation === :churn ? replacement.(warmkeys) : view(warmkeys, 1:residents)
        function setup(make)
            maps = [make() for _ in 1:GENERATION_CACHES]
            for map in maps
                if operation === :reserved
                    mapsizehint!(map)
                elseif operation in (:read, :delete, :churn, :threshold)
                    for i in 1:residents
                        mapinsert!(map, warmkeys[i])
                    end
                end
            end
            return maps
        end
        samples = [NTuple{3,Float64}[] for _ in makers]
        for make in makers
            measured(setup(make), keys, Val(operation))
        end
        for round in 1:ROUNDS
            GC.gc()
            for i in circshift(collect(eachindex(makers)), round)
                maps = setup(makers[i])
                push!(samples[i], measured(maps, keys, Val(operation)))
                for map in maps
                    expected = operation === :delete ? 0 : residents
                    if map isa AbstractDict
                        @assert length(map) == expected
                    elseif map isa JITArrays.DictMap
                        @assert map.count == expected
                    end
                    if operation !== :delete && !(operation === :churn && map isa AbstractJITArray)
                        @assert all(key -> hasvalue(map, key), checked)
                    end
                    if operation === :delete || (operation === :churn && !(map isa AbstractJITArray))
                        @assert all(key -> !hasvalue(map, key), warmkeys)
                    elseif operation === :churn
                        @assert count(key -> hasvalue(map, key), [warmkeys; checked]) == residents
                    end
                end
            end
        end
        for i in eachindex(makers)
            push!(results, (names[i], label, median(first.(samples[i]))))
        end
    end
    return results
end

# Optional external comparison: using DataStructures; comparetrees(SortedDict, Trie).
function comparetrees(SortedDict, Trie; maps=(:radixtree, :subtreetree))
    @assert !isempty(maps) && all(m -> m in (:radixtree, :subtreetree), maps) "Trees: radixtree, subtreetree"
    println("Julia ", VERSION, "; DataStructures ", pkgversion(parentmodule(SortedDict)), "; ", ROUNDS, " interleaved rounds.")
    domain = (typemin(Int):typemax(Int), typemin(Int):typemax(Int))
    constructors = (radixtree=JITArrays.RadixMap, subtreetree=JITArrays.SubtreeRadixMap)
    selected = Tuple(unique(maps))
    trees = map(selected) do storage
        Map = constructors[storage]
        () -> Map(Union{JITArrays.Empty,Value}, domain, (8, 8), JITArrays.EMPTY)
    end
    makers = (trees..., () -> SortedDict{NTuple{2,Int},Value}(), () -> Dict{NTuple{2,Int},Value}(), () -> Trie{Int,Value}())
    names = (map(storage -> string(nameof(constructors[storage])), selected)..., "SortedDict", "Dict", "Trie")
    rng = Xoshiro(2026)
    nearby = [(2mod(i, 32), 2div(i, 32)) for i in 1:CAPACITY]
    scattered = [(rand(rng, Int), rand(rng, Int)) for _ in 1:CAPACITY]
    mixed = [nearby[1:end-16]; scattered[1:16]]
    for (label, warm) in (("Nearby keys", nearby), ("Scattered signed keys", scattered),
        ("Mostly nearby keys, 16 large outliers", mixed))
        @assert length(unique(warm)) == CAPACITY
        @assert isempty(intersect(warm, [(key[1] + 128, key[2]) for key in warm]))
        if warm === mixed
            requests = [mod(i, 100) == 0 ? warm[CAPACITY - 15 + mod(div(i, 100), 16)] :
                warm[mod(i * 73, CAPACITY - 16) + 1] for i in 1:32768]
        else
            requests = [warm[mod(i * 73, CAPACITY) + 1] for i in 1:32768]
        end
        println("\n### ", label)
        results = Tuple{String,String,Float64}[]
        comparemaps(names, makers, warm, requests, (("Lookup", :read), ("Insert", :insert)), results)
        # DataStructures.Trie has no deletion operation.
        comparemaps(names[1:end-1], makers[1:end-1], warm, requests, (("Delete", :delete), ("Delete + insert", :churn)), results)
        benchmarktable(results)
    end
end

function benchmark(; maps=(:array, :dict, :radixtree, :subtreetree), caches=(:jit, :exact, :clock, :sieve))
    @assert !isempty(maps) && all(m -> m in (:array, :dict, :radixtree, :subtreetree), maps) "Maps: array, dict, radixtree, subtreetree"
    @assert !isempty(caches) && all(c -> c in (:jit, :exact, :clock, :sieve), caches) "Caches: jit, exact, clock, sieve"
    println("Julia ", VERSION, "; ", ROUNDS, " rotated rounds; ", GENERATION_CACHES, " containers per sample.")
    println("One timed loop, one matrix-element checksum/request; no output-array writes. Setup and prefill are untimed; full GC precedes each round. GC during operations is included.")
    println("Insertion/cache miss stores one precomputed matrix. Ready reads and misses use the same loop.")
    for storage in unique(maps)
        shape = storage in (:radixtree, :subtreetree) ? (8, 8) : nothing
        domain = storage === :array ? (4096,) : (typemin(Int):typemax(Int), typemin(Int):typemax(Int))
        options = (; storage=storage === :array ? nothing : storage, block_shape=shape)
        constructors = (jit=JITArray{Value}, exact=ExactLRUArray{Value}, clock=SampledLRUArray{Value}, sieve=SampledLRUArray{Value})
        names = (jit="JITArray", exact="ExactLRUArray", clock="SampledLRUArray (:clock)", sieve="SampledLRUArray (:sieve)")
        selected = Tuple(unique(caches))
        makers = map(selected) do policy
            constructor = constructors[policy]
            settings = policy === :jit ? options : policy === :exact ? (; options..., capacity=CAPACITY) : (; options..., capacity=CAPACITY, policy)
            () -> constructor(value, domain; settings...)
        end
        labels = map(policy -> names[policy], selected)
        warm = storage === :array ? collect(1:CAPACITY) : [(2mod(i, 32), 2div(i, 32)) for i in 1:CAPACITY]
        requests = [warm[mod(i * 73, CAPACITY) + 1] for i in 1:32768]
        if storage === :array
            rawlabels, rawmakers = ("Array",), (() -> Vector{Value}(undef, 4096),)
        elseif storage === :dict
            rawlabels = ("Dict",)
            rawmakers = (() -> Dict{NTuple{2,Int},Value}(),)
        else
            Map = storage === :radixtree ? JITArrays.RadixMap : JITArrays.SubtreeRadixMap
            rawlabels = (string(nameof(Map)),)
            rawmakers = (() -> Map(Union{JITArrays.Empty,Value}, domain, shape, JITArrays.EMPTY),)
        end
        println("\n## ", storage)
        results = Tuple{String,String,Float64}[]
        comparemaps((rawlabels..., labels...), (rawmakers..., makers...), warm, requests,
            (("Read", :read), ("Initial generation / insertion", :insert)), results)
        bounded = Tuple(findall(!=(:jit), selected))
        boundedlabels = map(i -> labels[i], bounded)
        boundedmakers = map(i -> makers[i], bounded)
        if storage !== :array
            comparemaps((rawlabels..., boundedlabels...), (rawmakers..., boundedmakers...), warm, requests,
                (("Generate + evict / replace", :churn),), results)
            if storage !== :dict
                comparemaps(rawlabels, rawmakers, warm, requests, (("Delete", :delete),), results)
            end
        elseif !isempty(bounded)
            comparemaps(boundedlabels, boundedmakers, warm, requests, (("Generate + evict", :churn),), results)
        end
        benchmarktable(results)
    end
end

function selections(args)
    options = Dict{Symbol,Tuple{Vararg{Symbol}}}(:maps => (:array, :dict, :radixtree, :subtreetree), :caches => (:jit, :exact, :clock, :sieve))
    for arg in args
        parts = split(arg, '='; limit=2)
        if length(parts) != 2 || !(parts[1] in ("--maps", "--caches"))
            error("Use --maps=array,dict,radixtree,subtreetree --caches=jit,exact,clock,sieve (default: all)")
        end
        options[Symbol(parts[1][3:end])] = Tuple(Symbol.(split(parts[2], ',')))
    end
    return (; options...)
end

# LAPACK DSYEVR, matching Julia's default symmetric eigensolver, with caller-owned buffers.
function matrixeigh!(W, eigenvalues, eigenvectors, support, work, iwork, found, info; query=false)
    n = size(W, 1)
    lwork, liwork = query ? (-1, -1) : (length(work), length(iwork))
    ccall((LinearAlgebra.BLAS.@blasfunc(dsyevr_), LinearAlgebra.LAPACK.libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BlasInt},
         Ptr{Float64}, Ref{LinearAlgebra.BlasInt}, Ref{Float64}, Ref{Float64},
         Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ref{Float64}, Ptr{LinearAlgebra.BlasInt},
         Ptr{Float64}, Ptr{Float64}, Ref{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt},
         Ptr{Float64}, Ref{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt},
         Ref{LinearAlgebra.BlasInt}, Clong, Clong, Clong),
        'V', 'A', 'U', n, W, stride(W, 2), 0.0, 0.0, 0, 0, -1.0, found,
        eigenvalues, eigenvectors, stride(eigenvectors, 2), support, work, lwork, iwork, liwork, info, 1, 1, 1)
    LinearAlgebra.LAPACK.chklapackerror(info[])
    return nothing
end

function matrixoperations!(inputs, C, W, eigenvalues, eigenvectors, support, work, iwork, found, info, ::Val{operation}) where operation
    checksum = 0.0
    for i in 1:4096
        A, B = @inbounds inputs[mod1(i, length(inputs))]
        if operation === :addition
            C .= A .+ B
            checksum += @inbounds C[1]
        elseif operation === :product
            mul!(C, A, B)
            checksum += @inbounds C[1]
        else
            copyto!(W, A)
            matrixeigh!(W, eigenvalues, eigenvectors, support, work, iwork, found, info)
            checksum += @inbounds eigenvalues[1]
        end
    end
    return checksum
end

"""Benchmark supplied-output matrix operations at powers of two from 2 through `maxsize` (default 16)."""
function benchmarkmatrices(maxsize::Int=16)
    maxsize >= 2 || throw(ArgumentError("maximum matrix size must be at least 2"))
    BLAS.set_num_threads(1)
    println("\n## Matrix operations with supplied outputs")
    println("\n| Size | Addition | Multiplication | eigh |")
    println("|---|---:|---:|---:|")
    rng = Xoshiro(2026)
    for power in 1:floor(Int, log2(maxsize))
        n = 2^power
        inputs = [(randn(rng, n, n), randn(rng, n, n)) for _ in 1:64]
        C, W, eigenvalues, eigenvectors = zeros(n,n), zeros(n,n), zeros(n), zeros(n,n)
        support = Vector{LinearAlgebra.BlasInt}(undef, 2n)
        work, iwork = zeros(1), zeros(LinearAlgebra.BlasInt, 1)
        found, info = Ref{LinearAlgebra.BlasInt}(0), Ref{LinearAlgebra.BlasInt}(0)
        matrixeigh!(W, eigenvalues, eigenvectors, support, work, iwork, found, info; query=true)
        resize!(work, Int(work[1]))
        resize!(iwork, Int(iwork[1]))
        buffers = (inputs, C, W, eigenvalues, eigenvectors, support, work, iwork, found, info)
        operations = (Val(:addition), Val(:product), Val(:eigh))
        samples = [Float64[] for _ in operations]
        bytes = [Float64[] for _ in operations]
        for operation in operations
            matrixoperations!(buffers..., operation)
        end
        A, B = inputs[end]
        @assert found[] == n
        @assert eigenvectors * Diagonal(eigenvalues) * eigenvectors' ≈ Matrix(Symmetric(A))
        @assert eigenvectors' * eigenvectors ≈ I
        @assert eigenvalues ≈ eigvals(Symmetric(A))
        matrixoperations!(buffers..., Val(:addition))
        @assert C ≈ A + B
        matrixoperations!(buffers..., Val(:product))
        @assert C ≈ A * B
        for round in 1:ROUNDS
            GC.gc()
            for offset in 0:2
                j = mod1(round + offset, 3)
                sample = @timed matrixoperations!(buffers..., operations[j])
                @assert isfinite(sample.value)
                push!(samples[j], sample.time * 1e9 / 4096)
                push!(bytes[j], sample.bytes / 4096)
            end
        end
        @assert all(iszero, median.(bytes))
        println("| ", n, "×", n, " | ", join(formattime.(median.(samples)), " | "), " |")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark(; selections(ARGS)...)
    benchmarkmatrices()
end
