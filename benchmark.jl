# Compare synchronous storage and cache operations with one timed loop.
# Setup, prefill, round-boundary GC and validation are outside timing.
using JITArrays
using Random

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

# Match JITArray's stored representation while excluding cache-miss handling.
@inline function mapread(map::JITArrays.DictMap{Union{JITArrays.Empty,JITArrays.JITEntry{Value}},N}, key) where N
    slot = JITArrays.store_lookup(map, key, JITArrays.store_hash(map, key))
    return slot === nothing ? nothing : JITArrays.slot_entry(slot).value
end
@inline function mapinsert!(map::JITArrays.DictMap{Union{JITArrays.Empty,JITArrays.JITEntry{Value}},N}, key) where N
    stored = JITArrays.JITEntry(PRECOMPUTED_VALUE)
    slot = JITArrays.store_reserve!(map, key, JITArrays.store_hash(map, key))
    JITArrays.slot_publish!(slot, stored)
    return stored.value
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

# Every implementation writes one result per request, using the same specialized loop.
function mapoperations!(output, maps, keys, operation)
    for map in maps, i in eachindex(keys)
        @inbounds output[i] = operation!(map, keys[i], operation)
    end
    return nothing
end

function measured(output, maps, keys, operation)
    sample = @timed mapoperations!(output, maps, keys, operation)
    requests = length(maps) * length(keys)
    return (sample.time * 1e9 / requests, sample.bytes / requests, sample.gctime * 1e9 / requests)
end

function row(workload, name, samples)
    times = sort(first.(samples))
    stats = map(i -> round(times[i]; digits=2), (cld(ROUNDS, 2), 2, ROUNDS - 1))
    bytes = sort(getindex.(samples, 2))[cld(ROUNDS, 2)]
    gc = sort(getindex.(samples, 3))[cld(ROUNDS, 2)]
    println("| ", workload, " | ", name, " | ", stats[1], " | ", stats[2], "–", stats[3],
        " | ", round(bytes; digits=2), " | ", round(gc; digits=2), " |")
end
function tableheader(unit)
    println("\n| Workload | Implementation | Median ns/", unit, " | p10–p90 ns/", unit, " | Bytes/", unit, " | GC ns/", unit, " |")
    println("|---|---|---:|---:|---:|---:|")
end

hasvalue(cache::AbstractJITArray, key) = isready(cache, key)
hasvalue(map, key) = mapread(map, key) === PRECOMPUTED_VALUE

function comparemaps(names, makers, warmkeys, requestkeys, operations)
    output = Vector{Value}(undef, max(length(warmkeys), length(requestkeys)))
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
            measured(output, setup(make), keys, Val(operation))
        end
        for round in 1:ROUNDS
            GC.gc()
            for i in circshift(collect(eachindex(makers)), round)
                maps = setup(makers[i])
                push!(samples[i], measured(output, maps, keys, Val(operation)))
                @assert all(x -> x === PRECOMPUTED_VALUE, view(output, eachindex(keys)))
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
            row(label, names[i], samples[i])
        end
    end
    return nothing
end

# Optional external comparison: using DataStructures; comparetrees(SortedDict, Trie).
function comparetrees(SortedDict, Trie; maps=(:msbtree, :lsbtree))
    @assert !isempty(maps) && all(m -> m in (:msbtree, :lsbtree), maps) "Trees: msbtree, lsbtree"
    println("Julia ", VERSION, "; DataStructures ", pkgversion(parentmodule(SortedDict)), "; ", ROUNDS, " interleaved rounds.")
    domain = (typemin(Int):typemax(Int), typemin(Int):typemax(Int))
    constructors = (msbtree=JITArrays.MSBRadixMap, lsbtree=JITArrays.LSBRadixMap)
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
        tableheader("op")
        comparemaps(names, makers, warm, requests, (("Lookup", :read), ("Insert", :insert)))
        # DataStructures.Trie has no deletion operation.
        comparemaps(names[1:end-1], makers[1:end-1], warm, requests, (("Delete", :delete), ("Delete + insert", :churn)))
    end
end

function benchmark(; maps=(:array, :dict, :msbtree, :lsbtree), caches=(:jit, :exact, :sampled))
    @assert !isempty(maps) && all(m -> m in (:array, :dict, :msbtree, :lsbtree), maps) "Maps: array, dict, msbtree, lsbtree"
    @assert !isempty(caches) && all(c -> c in (:jit, :exact, :sampled), caches) "Caches: jit, exact, sampled"
    println("Julia ", VERSION, "; ", ROUNDS, " rotated rounds; ", GENERATION_CACHES, " containers per sample.")
    println("One timed loop, one output write/request. Setup and prefill are untimed; full GC precedes each round. GC during operations is included.")
    println("Insertion/cache miss stores one precomputed matrix. Ready reads and misses use the same loop.")
    for storage in unique(maps)
        shape = storage in (:msbtree, :lsbtree) ? (8, 8) : nothing
        domain = storage === :array ? (4096,) : (typemin(Int):typemax(Int), typemin(Int):typemax(Int))
        options = (; storage=storage === :array ? nothing : storage, block_shape=shape)
        constructors = (jit=JITArray{Value}, exact=ExactLRUArray{Value}, sampled=SampledLRUArray{Value})
        names = (jit="JITArray", exact="ExactLRUArray", sampled="SampledLRUArray")
        selected = Tuple(unique(caches))
        makers = map(selected) do policy
            constructor = constructors[policy]
            settings = policy === :jit ? options : (; options..., capacity=CAPACITY)
            () -> constructor(value, domain; settings...)
        end
        labels = map(policy -> names[policy], selected)
        warm = storage === :array ? collect(1:CAPACITY) : [(2mod(i, 32), 2div(i, 32)) for i in 1:CAPACITY]
        requests = [warm[mod(i * 73, CAPACITY) + 1] for i in 1:32768]
        if storage === :array
            rawlabels, rawmakers = ("Array",), (() -> Vector{Value}(undef, 4096),)
        elseif storage === :dict
            rawlabels = ("Dict", "DictMap", "DictMap + JITEntry")
            rawmakers = (() -> Dict{NTuple{2,Int},Value}(),
                () -> JITArrays.DictMap(Union{JITArrays.Empty,Value}, Val(2), JITArrays.EMPTY),
                () -> JITArrays.DictMap(Union{JITArrays.Empty,JITArrays.JITEntry{Value}}, Val(2), JITArrays.EMPTY))
        else
            Map = storage === :msbtree ? JITArrays.MSBRadixMap : JITArrays.LSBRadixMap
            rawlabels = (string(nameof(Map)),)
            rawmakers = (() -> Map(Union{JITArrays.Empty,Value}, domain, shape, JITArrays.EMPTY),)
        end
        println("\n## ", storage)
        tableheader("request")
        comparemaps((rawlabels..., labels...), (rawmakers..., makers...), warm, requests,
            (("Ready lookup", :read), ("Insertion / cache miss", :insert)))
        bounded = Tuple(findall(!=(:jit), selected))
        boundedlabels = map(i -> labels[i], bounded)
        boundedmakers = map(i -> makers[i], bounded)
        if storage !== :array
            comparemaps((rawlabels..., boundedlabels...), (rawmakers..., boundedmakers...), warm, requests,
                (("Replacement / miss + eviction", :churn),))
            comparemaps(rawlabels, rawmakers, warm, requests, (("Delete", :delete),))
        elseif !isempty(bounded)
            comparemaps(boundedlabels, boundedmakers, warm, requests, (("Miss + eviction", :churn),))
        end
        if storage === :dict
            comparemaps(rawlabels, rawmakers, warm, requests,
                (("Insert, reserved", :reserved), ("Replace at load limit", :threshold)))
        end
    end
end

function selections(args)
    options = Dict{Symbol,Tuple{Vararg{Symbol}}}(:maps => (:array, :dict, :msbtree, :lsbtree), :caches => (:jit, :exact, :sampled))
    for arg in args
        parts = split(arg, '='; limit=2)
        if length(parts) != 2 || !(parts[1] in ("--maps", "--caches"))
            error("Use --maps=array,dict,msbtree,lsbtree --caches=jit,exact,sampled (default: all)")
        end
        options[Symbol(parts[1][3:end])] = Tuple(Symbol.(split(parts[2], ',')))
    end
    return (; options...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark(; selections(ARGS)...)
end
