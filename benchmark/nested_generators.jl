# julia --project=. benchmark/nested_generators.jl [output-directory]
# Set MEMOIZED_SOURCE to an isolated src/MemoizedArrays.jl to compare representations.
using LinearAlgebra, Random, Statistics, Printf, Profile, InteractiveUtils, Test
if haskey(ENV, "MEMOIZED_SOURCE")
    include(ENV["MEMOIZED_SOURCE"])
    using .MemoizedArrays
else
    using MemoizedArrays
end
BLAS.set_num_threads(1)
const Correction = Tuple{Matrix{ComplexF64},Vector{Float64}}
const SEGMENTS = 8

wrap_generator(g, ::Val{false}) = g
function wrap_generator(g, ::Val{true})
    generator_ref = Ref(g)
    return (indices...) -> generator_ref[](indices...)
end
buffers() = (; p=zeros(ComplexF64,4,4), tmp=zeros(ComplexF64,4,4),
    scaled=zeros(ComplexF64,4,4), rotation=zeros(ComplexF64,4,4))
polynomial(t, k) = k == 1 ? t : t * (t - 0.5)
function scale_columns!(out, vectors, angles, weight)
    for j in 1:4
        phase = cis(weight * angles[j])
        for i in 1:4
            out[i,j] = vectors[i,j] * phase
        end
    end
    return out
end
function apply_correction!(b, correction, t, k)
    vectors, angles = correction
    scale_columns!(b.scaled, vectors, angles, polynomial(t,k))
    mul!(b.rotation, b.scaled, adjoint(vectors))
    mul!(b.tmp, b.p, b.rotation)
    copyto!(b.p, b.tmp)
    return b.p
end
function evaluate_pairs!(b, u0, q1, q2, t)
    copyto!(b.p, u0)
    apply_correction!(b, q1, t, 1)
    apply_correction!(b, q2, t, 2)
    return real(b.p[1,1])
end
function evaluate!(b, exact, corrections, s, t)
    return evaluate_pairs!(b, exact[2s-1], corrections[s,1], corrections[s,2], t)
end
function evaluate_merged_pairs!(b, u0, q1, q2, merged, t)
    v1, a1 = q1
    v2, a2 = q2
    mul!(b.p, u0, v1)
    scale_columns!(b.p, b.p, a1, polynomial(t,1))
    mul!(b.tmp, b.p, merged)
    scale_columns!(b.tmp, b.tmp, a2, polynomial(t,2))
    mul!(b.p, b.tmp, adjoint(v2))
    return real(b.p[1,1])
end
function evaluate_merged!(b, exact, corrections, factors, s, t)
    return evaluate_merged_pairs!(b, exact[2s-1], corrections[s,1],
        corrections[s,2], factors[s], t)
end

function build_caches(H0, H1, x, wrapped)
    calls = zeros(Int,3)
    exact_generator = j -> begin
        calls[1] += 1
        exp(-im * (H0 + x[j]*H1))
    end
    exact = MemoizedArray(wrap_generator(exact_generator, wrapped),
        ArrayMap{Matrix{ComplexF64}}((length(x),)); concurrent=false)
    correction_generator = (s,k) -> begin
        calls[2] += 1
        t = k / 2
        b = buffers()
        copyto!(b.p, exact[2s-1])
        for earlier in 1:k-1
            apply_correction!(b, corrections[s,earlier], t, earlier)
        end
        residual = b.p' * exact[2s-1+k]
        eig = eigen(residual)
        (eig.vectors, angle.(eig.values) / polynomial(t,k))
    end
    # This self-reference is deliberately the ordinary closure construction:
    # both variants capture the same late-bound corrections variable.
    corrections = MemoizedArray(wrap_generator(correction_generator, wrapped),
        ArrayMap{Correction}((SEGMENTS,2)); concurrent=false)
    factor_generator = s -> begin
        calls[3] += 1
        corrections[s,1][1]' * corrections[s,2][1]
    end
    factors = MemoizedArray(wrap_generator(factor_generator, wrapped),
        ArrayMap{Matrix{ComplexF64}}((SEGMENTS,)); concurrent=false)
    # Controls: access through the same boxed binding as recursive generation,
    # versus a fresh binding to the completed cache. Neither generates new entries.
    captured_lookup = s -> real(corrections[s,2][1][1,1])
    stable_lookup = let cache = corrections
        s -> real(cache[s,2][1][1,1])
    end
    return (; exact, corrections, factors, calls, exact_generator,
        correction_generator, factor_generator, captured_lookup, stable_lookup)
end

# A scalar checksum keeps all results observable without collecting matrices.
@noinline function batch(f::F, repetitions) where F
    checksum = 0.0
    for i in 1:repetitions
        checksum += f(mod1(i,SEGMENTS))::Float64
    end
    return checksum
end
function measure(f::F; repetitions=10_000) where F
    batch(f, repetitions)
    bytes = @allocated batch(f, repetitions)
    times = Float64[]
    for _ in 1:7
        push!(times, @elapsed batch(f, repetitions))
    end
    return median(times)*1e9/repetitions, bytes/repetitions
end
function inspect_operation(f::F, path) where F
    batch(f, 1000)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1.0 batch(f, 1000)
    allocations = Profile.Allocs.fetch().allocs
    open(path * ".allocations.txt", "w") do io
        println(io, "1000 warmed calls; allocations=", length(allocations),
            "; bytes=", sum(a.size for a in allocations; init=0))
        counts = Dict{Any,Tuple{Int,Int}}()
        for a in allocations
            key = a.type
            n, bytes = get(counts, key, (0,0))
            counts[key] = (n+1, bytes+a.size)
        end
        for (kind, count) in sort!(collect(counts); by=pair -> string(first(pair)))
            println(io, count, " ", kind)
        end
        # One representative stack per allocation type.
        seen = Set{Any}()
        for a in allocations
            kind = a.type
            kind in seen && continue
            push!(seen,kind)
            println(io, "\n",kind, " bytes=", a.size)
            foreach(frame -> println(io,frame.func," at ",frame.file,":",frame.line), a.stacktrace)
        end
    end
    open(path * ".compiler.txt", "w") do io
        code_warntype(io, f, Tuple{Int}; debuginfo=:source)
        println(io,"\nLLVM for batch:")
        code_llvm(io, batch, Tuple{F,Int}; debuginfo=:source)
    end
end

function run_case(c, baseline, label, directory)
    (; exact, corrections, factors, calls) = c
    # Request k=2 first, exercising the recursive k=1 miss.
    for s in 1:SEGMENTS
        corrections[s,2]
        factors[s]
    end
    @test calls == [2SEGMENTS+1,2SEGMENTS,SEGMENTS]
    saved_calls = copy(calls)
    b = buffers()
    @testset "$label numerical checks" begin
        for s in 1:SEGMENTS, t in (0.0,0.21,0.5,0.79,1.0)
            evaluate!(b, exact, corrections, s, t)
            reference = copy(b.p)
            @test b.p' * b.p ≈ I atol=1e-10
            evaluate!(b, baseline.exact, baseline.corrections, s, t)
            @test b.p ≈ reference atol=1e-10
            evaluate_merged!(b, exact, corrections, factors, s, t)
            @test b.p ≈ reference atol=1e-10
            if t in (0.0,0.5,1.0)
                @test b.p ≈ exact[2s-1+Int(2t)] atol=1e-10
            end
        end
    end
    open(joinpath(directory,label * "_values.csv"),"w") do io
        for s in 1:SEGMENTS, t in (0.0,0.21,0.5,0.79,1.0)
            evaluate!(b, exact, corrections, s, t)
            for value in b.p
                println(io,real(value),",",imag(value))
            end
        end
    end
    # Ordinary arrays hold the same objects, isolating arithmetic from cache access.
    u = [exact[2s-1] for s in 1:SEGMENTS]
    q1 = [corrections[s,1] for s in 1:SEGMENTS]
    q2 = [corrections[s,2] for s in 1:SEGMENTS]
    operations = (
        ("correction_lookup", s -> real(corrections[s,2][1][1,1])),
        ("boxed_binding_lookup", c.captured_lookup),
        ("evaluate", s -> evaluate!(b,exact,corrections,s,0.37)),
        ("arithmetic_only", s -> evaluate_pairs!(b,u[s],q1[s],q2[s],0.37)))
    for (name, f) in operations
        ns, bytes = measure(f)
        @printf("%s,%s,%.2f,%.2f\n",label,name,ns,bytes)
        inspect_operation(f, joinpath(directory,label * "_" * name))
    end
    @test calls == saved_calls
    open(joinpath(directory,label * "_captures.txt"),"w") do io
        for name in (:exact_generator,:correction_generator,:factor_generator,
                     :captured_lookup,:stable_lookup)
            g = getproperty(c,name)
            println(io,name," fields: ",fieldtypes(typeof(g)))
        end
        println(io,"stored correction generator: ",typeof(corrections.generator))
        code_warntype(io, c.correction_generator, Tuple{Int,Int})
    end
end

# Each extra cache forwards the same merged factor. Only the generator's
# dependency graph grows; the matrix and work done on a warmed hit do not.
function dependency_layer(previous, wrapped)
    generator = s -> previous[s]
    return MemoizedArray(wrap_generator(generator,wrapped),
        ArrayMap{Matrix{ComplexF64}}((SEGMENTS,)); concurrent=false)
end
function evaluate_reader!(b, readers, exact, corrections, factors, s, t)
    u0 = readers[1](exact,2s-1)::Matrix{ComplexF64}
    q1 = readers[1](corrections,s,1)::Correction
    q2 = readers[1](corrections,s,2)::Correction
    merged = readers[1](factors,s)::Matrix{ComplexF64}
    return evaluate_merged_pairs!(b,u0,q1,q2,merged,t)
end
function run_boundaries(c, wrapped, label, directory)
    saved_calls = copy(c.calls)
    for depth in (0,16)
        dependency = c.factors
        for _ in 1:depth
            dependency = dependency_layer(dependency,wrapped)
        end
        for s in 1:SEGMENTS
            @test dependency[s] === c.factors[s]
        end
        measure_boundaries(c,dependency,depth,label,directory)
    end
    @test c.calls == saved_calls
end
function measure_boundaries(c, dependency, depth, label, directory)
    # Same getindex callback: a concrete tuple permits specialization, whereas
    # Vector{Function} models a runtime-selected callback with an abstract type.
    typed, dynamic = (getindex,), Function[getindex]
    b = buffers()
    @testset "$label depth=$depth callback equivalence" begin
        for s in 1:SEGMENTS, t in (0.0,0.21,0.5,0.79,1.0)
            evaluate_merged!(b,c.exact,c.corrections,c.factors,s,t)
            reference = copy(b.p)
            for readers in (typed,dynamic)
                evaluate_reader!(b,readers,c.exact,c.corrections,dependency,s,t)
                @test b.p ≈ reference atol=1e-10
            end
        end
    end
    for (name,readers) in (("typed_callback",typed),("dynamic_callback",dynamic))
        f = s -> real((readers[1](dependency,s)::Matrix{ComplexF64})[1,1])
        g = s -> evaluate_reader!(b,readers,c.exact,c.corrections,dependency,s,0.37)
        for (operation,func) in (("lookup",f),("evaluate",g))
            tag = "$(name)_$(operation)_depth$(depth)"
            ns,bytes = measure(func)
            @printf("%s,%s,%.2f,%.2f\n",label,tag,ns,bytes)
            flush(stdout)
            inspect_operation(func,joinpath(directory,label * "_" * tag))
        end
    end
end

function main()
    directory = isempty(ARGS) ? mktempdir(prefix="memoized-generators-") : abspath(ARGS[1])
    mkpath(directory)
    rng = Xoshiro(8128)
    h0, h1 = randn(rng,ComplexF64,4,4), randn(rng,ComplexF64,4,4)
    H0, H1 = Hermitian((h0+h0')/2), Hermitian((h1+h1')/2)
    x = collect(range(-0.8,0.8; length=2SEGMENTS+1))
    direct = build_caches(H0,H1,x,Val(false))
    wrapped = build_caches(H0,H1,x,Val(true))
    representation = ismutabletype(typeof(direct.exact)) ? "mutable" : "immutable"
    println("# Julia ",VERSION,"; CPU ",Sys.CPU_NAME,"; BLAS threads=1; ArrayMap; serial")
    println("# Profiles and compiler output: ",directory)
    println("variant,operation,nanoseconds,bytes_per_call")
    run_case(direct,direct,representation * "_direct",directory)
    run_case(wrapped,direct,representation * "_ref",directory)
    run_boundaries(direct,Val(false),representation * "_direct",directory)
    run_boundaries(wrapped,Val(true),representation * "_ref",directory)
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
