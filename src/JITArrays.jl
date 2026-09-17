# Package entry point: exports, shared types and source includes.
# Flow: cache[i, j] → ArrayInterface checks coordinates → the selected cache finds or generates a value.
# Synchronous execution: cache types manage retention; maps locate entries.

module JITArrays

"""
    VT{N,T} = Union{AbstractVector{T},NTuple{N,T}}

An abstract vector with element type `T`, or a length-`N` tuple of elements of type `T`.
"""
const VT{N,T} = Union{AbstractVector{T},NTuple{N,T}}

@inline as_tuple(x::Tuple{Vararg{Any,N}}, ::Val{N}) where N = x

@inline function as_tuple(x::AbstractVector, ::Val{N}) where {N}
    if length(x) != N
        throw(DimensionMismatch("expected $N parameters"))
    end
    return ntuple(i -> x[i], Val(N))
end

"""`@astuple N x...` converts named inputs to length-`N` tuples."""
macro astuple(N, names...)
    if !all(name -> name isa Symbol, names)
        throw(ArgumentError("@astuple expects variable names"))
    end
    return Expr(:block, (:($(esc(name)) = as_tuple($(esc(name)), Val($(esc(N))))) for name in names)...)
end

export AbstractJITArray, JITArray, LRUArray, ExactLRUArray, SampledLRUArray, generate!
export SparseJITArray, SparseLRUArray
export getready

"""Cache interface parameterized by value type, dimension count and storage type."""
abstract type AbstractJITArray{ValueType,N,M} <: AbstractArray{ValueType,N} end

abstract type SparseMap{T,N} end

include("caches/JITArray.jl")
include("caches/ExactLRUArray.jl")
include("caches/SampledLRUArray.jl")
include("maps/ArrayMap.jl")
include("maps/DictMap.jl")
include("maps/TreeMap.jl")
include("ArrayInterface.jl")

end # module JITArrays
