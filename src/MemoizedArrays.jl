# Package entry point: exports, shared types and source includes.
# Flow: cache[i, j] → ArrayInterface checks coordinates → the selected cache finds or generates a value.
# Cache types manage retention; maps locate entries. MemoizedArray supports opt-in concurrency for every map.

module MemoizedArrays

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

export AbstractMemoizedArray, MemoizedArray, LRUMemoizedArray, generate!
export getready, unbound, isunbound, queue_handle, queue_handles, with_queue
export ArrayMap, ExpandingArrayMap, DictMap, RadixMap, SubtreeRadixMap

abstract type MapConfig{V} end

"""Cache interface parameterized by value type, dimension count and map type."""
abstract type AbstractMemoizedArray{ValueType,N,M} <: AbstractArray{ValueType,N} end

abstract type SparseMap{T,N} end

struct Empty end
const EMPTY = Empty()


include("maps/ArrayMap.jl")
include("maps/ExpandingArrayMap.jl")
include("maps/DictMap.jl")
include("maps/TreeMap.jl")
include("eviction/MultiQueue.jl")
include("caches/ConcurrentGeneration.jl")
include("maps/ConcurrentTreeMap.jl")
include("eviction/PrivateLRU.jl")
include("eviction/SerialLRU.jl")
include("maps/LRUDictMap.jl")
include("maps/ConcurrentDictMap.jl")
include("caches/MemoizedArray.jl")
include("caches/LRUMemoizedArray.jl")
include("ArrayInterface.jl")

end # module MemoizedArrays
