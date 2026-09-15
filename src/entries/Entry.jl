struct Empty end
const _EMPTY = Empty()

abstract type AbstractEntry{V} end

mutable struct Entry{V} <: AbstractEntry{V}
    @atomic state::Union{Nothing,Threads.Condition,Bool,Empty}
    value::V
    Entry{V}() where V = new{V}(nothing)
end
