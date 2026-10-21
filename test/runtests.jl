using Test, MemoizedArrays
const JA = MemoizedArrays

include("helpers.jl")

include("Behavior_test.jl")
include("Concurrency_test.jl")

@testset "Internal invariants" begin
    include("internal/CoordinateMapping_test.jl")
    include("internal/Growth_test.jl")
    include("internal/LRUBookkeeping_test.jl")
    include("internal/TreePruning_test.jl")
    include("internal/DictionaryCollisions_test.jl")
end
