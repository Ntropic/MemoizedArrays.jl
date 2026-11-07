# Julia array operations: axes, bounds, indexing, views and copying.
# Also provides domain normalization and coordinate routing.

"""`AxisExtent(lo, hi)` defines inclusive axis bounds; coordinates index themselves."""
struct AxisExtent{Full} <: AbstractUnitRange{Int}
    lo::Int
    hi::Int
    AxisExtent(lo::Int, hi::Int) = new{lo == typemin(Int) && hi == typemax(Int)}(lo, hi)
end
Base.first(axis::AxisExtent) = axis.lo
Base.last(axis::AxisExtent) = axis.hi
Base.firstindex(axis::AxisExtent) = axis.lo
Base.lastindex(axis::AxisExtent) = axis.hi
Base.axes(axis::AxisExtent) = (axis,)
Base.isempty(axis::AxisExtent) = axis.hi < axis.lo
Base.length(axis::AxisExtent) = checked_extent(axis)
Base.size(axis::AxisExtent) = (length(axis),)
Base.checkindex(::Type{Bool}, axis::AxisExtent, i::Real) = first(axis) <= i <= last(axis)
# Every Int is valid on a full axis; specialization removes the bounds comparisons.
Base.checkindex(::Type{Bool}, ::AxisExtent{true}, ::Int) = true
Base.@propagate_inbounds function Base.getindex(axis::AxisExtent, i::Int)
    @boundscheck if !checkindex(Bool, axis, i)
        throw(BoundsError(axis, i))
    end
    return i
end
Base.iterate(axis::AxisExtent) = isempty(axis) ? nothing : (first(axis), first(axis))
Base.iterate(axis::AxisExtent, state::Int) = state == last(axis) ? nothing : (state + 1, state + 1)

const SparseArray{V,N} = AbstractMemoizedArray{V,N,M} where {M<:SparseMap}

Base.axes(cache::SparseArray) = cache.map.domain
Base.size(cache::SparseArray) = map(length, cache.map.domain)
Base.size(cache::SparseArray, d::Int) = d > ndims(cache) ? 1 : length(cache.map.domain[d])
Base.length(cache::SparseArray) = domain_length(cache.map.domain)
Base.IndexStyle(::Type{<:AbstractMemoizedArray}) = IndexCartesian()
Base.similar(cache::AbstractMemoizedArray, ::Type{T}, dims::Dims) where T = Array{T}(undef, dims)

Base.@propagate_inbounds function Base.getindex(cache::AbstractMemoizedArray{V,N}, indices::Vararg{Int,N}) where {V,N}
    return cache_get(cache, indices)::V
end
Base.@propagate_inbounds function cache_get(cache::AbstractMemoizedArray{V}, indices::Tuple) where V
    lookup = map_lookup(cache, indices)
    return cache_get(cache, lookup, indices)::V
end
Base.@propagate_inbounds Base.getindex(cache::AbstractMemoizedArray, key::CartesianIndex) = cache[Tuple(key)...]
# Prevent AbstractArray's linear-assignment fallback; supported setters require
# exactly one coordinate per dimension.
Base.setindex!(cache::AbstractMemoizedArray, value, indices::Vararg{Int}) =
    throw(Base.CanonicalIndexError("setindex!",typeof(cache)))
Base.@propagate_inbounds function Base.getindex(cache::AbstractMemoizedArray{V,N}, indices...) where {V,N}
    length(indices) == N || throw(ArgumentError("supply one index per cache dimension; linear indexing is unsupported"))
    return array_index(cache, to_indices(cache, indices))
end
Base.@propagate_inbounds function Base.view(cache::AbstractMemoizedArray{V,N}, indices...) where {V,N}
    length(to_indices(cache, indices)) == N || throw(ArgumentError("supply one index per cache dimension; linear views are unsupported"))
    selected = invoke(Base.view, Tuple{AbstractArray,Vararg{Any}}, cache, indices...)
    return tree_view(selected)
end

# Share the original cache and selection; the shortcut does not own cached values.
struct TreeView{V,N,S,T,P} <: AbstractArray{V,N}
    selection::S
    low::NTuple{P,UInt}
    high::NTuple{P,UInt}
    root::Base.RefValue{Union{TreeBranch{T,P},TreeLeaf{T,P}}}
    leaf::Base.RefValue{Union{Nothing,TreeLeaf{T,P}}}
end
Base.size(view::TreeView) = size(view.selection)
Base.axes(view::TreeView) = axes(view.selection)
Base.parent(view::TreeView) = parent(view.selection)
Base.parentindices(view::TreeView) = parentindices(view.selection)
Base.dataids(view::TreeView) = Base.dataids(parent(view))
Base.IndexStyle(::Type{<:TreeView}) = IndexCartesian()
Base.similar(::TreeView, ::Type{T}, dims::Dims) where T = Array{T}(undef, dims)

tree_view(selected) = selected
function tree_view(selected::SubArray{V,N,C}) where {V,N,T,P,C<:AbstractMemoizedArray{V,P,<:TreeMap{T,P}}}
    cache, indices = parent(selected), parentindices(selected)
    if P == 0 || !all(index -> index isa Int ||
            (index isa AbstractUnitRange && !isempty(index) &&
             !(first(index) == typemin(Int) && last(index) == typemax(Int))), indices)
        return selected
    end
    low = encode_coordinates(map(index -> index isa Int ? index : Int(first(index)), indices), cache_offset(cache))
    high = encode_coordinates(map(index -> index isa Int ? index : Int(last(index)), indices), cache_offset(cache))
    any(i -> low[i] > high[i], 1:P) && return selected
    root = Ref{Union{TreeBranch{T,P},TreeLeaf{T,P}}}(view_root(cache.map, low, high))
    leaf = Ref{Union{Nothing,TreeLeaf{T,P}}}(nothing)
    return TreeView{V,N,typeof(selected),T,P}(selected, low, high, root, leaf)
end

Base.@propagate_inbounds function Base.getindex(view::TreeView{V,N}, indices::Vararg{Int,N}) where {V,N}
    @boundscheck checkbounds(view, indices...)
    key = map(Int, Base.reindex(parentindices(view), indices))
    return tree_view_get(view, key)
end

@inline function tree_view_get(view::TreeView{V}, key) where V
    cache = parent(view)
    routing = cache_routing(cache, key)
    encoded = cache.map isa SubtreeRadixMap ? routing[1] : routing
    leaf = view.leaf[]
    lookup = if leaf !== nothing && leaf.used > 0 && tree_matches(cache.map, encoded, leaf.prefix, 0)
        (key, routing, leaf, (nothing, nothing, tree_address(cache.map, encoded, 0)))
    else
        node = view.root[]
        if !live_view_root(node) || !tree_matches(cache.map, encoded, node_prefix(node), node_level(node))
            node = view_root(cache.map, view.low, view.high)
            view.root[] = node
        end
        found = view_lookup(cache.map, key, routing, encoded, node)
        view.leaf[] = found[3]
        found
    end
    return cache_get(cache, lookup, key)::V
end
Base.@propagate_inbounds function Base.setindex!(view::TreeView{V,N}, value, indices::Vararg{Int,N}) where {V,N}
    @boundscheck checkbounds(view, indices...)
    key = map(Int, Base.reindex(parentindices(view), indices))
    parent(view)[key...] = value
    return view
end
Base.@propagate_inbounds Base.view(view::TreeView, indices...) = tree_view(Base.view(view.selection, indices...))

# Iteration keeps leaf references, but reads each value when requested so mutation stays visible.
mutable struct TreeViewIteration{K,T,O}
    const keys::K
    leaves::Union{Nothing,T}
    const origin::O
    const offset::O
end

function tree_view_iteration(view::TreeView{V,N,S,T,P}) where {V,N,S,T,P}
    cache = parent(view)
    bits = map(trailing_zeros, block_shape(cache.map))
    origin = map(>>, view.low, bits)
    shape = ntuple(i -> Int((view.high[i] >> bits[i]) - origin[i]) + 1, Val(P))
    Table = Array{Union{Nothing,TreeLeaf{T,P}},P}
    # Bound startup work and memory even when only the first value of a huge view is requested.
    budget = 65536
    for extent in shape
        budget ÷= extent
    end
    root = slot_entry(root_link(cache.map))::Union{TreeBranch{T,P},TreeLeaf{T,P}}
    leaves = if 1024 <= length(view) && budget > 0 && root !== cache.map.roots.empty
        table = Table(nothing, shape)
        root = view_root(cache.map, view.low, view.high)
        gather_slice_leaves!(table, root, cache.map, view.low, view.high, origin)
        table
    else
        nothing
    end
    keys = CartesianIndices(map(i -> i isa Int ? (i:i) : Int(first(i)):Int(last(i)), parentindices(view)))
    # Relative coordinates locate both the table entry and the slot within its leaf.
    offset = map((z,o,b) -> z - (o << b),
        encode_coordinates(ntuple(_ -> 0, Val(P)), cache_offset(cache)), origin, bits)
    return TreeViewIteration{typeof(keys),Table,typeof(origin)}(keys, leaves, origin, offset)
end

function Base.iterate(view::TreeView)
    traversal = tree_view_iteration(view)
    return iterate(view, (traversal, first(traversal.keys), isempty(traversal.keys)))
end

@inline function Base.iterate(view::TreeView{V}, state::Tuple{TreeViewIteration{K,T,O},I,Bool}) where {V,K,T,O,I}
    traversal, coordinate, done = state
    done && return nothing
    value = tree_iteration_get(view, traversal, Tuple(coordinate))
    next = iterate(traversal.keys, coordinate)
    following = next === nothing ? coordinate : next[1]
    return value, (traversal, following, next === nothing)
end

# Keeping lookup separate from state construction avoids a Subtree LRU code-generation regression.
@inline function tree_iteration_get(view::TreeView{V}, traversal, key) where V
    leaves = traversal.leaves
    leaves === nothing && return tree_view_get(view, key)
    cache = parent(view)
    relative = map((k,o) -> reinterpret(UInt,k) + o, key, traversal.offset)
    bits = map(trailing_zeros, block_shape(cache.map))
    index = map((x,b) -> Int(x >> b) + 1, relative, bits)
    leaf = @inbounds leaves[index...]
    if leaf !== nothing && leaf.used > 0
        stored = prepared_entry(leaf, (nothing, nothing, tree_address(cache.map, relative, 0)))
        if stored !== cache.map.empty
            return tree_ready_value(cache, stored)::V
        end
    end
    # A miss can generate or evict; use the scalar cursor for the rest of this pass.
    traversal.leaves = nothing
    return tree_view_get(view, key)
end

@inline function tree_ready_value(cache, stored)
    if cache isa LRUMemoizedArray
        stored.explicit || queue_touch!(cache.eviction, stored.id)
    elseif cache.map isa SubtreeRadixMap
        assume_initialized(isdefined(stored, :value))
    end
    return stored.value
end

# Keep Base's reduction wrappers for promotion, initial values, and left-to-right evaluation.
function Base._foldl_impl(op::F, init, view::TreeView) where F
    use_leaf_runs(view) || return invoke(Base._foldl_impl, Tuple{Any,Any,Any}, op, init, view)
    first = iterate(view)
    first === nothing && return init
    accumulator = op(init, first[1])
    state = first[2]
    traversal = state[1]
    if traversal.leaves === nothing
        return Base._foldl_impl(op, accumulator, Iterators.rest(view, state))
    end
    return fold_leaf_runs(op, accumulator, parent(view), traversal.keys, traversal.leaves, traversal.origin, 1)
end

function Base.mapreduce(f::F, op::O, view::TreeView; dims=:, init=Base._InitialValue()) where {F,O}
    dims isa Colon && return mapfoldl(f, op, view; init)
    return invoke(Base.mapreduce, Tuple{Any,Any,AbstractArray}, f, op, view; dims, init)
end

function Base.map!(f::F, dest::Array, source::TreeView) where F
    if use_leaf_runs(source) && length(dest) >= length(source) && !Base.mightalias(dest, source)
        traversal = tree_view_iteration(source)
        if traversal.leaves !== nothing && any(leaf -> leaf !== nothing, traversal.leaves)
            return copy_leaf_runs!(dest, parent(source), traversal.keys, traversal.leaves, traversal.origin, f)
        end
    end
    return invoke(Base.map!, Tuple{Any,AbstractArray,AbstractArray}, f, dest, source)
end

# Runs amortize their setup only when the selection spans a leaf along the first axis.
function use_leaf_runs(view::TreeView)
    axis = first(parentindices(view))
    map = parent(view).map
    return length(view) >= 1024 && axis isa AbstractUnitRange &&
        length(axis) >= first(block_shape(map)) && slot_entry(root_link(map)) !== map.roots.empty
end

@inline ready_value(cache, stored) = stored.value
@inline ready_entry(cache, stored) = stored !== cache.map.empty
"""`isready(cache, indices...)` checks for a cached value without generating or changing recency."""
Base.@propagate_inbounds Base.isready(cache::AbstractMemoizedArray{V,N}, indices::Vararg{Int,N}) where {V,N} = ready_entry(cache, entry(cache, cache_key(cache, indices)))
Base.@propagate_inbounds function Base.isready(cache::AbstractMemoizedArray{V,N}, key::VT{N,Int}) where {V,N}
    @astuple N key
    return isready(cache, key...)
end
Base.@propagate_inbounds Base.isready(cache::AbstractMemoizedArray, key::CartesianIndex) = isready(cache, Tuple(key))

"""
    generate!(cache, indices...)
    generate!(cache, requests)

Return one value, or generate a batch synchronously and return the cache. Integer tuples denote one coordinate.
"""
Base.@propagate_inbounds generate!(cache::AbstractMemoizedArray, indices::Vararg{Int,N}) where N = cache[indices...]
Base.@propagate_inbounds generate!(cache::AbstractMemoizedArray{V,N}, key::NTuple{N,Int}) where {V,N} = cache[key...]
Base.@propagate_inbounds generate!(cache::AbstractMemoizedArray{V,0}, key::Tuple{}) where V = cache[]
Base.@propagate_inbounds generate!(cache::AbstractMemoizedArray, key::CartesianIndex) = cache[key]
function generate!(cache::AbstractMemoizedArray{V,N},
    requests::Union{VT{B,NTuple{N,Int}},AbstractVector{Int},AbstractVector{CartesianIndex{N}}}) where {V,N,B}
    for key in requests
        generate!(cache, key)
    end
    return cache
end

# Copy coordinate selections without requesting linear cache indices.
const CacheView = SubArray{<:Any,<:Any,<:AbstractMemoizedArray}
const CacheSource = Union{AbstractMemoizedArray,CacheView,TreeView}
function Base.copyto!(dest::AbstractArray, source::CacheSource)
    cache, keys = source_keys(source)
    return copyvalues!(dest, cache, keys)
end
function Base.copyto!(dest::PermutedDimsArray, source::CacheSource)
    cache, keys = source_keys(source)
    return copyvalues!(dest, cache, keys)
end
Base.copy(source::CacheSource) = copyto!(Array{eltype(source)}(undef, size(source)), source)

# Helpers -------------------------------------------------------------------------------

"""`unbound(N)` returns a tuple of `N` unbounded axes (`:`)."""
function unbound(N::Integer)
    if N < 0
        throw(ArgumentError("dimension count must be nonnegative"))
    end
    return ntuple(_ -> Colon(), N)
end

"""`isunbound(axis)` tests whether an axis covers every supported integer coordinate."""
isunbound(::Colon) = true
isunbound(::Int) = false
isunbound(axis::AbstractUnitRange{Int}) = first(axis) == typemin(Int) && last(axis) == typemax(Int)

function axis_domain(axis::Int)
    if axis < 0
        throw(ArgumentError("dimensions must be nonnegative"))
    end
    return Base.OneTo(axis)
end
axis_domain(::Colon) = AxisExtent(typemin(Int), typemax(Int))
axis_domain(axis::AbstractUnitRange{Int}) = first(axis) == 1 ? Base.OneTo(last(axis)) : AxisExtent(first(axis), last(axis))
axis_domain(axis) = throw(ArgumentError("domain axes must be integer lengths, integer unit ranges, or :"))


# The precomputed offsets belong to the cache type, with no coordinate-time geometry work.
routing_offset(::Type) = ()
routing_offset(map) = routing_offset(typeof(map))
@generated function cache_offset(::AbstractMemoizedArray{V,N,M}) where {V,N,M}
    offset = routing_offset(M)
    return :(Val{$(QuoteNode(offset))}())
end
@inline cache_routing(cache, key) = cache_routing(cache, key, cache.map)
@inline cache_routing(cache, key, map) = map_routing(map, key)
@inline delete_entry!(cache, key) = delete_key!(cache.map, key)

# ArrayMap receives slot addresses; sparse maps receive coordinate tuples.
Base.@propagate_inbounds function cache_key(cache::SparseArray{V,N}, key::NTuple{N,Int}) where {V,N}
    @boundscheck checkbounds(cache, key...)
    return key
end
Base.@propagate_inbounds cache_key(cache::AbstractMemoizedArray, key::CartesianIndex) = cache_key(cache, Tuple(key))

# Coordinate selections use Base's lazy Cartesian index array.
Base.@propagate_inbounds array_index(cache::AbstractMemoizedArray, indices::Tuple{Vararg{Int}}) = cache[indices...]
Base.@propagate_inbounds function array_index(cache::AbstractMemoizedArray, indices::Tuple)
    _, source = source_keys(cache)
    keys = view(source, indices...)
    return copyvalues!(Array{eltype(cache)}(undef, size(keys)), cache, keys)
end
source_keys(cache::AbstractMemoizedArray) = (cache, CartesianIndices(axes(cache)))
source_keys(view::TreeView) = source_keys(view.selection)
function source_keys(source::CacheView)
    cache, keys = source_keys(parent(source))
    return cache, view(keys, parentindices(source)...)
end

# Extents can exceed Int even though both coordinate endpoints fit.
function checked_extent(axis::AxisExtent)
    extent = max(Int128(0), Int128(last(axis)) - Int128(first(axis)) + 1)
    if extent > typemax(Int)
        throw(OverflowError("axis extent exceeds Int; use coordinate indexing or a finite selection"))
    end
    return Int(extent)
end

function domain_length(ranges)
    if any(isempty, ranges)
        return 0
    end
    count = 1
    for axis in ranges
        count = Base.checked_mul(count, length(axis))
    end
    return count
end

@inline function entry(cache, key)
    slot = map_slot(cache.map, key, cache_routing(cache, key))
    return slot === nothing ? cache.map.empty : slot_entry(slot)
end

# Copy finite selections in coordinate order.
function copyvalues!(dest, cache, keys)
    if length(dest) < length(keys)
        throw(BoundsError(dest, eachindex(keys)))
    end
    if Base.mightalias(dest, keys)
        keys = copy(keys)
    end
    return copyvalues_unaliased!(dest, cache, keys)
end

function copyvalues_unaliased!(dest, cache, keys)
    for (i, key) in zip(eachindex(dest), keys)
        @inbounds dest[i] = cache[key]
    end
    return dest
end

function copyvalues_unaliased!(dest, cache::AbstractMemoizedArray{V,N,M}, keys) where {V,N,M<:TreeMap}
    # Reuse pays off for adjacent coordinates; scattered selections keep scalar lookup.
    selection = keys isa CartesianIndices ? keys.indices : keys isa SubArray ? parentindices(keys) : ()
    if isempty(selection) || !(first(selection) isa Union{AbstractUnitRange,Base.Slice})
        return invoke(copyvalues_unaliased!, Tuple{Any,Any,Any}, dest, cache, keys)
    end
    # The bounds table pays off on larger rectangles.
    if keys isa CartesianIndices && N > 0 && length(keys) >= 1024 &&
            all(axis -> axis isa AbstractUnitRange, keys.indices) &&
            all(i -> length(keys.indices[i]) >= block_shape(cache.map)[i], 1:N) &&
            slot_entry(root_link(cache.map)) !== cache.map.roots.empty
        low = encode_coordinates(Tuple(first(keys)), cache_offset(cache))
        high = encode_coordinates(Tuple(last(keys)), cache_offset(cache))
        # Unsigned wrap splits the region; the ordered cursor already handles it.
        if all(i -> low[i] <= high[i], 1:N)
            return bounded_copy!(dest, cache, keys, low, high)
        end
    end
    return copy_leaf_cursor!(dest, cache, keys)
end

function copy_leaf_cursor!(dest, cache::AbstractMemoizedArray{V}, keys) where V
    leaf = nothing
    # Measured worthwhile for narrow leaves; larger leaves amortize root traversal already.
    ancestors = first(block_shape(cache.map)) == 2 ? slice_ancestors(cache.map) : nothing
    for (i, coordinate) in zip(eachindex(dest), keys)
        key = Tuple(coordinate)
        routing = cache_routing(cache, key)
        encoded = cache.map isa SubtreeRadixMap ? routing[1] : routing
        # Eviction or nested generation can prune the previously resolved leaf.
        if leaf === nothing || leaf.used == 0 || !tree_matches(cache.map, encoded, leaf.prefix, 0)
            leaf, hint = prepare_slice!(ancestors, cache.map, key, routing, encoded)
        else
            hint = (nothing, nothing, tree_address(cache.map, encoded, 0))
        end
        missing = ancestors !== nothing && (leaf === nothing || prepared_entry(leaf, hint) === cache.map.empty)
        value = cache_get(cache, (key, routing, leaf, hint), key)::V
        @inbounds dest[i] = value
        # Generators and destination conversions may restructure the tree.
        if ancestors !== nothing && (missing || !(dest isa Array{V}))
            empty!(ancestors)
            leaf = nothing
        end
    end
    return dest
end

# Read the spatially collected leaves in the original coordinate order.
function bounded_copy!(dest, cache::AbstractMemoizedArray{V,N,M}, keys::CartesianIndices{N}, low, high) where {V,N,T,M<:TreeMap{T,N}}
    block = block_shape(cache.map)
    bits = map(trailing_zeros, block)
    origin = ntuple(i -> low[i] >> bits[i], Val(N))
    shape = ntuple(i -> Int((high[i] >> bits[i]) - origin[i]) + 1, Val(N))
    table = Array{Union{Nothing,TreeLeaf{T,N}}}(nothing, shape)
    root = view_root(cache.map, low, high)
    gather_slice_leaves!(table, root, cache.map, low, high, origin)
    if any(leaf -> leaf !== nothing, table)
        dest isa Array && return copy_leaf_runs!(dest, cache, keys, table, origin)
    elseif cache isa LRUMemoizedArray && cache.map isa SubtreeRadixMap
        return copy_leaf_cursor!(dest, cache, keys)
    end
    for (i, coordinate) in zip(eachindex(dest), keys)
        key = Tuple(coordinate)
        routing = cache_routing(cache, key)
        encoded = cache.map isa SubtreeRadixMap ? routing[1] : routing
        index = ntuple(d -> Int((encoded[d] >> bits[d]) - origin[d]) + 1, Val(N))
        leaf = @inbounds table[index...]
        lookup = if leaf === nothing || leaf.used == 0
            found = map_lookup(cache, key)
            @inbounds table[index...] = found[3]
            found
        else
            (key, routing, leaf, (nothing, nothing, tree_address(cache.map, encoded, 0)))
        end
        @inbounds dest[i] = cache_get(cache, lookup, key)::V
    end
    return dest
end

function copy_leaf_runs!(dest::Array, cache, keys, table, origin, f::F=identity) where F
    function write(i, value)
        mapped = f(value)
        @inbounds dest[i] = mapped
        return i + 1
    end
    fold_leaf_runs(write, 1, cache, keys, table, origin)
    return dest
end

# Bulk consumers advance through leaf slots without rebuilding each coordinate.
function fold_leaf_runs(op::F, accumulator, cache::AbstractMemoizedArray{V}, keys, table, origin, skipfirst=0) where {F,V}
    block = block_shape(cache.map)
    bits = map(trailing_zeros, block)
    for tail in CartesianIndices(Base.tail(keys.indices))
        x = first(keys.indices[1])
        remaining = length(keys.indices[1])
        while remaining > 0
            encoded = encode_coordinates((x, Tuple(tail)...), cache_offset(cache))
            index = map((e,b,o) -> Int((e >> b) - o) + 1, encoded, bits, origin)
            leaf = @inbounds table[index...]
            address = tree_address(cache.map, encoded, 0)
            count = min(remaining, block[1] - Int(encoded[1] & UInt(block[1] - 1)))
            for offset in skipfirst:count-1
                # User callbacks and destination conversion may mutate the cache between reads.
                if leaf !== nothing && leaf.used > 0
                    stored = slot_entry(map_slot(leaf.values, address + offset, nothing))
                    if stored !== cache.map.empty
                        accumulator = op(accumulator, tree_ready_value(cache, stored)::V)
                        continue
                    end
                end
                key = (x + offset, Tuple(tail)...)
                lookup = if leaf === nothing || leaf.used == 0
                    found = map_lookup(cache, key)
                    leaf = @inbounds table[index...] = found[3]
                    found
                else
                    (key, cache_routing(cache, key), leaf, (nothing, nothing, address + offset))
                end
                accumulator = op(accumulator, cache_get(cache, lookup, key)::V)
            end
            skipfirst = 0
            remaining -= count
            remaining == 0 && break
            x += count
        end
    end
    return accumulator
end
