# Geometry becomes type information before the cache chooses its entry layout.
struct RadixMapConfig{V,D,B,C} <: MapConfig{V}
    domain::D
end

struct SubtreeRadixMapConfig{V,D,B,C} <: MapConfig{V}
    domain::D
end

@inline function Base.getproperty(config::Union{RadixMapConfig{V,D,B,C},SubtreeRadixMapConfig{V,D,B,C}}, name::Symbol) where {V,D,B,C}
    name === :block_shape && return B
    name === :center && return C
    return getfield(config, name)
end
Base.propertynames(::Union{RadixMapConfig,SubtreeRadixMapConfig}, private::Bool=false) = (:domain, :block_shape, :center)

tree_axis(axis) = axis isa AxisExtent ? axis : AxisExtent(first(axis), last(axis))

# Sparse tree storage: one radix tree, optionally with shortcuts selected by coordinate bit depth.
# Allocates branches on demand and prunes unused storage after eviction.

abstract type TreeNode{T,N} end

# Prefixes are immutable; counters track occupied entries and children.
mutable struct TreeBranch{T,N} <: TreeNode{T,N}
    const prefix::NTuple{N,UInt}
    const level::Int
    const children::ArrayMap{TreeNode{T,N},N}
    used::Int
    occupied::Int
end

mutable struct TreeLeaf{T,N} <: TreeNode{T,N}
    const prefix::NTuple{N,UInt}
    const values::ArrayMap{T,N}
    used::Int
    # Initialize the sentinel too, so real leaves need no undefined-field checks.
    function TreeLeaf{T,N}(empty) where {T,N}
        return new{T,N}(ntuple(_ -> UInt(0), Val(N)), ArrayMap(T, ntuple(_ -> 0, Val(N)), empty), 0)
    end
    TreeLeaf{T,N}(prefix, values, used) where {T,N} = new{T,N}(prefix, values, used)
end

struct LeafSlot{T,N}
    leaf::TreeLeaf{T,N}
    reference::MemoryRef{T}
end

@inline LeafSlot(leaf::TreeLeaf, address::Int) = LeafSlot(leaf, map_slot(leaf.values, address, nothing))

@inline prepared_entry(leaf::TreeLeaf, state::Tuple) = slot_entry(map_slot(leaf.values, state[3], nothing))

"""
    RadixMap{T}(domain; block_shape, center=0.5)

Radix-tree lookup. `block_shape` gives branch and leaf slot dimensions,
one power of two ≥ 2 per axis. `center` is `0.5` or `-0.5`.
"""
struct RadixMap{T,N,BlockShape,D,E,PositiveCenter} <: SparseMap{T,N}
    roots::ArrayMap{TreeNode{T,N},1}
    empty::E
    domain::D
end

"""
    SubtreeRadixMap{T}(domain; block_shape, center=0.5)

Radix lookup with shortcuts around zero; every domain axis must contain zero.
Geometry follows `RadixMap`.
"""
struct SubtreeRadixMap{T,N,BlockShape,D,E,Levels,PositiveCenter} <: SparseMap{T,N}
    roots::ArrayMap{TreeNode{T,N},1}
    empty::E
    domain::D
    zero_path::NTuple{Levels,Int}
end

function RadixMap{V}(domain::VT; block_shape, center=0.5) where V
    domain = map(tree_axis, map(axis_domain, Tuple(domain)))
    block_shape = Tuple(block_shape)
    center = positive_center(center) ? 0.5 : -0.5
    return RadixMapConfig{V,typeof(domain),block_shape,center}(domain)
end

function SubtreeRadixMap{V}(domain::VT; block_shape, center=0.5) where V
    domain = map(tree_axis, map(axis_domain, Tuple(domain)))
    block_shape = Tuple(block_shape)
    center = positive_center(center) ? 0.5 : -0.5
    return SubtreeRadixMapConfig{V,typeof(domain),block_shape,center}(domain)
end

const TreeMap{T,N,B} = Union{RadixMap{T,N,B},SubtreeRadixMap{T,N,B}}

# Slots are filled with a sentinel at construction and are never unset. Keep Julia's
# normal GC-tracked loads, but let subtree reads omit undefined-reference guards.
@inline function assume_initialized(condition::Bool)
    Base.llvmcall(("""
        declare void @llvm.assume(i1)
        define void @entry(i8 %condition) alwaysinline {
            %ok = icmp ne i8 %condition, 0
            call void @llvm.assume(i1 %ok)
            ret void
        }
        """, "entry"), Cvoid, Tuple{Bool}, condition)
    return nothing
end

@inline slot_entry(slot::MemoryRef, ::TreeMap) = slot_entry(slot)
@inline function slot_entry(slot::MemoryRef, ::SubtreeRadixMap)
    assume_initialized(Core.memoryref_isassigned(slot, :not_atomic, false))
    return slot_entry(slot)
end

# Cache routing converts the checked coordinates once; traversal only sees unsigned offsets.
@inline map_routing(store::TreeMap{T,N}, key) where {T,N} = ntuple(i -> reinterpret(UInt, key[i]) + tree_offset(store)[i], Val(N))
@inline block_shape(::TreeMap{T,N,B}) where {T,N,B} = B
@inline tree_offset(::RadixMap{T,N,B,D,E,C}) where {T,N,B,D,E,C} = tree_offset(B, C)
@inline tree_offset(::SubtreeRadixMap{T,N,B,D,E,L,C}) where {T,N,B,D,E,L,C} = tree_offset(B, C)

# Alternating middle children keep each ancestor as close to the requested center
# as the already aligned children allow. Arithmetic wraps across the full UInt domain.
Base.@constprop :aggressive function tree_offset(block::NTuple{N,Int}, positive::Bool) where N
    return ntuple(Val(N)) do i
        width = block[i]
        bits = trailing_zeros(width)
        offset = UInt(0)
        digit = width ÷ 2 - positive
        for shift in 0:bits:8sizeof(UInt)-1
            offset |= UInt(digit) << shift
            digit = width - 1 - digit
        end
        offset
    end
end

function positive_center(center)
    if !(center isa Real && (center == 0.5 || center == -0.5))
        throw(ArgumentError("tree center must be 0.5 or -0.5"))
    end
    return center == 0.5
end
@inline function node_level(node::TreeNode{T,N}) where {T,N}
    if node isa TreeLeaf{T,N}
        return 0
    end
    return (node::TreeBranch{T,N}).level
end
@inline function node_prefix(node::TreeNode{T,N}) where {T,N}
    if node isa TreeLeaf{T,N}
        return node.prefix
    end
    return (node::TreeBranch{T,N}).prefix
end

@inline slot_entry(slot::LeafSlot) = slot_entry(slot.reference)
@inline slot_publish!(slot::LeafSlot, value) = slot_publish!(slot.reference, value)
map_retain!(::TreeMap, slot::LeafSlot) = (slot.leaf.used += 1)

# Validate local blocks without multiplying the global coordinate extents.
function block_length(shape, minimum::Int)
    count = 1
    for width in shape
        if width < minimum || !ispow2(width)
            throw(ArgumentError("block dimensions must be powers of two ≥ $minimum"))
        end
        count = Base.checked_mul(count, width)
    end
    return count
end

@inline tree_bits(block::Tuple, axis::Int, level::Int) = (level + 1) * trailing_zeros(block[axis])

@inline function tree_prefix(block::NTuple{N,Int}, key, level::Int) where N
    return ntuple(i -> key[i] & (typemax(UInt) << unsigned(tree_bits(block, i, level))), Val(N))
end

# Equal-width axes share one prefix shift after reducing their differing bits.
@inline function tree_matches(block::NTuple{N,Int}, key, prefix, level::Int) where N
    if N == 0
        return true
    elseif all(width -> width == first(block), block)
        difference = foldl(|, ntuple(i -> key[i] ⊻ prefix[i], Val(N)); init=UInt(0))
        return iszero(difference >> unsigned(tree_bits(block, 1, level)))
    end
    return tree_prefix(block, key, level) == prefix
end

# Minimum common level covering all encoded axes.
@inline function tree_level(block, key)
    if isempty(key)
        return 0
    end
    if all(width -> width == first(block), block)
        highest = foldl(|, key; init=UInt(0))
        bits = unsigned(8sizeof(UInt) - leading_zeros(highest | UInt(1)) - 1)
        return Int(bits ÷ unsigned(trailing_zeros(first(block))))
    end
    level = 0
    for i in eachindex(key)
        bits = unsigned(8sizeof(UInt) - leading_zeros(key[i] | UInt(1)) - 1)
        level = max(level, Int(bits ÷ unsigned(trailing_zeros(block[i]))))
    end
    return level
end

@inline root_index(::RadixMap, encoded) = 1
@inline root_index(store::SubtreeRadixMap{T,N}, encoded) where {T,N} = tree_level(store, ntuple(i -> encoded[i] ⊻ tree_offset(store)[i], Val(N))) + 1

# The last subtree slot owns the tree; earlier slots alias the path containing coordinate zero.
@inline root_link(store::TreeMap) = @inbounds map_slot(store.roots, length(store.roots.reference.mem), nothing)
refresh!(::RadixMap, depth) = nothing
# Skipped depths retain their nearest existing ancestor, so requests only descend.
function refresh!(store::SubtreeRadixMap{T,N}, depth::Int) where {T,N}
    zero = tree_offset(store)
    ancestor = slot_entry(map_slot(store.roots, depth + 2, nothing))
    node = ancestor
    for depth in depth:-1:0
        while node isa TreeBranch{T,N} && node.level > depth
            child = slot_entry(map_slot(node.children, store.zero_path[node.level + 1], nothing))
            if child === store.roots.empty || node_level(child) < depth ||
                    !tree_matches(store, zero, node_prefix(child), node_level(child))
                break
            end
            node = child
        end
        level = max(depth, node_level(node))
        alias = node === store.roots.empty || tree_prefix(store, node_prefix(node), level) != tree_prefix(store, zero, level) ? store.roots.empty : node
        slot = map_slot(store.roots, depth + 1, nothing)
        if slot_entry(slot) === alias
            if node !== ancestor
                break
            end
        else
            slot_publish!(slot, alias)
        end
    end
    return nothing
end

routing_offset(::Type{<:RadixMap{T,N,B,D,E,C}}) where {T,N,B,D,E,C} = tree_offset(B, C)
routing_offset(::Type{<:SubtreeRadixMap{T,N,B,D,E,L,C}}) where {T,N,B,D,E,L,C} = tree_offset(B, C)
@inline encode_coordinates(key::NTuple{N,Int}, ::Val{Offset}) where {N,Offset} =
    ntuple(i -> reinterpret(UInt, key[i]) + Offset[i], Val(N))

@inline cache_routing(cache, key, ::RadixMap) = encode_coordinates(key, cache_offset(cache))
@inline function cache_routing(cache, key, ::SubtreeRadixMap)
    offset = cache_offset(cache)
    encoded = encode_coordinates(key, offset)
    return encoded, tree_level(cache.map, ntuple(i -> encoded[i] ⊻ routing_offset(cache.map)[i], length(encoded))) + 1
end
Base.@propagate_inbounds function map_lookup(cache::AbstractMemoizedArray{V,N,M}, key::NTuple{N,Int}) where {V,N,M<:TreeMap}
    @boundscheck checkbounds(cache, key...)
    routing = cache_routing(cache, key)
    slot, hint = prepare(cache.map, key, routing)
    return key, routing, slot, hint
end

@inline function delete_entry!(cache::AbstractMemoizedArray{V,N,M}, key) where {V,N,M<:TreeMap}
    return delete_key!(cache.map, key, encode_coordinates(key, cache_offset(cache)))
end


# Flatten only a checked local block in column-major order.
@inline function tree_address(block::NTuple{N,Int}, key, level::Int) where N
    address, shift = UInt(0), 0
    for i in 1:N
        width = block[i]
        bitoffset = unsigned(level * trailing_zeros(width))
        if width == minimum(block)
            # The narrowest axes define the maximum level, so their shifts always fit.
            bitoffset &= UInt(8sizeof(UInt) - 1)
        end
        digit = (key[i] >> bitoffset) & UInt(width - 1)
        address |= digit << unsigned(shift)
        shift += trailing_zeros(width)
    end
    return Int(address) + 1
end

Base.@constprop :aggressive function RadixMap(::Type{T}, ranges::Tuple{Vararg{Any,N}}, block::VT{N,Int}, empty; center=0.5) where {T,N}
    @astuple N block
    block_length(block, 2)
    roots = ArrayMap(TreeNode{T,N}, (1,), TreeLeaf{T,N}(empty))
    return RadixMap{T,N,block,typeof(ranges),typeof(empty),positive_center(center)}(roots, empty, ranges)
end

Base.@constprop :aggressive function SubtreeRadixMap(::Type{T}, ranges::Tuple{Vararg{Any,N}}, block::VT{N,Int}, empty; center=0.5) where {T,N}
    @astuple N block
    block_length(block, 2)
    if any(axis -> !(first(axis) <= 0 <= last(axis)), ranges)
        throw(ArgumentError("SubtreeRadixMap requires every domain axis to contain zero"))
    end
    levels = isempty(block) ? 1 : cld(8sizeof(UInt), trailing_zeros(minimum(block)))
    positive = positive_center(center)
    zero = tree_offset(block, positive)
    zero_path = ntuple(i -> tree_address(block, zero, i - 1), Val(levels))
    roots = ArrayMap(TreeNode{T,N}, (levels,), TreeLeaf{T,N}(empty))
    return SubtreeRadixMap{T,N,block,typeof(ranges),typeof(empty),levels,positive}(roots, empty, ranges, zero_path)
end

Base.@constprop :aggressive function make_cache_map(kind, ::Type{V}, config::RadixMapConfig;
    concurrent::Bool=false, capacity::Int=0) where V
    domain = config.domain
    length(config.block_shape) == length(domain) || throw(DimensionMismatch("block_shape must have one width per domain axis"))
    if concurrent
        return concurrent_tree_map(kind, V, domain, config; capacity), domain
    end
    E, empty = map_entry(kind, V, NTuple{length(domain),Int}, config)
    return RadixMap(E, domain, config.block_shape, empty; center=config.center), domain
end

Base.@constprop :aggressive function make_cache_map(kind, ::Type{V}, config::SubtreeRadixMapConfig;
    concurrent::Bool=false, capacity::Int=0) where V
    domain = config.domain
    length(config.block_shape) == length(domain) || throw(DimensionMismatch("block_shape must have one width per domain axis"))
    if concurrent
        return concurrent_tree_map(kind, V, domain, config; capacity), domain
    end
    E, empty = map_entry(kind, V, NTuple{length(domain),Int}, config)
    return SubtreeRadixMap(E, domain, config.block_shape, empty; center=config.center), domain
end

function tree_leaf(store::TreeMap{T,N}, key) where {T,N}
    shape = block_shape(store)
    values = ArrayMap(T, shape, store.empty, Val(:nocheck))
    return TreeLeaf{T,N}(tree_prefix(store, key, 0), values, 0)
end

function tree_branch(store::TreeMap{T,N}, key, level::Int) where {T,N}
    empty = store.roots.empty::TreeLeaf{T,N}
    children = ArrayMap(TreeNode{T,N}, block_shape(store), empty, Val(:nocheck))
    return TreeBranch{T,N}(tree_prefix(store, key, level), level, children, 0, 0)
end

# Follow nodes and verify compressed prefixes.
@inline map_slot(store::TreeMap, key, encoded) = tree_find(store, encoded, Val(false))
@inline prepare(store::TreeMap, key, encoded) = tree_find(store, encoded, Val(true))

@inline map_slot(store::SubtreeRadixMap{T,N}, key, route::Tuple{NTuple{N,UInt},Int}) where {T,N} =
    tree_find(store, route[1], route[2], Val(false))
@inline prepare(store::SubtreeRadixMap{T,N}, key, route::Tuple{NTuple{N,UInt},Int}) where {T,N} =
    tree_find(store, route[1], route[2], Val(true))
@inline function reserve!(store::SubtreeRadixMap{T,N}, key, route::Tuple{NTuple{N,UInt},Int}, slot, state) where {T,N}
    link, parent, address = state
    if slot !== nothing
        if slot.used > 0
            return LeafSlot(slot, address)
        end
        # Keep retraversal for an invalidated hint out of the common insertion path.
        return @noinline map_reserve!(store, key, route)
    elseif parent === nothing
        return tree_reserve!(store, route[1], root_link(store), nothing, 1)
    elseif parent.used >= 2
        return tree_reserve!(store, route[1], link, parent, address)
    end
    return @noinline map_reserve!(store, key, route)
end
@inline map_reserve!(store::SubtreeRadixMap{T,N}, key, route::Tuple{NTuple{N,UInt},Int}) where {T,N} =
    map_reserve!(store, key, route[1], route[2])

@inline function tree_find(store::TreeMap{T,N}, encoded, ::Val{prepared}) where {T,N,prepared}
    return tree_find(store, encoded, root_index(store, encoded), Val(prepared))
end

@inline function tree_find(store::TreeMap{T,N}, encoded, index::Int, ::Val{prepared}) where {T,N,prepared}
    if store isa SubtreeRadixMap && index == 1
        # An existing central leaf is direct; otherwise this slot supplies its ancestor.
        link = @inbounds map_slot(store.roots, 1, nothing)
        leaf = slot_entry(link, store)
        if leaf isa TreeLeaf{T,N}
            if leaf === store.roots.empty
                return prepared ? (nothing, (link, nothing, 1)) : nothing
            end
            address = tree_address(store, encoded, 0)
            return prepared ? (leaf, (link, nothing, address)) : LeafSlot(leaf, address)
        end
    else
        link = @inbounds map_slot(store.roots, index, nothing)
    end
    level = store isa SubtreeRadixMap ? max(index - 1, node_level(slot_entry(link, store))) :
        tree_level(store, ntuple(_ -> typemax(UInt), Val(N)))
    return tree_find(store, encoded, link, level, Val(prepared))
end

# Validate a compressed transition before entering its node. Adjacent levels and
# a depth-zero shortcut already establish the destination's prefix.
@inline function tree_find(store::TreeMap{T,N}, encoded, link, expectedlevel::Int, ::Val{prepared}) where {T,N,prepared}
    empty = store.roots.empty::TreeLeaf{T,N}
    parent, address = nothing, 1
    while true
        node = slot_entry(link, store)
        if node === empty
            return prepared ? (nothing, (link, parent, address)) : nothing
        end
        if node isa TreeLeaf{T,N}
            if expectedlevel != 0 && !tree_matches(store, encoded, node.prefix, 0)
                return prepared ? (nothing, (link, parent, address)) : nothing
            end
            # Enter the established leaf and select its value slot.
            address = tree_address(store, encoded, 0)
            return prepared ? (node, (link, parent, address)) : LeafSlot(node, address)
        end
        branch = node::TreeBranch{T,N}
        if branch.level != expectedlevel && !tree_matches(store, encoded, branch.prefix, branch.level)
            return prepared ? (nothing, (link, parent, address)) : nothing
        end
        # Enter the established branch and select its child slot.
        address = tree_address(store, encoded, branch.level)
        expectedlevel = branch.level - 1
        parent = branch
        link = @inbounds map_slot(branch.children, address, nothing)
    end
end

@inline function reserve!(store::TreeMap, key, encoded, slot, state)
    link, parent, address = state
    if slot !== nothing
        if slot.used > 0
            return LeafSlot(slot, address)
        end
        return map_reserve!(store, key, encoded)
    end
    # Without a parent, resume from the owning root, never a shortcut slot.
    link = parent === nothing ? root_link(store) : link
    if parent === nothing || parent.used >= 2
        return tree_reserve!(store, encoded, link, parent, address)
    end
    return map_reserve!(store, key, encoded)
end

function tree_add_child!(branch::TreeBranch, address::Int, child)
    slot_publish!(map_slot(branch.children, address, nothing), child)
    branch.used += 1
    branch.occupied ⊻= address
    return nothing
end

@inline has_shortcut(::RadixMap, branch) = false
@inline function has_shortcut(store::SubtreeRadixMap, branch)
    # An off-zero owning root can still be aliased above its occupied prefix.
    depth = max(branch.level + 1, root_index(store, branch.prefix))
    return slot_entry(map_slot(store.roots, depth, nothing)) === branch
end
@inline zero_child(::RadixMap, branch) = 1
@inline zero_child(store::SubtreeRadixMap, branch) = @inbounds store.zero_path[branch.level + 1]

function tree_publish!(store::TreeMap, link, was_empty::Bool, node::TreeNode, parent, address::Int)
    slot_publish!(link, node)
    if was_empty && parent !== nothing
        parent.used += 1
        parent.occupied ⊻= address
    end
    if parent === nothing
        refresh!(store, length(store.roots.reference.mem)-2)
    elseif address == zero_child(store, parent) && has_shortcut(store, parent)
        refresh!(store, parent.level-1)
    end
    return nothing
end

# Walk node references; construct a writable link only at a missing position.
@inline map_reserve!(store::RadixMap{T,N}, key, encoded) where {T,N} =
    map_reserve!(store, key, encoded, slot_entry(map_slot(store.roots, 1, nothing)),
        tree_level(store, ntuple(_ -> typemax(UInt), Val(N))))

@inline function map_reserve!(store::TreeMap{T,N}, key, encoded, node, expectedlevel::Int) where {T,N}
    empty = store.roots.empty
    parent, address = nothing, 1
    # Carry the next adjacent-level index; compressed gaps recompute it below.
    nextaddress = tree_address(store, encoded, expectedlevel)
    while node isa TreeBranch{T,N}
        branch = node
        if branch.level != expectedlevel
            if !tree_matches(store, encoded, branch.prefix, branch.level)
                break
            end
            nextaddress = tree_address(store, encoded, branch.level)
        end
        parent = branch
        expectedlevel = branch.level - 1
        address = nextaddress
        nextaddress = tree_address(store, encoded, expectedlevel)
        node = slot_entry(map_slot(branch.children, address, nothing))
    end
    if node isa TreeLeaf{T,N} && node !== empty
        if expectedlevel == 0 || tree_matches(store, encoded, node.prefix, 0)
            return LeafSlot(node, tree_address(store, encoded, 0))
        end
    end
    link = parent === nothing ? root_link(store) : map_slot(parent.children, address, nothing)
    return tree_reserve!(store, encoded, link, parent, address)
end

@inline map_reserve!(store::SubtreeRadixMap, key, encoded) =
    map_reserve!(store, key, encoded, root_index(store, encoded))

@inline function map_reserve!(store::SubtreeRadixMap{T,N}, key, encoded, index::Int) where {T,N}
    if index == 1
        node = slot_entry(map_slot(store.roots, 1, nothing))
        if node isa TreeLeaf{T,N}
            if node !== store.roots.empty
                return LeafSlot(node, tree_address(store, encoded, 0))
            end
            return tree_reserve!(store, encoded, root_link(store), nothing, 1)
        end
    else
        node = slot_entry(map_slot(store.roots, index, nothing))
    end
    return map_reserve!(store, key, encoded, node, max(index-1, node_level(node)))
end

# known_missing is valid only when lookup has just established an empty or divergent link.
function tree_reserve!(store::TreeMap{T,N}, encoded, link, parent, address,
    ::Val{known_missing}=Val(false)) where {T,N,known_missing}
    empty = store.roots.empty::TreeLeaf{T,N}
    expectedlevel = parent === nothing ? tree_level(store, ntuple(_ -> typemax(UInt), Val(N))) : parent.level - 1
    while true
        node = slot_entry(link)
        if node === empty
            leaf = tree_leaf(store, encoded)
            tree_publish!(store, link, true, leaf, parent, address)
            return LeafSlot(leaf, tree_address(store, encoded, 0))
        elseif known_missing || (node_level(node) != expectedlevel &&
                !tree_matches(store, encoded, node_prefix(node), node_level(node)))
            prefix = node_prefix(node)
            difference = ntuple(i -> prefix[i] ⊻ encoded[i], Val(N))
            level = tree_level(store, difference)
            branch = tree_branch(store, encoded, level)
            leaf = tree_leaf(store, encoded)
            tree_add_child!(branch, tree_address(store, prefix, level), node)
            tree_add_child!(branch, tree_address(store, encoded, level), leaf)
            tree_publish!(store, link, false, branch, parent, address)
            return LeafSlot(leaf, tree_address(store, encoded, 0))
        elseif node isa TreeLeaf{T,N}
            return LeafSlot(node, tree_address(store, encoded, 0))
        else
            parent = node::TreeBranch{T,N}
            address = tree_address(store, encoded, parent.level)
            expectedlevel = parent.level - 1
            link = @inbounds map_slot(parent.children, address, nothing)
        end
    end
end

# Compressed branches have at least two children; only the leaf's parent can collapse.
function tree_remove!(store::TreeMap{T,N}, encoded, leaf) where {T,N}
    link = root_link(store)
    node = slot_entry(link)
    depth = length(store.roots.reference.mem)-2
    if node === leaf
        slot_publish!(link, store.roots.empty)
        refresh!(store, depth)
        return nothing
    end
    while true
        branch = node::TreeBranch{T,N}
        address = tree_address(store, encoded, branch.level)
        child = slot_entry(map_slot(branch.children, address, nothing))
        if child === leaf
            refresh = has_shortcut(store, branch)
            delete_key!(branch.children, address)
            branch.used -= 1
            branch.occupied ⊻= address
            if branch.used == 1
                remaining = slot_entry(map_slot(branch.children, branch.occupied, nothing))
                slot_publish!(link, remaining)
                if refresh
                    refresh!(store, depth)
                end
            elseif refresh && address == zero_child(store, branch)
                refresh!(store, branch.level-1)
            end
            return nothing
        end
        link = @inbounds map_slot(branch.children, address, nothing)
        node = child
        depth = branch.level-1
    end
end

function map_delete!(store::TreeMap, key, slot::LeafSlot)
    slot_publish!(slot, store.empty)
    slot.leaf.used -= 1
    if slot.leaf.used == 0
        tree_remove!(store, map_routing(store, key), slot.leaf)
    end
    return nothing
end

@inline delete_key!(map::TreeMap, key) = delete_key!(map, key, map_routing(map, key))

# Use depth shortcuts for removal too; only an emptied leaf needs root-based pruning.
@inline function delete_key!(map::SubtreeRadixMap, key, routing)
    slot = map_slot(map, key, routing)
    return @inline map_delete!(map, key, slot)
end

@inline function delete_key!(map::TreeMap{T,N}, key, encoded) where {T,N}
    link = root_link(map)
    node = slot_entry(link)
    depth = length(map.roots.reference.mem)-2
    parent, address = nothing, 1
    while node isa TreeBranch{T,N}
        parent = node
        address = tree_address(map, encoded, parent.level)
        child = slot_entry(map_slot(parent.children, address, nothing))
        if child isa TreeLeaf{T,N}
            node = child
            break
        end
        link = @inbounds map_slot(parent.children, address, nothing)
        node = child
        depth = parent.level-1
    end
    leaf = node::TreeLeaf{T,N}
    delete_key!(leaf.values, tree_address(map, encoded, 0))
    leaf.used -= 1
    if leaf.used == 0
        refresh = parent === nothing || has_shortcut(map, parent)
        if parent === nothing
            slot_publish!(link, map.roots.empty)
            refresh!(map, depth)
        else
            delete_key!(parent.children, address)
            parent.used -= 1
            parent.occupied ⊻= address
            if parent.used == 1
                remaining = slot_entry(map_slot(parent.children, parent.occupied, nothing))
                slot_publish!(link, remaining)
                if refresh
                    refresh!(map, depth)
                end
            elseif refresh && address == zero_child(map, parent)
                refresh!(map, parent.level-1)
            end
        end
    end
    return nothing
end
