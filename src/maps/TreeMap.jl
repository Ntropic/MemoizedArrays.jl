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

struct LeafSlot{L,R}
    leaf::L
    reference::R
end

@inline LeafSlot(leaf::TreeLeaf, address::Int) = LeafSlot(leaf, store_lookup(leaf.values, address, nothing))

@inline preparedentry(leaf::TreeLeaf, state::Tuple) = slot_entry(store_lookup(leaf.values, state[3], nothing))

"""Radix map: traverse high coordinate bits first."""
struct RadixMap{T,N,BlockShape,D,E} <: SparseMap{T,N}
    roots::ArrayMap{TreeNode{T,N},1}
    empty::E
    domain::D
end

"""One radix tree with depth-indexed shortcuts; storage option `:subtreetree`."""
struct SubtreeRadixMap{T,N,BlockShape,D,E} <: SparseMap{T,N}
    roots::ArrayMap{TreeNode{T,N},1}
    empty::E
    domain::D
end

const TreeMap{T,N,B} = Union{RadixMap{T,N,B},SubtreeRadixMap{T,N,B}}

# Unsigned zigzag includes typemin(Int) without signed overflow.
@inline signed2unsigned(i::Int) = (reinterpret(UInt, i) << 1) ⊻ reinterpret(UInt, i >> (8sizeof(Int) - 1))
@inline tree_key(key::NTuple{N,Int}) where N = ntuple(i -> signed2unsigned(key[i]), Val(N))
@inline store_hash(::SubtreeRadixMap, key) = tree_key(key)
# Cache routing converts the checked coordinates once; traversal only sees unsigned offsets.
@inline store_hash(store::RadixMap{T,N}, key) where {T,N} = ntuple(i -> reinterpret(UInt, key[i]) - reinterpret(UInt, first(store.domain[i])), Val(N))
@inline block_shape(::TreeMap{T,N,B}) where {T,N,B} = B
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
store_retain!(::TreeMap, slot::LeafSlot) = (slot.leaf.used += 1)

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

@inline tree_bits(store::TreeMap, axis::Int, level::Int) = (level + 1) * trailing_zeros(block_shape(store)[axis])

@inline function tree_prefix(store::TreeMap{T,N}, key, level::Int) where {T,N}
    return ntuple(i -> key[i] & (typemax(UInt) << unsigned(tree_bits(store, i, level))), Val(N))
end

# Equal-width axes share one prefix shift after reducing their differing bits.
@inline function tree_matches(store::TreeMap{T,N}, key, prefix, level::Int) where {T,N}
    block = block_shape(store)
    if N == 0
        return true
    elseif all(width -> width == first(block), block)
        difference = foldl(|, ntuple(i -> key[i] ⊻ prefix[i], Val(N)); init=UInt(0))
        return iszero(difference >> unsigned(tree_bits(store, 1, level)))
    end
    return tree_prefix(store, key, level) == prefix
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
@inline tree_level(store::TreeMap, key) = tree_level(block_shape(store), key)

@inline rootindex(::RadixMap, encoded) = 1
@inline rootindex(store::SubtreeRadixMap, encoded) = tree_level(store, encoded) + 1

# The last subtree slot owns the tree; earlier slots only alias its zero-prefix path.
@inline rootlink(store::TreeMap) = @inbounds store_lookup(store.roots, length(store.roots.reference.mem), nothing)
refresh!(::RadixMap) = nothing
function refresh!(store::SubtreeRadixMap{T,N}) where {T,N}
    node = slot_entry(rootlink(store))
    for depth in (length(store.roots.reference.mem)-2):-1:0
        while node isa TreeBranch{T,N} && node.level > depth
            node = slot_entry(store_lookup(node.children, 1, nothing))
        end
        slot_publish!(store_lookup(store.roots, depth + 1, nothing), node === store.roots.empty || any(!iszero, tree_prefix(store, node_prefix(node), depth)) ? store.roots.empty : node)
    end
    return nothing
end

# Flatten only a checked local block in column-major order.
@inline function tree_address(store::TreeMap{T,N}, key, level::Int) where {T,N}
    address, shift = UInt(0), 0
    block = block_shape(store)
    for i in 1:N
        width = block[i]
        bitoffset = unsigned(tree_bits(store, i, level - 1))
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

function RadixMap(::Type{T}, ranges::Tuple{Vararg{Any,N}}, block::VT{N,Int}, empty) where {T,N}
    @astuple N block
    block_length(block, 2)
    roots = ArrayMap(TreeNode{T,N}, (1,), TreeLeaf{T,N}(empty))
    return RadixMap{T,N,block,typeof(ranges),typeof(empty)}(roots, empty, ranges)
end

function SubtreeRadixMap(::Type{T}, ranges::Tuple{Vararg{Any,N}}, block::VT{N,Int}, empty) where {T,N}
    @astuple N block
    block_length(block, 2)
    levels = tree_level(block, ntuple(_ -> typemax(UInt), Val(N))) + 1
    roots = ArrayMap(TreeNode{T,N}, (levels,), TreeLeaf{T,N}(empty))
    return SubtreeRadixMap{T,N,block,typeof(ranges),typeof(empty)}(roots, empty, ranges)
end

function tree_leaf(store::TreeMap{T,N}, key) where {T,N}
    shape = block_shape(store)
    values = ArrayMap(T, shape, store.empty)
    return TreeLeaf{T,N}(tree_prefix(store, key, 0), values, 0)
end

function tree_branch(store::TreeMap{T,N}, key, level::Int) where {T,N}
    children = ArrayMap(TreeNode{T,N}, block_shape(store), store.roots.empty)
    return TreeBranch{T,N}(tree_prefix(store, key, level), level, children, 0, 0)
end

# Follow nodes and verify compressed prefixes.
@inline store_lookup(store::TreeMap, key, encoded) = treefind(store, encoded, Val(false))
@inline prepare(store::TreeMap, key, encoded) = treefind(store, encoded, Val(true))

@inline function treefind(store::TreeMap{T,N}, encoded, ::Val{prepared}) where {T,N,prepared}
    index = rootindex(store, encoded)
    link = @inbounds store_lookup(store.roots, index, nothing)
    maxlevel = tree_level(store, ntuple(_ -> typemax(UInt), Val(N)))
    return treefind(store, encoded, link, maxlevel, Val(prepared))
end

# Shortcuts enter the same top-down traversal; validate prefixes across compressed gaps.
@inline function treefind(store::TreeMap{T,N}, encoded, link, maxlevel::Int, ::Val{prepared}) where {T,N,prepared}
    # Adjacent levels already establish the child's prefix through the selected digit.
    empty = store.roots.empty::TreeLeaf{T,N}
    level = maxlevel + 1
    parent, address = nothing, 1
    while true
        node = slot_entry(link)
        if node === empty
            return prepared ? (nothing, (link, parent, address)) : nothing
        end
        if node isa TreeLeaf{T,N}
            if level != 1 && !tree_matches(store, encoded, node.prefix, 0)
                return prepared ? (nothing, (link, parent, address)) : nothing
            end
            address = tree_address(store, encoded, 0)
            return prepared ? (node, (link, parent, address)) : LeafSlot(node, address)
        end
        branch = node::TreeBranch{T,N}
        if branch.level + 1 != level && !tree_matches(store, encoded, branch.prefix, branch.level)
            return prepared ? (nothing, (link, parent, address)) : nothing
        end
        address = tree_address(store, encoded, branch.level)
        level = branch.level
        parent = branch
        link = @inbounds store_lookup(branch.children, address, nothing)
    end
end

@inline function reserve!(store::TreeMap, key, encoded, slot, state)
    link, parent, address = state
    if slot !== nothing
        if slot.used > 0
            return LeafSlot(slot, address)
        end
        return store_reserve!(store, key, encoded)
    end
    # Without a parent, resume from the owning root, never a shortcut slot.
    link = parent === nothing ? rootlink(store) : link
    if parent === nothing || parent.used >= 2
        return treereserve!(store, encoded, link, parent, address)
    end
    return store_reserve!(store, key, encoded)
end

function tree_add_child!(branch::TreeBranch, address::Int, child)
    slot_publish!(store_lookup(branch.children, address, nothing), child)
    branch.used += 1
    branch.occupied ⊻= address
    return nothing
end

function tree_publish!(store, link, old, node, parent, address::Int)
    slot_publish!(link, node)
    if old === store.roots.empty && parent !== nothing
        parent.used += 1
        parent.occupied ⊻= address
    end
    (parent === nothing || (address == 1 && all(iszero, parent.prefix))) && refresh!(store)
    return nothing
end

# Walk node references; construct a writable link only at a missing position.
@inline function store_reserve!(store::RadixMap{T,N}, key, encoded) where {T,N}
    empty = store.roots.empty
    node = slot_entry(store_lookup(store.roots, 1, nothing))
    level = tree_level(store, ntuple(_ -> typemax(UInt), Val(N))) + 1
    parent, address = nothing, 1
    # Carry the next adjacent-level index; compressed gaps recompute it below.
    nextaddress = tree_address(store, encoded, level - 1)
    while node isa TreeBranch{T,N}
        branch = node
        if branch.level + 1 != level
            if !tree_matches(store, encoded, branch.prefix, branch.level)
                break
            end
            nextaddress = tree_address(store, encoded, branch.level)
        end
        parent = branch
        level = branch.level
        address = nextaddress
        nextaddress = tree_address(store, encoded, level - 1)
        node = slot_entry(store_lookup(branch.children, address, nothing))
    end
    if node isa TreeLeaf{T,N} && node !== empty
        if level == 1 || tree_matches(store, encoded, node.prefix, 0)
            return LeafSlot(node, tree_address(store, encoded, 0))
        end
    end
    link = parent === nothing ? rootlink(store) : store_lookup(parent.children, address, nothing)
    return treereserve!(store, encoded, link, parent, address)
end

function store_reserve!(store::TreeMap{T,N}, key, encoded) where {T,N}
    link = rootlink(store)
    return treereserve!(store, encoded, link, nothing, 1)
end

function treereserve!(store::TreeMap{T,N}, encoded, link, parent, address) where {T,N}
    empty = store.roots.empty::TreeLeaf{T,N}
    while true
        node = slot_entry(link)
        if node === empty
            leaf = tree_leaf(store, encoded)
            tree_publish!(store, link, node, leaf, parent, address)
            return LeafSlot(leaf, tree_address(store, encoded, 0))
        elseif !tree_matches(store, encoded, node_prefix(node), node_level(node))
            prefix = node_prefix(node)
            difference = ntuple(i -> prefix[i] ⊻ encoded[i], Val(N))
            level = tree_level(store, difference)
            branch = tree_branch(store, encoded, level)
            leaf = tree_leaf(store, encoded)
            tree_add_child!(branch, tree_address(store, prefix, level), node)
            tree_add_child!(branch, tree_address(store, encoded, level), leaf)
            tree_publish!(store, link, node, branch, parent, address)
            return LeafSlot(leaf, tree_address(store, encoded, 0))
        elseif node isa TreeLeaf{T,N}
            return LeafSlot(node, tree_address(store, encoded, 0))
        else
            parent = node::TreeBranch{T,N}
            address = tree_address(store, encoded, parent.level)
            link = @inbounds store_lookup(parent.children, address, nothing)
        end
    end
end

# Compressed branches have at least two children; only the leaf's parent can collapse.
function tree_remove!(store::TreeMap{T,N}, encoded, leaf) where {T,N}
    link = rootlink(store)
    node = slot_entry(link)
    if node === leaf
        slot_publish!(link, store.roots.empty)
        refresh!(store)
        return nothing
    end
    while true
        branch = node::TreeBranch{T,N}
        address = tree_address(store, encoded, branch.level)
        child = slot_entry(store_lookup(branch.children, address, nothing))
        if child === leaf
            delete_key!(branch.children, address)
            branch.used -= 1
            branch.occupied ⊻= address
            if branch.used == 1
                slot_publish!(link, slot_entry(store_lookup(branch.children, branch.occupied, nothing)))
            end
            all(iszero, branch.prefix) && refresh!(store)
            return nothing
        end
        link = @inbounds store_lookup(branch.children, address, nothing)
        node = child
    end
end

function store_delete!(store::TreeMap, key, slot::LeafSlot)
    slot_publish!(slot, store.empty)
    slot.leaf.used -= 1
    if slot.leaf.used == 0
        tree_remove!(store, store_hash(store, key), slot.leaf)
    end
    return nothing
end

function delete_key!(map::TreeMap{T,N}, key) where {T,N}
    encoded = store_hash(map, key)
    link = rootlink(map)
    node = slot_entry(link)
    parent, address = nothing, 1
    while node isa TreeBranch{T,N}
        parent = node
        address = tree_address(map, encoded, parent.level)
        child = slot_entry(store_lookup(parent.children, address, nothing))
        if child isa TreeLeaf{T,N}
            node = child
            break
        end
        link = @inbounds store_lookup(parent.children, address, nothing)
        node = child
    end
    leaf = node::TreeLeaf{T,N}
    delete_key!(leaf.values, tree_address(map, encoded, 0))
    leaf.used -= 1
    if leaf.used == 0
        if parent === nothing
            slot_publish!(link, map.roots.empty)
        else
            delete_key!(parent.children, address)
            parent.used -= 1
            parent.occupied ⊻= address
            if parent.used == 1
                slot_publish!(link, slot_entry(store_lookup(parent.children, parent.occupied, nothing)))
            end
        end
        (parent === nothing || all(iszero, parent.prefix)) && refresh!(map)
    end
    return nothing
end
