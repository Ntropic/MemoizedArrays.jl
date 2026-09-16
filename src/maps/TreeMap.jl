# Sparse tree storage: one MSB tree or MSB roots selected by coordinate bit depth.
# Allocates branches on demand and prunes unused storage after eviction.

abstract type TreeNode{T,N} end

# Prefixes are immutable; counters track occupied entries and children.
mutable struct TreeBranch{T,N} <: TreeNode{T,N}
    const prefix::NTuple{N,UInt}
    const level::Int
    const children::Memory{Union{Nothing,TreeNode{T,N}}}
    used::Int
    occupied::Int
end

mutable struct TreeLeaf{T,N} <: TreeNode{T,N}
    const prefix::NTuple{N,UInt}
    const values::Memory{T}
    used::Int
end

struct LeafSlot{L}
    leaf::L
    address::Int
end

struct TreeHint{T,N}
    link::MemoryRef{Union{Nothing,TreeNode{T,N}}}
    parent::Union{Nothing,TreeBranch{T,N}}
    address::Int # Entry address for a found leaf, child address for a miss.
end

@inline preparedentry(leaf::TreeLeaf, hint::TreeHint) = @inbounds leaf.values[hint.address]

"""Most significant bit (MSB) radix map: traverse high coordinate bits first."""
struct MSBRadixMap{T,N,BlockShape} <: SparseMap{T,N}
    roots::Memory{Union{Nothing,TreeNode{T,N}}}
    empty::Empty
end

"""MSB trees selected by coordinate bit depth; storage option `:lsbtree`."""
struct LSBRadixMap{T,N,BlockShape} <: SparseMap{T,N}
    roots::Memory{Union{Nothing,TreeNode{T,N}}}
    empty::Empty
end

const RadixMap{T,N,B} = Union{MSBRadixMap{T,N,B},LSBRadixMap{T,N,B}}

# Unsigned zigzag includes typemin(Int) without signed overflow.
@inline signed2unsigned(i::Int) = (reinterpret(UInt, i) << 1) ⊻ reinterpret(UInt, i >> (8sizeof(Int) - 1))
@inline tree_key(key::NTuple{N,Int}) where N = ntuple(i -> signed2unsigned(key[i]), Val(N))
@inline store_hash(::RadixMap, key) = tree_key(key)
@inline block_shape(::RadixMap{T,N,B}) where {T,N,B} = B
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

@inline slot_entry(slot::LeafSlot) = @inbounds slot.leaf.values[slot.address]
@inline slot_publish!(slot::LeafSlot, value) = (@inbounds slot.leaf.values[slot.address] = value)
store_retain!(::RadixMap, slot::LeafSlot) = (slot.leaf.used += 1)

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

@inline tree_bits(store::RadixMap, axis::Int, level::Int) = (level + 1) * trailing_zeros(block_shape(store)[axis])

@inline function tree_prefix(store::RadixMap{T,N}, key, level::Int) where {T,N}
    return ntuple(i -> key[i] & (typemax(UInt) << unsigned(tree_bits(store, i, level))), Val(N))
end

# Minimum common level covering all encoded axes.
@inline function tree_level(block, key)
    level = 0
    for i in eachindex(key)
        bits = 8sizeof(UInt) - leading_zeros(key[i]) - trailing_zeros(block[i])
        if bits > 0
            level = max(level, cld(bits, trailing_zeros(block[i])))
        end
    end
    return level
end
@inline tree_level(store::RadixMap, key) = tree_level(block_shape(store), key)

@inline rootindex(::MSBRadixMap, encoded) = 1
@inline rootindex(store::LSBRadixMap, encoded) = tree_level(store, encoded) + 1

# Flatten only a checked local block in column-major order.
@inline function tree_address(store::RadixMap{T,N}, key, level::Int) where {T,N}
    address, shift = UInt(0), 0
    for i in 1:N
        width = block_shape(store)[i]
        digit = (key[i] >> unsigned(tree_bits(store, i, level - 1))) & UInt(width - 1)
        address |= digit << unsigned(shift)
        shift += trailing_zeros(width)
    end
    return Int(address) + 1
end

function MSBRadixMap(::Type{T}, ranges::Tuple{Vararg{Any,N}}, block::VT{N,Int}, empty::Empty) where {T,N}
    @astuple N block
    block_length(block, 2)
    roots = empty_map(Union{Nothing,TreeNode{T,N}}, (1,), nothing)
    return MSBRadixMap{T,N,block}(roots, empty)
end

function LSBRadixMap(::Type{T}, ranges::Tuple{Vararg{Any,N}}, block::VT{N,Int}, empty::Empty) where {T,N}
    @astuple N block
    block_length(block, 2)
    levels = tree_level(block, ntuple(_ -> typemax(UInt), Val(N))) + 1
    roots = empty_map(Union{Nothing,TreeNode{T,N}}, (levels,), nothing)
    return LSBRadixMap{T,N,block}(roots, empty)
end

function tree_leaf(store::RadixMap{T,N}, key) where {T,N}
    return TreeLeaf{T,N}(tree_prefix(store, key, 0), empty_map(T, block_shape(store), store.empty), 0)
end

function tree_branch(store::RadixMap{T,N}, key, level::Int) where {T,N}
    children = empty_map(Union{Nothing,TreeNode{T,N}}, block_shape(store), nothing)
    return TreeBranch{T,N}(tree_prefix(store, key, level), level, children, 0, 0)
end

# Follow nodes and verify compressed prefixes.
@inline store_lookup(store::RadixMap, key, encoded) = treefind(store, encoded, Val(false))
@inline prepare(store::RadixMap, key, encoded) = treefind(store, encoded, Val(true))

@inline function treefind(store::RadixMap{T,N}, encoded, ::Val{prepared}) where {T,N,prepared}
    index = rootindex(store, encoded)
    link = @inbounds memoryref(store.roots, index)
    # Adjacent levels already establish the child's prefix through the selected digit.
    level = store isa LSBRadixMap ? index : typemax(Int)
    parent, address = nothing, 1
    while true
        node = slot_entry(link)
        hint = TreeHint{T,N}(link, parent, address)
        if node === nothing
            return prepared ? (nothing, hint) : nothing
        end
        if node isa TreeLeaf{T,N}
            if level != 1 && tree_prefix(store, encoded, 0) != node.prefix
                return prepared ? (nothing, hint) : nothing
            end
            address = tree_address(store, encoded, 0)
            return prepared ? (node, TreeHint{T,N}(link, parent, address)) : LeafSlot(node, address)
        end
        branch = node::TreeBranch{T,N}
        if branch.level + 1 != level && tree_prefix(store, encoded, branch.level) != branch.prefix
            return prepared ? (nothing, hint) : nothing
        end
        address = tree_address(store, encoded, branch.level)
        level = branch.level
        parent = branch
        link = @inbounds memoryref(branch.children, address)
    end
end

@noinline function reserve!(store::RadixMap, key, encoded, slot, hint)
    if slot !== nothing
        if slot.used > 0
            return LeafSlot(slot, hint.address)
        end
        return store_reserve!(store, key, encoded)
    end
    link, parent, address = hint.link, hint.parent, hint.address
    if parent === nothing || parent.used >= 2
        return treereserve!(store, encoded, link, parent, address)
    end
    return store_reserve!(store, key, encoded)
end

function tree_add_child!(branch::TreeBranch, address::Int, child)
    @inbounds branch.children[address] = child
    branch.used += 1
    branch.occupied ⊻= address
    return nothing
end

function tree_publish!(link, old, node, parent, address::Int)
    slot_publish!(link, node)
    if old === nothing && parent !== nothing
        parent.used += 1
        parent.occupied ⊻= address
    end
    return nothing
end

function store_reserve!(store::RadixMap{T,N}, key, encoded) where {T,N}
    link = @inbounds memoryref(store.roots, rootindex(store, encoded))
    return treereserve!(store, encoded, link, nothing, 1)
end

function treereserve!(store::RadixMap{T,N}, encoded, link, parent, address) where {T,N}
    while true
        node = slot_entry(link)
        if node === nothing
            leaf = tree_leaf(store, encoded)
            tree_publish!(link, node, leaf, parent, address)
            return LeafSlot(leaf, tree_address(store, encoded, 0))
        elseif tree_prefix(store, encoded, node_level(node)) != node_prefix(node)
            prefix = node_prefix(node)
            difference = ntuple(i -> prefix[i] ⊻ encoded[i], Val(N))
            level = tree_level(store, difference)
            branch = tree_branch(store, encoded, level)
            leaf = tree_leaf(store, encoded)
            tree_add_child!(branch, tree_address(store, prefix, level), node)
            tree_add_child!(branch, tree_address(store, encoded, level), leaf)
            tree_publish!(link, node, branch, parent, address)
            return LeafSlot(leaf, tree_address(store, encoded, 0))
        elseif node isa TreeLeaf{T,N}
            return LeafSlot(node, tree_address(store, encoded, 0))
        else
            parent = node::TreeBranch{T,N}
            address = tree_address(store, encoded, parent.level)
            link = @inbounds memoryref(parent.children, address)
        end
    end
end

# Compressed branches have at least two children; only the leaf's parent can collapse.
function tree_remove!(store::RadixMap{T,N}, encoded, leaf) where {T,N}
    link = @inbounds memoryref(store.roots, rootindex(store, encoded))
    node = slot_entry(link)
    if node === leaf
        slot_publish!(link, nothing)
        return nothing
    end
    while true
        branch = node::TreeBranch{T,N}
        address = tree_address(store, encoded, branch.level)
        child = @inbounds branch.children[address]
        if child === leaf
            @inbounds branch.children[address] = nothing
            branch.used -= 1
            branch.occupied ⊻= address
            if branch.used == 1
                slot_publish!(link, @inbounds branch.children[branch.occupied])
            end
            return nothing
        end
        link = @inbounds memoryref(branch.children, address)
        node = child
    end
end

function store_delete!(store::RadixMap, key, slot::LeafSlot)
    slot_publish!(slot, store.empty)
    slot.leaf.used -= 1
    if slot.leaf.used == 0
        tree_remove!(store, tree_key(key), slot.leaf)
    end
    return nothing
end

function delete_key!(map::RadixMap{T,N}, key) where {T,N}
    encoded = tree_key(key)
    link = @inbounds memoryref(map.roots, rootindex(map, encoded))
    node = slot_entry(link)
    parent, address = nothing, 1
    while node isa TreeBranch{T,N}
        parent = node
        address = tree_address(map, encoded, parent.level)
        child = @inbounds parent.children[address]
        if child isa TreeLeaf{T,N}
            node = child
            break
        end
        link = @inbounds memoryref(parent.children, address)
        node = child
    end
    leaf = node::TreeLeaf{T,N}
    @inbounds leaf.values[tree_address(map, encoded, 0)] = map.empty
    leaf.used -= 1
    if leaf.used == 0
        if parent === nothing
            slot_publish!(link, nothing)
        else
            @inbounds parent.children[address] = nothing
            parent.used -= 1
            parent.occupied ⊻= address
            if parent.used == 1
                slot_publish!(link, @inbounds parent.children[parent.occupied])
            end
        end
    end
    return nothing
end
