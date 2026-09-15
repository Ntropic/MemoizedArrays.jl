abstract type _TreeNode{T,N} end

# Prefixes are immutable; only bounded caches maintain pruning counters.
mutable struct _TreeBranch{T,N} <: _TreeNode{T,N}
    const prefix::NTuple{N,UInt}
    const level::Int
    const children::AtomicMemory{Union{Nothing,_TreeNode{T,N}}}
    used::Int
    occupied::Int
end

mutable struct _TreeLeaf{T,N} <: _TreeNode{T,N}
    const prefix::NTuple{N,UInt}
    const values::AtomicMemory{T}
    used::Int
end

mutable struct _LSBNode{T,N}
    const values::AtomicMemory{T}
    const children::AtomicMemory{Union{Nothing,_LSBNode{T,N}}}
    used::Int
    children_used::Int
end

struct _LeafSlot{L}
    leaf::L
    address::Int
end

"""Most significant bit (MSB) radix map: traverse high coordinate bits first."""
struct MSBRadixMap{T,N,BlockShape}
    roots::AtomicMemory{Union{Nothing,_TreeNode{T,N}}}
    empty::Empty
end

"""Least significant bit (LSB) radix map: traverse low coordinate bits first."""
struct LSBRadixMap{T,N,BlockShape}
    root::_LSBNode{T,N}
    empty::Empty
end

const _RadixMap{T,N,B} = Union{MSBRadixMap{T,N,B},LSBRadixMap{T,N,B}}

# Unsigned zigzag includes typemin(Int) without signed overflow.
@inline _signed2unsigned(i::Int) = (reinterpret(UInt, i) << 1) ⊻ reinterpret(UInt, i >> (8sizeof(Int) - 1))
@inline _tree_key(key::NTuple{N,Int}) where N = ntuple(i -> _signed2unsigned(key[i]), Val(N))
@inline _store_hash(::_RadixMap, key) = _tree_key(key)
@inline _block_shape(::_RadixMap{T,N,B}) where {T,N,B} = B
@inline function _node_level(node::_TreeNode{T,N}) where {T,N}
    if node isa _TreeLeaf{T,N}
        return 0
    end
    return (node::_TreeBranch{T,N}).level
end
@inline function _node_prefix(node::_TreeNode{T,N}) where {T,N}
    if node isa _TreeLeaf{T,N}
        return node.prefix
    end
    return (node::_TreeBranch{T,N}).prefix
end

@inline _slot_entry(slot::_LeafSlot) = @inbounds @atomic :acquire slot.leaf.values[slot.address]
@inline _slot_publish!(slot::_LeafSlot, value) = (@inbounds @atomic :release slot.leaf.values[slot.address] = value)
@inline _slot_claim!(slot::_LeafSlot, empty, entry) =
    @inbounds @atomicreplace :acquire_release :acquire slot.leaf.values[slot.address] empty => entry
_store_retain!(::_RadixMap, slot::_LeafSlot) = (slot.leaf.used += 1)

# Validate local blocks without multiplying the global coordinate extents.
function _block_length(shape, minimum::Int)
    count = 1
    for width in shape
        if width < minimum || !ispow2(width)
            throw(ArgumentError("block dimensions must be powers of two ≥ $minimum"))
        end
        count = Base.checked_mul(count, width)
    end
    return count
end

@inline function _tree_bits(store::MSBRadixMap, axis::Int, level::Int)
    return (level + 1) * trailing_zeros(_block_shape(store)[axis])
end
@inline _tree_bits(::LSBRadixMap, axis::Int, level::Int) = 0

@inline function _tree_prefix(store::MSBRadixMap{T,N}, key, level::Int) where {T,N}
    return ntuple(i -> key[i] & (typemax(UInt) << unsigned(_tree_bits(store, i, level))), Val(N))
end

# Minimum common level covering all encoded axes.
@inline function _tree_level(block, key)
    level = 0
    for i in eachindex(key)
        bits = 8sizeof(UInt) - leading_zeros(key[i]) - trailing_zeros(block[i])
        if bits > 0
            level = max(level, cld(bits, trailing_zeros(block[i])))
        end
    end
    return level
end
@inline _tree_level(store::MSBRadixMap, key) = _tree_level(_block_shape(store), key)

# Flatten only a checked local block in column-major order.
@inline function _tree_address(store::_RadixMap{T,N}, key, level::Int) where {T,N}
    address, shift = UInt(0), 0
    for i in 1:N
        width = _block_shape(store)[i]
        digit = (key[i] >> unsigned(_tree_bits(store, i, level - 1))) & UInt(width - 1)
        address |= digit << unsigned(shift)
        shift += trailing_zeros(width)
    end
    return Int(address) + 1
end

function MSBRadixMap(::Type{T}, ranges::NTuple{N}, block::VT{N,Int}, empty::Empty) where {T,N}
    @astuple N block
    _block_length(block, 2)
    roots = _empty_map(Union{Nothing,_TreeNode{T,N}}, (1,), nothing)
    return MSBRadixMap{T,N,block}(roots, empty)
end

function _lsb_node(::Type{T}, ::Val{N}, block, empty) where {T,N}
    values = _empty_map(T, block, empty)
    children = _empty_map(Union{Nothing,_LSBNode{T,N}}, block, nothing)
    return _LSBNode{T,N}(values, children, 0, 0)
end
_lsb_node(store::LSBRadixMap{T,N}) where {T,N} =
    _lsb_node(T, Val(N), _block_shape(store), store.empty)

function LSBRadixMap(::Type{T}, ranges::NTuple{N}, block::VT{N,Int}, empty::Empty) where {T,N}
    @astuple N block
    _block_length(block, 2)
    return LSBRadixMap{T,N,block}(_lsb_node(T, Val(N), block, empty), empty)
end

function _tree_leaf(store::MSBRadixMap{T,N}, key) where {T,N}
    return _TreeLeaf{T,N}(_tree_prefix(store, key, 0), _empty_map(T, _block_shape(store), store.empty), 0)
end

function _tree_branch(store::MSBRadixMap{T,N}, key, level::Int) where {T,N}
    children = _empty_map(Union{Nothing,_TreeNode{T,N}}, _block_shape(store), nothing)
    return _TreeBranch{T,N}(_tree_prefix(store, key, level), level, children, 0, 0)
end

# Readers follow atomically published nodes and verify compressed prefixes.
@inline function _store_lookup(store::MSBRadixMap{T,N}, key, encoded) where {T,N}
    node = @inbounds @atomic :acquire store.roots[1]
    while node !== nothing
        if node isa _TreeLeaf{T,N}
            if _tree_prefix(store, encoded, 0) != node.prefix
                return nothing
            end
            return _LeafSlot(node, _tree_address(store, encoded, 0))
        end
        branch = node::_TreeBranch{T,N}
        if _tree_prefix(store, encoded, branch.level) != branch.prefix
            return nothing
        end
        address = _tree_address(store, encoded, branch.level)
        node = @inbounds @atomic :acquire branch.children[address]
    end
    return nothing
end

function _tree_add_child!(branch::_TreeBranch, address::Int, child, ::Val{Bounded}) where Bounded
    @inbounds @atomic :release branch.children[address] = child
    if Bounded
        branch.used += 1
        branch.occupied ⊻= address
    end
    return nothing
end

# Unbounded insertion retries the observed link; bounded insertion owns the outer lock.
@inline function _tree_publish!(link, old, node, parent, address::Int, ::Val{Bounded}) where Bounded
    if Bounded
        _slot_publish!(link, node)
        if old === nothing && parent !== nothing
            parent.used += 1
            parent.occupied ⊻= address
        end
        return true
    end
    return _slot_claim!(link, old, node).success
end

function _store_reserve!(store::MSBRadixMap{T,N}, key, encoded, bounded::Val{Bounded}) where {T,N,Bounded}
    link = @inbounds memoryref(store.roots, 1)
    parent, address = nothing, 1
    while true
        node = _slot_entry(link)
        if node === nothing
            leaf = _tree_leaf(store, encoded)
            if _tree_publish!(link, node, leaf, parent, address, bounded)
                return _LeafSlot(leaf, _tree_address(store, encoded, 0))
            end
        elseif _tree_prefix(store, encoded, _node_level(node)) != _node_prefix(node)
            prefix = _node_prefix(node)
            difference = ntuple(i -> prefix[i] ⊻ encoded[i], Val(N))
            level = _tree_level(store, difference)
            branch = _tree_branch(store, encoded, level)
            leaf = _tree_leaf(store, encoded)
            _tree_add_child!(branch, _tree_address(store, prefix, level), node, bounded)
            _tree_add_child!(branch, _tree_address(store, encoded, level), leaf, bounded)
            if _tree_publish!(link, node, branch, parent, address, bounded)
                return _LeafSlot(leaf, _tree_address(store, encoded, 0))
            end
        elseif node isa _TreeLeaf{T,N}
            return _LeafSlot(node, _tree_address(store, encoded, 0))
        else
            parent = node::_TreeBranch{T,N}
            address = _tree_address(store, encoded, parent.level)
            link = @inbounds memoryref(parent.children, address)
        end
    end
end

@inline function _lsb_fits(store::LSBRadixMap{T,N}, key) where {T,N}
    for i in 1:N
        if key[i] >= UInt(_block_shape(store)[i])
            return false
        end
    end
    return true
end
@inline _lsb_shift(store::LSBRadixMap{T,N}, key) where {T,N} =
    ntuple(i -> key[i] >> trailing_zeros(_block_shape(store)[i]), Val(N))

@inline function _store_lookup(store::LSBRadixMap, key, encoded)
    node = store.root
    while !_lsb_fits(store, encoded)
        address = _tree_address(store, encoded, 1)
        child = @inbounds @atomic :acquire node.children[address]
        if child === nothing
            return nothing
        end
        node = child
        encoded = _lsb_shift(store, encoded)
    end
    return _LeafSlot(node, _tree_address(store, encoded, 0))
end

function _store_reserve!(store::LSBRadixMap, key, encoded, ::Val{Bounded}) where Bounded
    node = store.root
    while !_lsb_fits(store, encoded)
        address = _tree_address(store, encoded, 1)
        link = @inbounds memoryref(node.children, address)
        child = _slot_entry(link)
        if child === nothing
            child = _lsb_node(store)
            if Bounded
                _slot_publish!(link, child)
                node.children_used += 1
            else
                claim = _slot_claim!(link, nothing, child)
                if !claim.success
                    child = claim.old
                end
            end
        end
        node = child
        encoded = _lsb_shift(store, encoded)
    end
    return _LeafSlot(node, _tree_address(store, encoded, 0))
end

_store_reserve!(store::_RadixMap, key, encoded) = _store_reserve!(store, key, encoded, Val(true))

# Occupancy XOR finds a sole MSB child without scanning its block.
function _tree_remove!(store::MSBRadixMap{T,N}, node, encoded, leaf) where {T,N}
    if node isa _TreeLeaf{T,N}
        if node !== leaf
            error("attempted to remove a detached cache leaf")
        end
        return nothing
    end
    branch = node::_TreeBranch{T,N}
    address = _tree_address(store, encoded, branch.level)
    child = @inbounds @atomic :acquire branch.children[address]
    replacement = _tree_remove!(store, child, encoded, leaf)
    @inbounds @atomic :release branch.children[address] = replacement
    if replacement === nothing
        branch.used -= 1
        branch.occupied ⊻= address
    end
    if branch.used == 0
        return nothing
    elseif branch.used == 1
        return @inbounds @atomic :acquire branch.children[branch.occupied]
    end
    return branch
end

function _store_delete!(store::MSBRadixMap, key, slot::_LeafSlot)
    _slot_publish!(slot, store.empty)
    slot.leaf.used -= 1
    if slot.leaf.used == 0
        root = @inbounds @atomic :acquire store.roots[1]
        root = _tree_remove!(store, root, _tree_key(key), slot.leaf)
        @inbounds @atomic :release store.roots[1] = root
    end
    return nothing
end

function _tree_remove!(store::LSBRadixMap, node::_LSBNode, encoded, leaf)
    if node !== leaf
        address = _tree_address(store, encoded, 1)
        child = @inbounds @atomic :acquire node.children[address]
        replacement = _tree_remove!(store, child, _lsb_shift(store, encoded), leaf)
        if replacement === nothing
            @inbounds @atomic :release node.children[address] = nothing
            node.children_used -= 1
        end
    end
    return node.used == 0 && node.children_used == 0 ? nothing : node
end

function _store_delete!(store::LSBRadixMap, key, slot::_LeafSlot)
    _slot_publish!(slot, store.empty)
    slot.leaf.used -= 1
    if slot.leaf.used == 0 && slot.leaf.children_used == 0
        _tree_remove!(store, store.root, _tree_key(key), slot.leaf)
    end
    return nothing
end
