# Readers retain atomic slot references while writers link fully initialized
# nodes by compare-and-swap. Memoized leaves and Subtree's central leaf are permanent. Other
# LRU leaves reserve their lifetime through publication and detach only once empty.
abstract type ConcurrentTreeNode{T,N} end

# Structural insertion shares the tree; pruning and shortcut maintenance exclude
# insertions briefly. Neither mode is held while generating or reading values.
mutable struct TreeStructure <: Base.AbstractLock
    const writer_lock::ReentrantLock
    @atomic inserting::Int
    @atomic pruning::Bool
    depth::Int
end
TreeStructure() = TreeStructure(ReentrantLock(), 0, false, 0)

function Base.lock(gate::TreeStructure)
    lock(gate.writer_lock)
    if gate.depth == 0
        @atomic :sequentially_consistent gate.pruning = true
        try
            while (@atomic :sequentially_consistent gate.inserting) != 0
                yield()
            end
        catch
            @atomic :sequentially_consistent gate.pruning = false
            unlock(gate.writer_lock)
            rethrow()
        end
    end
    gate.depth += 1
    return nothing
end

function Base.unlock(gate::TreeStructure)
    gate.depth -= 1
    gate.depth == 0 && (@atomic :sequentially_consistent gate.pruning = false)
    unlock(gate.writer_lock)
    return nothing
end

function tree_inserting(f, gate::TreeStructure)
    while true
        while @atomic :sequentially_consistent gate.pruning
            yield()
        end
        @atomic :sequentially_consistent gate.inserting += 1
        if !(@atomic :sequentially_consistent gate.pruning)
            break
        end
        @atomic :sequentially_consistent gate.inserting -= 1
    end
    try
        return f()
    finally
        @atomic :sequentially_consistent gate.inserting -= 1
    end
end

mutable struct TreeRetention
    # Occupied slots plus in-flight generations in removable leaves; negative means detached.
    @atomic count::Int
end

mutable struct ConcurrentTreeLeaf{T,N,R} <: ConcurrentTreeNode{T,N}
    const prefix::NTuple{N,UInt}
    const values::GenericMemoryRef{:atomic,T,Core.CPU}
    const retention::R
end

struct ConcurrentLeafSlot{T,N,R}
    leaf::ConcurrentTreeLeaf{T,N,R}
    reference::GenericMemoryRef{:atomic,T,Core.CPU}
end

mutable struct ConcurrentTreeBranch{T,N} <: ConcurrentTreeNode{T,N}
    const prefix::NTuple{N,UInt}
    const level::Int
    const children::GenericMemoryRef{:atomic,ConcurrentTreeNode{T,N},Core.CPU}
    @atomic used::Int
    @atomic occupied::Int
end

struct ConcurrentRadixMap{T,N,BlockShape,D,PositiveCenter,L,R} <: SparseMap{T,N}
    roots::GenericMemoryRef{:atomic,ConcurrentTreeNode{T,N},Core.CPU}
    empty_node::L
    empty::Empty
    domain::D
    structural::TreeStructure
    registry::R
end

struct ConcurrentSubtreeRadixMap{T,N,BlockShape,D,Levels,PositiveCenter,L,R} <: SparseMap{T,N}
    roots::GenericMemoryRef{:atomic,ConcurrentTreeNode{T,N},Core.CPU}
    empty_node::L
    empty::Empty
    domain::D
    structural::TreeStructure
    zero_path::NTuple{Levels,Int}
    registry::R
end

const ConcurrentTreeMap{T,N,B} = Union{ConcurrentRadixMap{T,N,B},ConcurrentSubtreeRadixMap{T,N,B}}

@inline block_shape(::ConcurrentTreeMap{T,N,B}) where {T,N,B} = B
@inline tree_offset(::ConcurrentRadixMap{T,N,B,D,C}) where {T,N,B,D,C} = tree_offset(B, C)
@inline tree_offset(::ConcurrentSubtreeRadixMap{T,N,B,D,L,C}) where {T,N,B,D,L,C} = tree_offset(B, C)
@inline tree_prefix(store::Union{TreeMap,ConcurrentTreeMap}, key, level::Int) = tree_prefix(block_shape(store), key, level)
@inline tree_matches(store::Union{TreeMap,ConcurrentTreeMap}, key, prefix, level::Int) = tree_matches(block_shape(store), key, prefix, level)
@inline tree_level(store::Union{TreeMap,ConcurrentTreeMap}, key) = tree_level(block_shape(store), key)
@inline tree_address(store::Union{TreeMap,ConcurrentTreeMap}, key, level::Int) = tree_address(block_shape(store), key, level)

routing_offset(::Type{<:ConcurrentRadixMap{T,N,B,D,C}}) where {T,N,B,D,C} = tree_offset(B, C)
routing_offset(::Type{<:ConcurrentSubtreeRadixMap{T,N,B,D,L,C}}) where {T,N,B,D,L,C} = tree_offset(B, C)
@inline map_routing(store::ConcurrentTreeMap{T,N}, key) where {T,N} =
    ntuple(i -> reinterpret(UInt, key[i]) + tree_offset(store)[i], Val(N))
@inline cache_routing(cache, key, ::ConcurrentRadixMap) = encode_coordinates(key, cache_offset(cache))
@inline function cache_routing(cache, key, ::ConcurrentSubtreeRadixMap)
    encoded = encode_coordinates(key, cache_offset(cache))
    return encoded, root_index(cache.map, encoded)
end

@inline function node_level(node::ConcurrentTreeNode{T,N}) where {T,N}
    if node isa ConcurrentTreeBranch{T,N}
        return node.level
    end
    return 0
end
@inline function node_prefix(store::ConcurrentTreeMap{T,N}, node::ConcurrentTreeNode{T,N}) where {T,N}
    if node isa ConcurrentTreeBranch{T,N}
        return node.prefix
    end
    return (node::typeof(store.empty_node)).prefix
end
@inline root_index(::ConcurrentRadixMap, encoded) = 1
@inline root_index(store::ConcurrentSubtreeRadixMap{T,N}, encoded) where {T,N} =
    tree_level(store, ntuple(i -> encoded[i] ⊻ tree_offset(store)[i], Val(N))) + 1
@inline root_link(store::ConcurrentTreeMap) = @inbounds memoryref(store.roots, length(store.roots.mem))

Base.@constprop :aggressive function concurrent_tree_map(kind::Type{K}, ::Type{V}, domain::NTuple{N,Any}, config::Union{RadixMapConfig,SubtreeRadixMapConfig}; capacity::Int=0) where {K,V,N}
    block = config.block_shape
    block_length(block, 2)
    positive = positive_center(config.center)
    S = concurrent_value_type(V)
    if kind === MemoizedEntry
        T = Union{Empty,Generating,GenerationFailure,S}
        retention, registry = nothing, nothing
    else
        T = Union{Empty,Generating,GenerationFailure,GenerateWait{S},SharedLRUEntry{S,NTuple{N,Int}}}
        retention = TreeRetention(typemin(Int))
        registry = Memory{ConcurrentLeafSlot{T,N,TreeRetention}}(undef, capacity)
    end
    empty = ConcurrentTreeLeaf{T,N,typeof(retention)}(ntuple(_ -> UInt(0), Val(N)), memoryref(AtomicMemory{T}(undef, 0)), retention)
    if config isa RadixMapConfig
        roots = memoryref(empty_map(ConcurrentTreeNode{T,N}, (1,), empty, Val(true), Val(:nocheck)))
        return ConcurrentRadixMap{T,N,block,typeof(domain),positive,typeof(empty),typeof(registry)}(roots, empty, EMPTY, domain, TreeStructure(), registry)
    end
    if any(axis -> !(first(axis) <= 0 <= last(axis)), domain)
        throw(ArgumentError("SubtreeRadixMap requires every domain axis to contain zero"))
    end
    levels = isempty(block) ? 1 : cld(8sizeof(UInt), trailing_zeros(minimum(block)))
    zero = tree_offset(block, positive)
    zero_path = ntuple(i -> tree_address(block, zero, i - 1), Val(levels))
    roots = memoryref(empty_map(ConcurrentTreeNode{T,N}, (levels,), empty, Val(true), Val(:nocheck)))
    store = ConcurrentSubtreeRadixMap{T,N,block,typeof(domain),levels,positive,typeof(empty),typeof(registry)}(
        roots, empty, EMPTY, domain, TreeStructure(), zero_path, registry)
    central = tree_leaf(store, zero)
    for index in eachindex(roots.mem)
        slot_publish!(@inbounds(memoryref(roots, index)), central)
    end
    return store
end

function tree_leaf(store::ConcurrentTreeMap{T,N}, encoded) where {T,N}
    # Boxed atomic slots occupy one pointer each. Keep the allocation as the
    # MemoryRef owner and select its first cache-line-aligned slot.
    count = prod(block_shape(store))
    width = sizeof(Ptr{Cvoid})
    values = empty_map(T, (count + 64 ÷ width - 1,), EMPTY, Val(true), Val(:nocheck))
    reference = GC.@preserve values begin
        offset = Int((-UInt(pointer(values))) & UInt(63)) ÷ width
        memoryref(values, offset + 1)
    end
    retention = store.registry === nothing ? nothing : TreeRetention(0)
    return ConcurrentTreeLeaf{T,N,typeof(retention)}(tree_prefix(store, encoded, 0), reference, retention)
end

function tree_branch(store::ConcurrentTreeMap{T,N}, encoded, level::Int) where {T,N}
    children = empty_map(ConcurrentTreeNode{T,N}, block_shape(store), store.empty_node, Val(true), Val(:nocheck))
    return ConcurrentTreeBranch{T,N}(tree_prefix(store, encoded, level), level, memoryref(children), 0, 0)
end

function tree_add_child!(branch::ConcurrentTreeBranch, address::Int, child)
    slot_publish!(@inbounds(memoryref(branch.children, address)), child)
    @atomic :monotonic branch.used = (@atomic :monotonic branch.used) + 1
    @atomic :monotonic branch.occupied = (@atomic :monotonic branch.occupied) ⊻ address
    return nothing
end

Base.@propagate_inbounds function map_lookup(cache::AbstractMemoizedArray{V,N,M}, key::NTuple{N,Int}) where {V,N,M<:ConcurrentTreeMap}
    @boundscheck checkbounds(cache, key...)
    routing = cache_routing(cache, key)
    if cache.map.registry === nothing
        return key, routing, map_slot(cache.map, key, routing), nothing
    end
    location = tree_lookup_location(cache.map, routing)
    return location === nothing ? (key, routing, nothing, nothing) :
        (key, routing, location.reference, location.leaf)
end

@inline map_slot(store::ConcurrentTreeMap, key, encoded) = tree_find(store, encoded, root_index(store, encoded))
@inline map_slot(store::ConcurrentSubtreeRadixMap{T,N}, key, route::Tuple{NTuple{N,UInt},Int}) where {T,N} =
    tree_find(store, route[1], route[2])

@inline tree_result(leaf, address, ::Val{false}) = @inbounds memoryref(leaf.values, address)
@inline tree_result(leaf, address, ::Val{true}) = ConcurrentLeafSlot(leaf, @inbounds memoryref(leaf.values, address))
@inline tree_lookup_location(store::ConcurrentTreeMap, encoded) = tree_find(store, encoded, root_index(store, encoded), Val(true))
@inline tree_lookup_location(store::ConcurrentSubtreeRadixMap{T,N}, route::Tuple{NTuple{N,UInt},Int}) where {T,N} =
    tree_find(store, route[1], route[2], Val(true))

@inline function tree_find(store::ConcurrentTreeMap{T,N}, encoded, index::Int, location::Val=Val(false)) where {T,N}
    node = slot_entry(@inbounds memoryref(store.roots, index))
    if store isa ConcurrentSubtreeRadixMap && index == 1
        return tree_result(node::typeof(store.empty_node), tree_address(store, encoded, 0), location)
    end
    expectedlevel = store isa ConcurrentSubtreeRadixMap ? max(index - 1, node_level(node)) :
        tree_level(store, ntuple(_ -> typemax(UInt), Val(N)))
    while node !== store.empty_node
        if node isa typeof(store.empty_node)
            if expectedlevel != 0 && !tree_matches(store, encoded, node.prefix, 0)
                return nothing
            end
            return tree_result(node, tree_address(store, encoded, 0), location)
        end
        branch = node::ConcurrentTreeBranch{T,N}
        if branch.level != expectedlevel && !tree_matches(store, encoded, branch.prefix, branch.level)
            return nothing
        end
        address = tree_address(store, encoded, branch.level)
        expectedlevel = branch.level - 1
        node = slot_entry(@inbounds memoryref(branch.children, address))
    end
    return nothing
end

refresh!(::ConcurrentRadixMap, depth, ::Val=Val(false)) = nothing
# Refresh under the structural lock. Stale shortcuts remain valid subtrees;
# readers that miss through them re-resolve from the owning root before insertion.
function refresh!(store::ConcurrentSubtreeRadixMap{T,N}, depth::Int, ::Val{Full}=Val(false)) where {T,N,Full}
    zero = tree_offset(store)
    ancestor = slot_entry(@inbounds memoryref(store.roots, depth + 2))
    node = ancestor
    for depth in depth:-1:0
        while node isa ConcurrentTreeBranch{T,N} && node.level > depth
            child = slot_entry(@inbounds memoryref(node.children, store.zero_path[node.level + 1]))
            if child === store.empty_node || node_level(child) < depth ||
                    !tree_matches(store, zero, node_prefix(store, child), node_level(child))
                break
            end
            node = child
        end
        level = max(depth, node_level(node))
        alias = node === store.empty_node || tree_prefix(store, node_prefix(store, node), level) != tree_prefix(store, zero, level) ? store.empty_node : node
        slot = @inbounds memoryref(store.roots, depth + 1)
        if slot_entry(slot) === alias
            if !Full && node !== ancestor
                break
            end
        else
            slot_publish!(slot, alias)
        end
    end
    return nothing
end

@inline has_shortcut(::ConcurrentRadixMap, branch) = false
@inline function has_shortcut(store::ConcurrentSubtreeRadixMap, branch)
    depth = max(branch.level + 1, root_index(store, branch.prefix))
    return slot_entry(@inbounds memoryref(store.roots, depth)) === branch
end
@inline zero_child(::ConcurrentRadixMap, branch) = 1
@inline zero_child(store::ConcurrentSubtreeRadixMap, branch) = @inbounds store.zero_path[branch.level + 1]

@inline map_reserve!(store::ConcurrentSubtreeRadixMap{T,N}, key, route::Tuple{NTuple{N,UInt},Int}, location::Val=Val(false)) where {T,N} =
    map_reserve!(store, key, route[1], location)

@noinline function map_reserve!(store::ConcurrentTreeMap{T,N}, key, encoded, location::Val=Val(false)) where {T,N}
    leaf = tree_leaf(store, encoded)
    slot, refresh = if store.registry === nothing
        tree_cas_insert!(store, encoded, leaf, location)
    else
        tree_inserting(store.structural) do
            tree_cas_insert!(store, encoded, leaf, location)
        end
    end
    if refresh
        lock(store.structural) do
            refresh!(store, length(store.roots.mem) - 2, Val(true))
        end
    end
    return slot
end

# Memoized never detaches nodes; LRU's shared insertion phase excludes pruning.
# A retained child reference stays reachable when another caller inserts an
# ancestor. Only the contested child slot needs a CAS.
function tree_cas_insert!(store::ConcurrentTreeMap{T,N}, encoded, leaf, location) where {T,N}
    link, parent, address = root_link(store), nothing, 1
    expectedlevel = tree_level(store, ntuple(_ -> typemax(UInt), Val(N)))
    while true
        node = slot_entry(link)
        if node === store.empty_node
            replacement = leaf
        elseif node_level(node) != expectedlevel &&
                !tree_matches(store, encoded, node_prefix(store, node), node_level(node))
            prefix = node_prefix(store, node)
            level = tree_level(store, ntuple(i -> prefix[i] ⊻ encoded[i], Val(N)))
            replacement = tree_branch(store, encoded, level)
            tree_add_child!(replacement, tree_address(store, prefix, level), node)
            tree_add_child!(replacement, tree_address(store, encoded, level), leaf)
        elseif node isa typeof(store.empty_node)
            return tree_result(node, tree_address(store, encoded, 0), location), false
        else
            parent = node::ConcurrentTreeBranch{T,N}
            address = tree_address(store, encoded, parent.level)
            expectedlevel = parent.level - 1
            link = @inbounds memoryref(parent.children, address)
            continue
        end
        claim = claim_slot!(link, node, replacement)
        if claim.success
            if store.registry !== nothing && parent !== nothing && node === store.empty_node
                @atomic :monotonic parent.used += 1
                @atomic :monotonic parent.occupied ⊻= address
            end
            refresh = store isa ConcurrentSubtreeRadixMap &&
                (parent === nothing || (address == zero_child(store, parent) && has_shortcut(store, parent)))
            return tree_result(leaf, tree_address(store, encoded, 0), location), refresh
        end
    end
end

# Cold LRU operations keep only the leaf and value ref. The cache already
# supplies the map and key when a retired location needs to be followed.
@noinline function relocated_slot(store::ConcurrentTreeMap, key)
    # Starting at the owning root in the shared insertion phase avoids a shortcut
    # that still points to a leaf retired by another caller.
    return map_reserve!(store, key, map_routing(store, key), Val(true))
end

# Retired leaves contain only EMPTY; the next claim follows their replacement.
@inline slot_entry(location::ConcurrentLeafSlot) = slot_entry(location.reference)

@inline permanent_leaf(::ConcurrentRadixMap, leaf) = false
@inline permanent_leaf(store::ConcurrentSubtreeRadixMap, leaf) =
    leaf.prefix == tree_prefix(store, tree_offset(store), 0)

@inline function reserve_leaf!(leaf::ConcurrentTreeLeaf{T,N,TreeRetention}) where {T,N}
    # The negative range keeps a detached leaf detached while concurrent failed
    # reservations undo their increments. A live zero count becomes positive.
    count = @atomic :acquire_release leaf.retention.count += 1
    count > 0 && return true
    @atomic :acquire_release leaf.retention.count -= 1
    return false
end

@inline function release_leaf!(store, leaf)
    permanent_leaf(store, leaf) && return nothing
    remaining = @atomic :acquire_release leaf.retention.count -= 1
    remaining == 0 && retire_leaf!(store, leaf)
    return nothing
end

@inline function claim_slot!(store::ConcurrentTreeMap, key, slot::ConcurrentLeafSlot, expected, desired)
    while true
        if expected === EMPTY
            if !permanent_leaf(store, slot.leaf) && !reserve_leaf!(slot.leaf)
                slot = relocated_slot(store, key)
                continue
            end
            claimed = claim_slot!(slot.reference, expected, desired)
            claimed.success || release_leaf!(store, slot.leaf)
        elseif !permanent_leaf(store, slot.leaf) && (@atomic :acquire slot.leaf.retention.count) < 0
            slot = relocated_slot(store, key)
            continue
        else
            claimed = claim_slot!(slot.reference, expected, desired)
        end
        return (old=claimed.old, success=claimed.success, location=slot)
    end
end

@inline claimed_slot(::ConcurrentLeafSlot, claim) = claim.location

# Ownership pins the actual claimed leaf through success or retained failure.
# Keep that location across generation. Entry state stays exclusive until publication.
@inline function managed_publish!(store::ConcurrentTreeMap, location::ConcurrentLeafSlot, stored, outcome)
    if stored isa SharedLRUEntry
        @inbounds store.registry[stored.id] = location
    end
    return managed_publish!(location.reference, stored, outcome)
end

# Entry ownership already excludes another eviction of this mapping. A new
# generation can reserve this leaf immediately after the slot becomes empty.
function remove_mapping!(store::ConcurrentTreeMap, entry)
    location = @inbounds store.registry[entry.id]
    slot_publish!(location.reference, EMPTY)
    release_leaf!(store, location.leaf)
    return nothing
end

@noinline function retire_leaf!(store::ConcurrentTreeMap, leaf)
    permanent_leaf(store, leaf) && return nothing
    lock(store.structural) do
        claimed = @atomicreplace :acquire_release :acquire leaf.retention.count 0 => typemin(Int)
        claimed.success || return nothing
        detach_leaf!(store, root_link(store), leaf)
    end
    return nothing
end

# Structural removal walks only the leaf's own path. Collapsed branches remain
# valid for readers retaining old references; their removed child is already empty.
function detach_leaf!(store::ConcurrentTreeMap{T,N}, link, leaf) where {T,N}
    node = slot_entry(link)
    depth = length(store.roots.mem) - 2
    if node === leaf
        slot_publish!(link, store.empty_node)
        refresh!(store, depth, Val(true))
        return nothing
    end
    while true
        branch = node::ConcurrentTreeBranch{T,N}
        address = tree_address(store, leaf.prefix, branch.level)
        childlink = @inbounds memoryref(branch.children, address)
        child = slot_entry(childlink)
        if child === leaf
            shortcut = has_shortcut(store, branch)
            slot_publish!(childlink, store.empty_node)
            remaining = (@atomic :monotonic branch.used) - 1
            occupied = (@atomic :monotonic branch.occupied) ⊻ address
            @atomic :monotonic branch.used = remaining
            @atomic :monotonic branch.occupied = occupied
            if remaining == 1
                survivor = slot_entry(@inbounds memoryref(branch.children, occupied))
                slot_publish!(link, survivor)
                shortcut && refresh!(store, depth, Val(true))
            elseif shortcut && address == zero_child(store, branch)
                refresh!(store, branch.level - 1, Val(true))
            end
            return nothing
        end
        link, node, depth = childlink, child, branch.level - 1
    end
end

queue_key(::ConcurrentTreeMap{T,N}) where {T,N} = ntuple(_ -> 0, Val(N))


function assignment_slot(map::ConcurrentTreeMap,key,routing,reference,leaf)
    reference === nothing && return map_reserve!(map,key,routing,Val(true))
    return ConcurrentLeafSlot(leaf,reference)
end
