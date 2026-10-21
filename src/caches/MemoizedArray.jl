# Cache without eviction, entries, constructors and serial/concurrent access.


struct MemoizedEntry{V}
    value::V
end

# Mutable identity distinguishes the shared empty entry from every generated value.
mutable struct ArrayEntry{V}
    const value::V
    ArrayEntry{V}() where V = new{V}()
    ArrayEntry{V}(value) where V = new{V}(value)
end

"""
    MemoizedArray(generator, map; concurrent=false)

Cache values on demand without eviction. The map specifies the value type, domain and lookup strategy.
Set `concurrent=true` to coordinate shared access.
"""
struct MemoizedArray{V,N,M,D,G,Concurrent} <: AbstractMemoizedArray{V,N,M}
    map::M
    domain::D
    generator::G
end

Base.@constprop :aggressive function MemoizedArray(generator::G, config::MapConfig; concurrent::Bool=false) where G
    return MemoizedArray(generator, config, Val(concurrent))
end

Base.@constprop :aggressive function MemoizedArray(generator::G, config::MapConfig{V}, ::Val{Concurrent}) where {G,V,Concurrent}
    map, domain = make_cache_map(MemoizedEntry, V, config; concurrent=Concurrent)
    return MemoizedArray{V,length(domain),typeof(map),typeof(domain),G,Concurrent}(map, domain, generator)
end



@inline entrytype(cache::MemoizedArray{V}) where V = cache.map.empty === EMPTY ? MemoizedEntry{V} : typeof(cache.map.empty)

# Pass encoded coordinates as scalars; recover generator coordinates only on a miss.
# Inline generation into an existing leaf; keep structural misses out of the read path.
@noinline function treemiss!(cache::MemoizedArray{V,N,M,D,G}, leaf, parent, address,
    encoded::Vararg{UInt,N}) where {V,N,M,D,G}
    map = cache.map
    slot = if leaf === nothing
        link = parent === nothing ? root_link(map) : map_slot(parent.children, address, nothing)
        if map isa RadixMap || parent !== nothing
            # Lookup already proved this link empty or its compressed prefix different.
            tree_reserve!(map, encoded, link, parent, address, Val(true))
        else
            # A parentless subtree hint can alias only part of the owning tree.
            tree_reserve!(map, encoded, link, parent, address)
        end
    else
        LeafSlot(leaf, address)
    end
    offset = routing_offset(typeof(map))
    key = ntuple(i -> reinterpret(Int, encoded[i] - offset[i]), Val(N))
    stored = entrytype(cache)(cache.generator(key...))
    # Nested generation or conversion can fill this same retained slot.
    if slot_entry(slot) === map.empty
        map_retain!(map, slot)
    end
    slot_publish!(slot, stored)
    return stored.value
end

@inline function cache_get(cache::MemoizedArray{V,N,M}, lookup, indices::Tuple) where {V,N,M<:ConcurrentTreeMap}
    key, routing, ref, _ = lookup
    if ref !== nothing
        stored = slot_entry(ref)
        if stored isa concurrent_value_type(V)
            return decode_value(V, stored)
        end
    end
    return concurrent_tree_miss!(cache, ref, indices...)
end

# Passing memory and offset keeps the cold call from boxing the hot slot reference.
@inline concurrent_tree_miss!(cache, ref, indices...) = concurrent_tree_miss!(cache,
    ref === nothing ? nothing : ref.mem, ref === nothing ? 0 : Core.memoryrefoffset(ref), indices...)

@noinline function concurrent_tree_miss!(cache::MemoizedArray{V,N}, mem, address::Int, indices::Vararg{Int,N}) where {V,N}
    ref = mem === nothing ? nothing : memoryref(mem, address)
    if ref === nothing
        ref = map_reserve!(cache.map, indices, cache_routing(cache, indices))
    end
    return concurrent_generate!(cache, ref, indices)
end

@inline function cache_get(cache::MemoizedArray{V,N,M}, lookup, indices::Tuple) where {V,N,M<:ConcurrentDictMap}
    key, hashed, record, _ = lookup
    if record !== nothing
        stored = slot_entry(record)
        if stored isa concurrent_value_type(V)
            return decode_value(V, stored)
        end
    end
    return concurrent_dict_miss!(cache, key, hashed, record, indices)
end

@noinline function concurrent_dict_miss!(cache, key, hashed, record, indices)
    if record === nothing
        record = map_reserve!(cache.map, key, hashed)
    end
    return concurrent_generate!(cache, record, indices)
end

@inline function cache_get(cache::MemoizedArray{V,N,M}, lookup, indices::Tuple) where {V,N,M<:ConcurrentArrayMap}
    key, _, ref, _ = lookup
    stored = slot_entry(ref)
    stored isa concurrent_value_type(V) && return decode_value(V, stored)
    return concurrent_miss!(cache, key, indices)
end

@noinline function concurrent_miss!(cache::MemoizedArray{V}, key, indices) where V
    ref = map_slot(cache.map, key, nothing)
    return concurrent_generate!(cache, ref, indices)
end

@inline function concurrent_generate!(cache::MemoizedArray{V}, ref, indices) where V
    stored = slot_entry(ref)
    if stored === EMPTY
        claimed = claim_slot!(ref, EMPTY, GENERATING)
        if claimed.success
            value = try
                result = cache.generator(indices...)
                result isa V ? result : convert(V, result)
            catch error
                slot_publish!(ref, GenerationFailure(error))
                rethrow()
            end
            slot_publish!(ref, encode_value(V, value))
            return value
        end
        stored = claimed.old
    end
    stored isa GenerationFailure && throw(stored.error)
    stored === GENERATING && return wait_generated(V, ref)
    return decode_value(V, stored)
end


@inline ready_entry(cache::MemoizedArray{V,N,M,D,G,true}, stored) where {V,N,M,D,G} = stored isa concurrent_value_type(V)

@inline function ready_value(cache::MemoizedArray{V,N,M}, stored) where {V,N,M<:Union{ConcurrentArrayMap,ConcurrentExpandingArrayMap}}
    stored isa GenerationFailure && throw(stored.error)
    stored isa concurrent_value_type(V) || throw(UndefRefError())
    return decode_value(V, stored)
end

@inline function cache_get(cache::MemoizedArray{V,N,M}, lookup, indices::Tuple) where {V,N,M<:ConcurrentExpandingArrayMap}
    key, allocation, ref, _ = lookup
    if ref !== nothing
        stored = slot_entry(ref)
        if stored isa concurrent_value_type(V)
            return decode_value(V, stored)
        end
    end
    return expanding_miss!(cache, key, allocation, ref, indices)
end

@inline expanding_miss!(cache, key, allocation, ref, indices) = expanding_miss!(cache, key, allocation,
    ref === nothing ? nothing : ref.mem, ref === nothing ? 0 : Core.memoryrefoffset(ref), indices)

@noinline function expanding_miss!(cache::MemoizedArray{V}, key, allocation, mem, address::Int, indices) where V
    ref = mem === nothing ? nothing : memoryref(mem, address)
    map = cache.map
    if ref === nothing
        allocation = grow_allocation!(map, key)
        ref = allocation_slot(allocation, key)
    end
    polls = 0
    while true
        stored = slot_entry(ref)
        if stored === EMPTY
            claimed = claim_slot!(ref, EMPTY, GENERATING)
            if claimed.success
                value, encoded = try
                    result = cache.generator(indices...)
                    value = result isa V ? result : convert(V, result)
                    value, encode_value(V, value)
                catch error
                    publish_generated!(map, key, allocation, ref, GenerationFailure(error))
                    rethrow()
                end
                publish_generated!(map, key, allocation, ref, encoded)
                return value
            end
            stored = claimed.old
        end
        if stored === RELOCATED
            allocation, ref = relocated_slot(map, allocation, key)
        elseif stored === GENERATING
            polls = poll_step(polls)
        elseif stored isa GenerationFailure
            throw(stored.error)
        else
            return decode_value(V, stored)
        end
    end
end

"""`getready(cache, indices...)` reads an already populated ArrayMap-backed MemoizedArray without generating."""
Base.@propagate_inbounds function getready(cache::MemoizedArray{V,N,M}, indices::Vararg{Int,N}) where {V,N,M<:DenseMap}
    return ready_value(cache, slot_entry(map_slot(cache.map, cache_key(cache, indices), nothing)))::V
end

function map_entry(::Type{MemoizedEntry}, ::Type{V}, Key, config) where V
    if config isa DictMapConfig
        # Absence lives in the table; serial Memoized records are created completed.
        return MemoizedEntry{V}, EMPTY
    end
    return ArrayEntry{V}, ArrayEntry{V}()
end

@inline function cache_get(cache::MemoizedArray{V,N,M,D,G,false}, lookup, indices::Tuple) where {V,N,M,D,G}
    key, routing, slot, hint = lookup
    if slot !== nothing
        stored = if cache.map isa SubtreeRadixMap
            slot_entry(map_slot(slot.values, hint[3], nothing), cache.map)
        else
            prepared_entry(slot, hint)
        end
        if stored !== cache.map.empty
            if cache.map isa SubtreeRadixMap
                # Only the sentinel has an uninitialized value field.
                assume_initialized(isdefined(stored, :value))
            end
            return stored.value
        end
        if cache.map isa TreeMap
            encoded = cache.map isa SubtreeRadixMap ? routing[1] : routing
            return @inline treemiss!(cache, slot, nothing, hint[3], encoded...)
        end
    end
    if cache.map isa TreeMap
        encoded = cache.map isa SubtreeRadixMap ? routing[1] : routing
        return treemiss!(cache, slot, hint[2], hint[3], encoded...)
    end
    stored = entrytype(cache)(cache.generator(indices...))
    slot = if cache.map isa DictMap
        # Store the completed payload inline in its stable record.
        reserve!(cache.map, key, routing, slot, hint, stored)
    else
        reserve!(cache.map, key, routing, slot, hint)
    end
    map_retain!(cache.map, slot)
    slot_publish!(slot, stored)
    return stored.value
end

# Serial assignment publishes a completed value through the map's ordinary
# reservation path. It is useful for seeding an acyclic recursive generator.
@inline function cache_set!(cache::MemoizedArray{V,N,M,D,G,false}, value, lookup) where {V,N,M,D,G}
    value = value isa V ? value : (convert(V, value)::V)
    stored = entrytype(cache)(value)
    key, routing, slot, hint = lookup
    map = cache.map
    if map isa DictMap
        slot = reserve!(map, key, routing, slot, hint, stored)
    else
        slot = reserve!(map, key, routing, slot, hint)
        if slot_entry(slot) === map.empty
            map_retain!(map, slot)
        end
    end
    slot_publish!(slot, stored)
    return value
end

Base.@propagate_inbounds function Base.setindex!(cache::MemoizedArray{V,N,M,D,G,false}, value,
    indices::Vararg{Int,N}) where {V,N,M,D,G}
    return cache_set!(cache, value, map_lookup(cache, indices))
end

Base.@propagate_inbounds Base.setindex!(cache::MemoizedArray, value, key::CartesianIndex) =
    setindex!(cache, value, Tuple(key)...)
