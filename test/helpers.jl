test_map_config(kind, shape=(8,); center=0.5) =
    kind === :array ? ((V, domain) -> ArrayMap{V}(domain)) : kind === :dict ? ((V, domain) -> DictMap{V}(domain)) :
    kind === :radix ? ((V, domain) -> RadixMap{V}(domain; block_shape=shape, center)) : ((V, domain) -> SubtreeRadixMap{V}(domain; block_shape=shape, center))

function readyallocations(cache)
    cache[ntuple(_ -> 1, ndims(cache))...]
    return @allocated cache[ntuple(_ -> 1, ndims(cache))...]
end

request_allocations(cache, key) = @allocated cache[key]

function test_result(f)
    try
        return f()
    catch error
        return error
    end
end

function concurrent_test_cache(C, V, generator, domain; capacity=8, queues=8, map_factory=((V, domain) -> ArrayMap{V}(domain)))
    if C === MemoizedArray
        return C(generator, map_factory(V, domain); concurrent=true)
    end
    return C(generator, map_factory(V, domain); capacity, queues, concurrent=true)
end

concurrent_test_handle(cache::MemoizedArray, _) = cache
concurrent_test_handle(cache::LRUMemoizedArray, i) = queue_handle(cache, i)

function concurrent_test_readallocations(cache)
    cache[1]
    return @allocated cache[1]
end

# Keep allocation probes identical across map implementations.
request_allocations(cache, key::Tuple) = @allocated cache[key...]
function read_allocations(cache, key::Tuple)
    cache[key...]
    return @allocated cache[key...]
end

struct TestConversion{F}
    convert::F
end
Base.convert(::Type{Int}, value::TestConversion) = value.convert()
