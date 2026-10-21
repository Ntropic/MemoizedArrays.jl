# Shared generation states, value encoding and cached-reference waiting.

struct Generating end
const GENERATING = Generating()
struct GenerationFailure
    error::Any
end

struct GenerateWait{S}
    reference::GenericMemoryRef{:atomic,Union{Generating,GenerationFailure,S},Core.CPU}
end
function GenerateWait{S}() where S
    slots = AtomicMemory{Union{Generating,GenerationFailure,S}}(undef, 1)
    @atomic :monotonic slots[1] = GENERATING
    return GenerateWait{S}(memoryref(slots))
end

# Keep common concrete values direct. Broad V (including Any) needs a wrapper
# so generated values can never be confused with internal states.
@inline function concurrent_value_type(::Type{V}) where V
    markers = Union{Empty,Generating,GenerationFailure,GenerateWait,Relocated}
    return isconcretetype(V) && typeintersect(V, markers) === Union{} ? V : MemoizedEntry{V}
end
@inline encode_value(::Type{V}, value) where V = concurrent_value_type(V) === V ? value : MemoizedEntry{V}(value)
@inline decode_value(::Type{V}, stored) where V = concurrent_value_type(V) === V ? stored::V : stored.value::V

@inline function poll_step(polls)
    GC.safepoint()
    polls += 1
    if polls == 1024
        yield()
        return 0
    end
    return polls
end

# Immediate polling uses one cached reference. Yield occasionally so a paused
# owner can resume even on one Julia thread; no timer or sleeping interval is used.
@noinline function wait_generated(::Type{V}, ref) where V
    stored = slot_entry(ref)
    polls = 0
    while stored === GENERATING
        polls = poll_step(polls)
        stored = slot_entry(ref)
    end
    stored isa GenerationFailure && throw(stored.error)
    return decode_value(V, stored)
end
