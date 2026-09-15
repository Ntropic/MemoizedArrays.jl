"""`CoarseClock()` uses the operating system's approximate monotonic clock."""
struct CoarseClock end

"""`NanosecondClock()` uses Julia's `time_ns()`."""
struct NanosecondClock end

"""`LossyCounter()` counts accesses without synchronization, allowing concurrent increments to be lost."""
mutable struct LossyCounter
    value::UInt64
    LossyCounter() = new(UInt64(0))
end

const _CacheClock = Union{CoarseClock,NanosecondClock,LossyCounter}

# Resolve symbols once, when constructing the cache.
_cache_clock(clock::_CacheClock) = clock
function _cache_clock(clock::Symbol)
    if clock === :coarse
        return CoarseClock()
    elseif clock === :lossy
        return LossyCounter()
    elseif clock === :nanosecond
        return NanosecondClock()
    end
    throw(ArgumentError("unknown clock $clock; use :coarse, :lossy, or :nanosecond"))
end

# Read the current stamp without advancing an access counter.
@inline _clock_now(::NanosecondClock) = time_ns()
@inline _clock_now(clock::LossyCounter) = clock.value

# Select the native implementation during compilation; no OS branch on each read.
@static if Sys.islinux() && Sys.WORD_SIZE == 64
    @inline function _clock_now(::CoarseClock)
        stamp = Ref{NTuple{2,Clong}}()
        result = ccall(:clock_gettime, Cint, (Cint, Ref{NTuple{2,Clong}}), 6, stamp) # CLOCK_MONOTONIC_COARSE
        if result != 0
            systemerror("clock_gettime", true)
        end
        seconds, nanoseconds = stamp[]
        return UInt64(seconds) * UInt64(1_000_000_000) + UInt64(nanoseconds)
    end
elseif Sys.isapple()
    @inline _clock_now(::CoarseClock) = ccall(:mach_approximate_time, UInt64, ())
elseif Sys.iswindows()
    @inline _clock_now(::CoarseClock) = ccall((:GetTickCount64, "kernel32"), stdcall, UInt64, ())
else
    @inline _clock_now(::CoarseClock) = time_ns()
end

# Clock reads have no local state to update.
@inline _clock_tick!(clock::Union{CoarseClock,NanosecondClock}, count::Int=1) = _clock_now(clock)

# Unsynchronized increments intentionally allow lost updates.
@inline function _clock_tick!(clock::LossyCounter, count::Int=1)
    return clock.value += UInt64(count)
end
