@testset "Completion notification preserves entries and releases bookkeeping" begin
    for (constructor, stamp, succeeds) in ((ExactLRUArray, nothing, true),
        (SampledLRUArray, nothing, true), (SampledLRUArray, 0, false))
        started, release = Channel{Nothing}(1), Channel{Nothing}(1)
        cache = constructor{Int}((2,), 1, i -> begin
            if i == 1
                put!(started, nothing)
                take!(release)
            end
            return i
        end)
        # An unsupported stamp fails sampled insertion before its bookkeeping changes.
        owner = @async try
            JITArrays._cache_get(cache, 1, stamp)
        catch exception
            exception
        end
        entered = timedwait(() -> isready(started) || istaskdone(owner), 5)
        @test entered == :ok
        @test isready(started)
        if !isready(started)
            continue
        end
        take!(started)
        entry = JITArrays._entry(cache, 1)
        follower = @async try
            cache[1]
        catch exception
            exception
        end
        registered = timedwait(() -> (@atomic :acquire entry.state) isa Threads.Condition, 5)
        @test registered == :ok
        if registered != :ok
            put!(release, nothing)
            continue
        end
        condition = @atomic :acquire entry.state
        lock(condition)
        try
            put!(release, nothing)
            published = timedwait(() -> begin
                state = @atomic :acquire entry.state
                return succeeds ? state === true : state === JITArrays._EMPTY
            end, 5)
            @test published == :ok
            @test !islocked(cache.available)
            if published == :ok && !islocked(cache.available)
                @test cache[2] == 2
                if succeeds
                    # Interrupt the old owner after eviction while notification is blocked.
                    schedule(owner, InterruptException(); error=true)
                    completed = timedwait(() -> istaskdone(owner), 5)
                    @test completed == :ok
                    if completed == :ok
                        @test fetch(owner) isa InterruptException
                    end
                    @test (@atomic :acquire entry.state) === true
                    @test entry.value == 1
                else
                    @test (@atomic :acquire entry.state) === JITArrays._EMPTY
                end
                @test isready(cache, 2)
                if cache isa SampledLRUArray
                    @test cache.ready == 1 && cache.pending == 0
                else
                    @test cache.count == 1
                end
            end
        finally
            unlock(condition)
            JITArrays._notify_waiters(condition)
        end
        completed = timedwait(() -> all(istaskdone, (owner, follower)), 5)
        @test completed == :ok
        if completed == :ok
            if succeeds
                @test fetch(follower) == 1
            else
                @test fetch(owner) isa MethodError
                @test fetch(follower) isa ErrorException
            end
        end
    end
end
