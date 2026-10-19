struct MultiQueue{Q,L}
    queues::Q
    claims::L
end
MultiQueue(queues::Q) where Q = MultiQueue{Q,Vector{ReentrantLock}}(
    queues, [ReentrantLock() for _ in eachindex(queues)])
