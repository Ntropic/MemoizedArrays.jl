# Caches

```@meta
CurrentModule = MemoizedArrays
```

Choose retention and opt into concurrency. See [Examples](../examples.md) for use.

## Constructors

```@docs
MemoizedArray
LRUMemoizedArray
```

## Queue handles

Concurrent LRU access uses one active caller per queue. See the
[concurrent LRU example](../examples.md#Concurrent-LRUMemoizedArray).

```@docs
queue_handle
queue_handles
with_queue
```

Queues reference shared entries. On eviction, a queue removes the map entry only if its
last-access hint still names that queue; otherwise it drops only its own reference.
