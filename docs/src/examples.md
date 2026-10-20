# Examples

```@meta
CurrentModule = MemoizedArrays
```


## MemoizedArray
In the simplest case we define a  generate a conventional array via `ArrayMap`
```@example basic
using MemoizedArrays # hide
generator(i, j) = i + j;
values = MemoizedArray(generator, ArrayMap{Int}((100, 100)));
values[2, 3]
```
`MemoizedArray` stores any generated values - it does not evict old and unused values. Use `ArrayMap` for a fixed finite domain.

The generator can be selfreferential, requesting values of the array it constructs, by using the `do` syntax 

```@example recursive
using MemoizedArrays # hide
values = MemoizedArray(ArrayMap{Int}((8,))) do i
    values[i - 1] + 1
end;
values[1] = 1;
values[8]
```

Dependencies must be acyclic: a generator cannot request its own unfinished value.

## LRUMemoizedArray

```@example lru
using MemoizedArrays # hide
values = LRUMemoizedArray(identity, ArrayMap{Int}((10,)); capacity=2);
values[1], values[2], values[1], values[3]
```

```@example lru
isready(values, 2)
```

`LRUMemoizedArray` retains at most `capacity` values and regenerates evicted ones.

## ExpandingArrayMap

![An expanding array map copies existing values into a larger grid and generates only the requested new value](assets/expanding-array-map.svg)

```@example expanding
using MemoizedArrays # hide
values = MemoizedArray(+, ExpandingArrayMap{Int}((3, 3); increment=2));
values[5, 2]
```

```@example expanding
size(values)
```

Growth applies only above the initial upper bounds. `increment` accepts one integer or one
integer per axis, such as `(2, 4)`.

## RadixMap

![A radix lookup follows selected slots through branch arrays into a leaf array, with dots indicating further levels above](assets/radix-slots.svg)

```@example radix
using MemoizedArrays # hide
values = MemoizedArray(tuple, RadixMap{Tuple{Int,Int}}(unbound(2); block_shape=(8, 8)));
values[-200, 300]
```

Use `SubtreeRadixMap` for a tree map whose requests cluster near zero:

```@example subtree
using MemoizedArrays # hide
values = MemoizedArray(tuple, SubtreeRadixMap{Tuple{Int,Int}}(unbound(2); block_shape=(8, 8)));
values[0, -1]
```

## DictMap

```@example dict
using MemoizedArrays # hide
values = LRUMemoizedArray(identity, DictMap{Int}(unbound(1)); capacity=512);
values[-20]
```

## Indexing, views and generation

```@example indexing
using MemoizedArrays # hide
values = MemoizedArray(tuple, ArrayMap{Tuple{Int,Int}}((-3:3, 0:4)));
view(values, -2:2, 0:3)[1, 1]
```

```@example indexing
generate!(values, ((-3, 4), (3, 0))) === values
```

Slices return ordinary arrays. A view generates only when it is read.

## Concurrent MemoizedArray

Start Julia with, for example, `julia --threads=8`.

```julia
values = MemoizedArray(i -> fill(Float64(i), 4, 4),
                  ArrayMap{Matrix{Float64}}((1000,)); concurrent=true)
tasks = [Threads.@spawn values[7] for _ in 1:4]
fetch.(tasks)
```

Concurrent requests for the same missing coordinate generate it once; other callers wait.

## Concurrent LRUMemoizedArray

The entry stores both the value and a `last_access` hint. Queues hold references to it.

```@raw html
<iframe src="assets/lru-animation.html" title="LRU access and eviction animation"
    style="display:block; width:100%; height:440px; border:0; background:transparent;"></iframe>
```

```julia
values = LRUMemoizedArray(identity, ArrayMap{Int}((1000,));
                  capacity=512, concurrent=true, queues=8)
handles = queue_handles(values)
tasks = [Threads.@spawn handles[i][i] for i in eachindex(handles)]
fetch.(tasks)
```

Or let the cache claim a queue for one operation:

```julia
with_queue(values) do handle
    handle[1]
end
```

Use one caller per queue. Do not mix automatic selection with explicit handles concurrently
on the same cache.
