# Maps

```@meta
CurrentModule = MemoizedArrays
```

`*Map{T}(domain; options...)` returns a map configuration; `T` is required. Pass it to `MemoizedArray(generator, map)`
or `LRUMemoizedArray(generator, map; capacity)`. Each cache allocates its own map.
See [Examples](../examples.md) for use.

## ArrayMap

```@docs
ArrayMap
```

## ExpandingArrayMap

```@docs
ExpandingArrayMap
```

See the [expansion example](../examples.md#ExpandingArrayMap).

## Tree maps

```@docs
RadixMap
SubtreeRadixMap
```

See the [radix examples](../examples.md#RadixMap).

## DictMap

```@docs
DictMap
```

## Domains

A domain is a tuple or vector of axes; mixed axis forms are allowed.

- Integer length `n` → coordinates `1:n`.
- Integer unit range → its given inclusive bounds.
- `:` → every signed `Int` coordinate; `unbound(N)` gives N such axes.
- `()` → a zero-dimensional domain.
- Fixed and expanding array maps require finite domains; expansion keeps lower bounds fixed.

```@docs
unbound
isunbound
```
