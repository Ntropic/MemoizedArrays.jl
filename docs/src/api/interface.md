# Array interface

```@meta
CurrentModule = MemoizedArrays
```

Supports integer coordinates, `CartesianIndex`, ranges, index vectors, `:` slices and views.
Views are lazy and share the parent cache; rectangular serial tree views retain bounded subtree shortcuts.

## Readiness and generated values

```@docs
isready
getready
```

## Generation

```@docs
generate!
```
