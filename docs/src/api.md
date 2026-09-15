# API reference

```@meta
CurrentModule = JITArrays
```

`V` is the generated value type; `R <: AbstractUnitRange{Int}` denotes a signed integer range.

| API | Contents |
|---|---|
| [Caches](api/caches.md) | Unbounded retention, bounded LRU, and recency clocks |
| [Storage](api/storage.md) | Dense arrays, dictionaries, and the two tree backends |
| [Array interface](api/interface.md) | Shared array behavior and asynchronous batches |

```@docs
AbstractJITArray
```

## Signature notation

```@docs
VT
```
