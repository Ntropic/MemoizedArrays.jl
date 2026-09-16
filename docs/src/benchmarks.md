# Benchmarks

## Run

```sh
# All maps and caches (default).
julia --startup-file=no --compiled-modules=existing --project=. benchmark.jl

# Select maps and caches independently.
julia --startup-file=no --compiled-modules=existing --project=. benchmark.jl --maps=dict
julia --startup-file=no --compiled-modules=existing --project=. benchmark.jl --maps=array,lsbtree --caches=jit,exact
```

Maps: `array`, `dict`, `msbtree`, `lsbtree`. Caches: `jit`, `exact`, `sampled`.
Tree benchmarks use `(8, 8)` blocks.

## What each row measures

| Workload | Raw storage | Cache |
|---|---|---|
| Read | Retrieve an existing value | Retrieve an existing value and update recency where applicable |
| Insert / miss | Insert the supplied matrix reference | Look up a missing key, call the generator, create an entry and insert it |
| Replace / evict | Delete the supplied key and insert its replacement | Request a missing key and let the cache choose an eviction victim |

The generator returns one precomputed 4×4 matrix. `DictMap + JITEntry` stores the same entry wrapper
as JITArray, but calls map insertion directly. It separates the cost of that representation from
cache-miss handling. `Insert, reserved` preallocates table capacity before timing.

All implementations now use **one timed loop**, write one result per request, and process eight
containers per sample. Reads use 32,768 requests per container; insertions and replacements use 512.
Setup, prefill, validation and full GC at the start of each round are untimed. GC triggered inside
the request loop is included and reported separately. Allocation counts come from the same timed
sample. Fifteen rounds rotate implementation order; the script prints median, approximate p10–p90,
bytes/request and GC ns/request.

Earlier tables used different loops and output writes for raw insertion and cache generation.
Those measurements have been replaced. Compare implementations within a table; timing differences
between separate runs also reflect compilation, allocation layout and CPU conditions.

## Matrix operations for comparison

Julia 1.12.7, AMD Ryzen AI 7 350, 2026-09-16. Ordinary `Matrix{Float64}` inputs,
OpenBLAS with one thread. Median time per operation, including result allocation.

| Matrix size | Addition `A+B` | Product `A*B` | `eigh` |
|---|---:|---:|---:|
| 2×2 | 117 ns | 139 ns | 2.31 µs |
| 4×4 | 183 ns | 331 ns | 11.91 µs |
| 8×8 | 325 ns | 506 ns | 33.62 µs |
| 16×16 | 1.53 µs | 1.76 µs | 100.61 µs |

`eigh` uses `eigen(Symmetric(A))`, computing both eigenvalues and eigenvectors.
Each measurement uses 15 rounds of 4,096 operations, cycling through 64 input pairs and
retaining results. Compilation, input generation and round-boundary GC are excluded;
GC during operations is included. Operation order rotates between rounds. Timings vary,
so these are approximate comparisons with the separately measured cache operations.

These operations compute and allocate their results. The cache benchmarks return a
precomputed matrix; a generator performing these operations adds their cost to the cache miss.

## Baseline before entry reuse

These tables predate the direct-counter and entry-reuse changes. The paired entry-reuse comparison is
recorded in the repository at `experiments/reuse.md`.

Julia 1.12.7, AMD Ryzen AI 7 350, 2026-09-15. Synchronous execution. Median **ns/request**;
allocation columns are **bytes/request**. All ready reads allocated zero bytes.

Sparse keys are `(2mod(i, 32), 2div(i, 32))`, `i=1:512`. Replacement adds 128 to the first coordinate.
Array storage uses keys `1:512`, replacing them with `513:1024`. Construction of the matrix is excluded.


### ArrayMap

| Implementation | Read ns | Insert / miss ns | Replace / evict ns | Insert bytes | Replace bytes |
|---|---:|---:|---:|---:|---:|
| Array | 4.75 | 2.71 | — | 0.0 | — |
| JITArray | 5.55 | 21.19 | — | 16.0 | — |
| ExactLRUArray | 16.28 | 29.75 | 50.78 | 48.0 | 48.0 |
| SampledLRUArray | 8.35 | 29.79 | 231.3 | 48.0 | 48.0 |

### DictMap

| Implementation | Read ns | Insert / miss ns | Replace / evict ns | Insert bytes | Replace bytes |
|---|---:|---:|---:|---:|---:|
| Dict | 24.64 | 95.99 | 203.64 | 67.34 | 200.23 |
| DictMap | 20.41 | 90.29 | 122.27 | 68.16 | 64.08 |
| DictMap + JITEntry | 22.22 | 113.45 | 136.56 | 84.16 | 80.08 |
| JITArray | 24.93 | 134.24 | — | 84.16 | — |
| ExactLRUArray | 43.1 | 139.65 | 208.67 | 116.16 | 112.08 |
| SampledLRUArray | 27.69 | 135.15 | 503.23 | 132.16 | 128.08 |

Additional raw operations, ns/request (bytes/request in parentheses):

| Operation | Dict | DictMap | DictMap + JITEntry |
|---|---:|---:|---:|
| Delete | 60.18 (0.0) | 22.9 (0.0) | 23.95 (0.0) |
| Insert, reserved | 73.09 (0.0) | 65.93 (48.0) | 82.81 (64.0) |
| Replace at load limit | 80.21 (0.0) | 99.53 (48.0) | 117.64 (64.0) |

The load-limit case repeatedly deletes and reinserts keys into a table with 48 resident entries.

For insertion, DictMap uses 68.16 bytes/request; wrapping the value in `JITEntry` raises that to
84.16 bytes, the same allocation as the full JITArray miss. The corresponding medians are
90.29, 113.45 and 134.24 ns. Their p10–p90 ranges are 82.99–159.91, 99.99–147.68 and
107.35–166.85 ns: these overlap, so subtracting medians does not precisely price individual steps.

### MSB tree

| Implementation | Read ns | Insert / miss ns | Replace / evict ns | Insert bytes | Replace bytes |
|---|---:|---:|---:|---:|---:|
| MSBRadixMap | 30.34 | 105.44 | 129.25 | 152.72 | 151.53 |
| JITArray | 37.99 | 122.82 | — | 168.72 | — |
| ExactLRUArray | 58.25 | 133.42 | 196.51 | 200.72 | 199.53 |
| SampledLRUArray | 42.82 | 132.04 | 626.2 | 216.72 | 215.53 |

Raw deletion: 39.79 ns/request, zero allocation.

### LSB tree

| Implementation | Read ns | Insert / miss ns | Replace / evict ns | Insert bytes | Replace bytes |
|---|---:|---:|---:|---:|---:|
| LSBRadixMap | 35.55 | 129.78 | 193.46 | 152.72 | 151.53 |
| JITArray | 45.68 | 139.44 | — | 168.72 | — |
| ExactLRUArray | 66.96 | 149.74 | 215.14 | 200.72 | 199.53 |
| SampledLRUArray | 47.95 | 157.65 | 706.74 | 216.72 | 215.53 |

Raw deletion: 43.18 ns/request, zero allocation.

## Effect of this optimization

These separate paired runs load the synchronous source before and after this change into the
same process and use the corrected harness for both. They measure implementation changes,
not removal of concurrency. “Before” already includes the LSB forest of trees by bit depth.

| Map / cache | Operation | Before ns | After ns | Reduction |
|---|---|---:|---:|---:|
| DictMap / JITArray | Miss | 119.30 | 109.04 | 9% |
| MSB / JITArray | Miss | 133.66 | 113.83 | 15% |
| MSB / ExactLRUArray | Miss | 143.87 | 127.81 | 11% |
| MSB / SampledLRUArray | Miss | 146.69 | 124.86 | 15% |
| LSB / JITArray | Miss | 155.69 | 137.44 | 12% |
| LSB / ExactLRUArray | Miss | 149.22 | 136.45 | 9% |
| LSB / SampledLRUArray | Miss | 152.19 | 133.47 | 12% |
| LSB / ExactLRUArray | Miss + eviction | 262.46 | 207.60 | 21% |

ArrayMap and DictMap LRU results were broadly unchanged. Allocation counts did not increase.
This is not a uniform read improvement: MSB ready reads changed from 36.51 to 37.76 ns for JIT,
52.99 to 59.28 ns for exact LRU, and 40.60 to 42.10 ns for sampled LRU. Retaining insertion
information has a cost even when the value is present. All remain allocation-free.

Mutation timings vary substantially: LSB exact-LRU eviction had p10–p90 ranges of
254.15–386.78 ns before and 199.69–270.45 ns after. Sampled-LRU eviction medians improved
on both trees, but their broad overlapping ranges do not support a precise speedup claim.

## Reusing searches

Cache misses now retain the position found by lookup instead of immediately throwing it away.
Array slots are reused directly. DictMap retains its table, position and insertion counter. Trees
retain a concrete `TreeHint` with the link, parent and address; ready reads use the leaf directly,
avoiding allocation of a temporary `LeafSlot`.

Some work remains necessary:

- **Nested dictionary insertion or resize:** the generator can insert into the same cache and
  change the probe chain or table. Search again when the saved position was invalidated, reusing
  the already computed hash. Julia's `Dict.get!` uses a similar mutation check around its generator.
- **Dictionary eviction:** deletion leaves a tombstone, so it can reuse an absent-key insertion
  position. An insertion that triggers table rebuilding needs a position in the replacement table.
- **Tree pruning:** an evicted leaf can collapse its parent. A detached parent's link cannot be
  reused. Resume at a retained, attached link, or start again at the root when necessary.
- **Eviction of another key:** the victim has its own coordinates/hash. The incoming key's routing
  data cannot locate it. Tree eviction now encodes the victim once and locates and removes it in
  one descent. Deletion given only a leaf slot still needs a parent search when the leaf becomes empty.
- **Entry allocation and recency:** each miss still creates its cache entry; DictMap also creates a
  stable hash record. Exact LRU maintains its linked list and sampled LRU maintains its recency and
  sampling state. These are separate operations from finding the insertion position.

## Why fewer LSB levels did not guarantee faster reads

The current `:lsbtree` layout selects independent compressed MSB trees by coordinate bit depth.
MSB was already compressed: it did not traverse every possible bit group for every key. The
nearby-only workload therefore saves only about half a node visit per read, while LSB adds root
selection. With large outliers, keeping the small-coordinate roots independent saves much more.

The previous low-bit-first LSB layout used one concrete node type and no compressed-prefix checks.
It was fast on small coordinates, but allocated value and child arrays at every traversed level.
The current layout uses fewer nodes and much less memory for large coordinates. It must distinguish
branches from leaves and check prefixes where compression skips levels. Checks between adjacent
levels have been removed because the previous routing step already established those bits.

The node counts below use 512 entries. The mixed workload has 496 nearby keys and 16 large outliers;
99% of reads target nearby keys. Resident size includes reachable map storage and the shared matrix.

| Workload | Layout | Read ns | Mean nodes visited | Resident bytes |
|---|---|---:|---:|---:|
| Nearby | Old low-bit LSB | 15.25 | 2.496 | 284,152 |
| Nearby | Before depth roots | 40.16 | 2.494 | 74,344 |
| Nearby | After depth roots | 34.96 | 2.494 | 74,344 |
| Nearby | MSB | 29.57 | 2.998 | 74,176 |
| Scattered | Old low-bit LSB | 1473.23 | 21.783 | 11,061,880 |
| Scattered | Before depth roots | 73.3 | 3.662 | 379,832 |
| Scattered | After depth roots | 68.73 | 3.662 | 379,832 |
| Scattered | MSB | 57.42 | 3.879 | 379,664 |
| Mixed | Old low-bit LSB | 28.8 | 2.673 | 624,696 |
| Mixed | Before depth roots | 36.54 | 2.002 | 85,048 |
| Mixed | After depth roots | 34.48 | 2.002 | 85,048 |
| Mixed | MSB | 54.52 | 4.98 | 85,456 |

“Old low-bit LSB” is the earlier synchronous low-bit-first trie, included only to diagnose the
layout tradeoff. “Before/After depth roots” isolates this optimization of the current forest.
Skipping redundant prefix checks reduced nearby lookup from 40.16 to 34.96 ns. LSB still trails
MSB for nearby-only and scattered keys. In the mixed workload it saves about three node visits
and wins. Compared with the old low-bit trie, the forest uses about 29× less resident storage
for scattered keys and reduces lookup from 1,473 to 69 ns.

## External tree comparison

DataStructures 0.19.6: `SortedDict` (ordered tree) and `Trie`. The trie consumes the two integer
coordinates as its path, so it is a dictionary-backed coordinate trie, not a bitwise radix tree.
Trie deletion is omitted. These are raw storage operations with the same timed loop; they do not
include cache entry creation or eviction policy.
Median ns/request, from a separate comparison run.

```sh
JULIA_LOAD_PATH="@:/path/to/environment:@stdlib" \
julia --startup-file=no --compiled-modules=existing --project=. \
  -e 'using DataStructures; include("benchmark.jl"); comparetrees(SortedDict, Trie)'
```

DataStructures is optional. Select either tree with `maps=(:msbtree,)` or `maps=(:lsbtree,)`;
omitted means both. The mixed workload directs 99% of reads to nearby keys.

### Nearby keys

| Storage | Lookup | Insert | Delete | Delete + insert |
|---|---:|---:|---:|---:|
| MSBRadixMap | 30.55 | 98.41 | 36.82 | 130.78 |
| LSBRadixMap | 36.96 | 125.21 | 43.91 | 181.7 |
| SortedDict | 58.98 | 333.0 | 314.39 | 666.58 |
| Dict | 25.46 | 91.78 | 61.65 | 215.69 |
| Trie | 45.28 | 158.67 | — | — |

### Scattered signed keys

| Storage | Lookup | Insert | Delete | Delete + insert |
|---|---:|---:|---:|---:|
| MSBRadixMap | 54.98 | 376.02 | 263.36 | 918.35 |
| LSBRadixMap | 66.68 | 421.91 | 305.63 | 1000.32 |
| SortedDict | 50.7 | 318.01 | 356.22 | 742.15 |
| Dict | 23.38 | 97.85 | 65.65 | 211.88 |
| Trie | 55.35 | 345.06 | — | — |

### Mostly nearby keys, 16 large outliers

| Storage | Lookup | Insert | Delete | Delete + insert |
|---|---:|---:|---:|---:|
| MSBRadixMap | 51.5 | 115.16 | 72.71 | 188.76 |
| LSBRadixMap | 32.27 | 129.68 | 53.52 | 200.07 |
| SortedDict | 56.27 | 326.41 | 380.65 | 692.07 |
| Dict | 24.06 | 90.81 | 64.27 | 215.07 |
| Trie | 44.78 | 177.58 | — | — |
