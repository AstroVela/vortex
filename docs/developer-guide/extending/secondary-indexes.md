# Secondary Indexes

:::{warning}
This is an experimental design and interface prototype on AstroVela's Vane
development branch. It is not a stable API, a new Vortex file-format feature,
or an index-enabled SQL release. SPFresh and persistent catalog integration
are subsequent milestones.
:::

## Scope and ownership

An index is optional derived data over a fixed source snapshot. The source data
remains authoritative. Scalar predicates and ranked vector searches have distinct
contracts; a backend need not support both, or support mutation at all.

The design borrows the separation of index segments, coverage, and table snapshots
from [Lance's index specification](https://lance.org/format/index/), without
depending on Lance code or its internal interfaces. It does not reuse or migrate
the earlier `vendor/vortex-index` prototype.

| Component | Responsibility |
|---|---|
| `vortex-index` | Common metadata, registry, source/storage traits, scalar/vector query contracts, reference implementation |
| Future `vortex-index-spfresh` | Native bridge, SPFresh format/configuration, ID mapping, backend capabilities |
| `vortex-duckdb` in this monorepo | SQL binding, query planning, permission checks, result conversion |
| External `duckdb-vortex` repository | Extension composition, build/package configuration, pinned dependency revisions |
| Source/table adapter | Snapshot identity, visibility, catalog publication, conflict detection, cleanup |

Dependency direction is from integrations and backends to `vortex-index`, and
from `vortex-index` to the existing array/scan primitives and, with its opt-in
`file` feature, the file reader. Array, layout, and file readers do not acquire
native ANN dependencies. No backend is registered by default, and no C++
dependency is needed to test the common contracts.

Do not add a full `vortex-dataset` transaction engine as a prerequisite. Initially
an adapter can expose a frozen collection of Vortex files. An Iceberg or another
table adapter can later provide the same snapshot contract. Table-specific
commit rules do not belong in an index algorithm.

## Snapshot and addressing

`Snapshot` contains a dataset namespace, visibility version, schema fingerprint,
and an ordered inventory of file IDs, locations, immutable versions and physical
row counts. A file version is a content identity or immutable object version;
path, length and modification time alone are insufficient. The adapter must
verify/pin these versions when reading. Descriptor validation cannot detect an
adapter that lies about the underlying objects.

`RowAddress { file_id, row_offset }` is a physical address within that snapshot.
It is not a globally stable logical row ID. IDs are not file-list positions,
distributed partition numbers, or SPFresh internal vector IDs. A backend with
dense/native IDs owns an explicit mapping to public row addresses.

The initial reader requires an exact snapshot match, including schema and
visibility. Append, compaction, replacement, and deletion-mask changes reject
old metadata. Reusing unchanged coverage across snapshots requires a later
compatibility proof supplied by the table layer; comparing only vector counts
does not suffice. Stable logical IDs can be introduced separately through a
snapshot-bound resolver, not by changing the meaning of existing addresses.

## Metadata and artifacts

`IndexMetadata` is a draft version-1 sidecar descriptor. It records a logical
name, immutable generation, backend identifier and format version, source
snapshot, dependent top-level fields, coverage, and artifact inventory. Field
names are safe only together with the exact schema identity; stable field IDs
and nested-field evolution are deferred.

Coverage is currently whole-file. An index must include every eligible visible
row in each covered file, with each backend defining its supported input types
and NULL policy. An index can cover only part of the file inventory. Uncovered
files must be searched separately, not treated as containing no matches.

Artifacts have relative paths, lengths and algorithm-tagged checksums. Paths are
confined to a generation, including symlink confinement in a local store. The
provider/store verifies bytes and rejects corruption or missing artifacts.
Metadata validation alone is not checksum verification. Native SPFresh files
may retain their own format; mapping and auxiliary columns may use Vortex files.

`IndexStore` initially specifies whole-object IO to keep the prototype small.
Production backends need a range-read or pinned-local-artifact capability to
avoid downloading/materializing an entire native index. An object-store catalog
does not imply that SPFresh can query objects remotely. Unsupported storage
capabilities must be rejected explicitly.

## Read interfaces

`IndexProvider` identifies and opens an implementation. `IndexRegistry` rejects
duplicate names, validates snapshot/format support before open, and verifies
that the returned handle describes the requested generation. Registration is
explicit and case-sensitive. Lazy handle caching belongs to a runtime/session
integration, with keys including dataset, source version, index generation and
backend format. The prototype does not install a global cache.

`Index` exposes capability-specific interfaces. `VectorIndex` uses finite f32
query vectors, explicit dimensions/metric, top-k, exact/approximate mode, and an
opaque backend-owned options payload. Do not add `max_check`, HNSW parameters,
or SPDK settings to common metadata. Version and validate those within their
own backend. Future native dtypes/batch layouts can be added without making
SPFresh's wire buffers the shared API.

Vector results contain snapshot-relative row addresses and finite distances,
ordered by distance then row address. The default batch method preserves query
order and runs sequentially; a native provider can override it. Approximate mode
permits ANN retrieval, not stale data or relaxed predicates. Exact mode requires
an exact implementation or an explicit rejection/fallback.

`RowFilter` binds allow/exclude sets to the same snapshot. An absent allow list
means all rows, an empty list means no rows, and exclusions always win. The
source adapter must include tombstones in this filter. Backends without filtered
retrieval reject restricted requests. Truncating to k before filtering is not a
valid implementation of filtered top-k. Over-fetch/refill may be an ANN policy
in a later backend, but it cannot claim exact top-k merely because it returned k
rows.

`ScalarIndex` accepts Vortex expressions and returns either exact candidates or
a no-false-negative superset within its coverage. Supersets require residual
predicate evaluation. This is distinct from ANN, which may miss nearer rows.
No scalar backend is implemented in the first milestone.

## Source scan and materialization

`IndexSource` exposes a pinned snapshot, projected streaming scans, and ordered
row retrieval. Each source batch carries physical addresses aligned with its
Vortex array. Filtering must not renumber offsets. `take` preserves request
order and duplicates; missing or invisible rows are errors, not silently
shortened arrays.

Existing Vortex row selections are reused. Candidate addresses are grouped by
file, sorted and deduplicated before scanning. A selection cannot be broadcast
unchanged over every partition in a multi-file scan. Ranked retrieval retains
the original hit list so it can restore distance ordering after file-local IO.

### Implemented local adapter

The optional `vortex-index/file` feature provides `file::LocalFileSource` for a
frozen inventory without deletion masks. Every physical row is visible. Its
`open(snapshot, dtype, max_pinned_bytes, session)` checks every file before
returning; it does not discover a new snapshot or substitute current versions.
The session must provide registered file encodings/layouts and a live runtime.

- Locations are absolute native filesystem paths, not `file://` or cloud URLs.
- File versions are lowercase `sha256:<hex>` digests of the complete encoded
  bytes, generated by `file::file_version`.
- `file::schema_fingerprint` hashes the compact DType JSON representation with
  a `vortex-dtype-json-v1:` prefix. This experimental scheme is not a stable
  cross-release file-format fingerprint or a table's persistent field-ID scheme.
- All files must match the supplied non-nullable top-level struct dtype and
  physical row counts. Nested and nullable fields are supported. Empty file
  inventories still require an explicit dtype.
- `scan` preserves requested file order and physical row order. Repeated or
  unknown file IDs, and repeated or unknown projection fields, are errors.
  `take` sorts and deduplicates file-local selections, then restores caller
  order and multiplicity. Empty requests and zero-column projections are valid.

Opening copies and retains all encoded file bytes, after bounded reads and hash
verification. Scans stream projected batches from these pinned bytes, not from
the original paths. Later replacements, deletions, or in-place writes therefore
cannot mix file versions within the opened source. Reopening against the old
descriptor rejects changed or missing files. The catalog must supply trusted
expected digests; checksums alone do not authenticate a descriptor.

`max_pinned_bytes` limits the sum of encoded file lengths, not allocation
capacity, decoded arrays or total RSS. This is a bounded local prototype, not
an out-of-core source or a native index artifact store. A later scalable adapter
must preserve the same pinning guarantee through immutable objects or managed
local snapshots; holding an ordinary mutable file descriptor is insufficient.

The intended vector execution path is:

```text
pinned snapshot + predicate/visibility mask
    -> eligible index generations + scans of uncovered files
    -> backend candidate searches and exact scans
    -> snapshot-bound row resolution, deduplication, optional refinement
    -> global top-k under one consistent metric
    -> Vortex row retrieval and rank restoration
```

ANN branches keep the overall query approximate even if a refinement step
recomputes exact distances for retrieved candidates. An optimizer must never
replace exact SQL distance ordering with ANN without explicit approximate
semantics. Missing/unsupported indexes may trigger scan fallback; corruption
must remain observable. A direct request for an unavailable named index should
return an error instead of silently returning no rows.

## Build, publication and future updates

`IndexBuilder` writes a new private generation from a pinned source and returns
sealed metadata. It does not publish the descriptor. The catalog owner must:

1. Write and durably finalize all generation artifacts.
2. Validate the inventory, source snapshot and backend metadata.
3. Atomically publish metadata using an expected-version check or transaction.
4. Preserve generations still referenced by active readers/snapshots.

On a local filesystem, publication requires an appropriate writer lock/version
check and durable rename protocol; on object storage, conditional writes or a
transactional catalog are required. An ordinary rename is not a portable
object-store commit. An interrupted build may leave unreferenced artifacts, but
must never expose a partially built index. Drop first removes the catalog
reference; physical deletion follows safe reclamation.

Mutable SPFresh operations do not imply table-level ACID. A future writer API
must define sequence numbers, replay/idempotency, checkpoint durability and
generation publication before exposing updates through SQL. Published reader
generations remain immutable; mutation occurs in private state. The prototype
deliberately does not publish an `insert/delete/flush` trait with unspecified
recovery semantics.

## Delivery plan

1. **Implemented foundation:** draft contracts and metadata validation; registry;
   per-file candidate selections; in-memory exact squared-L2 reference; unit and
   conformance tests. `FlatIndex` only accepts dense covered files without NULL
   vectors, is not persistent, and is not a production query path.
2. **Vortex/SPFresh end-to-end:** the optional, memory-pinned local file source is
   implemented, including file scan / Flat search / ranked row retrieval tests.
   Next implement a generation store, port the separately tested native bridge
   into an optional backend, persist native-ID mappings, then validate
   build/open/search/take across multiple files.
   First qualify local storage and immutable checkpoints. Fix native PIC and
   dependency isolation before qualifying shared-extension packaging.
3. **Engine integration:** add common SQL entrypoints through `vortex-duckdb`,
   leave current SPFresh SQL names as compatibility adapters, and update the
   external extension's pinned Vortex revision. Do not duplicate index logic in
   C++ SQL binding code. Existing Vane branches remain unchanged until this step.
4. **Lifecycle and distribution:** qualify partial coverage across snapshots,
   updates, crash recovery, compaction and reader-safe reclamation before Ray
   execution. Workers reopen by immutable descriptor, never serialized native
   pointers or coordinator-local paths. Merge shard results under one metric.

Foundation tests cover metadata round trips, ambiguous/stale identities,
coverage, path constraints, registry dispatch, filtered top-k, query validation,
batch ordering and scan selection grouping. Later acceptance must add crash
injection, concurrent publication, a second real provider, and the existing
SPFresh native ID/distance/recall parity fixtures. The SPDK memory shim is not
NVMe qualification.

Local file-source tests cover missing/replaced/corrupt files, schema and row-count
mismatches, byte budgets, multi-batch offsets, cross-file ordered retrieval with
duplicates, nested/nullable projections and empty inputs. They do not qualify
out-of-core memory use, cloud storage or concurrent catalog commits.
