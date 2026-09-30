# Secondary Indexes

:::{warning}
This is an experimental design and interface prototype on AstroVela's Vane
development branch. It is not a stable API, a new Vortex file-format feature,
or a stable index-enabled SQL release. Immutable SPFresh readers and an opt-in
static initial builder are available behind an explicit native feature. The
optional DuckDB adapter supports explicit local static-index operations;
persistent table catalogs and automatic query planning remain future work.
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
| `vortex-index-spfresh` | Optional static native builder and immutable readers, pinned SPFresh format, ID mapping, backend capabilities |
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

`IndexStore` provides whole-object IO and optional `as_local_files()` access for
path-based backends. `LocalIndexFiles` imports files with bounded streaming IO
and materializes verified, isolated local copies held by a `LocalArtifactLease`.
Stores without this capability return `None`; providers requiring paths must
reject them explicitly. Range reads and shared local caching remain future work.
An object-store catalog does not imply that SPFresh can query objects remotely.

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
complete artifact metadata. It does not seal or publish the descriptor. The
catalog owner must:

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
must define sequence numbers, replay/idempotency, durable native state and
generation publication before exposing updates through SQL. Published reader
generations remain immutable; mutation occurs in private state. The prototype
deliberately does not publish an `insert/delete/flush` trait with unspecified
recovery semantics.

### Implemented local generation store

The independent `vortex-index/local-store` feature provides Unix
`store::LocalIndexStore`. It implements immutable local artifacts, not catalog
publication. Its optional file-path capability is described below.

1. `create` exclusively allocates a new generation under an existing durable,
   absolute, symlink-free root. There is no reopen-for-write or name reuse.
2. `write` and `import_file` accept portable relative artifact paths. They create
   files without clobbering, sync file bytes and directory entries, and record
   size/SHA-256 identities. Artifact traversal uses directory descriptors and no-follow opens;
   special-file reads are nonblocking and rejected after descriptor validation.
3. `seal` requires the exact successful artifact inventory and rechecks its
   contents. It syncs a pending manifest, atomically hard-links it into its final
   name without replacement, removes the pending link, and syncs the directory.
   The handle then rejects writes. IO failures invalidate the writer.
4. The catalog may publish the returned `LocalGeneration` with its expected
   snapshot through its own transaction. No latest pointer is written by the
   store. An error after manifest installation can leave a complete but
   unreferenced generation; the store does not automatically adopt or delete it.
5. `open` verifies the manifest against the trusted descriptor, checks metadata
   generation and exact snapshot identity, and streams verification of every
   artifact. Reads validate inventory membership and recheck bytes before
   returning them. A changed file is an error, never a new implicit version.

IO is blocking, including async trait methods, and belongs on a blocking worker.
Explicit limits apply to each artifact and manifest, not total memory. The root
and its children must be owner-managed: this is not protection against a
same-user process moving directories or rewriting files. Returned bytes are
pinned, and leased copies are isolated from generation changes; a mutable file
descriptor alone is not. Local filesystem sync and atomic hard-link semantics
are required. Network storage and power-loss recovery
are not qualified, and cleanup requires a later reader-safe reclamation policy.

A test-only persistent Flat provider covers independent builder and reader
processes, multiple source files, exact/filter search, and ranked row retrieval.
Storage tests cover corruption, missing objects, limits, path escape/symlinks,
FIFOs, concurrent creation/writes, interrupted manifests, injected write/sync
boundary failures and a builder killed before seal. This does not turn the
in-memory `FlatIndex` into a supported persistent or production backend.

### Native file adaptation

`LocalIndexStore` implements `LocalIndexFiles` under the same `local-store`
feature. No C++ dependency or backend-specific configuration enters the common
interfaces. This adapts the existing generation/artifact lifecycle; it does not
introduce a separate checkpoint abstraction.

Before importing, the backend must stop writers and finalize its complete file
set. `import_file(relative_path, absolute_source)` rejects symlinks in every
source path component and non-regular files, including FIFOs without a writer.
It copies bytes through a 64 KiB buffer, hashes them, enforces the per-artifact
limit, and uses the same durable, exclusive write protocol as `write`. It never
adopts a source file or hard-links one into the generation. The caller must keep
the source unchanged during copying: this is not a snapshot of a live native
writer, and same-length concurrent rewrites are not reliably detected. Imports
and sealing serialize on the same writer state; a partial-copy or sync failure
invalidates that writer. Preflight validation failures leave it usable.

After sealing, `materialize(artifacts, scratch_root, max_bytes)` creates a fresh
owner-only scratch directory. It checks inventory membership, rejects repeated
paths, and limits the sum of requested artifact sizes before copying. Each file
is streamed and reverified against its declared length and SHA-256, retains its
relative path, and becomes read-only before the lease is returned. No partially
verified lease is returned on failure. The implementation uses ordinary copies,
not hard links: mutations or deletion of the original generation do not affect
an existing lease, and a native library modifying one lease cannot modify the
generation or another lease through shared inodes.

The provider must retain `Box<dyn LocalArtifactLease>` until all native handles,
background readers and mappings are closed, including libraries that reopen
files lazily. A copied path string does not retain that lifetime. The lease does
not depend on the store handle remaining alive. Native readers must not mutate
their leased files. Paths inside configuration files are opaque to the store;
the backend must validate/resolve them and ensure that every required file is
included. Read-only file permissions are not a sandbox for native code.

Scratch roots must be absolute, symlink-free, exclusively owner-managed and
stable for the entire lease lifetime, including temporary-directory creation
and cleanup. Dropping the lease attempts to remove its directory. Scratch copies
are not durable, and a process crash can leave orphan directories; the owner
must reclaim them without removing active leases. Every lease costs a full disk
copy of the requested files. The total byte limit is per call and excludes
filesystem overhead, other leases, page cache and native-library allocations.
There is no shared cache, disk reservation, global quota or cloud staging yet.

Tests cover bounded/short/interrupted streaming reads, import/seal races, partial
copy and sync failures, invalid paths, symlinks and FIFOs, corruption, byte limits,
independent concurrent leases, mutation isolation, lifetime cleanup, and import /
seal / reopen / file-path access in separate processes. These exercise the file
adapter, not SPFresh format compatibility or native search correctness.

## Delivery plan

1. **Implemented foundation:** draft contracts and metadata validation; registry;
   per-file candidate selections; in-memory exact squared-L2 reference; unit and
   conformance tests. `FlatIndex` only accepts dense covered files without NULL
   vectors, is not persistent, and is not a production query path.
2. **Vortex/SPFresh end-to-end:** the optional, memory-pinned local file source is
   implemented, including file scan / Flat search / ranked row retrieval tests.
   The local generation store now supports durable immutable artifacts and
   trusted manifest reopening, exercised across processes by a test-only
   reference backend. Streaming imports and leased local native artifact copies
   are implemented. The optional `vortex-index-spfresh/native` backend now
   imports frozen Float32/L2 static bundles, persists dense native-ID mappings,
   opens verified leased copies, and implements unfiltered ANN search/batch.
   Real native tests cover C ABI parity, Flat recall, multi-file retrieval,
   independent processes and concurrent reader lifetimes. The pinned native
   build checks PIC by linking a shared bridge. `SpFreshIndexBuilder` now supports
   bounded in-memory initial construction from a pinned `IndexSource`, preserving
   physical row addresses and validating native coverage before import. It is
   explicitly attached to a provider; the owner still seals and publishes the
   generation. Only non-nullable Float32/L2 static construction is supported;
   dynamic updates, SPDK and shared DuckDB extension packaging remain unimplemented.
3. **Engine integration:** `vortex-duckdb/index` provides explicit static
   `vortex_index_build` and `vortex_index_search` functions. The owner supplies
   provider factories; common SQL code has no native backend dependency.
   Construction seals and qualifies a reader before exclusively publishing a
   reference containing the exact source schema/snapshot and manifest identity.
   Queries require full coverage, recheck source/artifact identity, and use
   ordered `take` to return original rows with ANN distances. Nullable vector
   schema markers are accepted only after checking actual NULLs. External
   `duckdb-vortex` composes the SPFresh provider and native build behind an
   explicit feature. Automatic planning, filtered ANN, mutable table catalogs,
   handle caching and distributed index execution remain follow-up work.
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
