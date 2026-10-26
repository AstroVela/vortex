# vortex-index

Experimental secondary index contracts for Vortex. This crate is not published,
registered in default Vortex sessions, or wired into SQL query planning.

It defines snapshot-bound row addresses, index metadata, backend registration,
source/storage interfaces, and separate scalar/vector query contracts. The
in-memory `FlatIndex` is a small exact squared-L2 reference implementation, not
a production ANN index or a persistent backend.

## Local file source

Enable `file` to use `vortex_index::file::LocalFileSource`. It verifies a frozen
inventory of local Vortex files against whole-file SHA-256 versions, physical row
counts and a shared schema. Use `file::file_version` and
`file::schema_fingerprint` to prepare those identities. The catalog must supply
trusted expected identities; opening never adopts the files' current versions.

Paths must be absolute native filesystem paths, not URLs. This source exposes
every physical row; use `snapshot::VisibleSource` for table-owned visibility.
Projected `scan` preserves physical addresses, and `take` groups selections per
file before restoring request order and duplicate rows. Empty projections are supported.

The source copies and pins **all encoded file bytes in memory** at open time,
subject to an explicit total byte budget. This prevents subsequent replacement,
deletion or in-place writes from changing a read. The budget is not a total RSS
limit: decoding and result arrays use additional memory. This adapter is for
bounded integration tests and local prototypes, not out-of-core workloads.

```sh
cargo test -p vortex-index --features file
```

## Explicit base and delta queries

Enable `file,snapshot` to query local Vortex files through `LocalFileSource`,
`VisibleSource`, `SnapshotIndex` and `SnapshotVectorSearch`. The caller supplies
an explicit file inventory, visibility mask, base generations and delta file IDs;
no cloud storage or running catalog service is required for this local path.
The local source retains its encoded-file memory budget described above.

Vortex remains a general columnar format: payload columns keep their existing
types, and vector indexes remain optional derived data. The application/table
layer owns logical IDs, deletion markers and superseded row versions. The query
layer operates on those supplied snapshots; it does not create or commit updates.

1. Wrap the pinned source in `VisibleSource` with the complete current `RowFilter`.
   Invisible rows are excluded from scans and rejected by `take`; physical row
   offsets never change when rows are filtered.
2. Supply each opened base generation as a `SnapshotIndex`, together with its
   **original build visibility** and backend query options. Persist that visibility
   alongside the generation; substituting today's mask can omit newly visible rows.
3. Pass the base generations and an **explicit list of delta file IDs** to
   `SnapshotVectorSearch::try_new`. Every snapshot file must occur exactly once
   in base coverage or in that delta list. An initial delta-only snapshot must
   explicitly list all its files.
4. Search applies current visibility and the query predicate before each base
   index's top-k. It scans only the declared delta vector columns and merges
   results using a bounded top-k heap. Retrieve other columns afterward through
   `search.source().take(...)`, preserving hit order and duplicates.

Base generations can span successive table snapshots only while their covered
file descriptors (ID, local path, version and row count) remain identical
and current visibility is a subset of build visibility. A new delta file or a
deletion does not itself require rebuilding unchanged base files. Changing a base
file or making previously hidden rows visible requires a new generation/plan.

There is **no automatic scan fallback**. Stale generations, uncovered files,
overlapping coverage, unsupported capabilities, corrupt artifacts and IO failures
are errors. An exact query requires every base index to support exact search;
a restricted query requires filter support. Approximate queries may use ANN
generations but never relax visibility. SQL predicate planning and evaluation of
predicates over new delta rows remain the caller's responsibility; an old
attribute/text allow-list must not omit matching delta rows.

The current vector path supports finite, non-null `FixedSizeList<Float32>` and
squared L2, with a shared schema fingerprint. It does not implement schema
evolution or promise backward compatibility for these experimental APIs or
catalog representations. Validation of identities and visibility is required for
correctness, independently of format/API compatibility.

## Future table publication and recovery integration

Local queries can run against explicitly supplied snapshots. When a durable
table/catalog layer is added, it must own the following protocol. These are
integration requirements, not a publication, recovery or compaction service
implemented by this crate.

1. Write immutable source files and collect their expected identities. Publish
   one atomic table revision containing the source snapshot, complete visibility,
   base generation references and explicit delta inventory. Data and visibility
   changes always get a new snapshot version, including deletion-only changes.
2. Build a generation against pinned source files and build visibility. Complete
   and verify every artifact before sealing it. Persist the trusted generation
   descriptor, build snapshot and build visibility together. A sealed generation
   is not published merely because its files exist.
3. Attach it with a catalog compare-and-swap against the revision used to plan the
   build, atomically replacing the relevant delta/base entries. On conflict, load
   the current revision and validate a new plan before another publication attempt.
   Readers pin a complete revision and never combine its data with another
   revision's visibility or index references.
4. Background compaction writes new columnar files and matching indexes before
   atomically replacing base entries. Old physical row addresses are never carried
   into rewritten files. Retain old source files and artifacts until no retained
   snapshot/reader references them; cleanup belongs to the table's GC policy.
5. After a crash, reopen only catalog-referenced sealed descriptors and pinned
   files. Unpublished builds are orphans, never candidates discovered by listing
   directories. Missing or corrupt published state fails explicitly; recovery
   repairs or republishes a complete revision instead of switching to latest data
   or silently scanning.

Tests exercise real local Vortex files, multiple base generations, explicit delta
scans, deletion visibility, ordered payload retrieval and rejection without
fallback. The snapshot query tests do not qualify a future table/catalog layer's
crash durability.

```sh
cargo test -p vortex-index --features file,snapshot
```

## Local generation store

Enable `local-store` on Unix to use `store::LocalIndexStore`, independently of
the `file` source feature. The default feature set stays empty.

- `create(root, generation, limits)` exclusively creates a private generation.
  The root must already exist durably, be absolute, and have no symlink
  components. Existing generation names are never reused, including failed or
  abandoned builds.
- `IndexStore::write` creates artifacts without overwriting, syncs their bytes
  and containing directories, and returns exact length and SHA-256 identities.
  Nested relative paths are allowed; traversal and symlinks are rejected.
- `seal(metadata)` requires exactly the successful write inventory, verifies
  every artifact, and atomically installs a synced manifest without clobbering
  an existing one. The writer becomes read-only. IO failure makes the writer
  unusable; retry in a new generation.
- `open(root, descriptor, snapshot, limits)` checks a trusted manifest digest,
  generation and snapshot, then streams verification of every artifact. Later
  reads verify again before returning bytes. Missing, modified or non-regular
  files are errors, not empty results.
- `prepare_open(root, descriptor, snapshot, limits)` authenticates the manifest
  and snapshot without exposing a store or reading artifact contents. Inspect
  its metadata to reserve resources, then consume it with `verify()` for the
  ordinary sealed store or `materialize(scratch_root, max_bytes)` for a private
  store. Materialization copies and verifies all artifacts in one streaming pass;
  any error removes the partial copies. No unverified sealed store is returned.

A materialized store is read-only and independent of later changes to the
canonical generation. Its first complete-inventory `LocalIndexFiles::materialize`
call into the same scratch root transfers the existing verified lease. Limits,
inventory and directory identity are checked before transfer. Subsequent calls,
subsets and other scratch roots receive separately copied and verified leases.
The store and transferred lease retain the private directory until both are
dropped; the provider must close native handles before releasing its lease.

Providers can opt into this path with `IndexProvider::local_materialization_limit`.
The hint declares a total artifact byte budget; it does not bypass backend format
checks or require query-engine knowledge of SPFresh. Providers must also accept
ordinary stores and keep enforcing their own limits during open.

The returned `LocalGeneration` descriptor binds the manifest bytes, which in
turn bind metadata and all artifact identities. The catalog must durably store
that descriptor with the expected source snapshot; never discover a generation
by hashing whatever files are currently present. Sealing does **not** publish
an index or implement a source/catalog transaction. There is no mutable latest
pointer, automatic cleanup or writer resume.

This implementation uses blocking IO, including its async trait methods; run
it on a blocking worker. `LocalStoreLimits` bounds each artifact and manifest,
not total RSS. Filesystem durability assumes local file/directory sync and
atomic hard-link support; network filesystems and power-loss recovery are not
qualified. Roots are owner-managed, not an adversarial same-user filesystem
sandbox. Returned `Bytes` are pinned; ordinary on-disk files are not.

Tests cover corruption, limits, path confinement, FIFOs, concurrent creation and
writes, injected IO failures, and killed builders. A **test-only** persisted Flat
provider exercises `build -> process exit -> reopen -> search -> take` against
multiple Vortex files. It is not an exported backend or a supported format.

```sh
cargo test -p vortex-index --all-features
cargo test -p vortex-index --no-default-features --features local-store
```

Both adapters use the same accelerated SHA-256 implementation for complete
encoded files, manifests and streaming artifact verification. The existing
`sha256:<lowercase hex>` identities and serialized formats are unchanged; no
content check is replaced with a path, timestamp or length-only check. The
`ring` dependency is enabled only by the optional file/storage features.
The previous RustCrypto implementation is retained as a test and benchmark
oracle for padding and IO chunk boundaries.

```sh
cargo bench -p vortex-index --features file --bench checksum
```

The [SPFresh provider](../vortex-index-spfresh/README.md) uses constrained local
artifact access. SQL wiring remains separate; whole-object `Bytes` IO is not a
scalable substitute for native checkpoint file access.

The crate has no DuckDB, SPFresh, SPTAG, or SPDK dependency. No code was imported
from the earlier `vendor/vortex-index` prototype. See the
[index design](../docs/developer-guide/extending/secondary-indexes.md) for
ownership, consistency rules, limitations, and the SPFresh integration plan.

## Attribute prefilters

`AttributeIndex::filter` evaluates typed `attributes::AttributePredicate` values
and returns an exact `RowFilter`. `FullTextIndex::filter_text` returns all matching
visible text rows, without a BM25 top-k limit. Both can prefilter vector search.
The implementation lives in the existing [Tantivy provider](../vortex-fts/README.md#tantivy-prefilters-for-vector-search):
text and scalar attributes share one Tantivy generation, document row mapping,
and artifact lifecycle. There is no separate custom attribute postings format.

Use `Index::as_attributes()` or `as_full_text()` after opening a generation through
`IndexProvider`. Intersect text and attribute predicates by passing the text
filter as visibility to the attribute filter. Snapshot mismatches and incomplete
file coverage are errors; the allow-list must not silently omit unindexed files.
SPFresh translates the resulting physical addresses through its own row map.
The two indexes may have different internal ID orders for the same snapshot.

Attribute/text providers still require the exact snapshot supplied to them.
The base/delta vector layer does not automatically extend their allow-lists to
new files. Catalog publication and SQL predicate planning remain the caller's
responsibility.
