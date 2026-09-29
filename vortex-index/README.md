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

Paths must be absolute native filesystem paths, not URLs. Every row is visible;
deletion masks and table transactions are not implemented. Projected `scan`
preserves physical addresses, and `take` groups selections per file before
restoring request order and duplicate rows. Empty projections are supported.

The source copies and pins **all encoded file bytes in memory** at open time,
subject to an explicit total byte budget. This prevents subsequent replacement,
deletion or in-place writes from changing a read. The budget is not a total RSS
limit: decoding and result arrays use additional memory. This adapter is for
bounded integration tests and local prototypes, not out-of-core workloads.

```sh
cargo test -p vortex-index --features file
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

Native SPFresh integration and SQL wiring remain separate milestones. A native
backend will also need constrained local artifact access; whole-object `Bytes`
IO is not a scalable substitute for native checkpoint files or range reads.

The crate has no DuckDB, SPFresh, SPTAG, or SPDK dependency. No code was imported
from the earlier `vendor/vortex-index` prototype. See the
[index design](../docs/developer-guide/extending/secondary-indexes.md) for
ownership, consistency rules, limitations, and the SPFresh integration plan.
