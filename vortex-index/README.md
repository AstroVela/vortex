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

The default feature set stays empty. Persistent generation stores, native
SPFresh integration and SQL wiring are separate milestones.

The crate has no DuckDB, SPFresh, SPTAG, or SPDK dependency. No code was imported
from the earlier `vendor/vortex-index` prototype. See the
[index design](../docs/developer-guide/extending/secondary-indexes.md) for
ownership, consistency rules, limitations, and the SPFresh integration plan.
