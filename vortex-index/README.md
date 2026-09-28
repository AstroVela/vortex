# vortex-index

Experimental secondary index contracts for Vortex. This crate is not published,
registered in default Vortex sessions, or wired into SQL query planning.

It defines snapshot-bound row addresses, index metadata, backend registration,
source/storage interfaces, and separate scalar/vector query contracts. The
in-memory `FlatIndex` is a small exact squared-L2 reference implementation, not
a production ANN index or a persistent backend.

The crate has no DuckDB, SPFresh, SPTAG, or SPDK dependency. No code was imported
from the earlier `vendor/vortex-index` prototype. See the
[index design](../docs/developer-guide/extending/secondary-indexes.md) for
ownership, consistency rules, limitations, and the SPFresh integration plan.
