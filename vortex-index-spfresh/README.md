# SPFresh Index Provider

Experimental static backend for `vortex-index`, with immutable readers and an
opt-in initial builder. This is not a SQL integration, an online SPFresh writer,
or a stable portable native file format.

## Build

Default features do not compile, download or link SPFresh. Native support is
explicit and currently qualified on Linux x86_64 with GCC:

```sh
# Debian/Ubuntu build prerequisites, in addition to the Vortex Rust toolchain.
sudo apt-get install cmake g++ libzstd-dev libclang-dev pkg-config
bash vortex-index-spfresh/native/build.sh
export VORTEX_SPFRESH_NATIVE="$PWD/target/spfresh-native/build"
cargo test --locked -p vortex-index-spfresh --all-features
cargo clippy --locked -p vortex-index-spfresh --all-targets --all-features -- -D warnings
```

The explicit build script fetches SPFresh/SPFresh revision
`5893eb61ee3b18610b6b00f1939be7dae1af8904`, checks it, and applies
`native/static-only.patch`. It refuses unexpected existing source changes.
Use a fresh build directory when changing the revision/patch. Cargo never fetches
native sources. The patch disables dynamic/SPDK/RocksDB branches and the external
delete-map load, makes NUMA optional, and fixes destruction of uninitialized
datasets. It selects upstream's checked synchronous posting reader (the AIO
completion path ignores read errors) and makes query-result ownership exception
safe. Float atomic accumulation during construction uses a correctly sized
OpenMP operation rather than a platform-dependent `long` cast; an ordinary native
test checks neighboring data and concurrent increments. SIMD intrinsics have
per-function ISA targets; dispatch checks CPU and OS
support without applying AVX flags to baseline code. No SPDK memory shim,
RocksDB, TBB, Boost or git submodules are needed.
The upstream MIT notice is in `native/SPFresh.LICENSE`.

Core archives are PIC and a shared bridge is actually linked with no unresolved
symbols during qualification. The fixture executable is a test-only builder, not
an exposed `IndexBuilder`. Native source revision and patch fingerprints are
checked by Cargo. CMake/system dependencies are not hermetically vendored.

## Frozen Bundle Contract

Backend ID: `spfresh.static`. Backend format version: `1`.

Only unquantized Float32 squared L2, BKT heads, excluded heads, **one** local static
posting file, no deleted vectors, and no compression, posting rearrangement or
delta encoding are supported. Native IDs must be dense from zero. This uses
SPFresh's static SPANN search path, not its online incremental-update path.

The source directory passed to `import_bundle` contains:

| Name | Contents |
|---|---|
| `vectors.bin` | BKT head vectors |
| `tree.bin` | BKT search trees |
| `graph.bin` | BKT neighborhood graph |
| `deletes.bin` | BKT delete labels, with no deleted heads |
| `head_ids.bin` | Head-to-full-native-ID translation |
| `postings.bin` | Single uncompressed static posting file |

The first four normally come from SPFresh's `HeadIndex` directory; the last two
are its head-vector-ID and SSD-index outputs, renamed as above. An arbitrary
SPFresh configuration or dynamic-store checkpoint is **not** a supported import.
The producer must assert the pinned revision, format, dimension, L2 metric,
original native ID order and frozen source snapshot. Native binaries do not
reliably encode all of those semantics, so the importer cannot infer them.

All writers must be stopped throughout import. Import copies the six files into
`spfresh/native/` and writes two additional artifacts:

- `spfresh/bundle.json`: closed, versioned descriptor including the pinned
  revision, Float32/L2 shape, snapshot, field and whole-file coverage binding.
- `spfresh/rows.bin`: eight-byte `VXSFROW1` magic, little-endian u64 row count,
  then pairs of little-endian u64 file ID and u64 physical offset, in native ID
  order. This is a backend format, not a new Vortex file-format extension.

Mapping must be a bijection over **all physical rows** in the covered files.
Sparse native IDs, nullable/invalid vectors and excluded/deleted source rows
require a future format/eligibility contract. File IDs need not be contiguous.
Mapping validation takes O(rows log rows) time and O(rows) memory.

The owner calls `LocalIndexStore::seal`, retains its trusted `LocalGeneration`,
and opens the backend successfully before catalog publication. Import alone
does not open/qualify native binaries, seal or publish anything. It is usable
without the native feature. Providers advertise no builder unless explicitly
configured with `SpFreshProvider::with_builder`.

## Initial Static Build

`SpFreshIndexBuilder::try_new(scratch_root, session, limits)` implements the common
`IndexBuilder` trait. It requires the `native` feature, an existing absolute
symlink-free scratch directory, a session supporting the source encodings, and a
local-file-capable private store. The supplied metadata must match the source's
pinned snapshot, declare one top-level vector field, and have no artifacts.

Serialize `SpFreshBuildOptions` into `IndexBuildRequest.backend_options`. All
fields are required; unknown fields and versions fail:

```json
{"format_version":1,"dimension":8,"head_count":64,"posting_page_limit":12,"replicas":4}
```

This version accepts at least 64 covered rows, 32 <= head_count < rows,
1 <= dimension <= 4096, 1 <= posting_page_limit <= 4096 and 1 <= replicas <= 8.
It uses random head selection, BKT heads, one static posting file and fixed
single-threaded native construction. The metric is always squared L2. Source
batches must be non-nullable structs projecting exactly the requested field as
non-nullable FixedSizeList<Float32>, with no non-finite components. Nullable
schemas are rejected even if their current values are all valid.

The builder scans only the requested files, checks row/data alignment and full
physical coverage, and preserves scan order in the dense native-ID mapping.
Missing/deleted/duplicate rows are errors, not silently renumbered exclusions.
It constructs in a fresh private scratch directory, closes native writers,
validates native structure and coverage, and reopens the native index before
importing the closed files. If posting truncation loses any source row, the build
fails; increase head_count or posting_page_limit and retry in a fresh generation.

Successful build returns durable artifact metadata, **not** a sealed store or a
published index. The owner calls `LocalIndexStore::seal`, opens through the
provider, and publishes only against the expected snapshot. Scan/native failures
do not import artifacts. Failed imports can leave unreferenced partial artifacts;
retry in a fresh generation without overwriting existing files. Scratch is
removed on success, ordinary errors and cancellation between scan batches;
process-crash cleanup belongs to the owner.

The first builder buffers vectors and the row map in memory. Default limits are
1,000,000 rows, 256 MiB of flattened Float32 input, and 1 GiB of the six native
output files. Output size is checked after construction, before validation/import.
These are **not** native RSS or temporary-disk quotas; source batch execution,
head graphs, replica selection, temporary files and metadata need additional
resources. The builder is not out-of-core. Native build is synchronous, cannot
be interrupted by dropping an async future mid-call, and serializes with searches.

## Open, Search And Lifetime

Register `SpFreshProvider` explicitly. It requires `IndexStore::as_local_files`
and an absolute owner-managed scratch directory. Every open materializes exactly
the eight declared, checksum-verified artifacts. It rejects missing/extra files,
descriptor/snapshot/coverage mismatches, invalid mappings, malformed dimensions,
non-finite stored vectors, invalid posting IDs and incomplete native coverage.
Binary shape checks precede native allocation.

The bridge constructs configuration internally. It never reads imported INI
files, an original source directory, external delete maps, SPDK environment
variables or dynamically named posting shards. Only fixed files exist in the
lease. The opened native handle is destroyed before its lease. Deleting the
source or canonical generation cannot change an already-open reader.

Search supports Approximate only, with unrestricted filters bound to exactly the
indexed snapshot. Exact, allow/exclude masks and unknown backend options fail
explicitly. k is 1..=4096 and is capped to the covered row count. Results are
deduplicated and sorted by squared distance then physical row address.
Batch search preserves query order; empty batches still validate options/filter.

Empty backend options use max_check=4096, internal_results=max(64,k), and the
bundle's page limit. Explicit JSON must specify exactly:

```json
{"max_check":4096,"internal_results":64,"search_pages":12}
```

Require k <= internal_results <= 4096, internal_results <= max_check <= 1048576,
and search_pages == the bundle page limit. Posting reads are truncated by the
bundle page limit when the index is opened; in this static path a smaller
per-query search_pages would shrink the native read buffers without reducing
IO, so values below the page limit are rejected. The provider limits
materialized bytes, mapping rows, batch vectors and posting-buffer allocation
separately. Those are not a total native RSS limit or global disk quota. Each
handle costs a full artifact copy and retains its own in-memory head index and
row map.

Open/import/search are **blocking**, despite common async trait signatures.
Run them on blocking workers. The bridge serializes all native calls across
handles and clears SPANN/BKT thread-local workspaces at call boundaries, since
upstream shares them across Float32 indexes with potentially different sizes.
Posting reads use the upstream synchronous path and fail on short/error reads;
there are no outstanding native IO tasks when a call returns. This conservative
first version favors isolation over concurrent throughput.
Do not link another incompatible SPTAG build into the same process.

Artifacts and their producer remain trusted: checksums and structural checks
are not authentication or a sandbox for upstream C++ parsers. Scratch directories
must not be changed by another same-user process while a reader is alive.
Crash-left scratch cleanup belongs to the owner, not this provider.

## Qualification

Native tests build actual indexes, compare C ABI IDs/distances to mapped provider
results, check recall against FlatIndex, and use ranked addresses to retrieve
rows from two Vortex files. They exercise independent build/read processes after
deleting original native inputs, multiple handles with changing probe counts,
lease cleanup, checksum corruption and semantic corruption with valid checksums.
Initial-builder tests cover actual multi-file Vortex input, Flat recall, ranked
take, independent build/read processes, invalid source/options, byte limits and
real multipage postings from 30,000 eight-dimensional vectors. CI runs these
ordinary tests and all-target Clippy after the pinned native build.
This is correctness qualification, not a large-scale latency/recall benchmark
or NVMe/SPDK qualification.
