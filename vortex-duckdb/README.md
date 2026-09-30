# Vortex DuckDB

Rust bindings for DuckDB. Supports DuckDB precompiled libraries for fast builds and from source builds for debugging.

## Prerequisites

- **Ninja**: `brew install ninja` (macOS) | `apt-get install ninja-build` (Ubuntu)
- **CMake**: `brew install cmake` (macOS) | `apt-get install cmake` (Ubuntu)
- **C++20 compatible compiler**: GCC or Clang

## Build Modes

### Default (Release)

Link against the precompiled DuckDB release build.

```bash
cargo build -p vortex-duckdb
```

### Debug Build

Opt into DuckDB debug build: `VX_DUCKDB_DEBUG=1`.

```bash
VX_DUCKDB_DEBUG=1 cargo build -p vortex-duckdb
```

### Vane Distributed Scan Build

The explicit `VORTEX_VANE_DISTRIBUTED=1` build mode adds Vane's distributed
table-scan protocol. It requires an exact Vane DuckDB source tree containing
`duckdb/function/distributed_table_function.hpp`; it never substitutes or
downloads another DuckDB implementation.

```bash
VORTEX_VANE_DISTRIBUTED=1 \
DUCKDB_SOURCE_DIR=/path/to/vane/external/duckdb \
DUCKDB_VERSION=v1.5.5-vane.64ed91c7e7 \
cargo build -p vortex-duckdb
```

Without that explicit mode, including under Cargo's `--all-features`, the
normal precompiled and source-build paths above are unchanged.

Vane registration crosses the Rust library boundary through one public Rust
entry point. The underlying C++ registrar remains internal, so both staticlib
and cdylib consumers use the same exported symbol.

The distributed bind protocol records each object's size plus its storage
version and/or ETag. Worker metadata checks and every subsequent range read are
pinned to that identity, so execution cannot silently switch to different
same-length contents after planning: versioned stores read the selected version
and ETag-protected stores reject an identity mismatch. A backend that provides
neither a version nor an ETag is rejected during bind; there is no path-and-size
fallback. Protocol version 2 intentionally does not decode older bind payloads.

### AddressSanitizer & ThreadSanitizer

Enable both ASAN & TSAN: `VX_DUCKDB_SAN=1`.

```bash
VX_DUCKDB_DEBUG=1 VX_DUCKDB_SAN=1 cargo build -p vortex-duckdb
```

## Environment Variables

| Variable          | Effect                          |
| ----------------- | ------------------------------- |
| `VX_DUCKDB_DEBUG` | Build from source in debug mode |
| `VX_DUCKDB_ASAN`  | Enable AddressSanitizer         |

## Running Tests

```bash
# By default, link against the precompiled DuckDB release build.
cargo test -p vortex-duckdb

# Link against the DuckDB debug build from source.
VX_DUCKDB_DEBUG=1 cargo test -p vortex-duckdb

# Link against the DuckDB debug build from source with ASAN & TSAN.
ASAN_OPTIONS=detect_container_overflow=0 VX_DUCKDB_DEBUG=1 VX_DUCKDB_SAN=1 cargo test -p vortex-duckdb
```

## Testing the extension with DuckDB

By default, our tests use a precompiled build which means you don't get an
.extension which you can load in DuckDB. If you want to test a full setup,

1. Clone [duckdb-vortex](https://github.com/vortex-data/duckdb-vortex)
   repository.

2. If there is an api difference between duckdb-vortex's duckdb submodule and
   vortex's vortex-duckdb/duckdb submodule, checkout duckdb-vortex to previous
   commit. For example, if duckdb-vortex's HEAD uses 1.6 API but vortex's HEAD
   uses 1.5.2, checkout duckdb-vortex at 8a41ee6ebd9.

3. Update duckdb-vortex's submodules. Replace vortex/ submodule by a softlink to
   your local vortex repository.
4. Inside duckdb-vortex, run make -j.

./target/release/duckdb will be a duckdb instance with vortex-duckdb already
loaded.

## Testing a custom DuckDB tag

Change `DUCKDB_VERSION` environment variable value to a preferred hash or commit
(local build), or change build.rs (for testing in CI).

## Static Index SQL

The optional Unix `index` feature registers `vortex_index_build` and
`vortex_index_search`. An extension supplies backend factories through
`index::register_index_provider_factory`; this crate does not link SPFresh.
The external `duckdb-vortex` extension's `index-spfresh` feature supplies the
`spfresh.static` provider and its qualified native build.

Construction takes an explicit ordered list of absolute local Vortex paths,
an absolute reference filename under an existing owner-managed directory,
a vector field, backend ID, and backend-owned build JSON. Files must share a
schema and the selected field must contain fixed-size Float32 vectors without
NULL rows or elements. Nullable schema markers are accepted only after checking
the actual vectors. Unsupported field types return SQL errors before canonical
conversion or generation creation, including for empty sources. File IDs are
assigned starting at one in the supplied order.

```sql
SELECT * FROM vortex_index_build(
    ['/data/a.vortex', '/data/b.vortex'], '/indexes/embedding.json',
    'embedding', 'spfresh.static',
    '{"format_version":1,"dimension":8,"head_count":64,"posting_page_limit":12,"replicas":4}'
);

SELECT rank, file_id, row_offset, distance, "row".id
FROM vortex_index_search(
    '/indexes/embedding.json', [1,2,3,4,5,6,7,8]::FLOAT[], 10
)
ORDER BY rank;
```

Build executes during query initialization, not bind or EXPLAIN. It builds a
private generation, seals it, verifies a reopened reader, rechecks the source,
and durably publishes a new reference without replacing an existing name.
The result contains `reference`, `generation`, and `rows`. Side effects are
external to DuckDB transactions; rollback does not remove a completed index.
Failed builds/imports may leave unreferenced generations for owner cleanup.

Search returns ANN candidates with one-based `rank`, physical `file_id` and
`row_offset`, squared-L2 `distance`, and a `row` struct containing all original
columns fetched through `IndexSource::take`. Use `ORDER BY rank` for SQL result
ordering. Optional `backend_options` is passed unchanged to the provider.
Each execution verifies the reference, source contents, sealed manifest and
artifacts anew. Source replacement, deletion or corruption fails; prepared
queries also reject changed reference bytes instead of silently adopting them.
Reference identity is pinned per prepared statement at its first resolved bind
and survives automatic parameter and catalog rebinding, including replacement
before the first execution when WHERE, LIMIT or projection parameters cause
DuckDB to discard the original query plan. Query vectors, `k`, and
backend options may change between executions; each reference path retains its
first identity. Prepare a new statement to accept a replacement reference.
Pins belong to the original prepared handle, including C API prepares of
`CALL` table macros and `SET VARIABLE` with index subqueries. Binding an
unexecuted relation's schema does not pin a later independent query.
Successful and failed C API prepares of `EXECUTE` and `EXPLAIN EXECUTE` release
their temporary pin borrows when planning ends. Index-bearing EXECUTE wrappers
capture their own identity even when the SQL owner has a cached plan, and rebind
before execution rather than reuse a plan borrowed from a deallocated SQL owner.

The initial path requires full coverage of a frozen local source, enabled
external access and a registered provider. SQL WHERE clauses filter returned
candidates after ANN retrieval; they do not implement filtered top-k. The
adapter does not rewrite exact SQL distance ordering, reuse native handles,
support online mutation, or serialize native state to Ray workers.
Native work runs synchronously on the DuckDB query worker.

Limits are 512 MiB of pinned encoded source bytes, 16 MiB of reference/manifest
bytes, 1 GiB per artifact, 64 KiB of backend options, at most 4096 input files,
and `1 <= k <= 10000`. Backend build limits also apply. These are input/object
budgets, not total RSS or temporary-disk quotas. Reference and index directories
are owner-managed trusted metadata, and scratch directories are private.
