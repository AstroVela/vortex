# Static hnswlib indexes

Experimental, opt-in `hnswlib.static` implementation of `vortex-index`.
The generic index interfaces and DuckDB SQL adapter do not depend on this crate.
Default features contain only bundle metadata and import support.

## Native build

The `native` feature requires Linux x86_64, a C++17 compiler and a clean
hnswlib v0.9.0 checkout at `d9b3608c83d83b46c96e25088cb1d729b29dcfe9`:

```bash
export VORTEX_HNSWLIB_SOURCE=/absolute/path/to/hnswlib
cargo test -p vortex-index-hnswlib --features native
```

AVX2, FMA and F16C are required and checked before entering native code. This
initial implementation does not provide CPU dispatch or portable native files.
The upstream Apache-2.0 headers are compiled from the external checkout,
not copied or patched here.

## Contract

- Static, immutable Float32, non-null fixed-size vectors and squared L2 only.
- Initial construction explicitly attaches `HnswIndexBuilder` to the provider;
  sealing and reference publication remain the owner's responsibility.
- Required build JSON: `{"format_version":1,"dimension":128,"m":32,
  "ef_construction":200,"seed":100,"threads":1}`. Only `threads=1` is accepted
  because upstream parallel insertion races on its random number generator.
  Independent indexes can still be built concurrently.
  Construction buffers vectors and owns a native graph;
  vector/artifact limits are not a process memory quota.
- Optional query JSON: `{"ef":136}`. Empty options use `max(k,64)`, which is not
  a dataset-independent recall guarantee. `ef >= k`, `k <= 4096`, finite values,
  matching snapshots and all configured resource limits are enforced.
- Exact search, visibility restrictions, unknown parameters, deletes,
  incremental writes and unsupported metrics are rejected, not silently ignored.
- Bundles bind the complete physical row mapping, source snapshot, coverage,
  upstream revision and graph shape. Opening verifies a private artifact lease
  and validates the pinned native file structure before deserialization.
- Queries synchronize ef changes and search on the same native handle; different
  generations do not share a global query lock. Per-handle distance dispatch uses
  upstream stateless kernels without `L2Space`'s mutable global SIMD pointer.
  Blocking build/open/search must run on an appropriate blocking worker.

Defaults allow 1M rows, 512 MiB buffered input vectors and 1 GiB materialized
artifacts. These do not change DuckDB's separate default 256 MiB prepared-index
retention budget. A SIFT1M full-precision HNSW graph exceeds that cache budget;
benchmarks using an isolated 1 GiB retention candidate must label it separately.

`bench_provider` opens a sealed local reference once and records individual ANN
queries, without SQL or original-vector retrieval. Config and output paths are
its two arguments. `--barriers` enables phase markers with a stdin `continue`
acknowledgment for an external resource monitor. Timings exclude JSON emission.

## Native concurrency regression

The standalone native test exercises rejected parallel insertion, independent
builds, concurrent opens and search during build/open at 17, 128 and 129 dimensions.
The hnswlib CI job runs it with ThreadSanitizer. To run locally from the repository
root with the pinned source and a supported CPU:

```bash
work=$(mktemp -d)
c++ -std=c++17 -O1 -g -fsanitize=thread -fno-omit-frame-pointer \
  -mavx2 -mfma -mf16c -pthread -I "$VORTEX_HNSWLIB_SOURCE" \
  vortex-index-hnswlib/native/bridge.cpp \
  vortex-index-hnswlib/native/tests/concurrency.cpp -o "$work/concurrency"
TSAN_OPTIONS=halt_on_error=1:detect_deadlocks=0 "$work/concurrency" "$work"
```

Data-race detection remains enabled. The separate lock-order detector is disabled
because upstream insertion acquires its private graph locks in different orders
even on the serial control. If TSan exits with `unexpected memory mapping`, run
the test with `setarch x86_64 -R` where supported to disable ASLR for that process.
No system-wide kernel changes or race suppressions are needed.
