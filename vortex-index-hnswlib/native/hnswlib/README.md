# Payload graph extension

`payload_graph.h` extends the pinned hnswlib v0.9.0 headers in this repository.
It uses hnswlib's internal node IDs, distance functions and visited-list pool;
the C bridge remains responsible only for ownership and argument translation.
The base index must be closed and immutable. This is a Vortex-maintained
extension, not an upstream hnswlib feature.

Each attribute block builds a single-level NSW graph over borrowed base vectors.
`payload_m` bounds forward neighbors to M and reverse neighbors to 2M **per
block**, not the total degree across overlapping blocks. The builder merges and
deduplicates edges, removes links already present in the base graph and rejects
configured block, membership, total-edge or per-node degree limit violations.
It does not silently truncate blocks. A failed builder must be discarded.

The immutable CSR overlay is logically part of level zero. Filtered search
descends upper levels normally, then visits base and payload neighbors with a
shared visited list. Rejected labels may navigate, but never enter the result
heap. The algorithm has no ACORN expansion or automatic exact-scan path.
Unfiltered queries use the original graph search. Neither graph changes during
search; callers serialize changes to `ef` with queries, as the Vortex bridge does.

The approach follows Qdrant's
[attribute block construction](https://github.com/qdrant/qdrant/blob/222b2e44ab1c3f17e8246396ead6558e45041df7/lib/segment/src/index/hnsw_index/hnsw/build/additional_links/block.rs)
and level-zero merge, with no graph per predicate combination.

Run the native tests against the pinned checkout:

```sh
c++ -std=c++17 -O1 -g -DNO_MANUAL_VECTORIZATION -fsanitize=address,undefined -fno-omit-frame-pointer \
  -pthread -I "$VORTEX_HNSWLIB_SOURCE" \
  vortex-index-hnswlib/native/tests/payload_graph.cpp -o /tmp/payload-test
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 /tmp/payload-test
```

Tests cover block isolation, overlapping blocks, unchanged base adjacency,
non-identity labels, exact-oracle parity at exhaustive ef, empty and singleton
filters, rejected navigation nodes, payload-only paths and invalid CSR/budgets.
This small deterministic fixture is a correctness test, not a production recall
or latency qualification.

The sanitizer run uses upstream scalar kernels because v0.9.0 SSE construction
prefetch reads a neighbor past capacity (`hnswalg.h:277`). CI also runs the same
fixture with AVX2 kernels enabled. The new extension is fully instrumented.
