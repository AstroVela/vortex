# Vortex full-text search

`vortex-fts` builds a persistent [Tantivy 0.25](https://docs.rs/tantivy/0.25.0/tantivy/)
sidecar for immutable Vortex data. It includes a multi-file `TantivyProvider` for
`vortex-index`, DuckDB SQL support (with `vortex-duckdb/extension` in this workspace), and a single-file Rust API and the
`vortex-fts` CLI: multi-column text indexing, field-qualified/phrase/Boolean
queries, BM25 top-k results, and physical row IDs for Vortex row selection.

```sh
cargo run -p vortex-fts --example quickstart
cargo run -p vortex-fts -- build --source articles.vortex --index articles.fts --columns title,body
cargo run -p vortex-fts -- search --source articles.vortex --index articles.fts --query '"full text"' --limit 10
```

The destination must not exist. Source files and completed index directories must
remain immutable during use. Every search validates a full source SHA-256 digest;
this intentionally adds a sequential source-file read and is not a low-latency
large-file search API. The implementation does not modify the Vortex file format.

- [用户文档 / User guide](https://github.com/AstroVela/vortex/pull/20#fts-user-guide)
- [设计文档 / Design](https://github.com/AstroVela/vortex/pull/20#fts-design)
- [Executable Rust example](examples/quickstart.rs)

## Development

```sh
cargo test -p vortex-fts
cargo test --doc -p vortex-fts
cargo clippy -p vortex-fts --all-targets --all-features
cargo +nightly fmt --all -- --check
```

DuckDB SQL、多文件 provider 和 Iceberg 快照适配请参阅 [用户文档](https://github.com/AstroVela/vortex/pull/20#fts-user-guide) 与 [设计文档](https://github.com/AstroVela/vortex/pull/20#fts-design)。


## Single-repository integrations

- [Loadable DuckDB extension](../vortex-duckdb/extension/README.md)
- [Iceberg snapshot adapter](integrations/iceberg/README.md)
- [Iceberg user guide](https://github.com/AstroVela/vortex/pull/20#iceberg-fts-user-guide) and [design](https://github.com/AstroVela/vortex/pull/20#iceberg-fts-design)

All integration changes are delivered in this repository; no companion PR is required.

## Tantivy prefilters for vector search

The existing multi-file `TantivyProvider` now supports full-text retrieval and
structured attribute filtering in the **same Tantivy index generation**. The
builder, document-to-`RowAddress` mapping, `IndexStore` persistence, checksums,
sealed generation and local reader leases are shared. Attributes use Tantivy
terms/postings and Boolean/range queries; there is no independent Roaring
attribute index or separate attribute artifact format.

- Backend format **1** remains readable and keeps its original text schema and
  empty build options. It also supports `FullTextIndex::filter_text` for complete
  text-match enumeration. Adding scalar fields requires a new format-2 build.
- Backend format **2** accepts serialized `TantivyBuildOptions`. `text_fields`
  retain the default text analyzer. `attribute_fields` map column names to
  `AttributeType::{String, Bool, Int, UInt, Float}` for Utf8, Bool, I64, U64 and
  finite F64 respectively. `IndexMetadata.fields` is the union of both sets.
  A string column can serve both roles, or a generation can be attributes-only.
- String attributes are raw, case-sensitive whole values. For example `tech`,
  `Tech` and `tech news` are distinct attribute terms even though the full-text
  analyzer may match all three for `tech`. Empty strings are present values.
  Null columns and null parent structs have no attribute term. Strings exceeding
  Tantivy's `MAX_TOKEN_LEN` (65,530 UTF-8 bytes) fail the build/query explicitly
  rather than silently losing an exact-match term.
- `AttributeIndex::filter` supports `Eq`, `In`, inclusive `Range`, `Exists`,
  `And` and `Or`. No type coercion is performed; wrong types, unknown fields and
  reversed bounds fail. Floats must be finite; signed zeros compare equal.
  Empty AND matches all visible documents, including null rows; empty OR/IN
  matches none. Queries allow at most 64 nesting levels and 4,096 clauses/values.
- `FullTextIndex::filter_text` and attribute filters enumerate **all** matches
  with scoring disabled, intersect caller visibility and return a `RowFilter`.
  They do not call BM25 top-k or apply the full-text 10,000-hit result limit.
  Both reject stale snapshots and partial file coverage. Ranked BM25 search is
  still available through `search_text`.

```rust,ignore
let config = TantivyBuildOptions {
    text_fields: vec!["title".into()],
    attribute_fields: [
        ("category".into(), AttributeType::String),
        ("price".into(), AttributeType::Float),
    ].into(),
};
// Build with backend = TANTIVY_ID, backend_version = 2,
// fields = ["title", "category", "price"], and JSON-encoded config as backend_options.
// Seal and reopen with the same existing IndexStore / TantivyProvider APIs.
let text_rows = opened.as_full_text().unwrap()
    .filter_text("database", &visibility).await?;
let allowed = opened.as_attributes().unwrap().filter(
    &AttributePredicate::Eq("category".into(), AttributeValue::String("tech".into())),
    &text_rows,
).await?;
let hits = vectors.search(&query, &options, &allowed).await?;
```

The ANN side translates `RowAddress` to its own native-ID mask and rejects
unmatched candidates before vector distance calculation. This follows the
external metadata-filter composition used by OpenData Vector, while reusing
this repository's existing Tantivy backend. It is not an automatic SQL planner
or an online LIRE update implementation. Full-text-only CLI/single-file APIs
remain unchanged; the new capability is on the multi-file provider API.

The existing provider limits still apply: 16 distinct source fields, one million
indexed visible rows and 1 GiB of native index artifacts. Match collection
materializes a `BTreeSet<RowAddress>` and costs O(matches) memory; these limits
are not an overall RSS guarantee. Source adapters must scan all visible rows
in declared coverage. Publish the Tantivy and vector generations against the
same immutable snapshot; changing attributes or data requires rebuilding.
