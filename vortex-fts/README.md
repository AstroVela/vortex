# Vortex full-text search

`vortex-fts` builds a persistent [Tantivy 0.25](https://docs.rs/tantivy/0.25.0/tantivy/)
sidecar for immutable Vortex data. It includes a multi-file `TantivyProvider` for
`vortex-index`, DuckDB SQL support (with the companion extension), and a single-file Rust API and the
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

- [用户文档 / User guide](docs/user-guide.md)
- [设计文档 / Design](docs/design.md)
- [Executable Rust example](examples/quickstart.rs)

## Development

```sh
cargo test -p vortex-fts
cargo test --doc -p vortex-fts
cargo clippy -p vortex-fts --all-targets --all-features
cargo +nightly fmt --all -- --check
```

DuckDB SQL、多文件 provider 和 Iceberg 快照适配请参阅 [用户文档](docs/user-guide.md#duckdb多文件-sql-检索) 与 [设计文档](docs/design.md#通用索引能力与多文件后端)。
