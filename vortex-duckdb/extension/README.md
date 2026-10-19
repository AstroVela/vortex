# In-workspace Tantivy DuckDB extension

This package builds a loadable DuckDB extension from the current Vortex workspace,
registering Vortex IO plus `vortex_fts_build` and `vortex_fts_search`. It does not
depend on a fork or companion PR in another repository.

```sh
cargo build -p vortex-duckdb-fts-extension
# Example: DuckDB v1.5.5 on macOS arm64; use .so and the matching platform on Linux.
python3 vortex-duckdb/extension/package.py \
  --library target/debug/libvortex_fts_extension.dylib \
  --output /absolute/output/vortex.duckdb_extension \
  --duckdb-version v1.5.5 --platform osx_arm64
```

Use `SELECT version()` and `PRAGMA platform` on the target DuckDB. The output's
parent must exist; the output must not. The output name determines the exported
`vortex_duckdb_cpp_init` entrypoint. This unsigned development build requires
`duckdb -unsigned` or `allow_unsigned_extensions=true` in Python. Retain the linked
DuckDB library in the build directory; the footer is not a dependency bundler.

```sql
LOAD '/absolute/output/vortex.duckdb_extension';
SELECT function_name FROM duckdb_functions() WHERE function_name LIKE 'vortex_fts%';
```

The default dependency build downloads matching DuckDB 1.5.5 headers/library.
Custom native builds must provide matching source/library/linker configuration;
this wrapper does not qualify arbitrary DuckDB/Vane ABIs. Do not load another
Vortex extension into the same database.

See the [user guide](https://github.com/AstroVela/vortex/pull/20#fts-user-guide) and
[design](https://github.com/AstroVela/vortex/pull/20#fts-design).
