# Iceberg snapshot adapter

`iceberg_fts.py` provides Python `build`/`search` APIs and a CLI. It scans an explicit
Iceberg snapshot with the standard reader, materializes visible rows to Vortex,
and calls the native Tantivy SQL functions. Snapshot, schema and field IDs are
validated before search. New snapshots require rebuilding in a new directory.

All adapter code and test fixtures are in this repository. Build the native
extension from `vortex-duckdb/extension`; supply a compatible Iceberg extension.

From the repository root:

```sh
export PYTHONPATH="$PWD/vortex-fts/integrations/iceberg"
python3 -m unittest discover -s vortex-fts/integrations/iceberg/tests -v
VORTEX_EXTENSION=/absolute/vortex.duckdb_extension \
ICEBERG_EXTENSION=/absolute/iceberg.duckdb_extension \
python3 -m unittest discover -s vortex-fts/integrations/iceberg/tests -v
```

The native suite requires the matching Python `duckdb` package. It includes a
local REST fixture and bundled equality-delete data, so no other repository
checkout is needed. `ICEBERG_VORTEX_FORMAT=1` enables an additional test only for
Iceberg readers that support Vortex data files. Skipped tests are not passes.

See the [user guide](../../docs/iceberg-user-guide.md) and
[design](../../docs/iceberg-design.md).
