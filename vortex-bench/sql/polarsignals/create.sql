-- Polar Signals table registration, the companion of this benchmark's query files.
--
-- The harness (vortex-bench/src/pipeline/create_sql.rs) substitutes these placeholders before
-- executing the section matching the engine under test:
--
--   {object}  TABLE or VIEW, depending on whether the engine loads the data or reads it in place
--   {dir}     absolute path (or URL) of the directory holding the requested format's files
--   {ext}     file extension of that format: parquet or vortex
--   {read}    DuckDB reader for that format: read_parquet or read_vortex
--   {format}  DataFusion STORED AS name for that format: PARQUET or VORTEX
--
-- The one table is a single stacktraces.{ext} file.
-- Statements are split on semicolons, so a comment must never contain one.

-- @engine duckdb
CREATE {object} IF NOT EXISTS stacktraces AS SELECT * FROM {read}('{dir}/stacktraces.{ext}');

-- @engine datafusion
CREATE EXTERNAL TABLE IF NOT EXISTS stacktraces STORED AS {format} LOCATION '{dir}/stacktraces.{ext}';
