-- TPC-H table registration, the companion of the q*.sql query files in this directory.
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
-- Every table lives in one shared directory, so each is scoped to its own {table}_*.{ext} files.
-- Statements are split on semicolons, so a comment must never contain one.

-- @engine duckdb
CREATE {object} IF NOT EXISTS customer AS SELECT * FROM {read}('{dir}/customer_*.{ext}');
CREATE {object} IF NOT EXISTS lineitem AS SELECT * FROM {read}('{dir}/lineitem_*.{ext}');
CREATE {object} IF NOT EXISTS nation AS SELECT * FROM {read}('{dir}/nation_*.{ext}');
CREATE {object} IF NOT EXISTS orders AS SELECT * FROM {read}('{dir}/orders_*.{ext}');
CREATE {object} IF NOT EXISTS part AS SELECT * FROM {read}('{dir}/part_*.{ext}');
CREATE {object} IF NOT EXISTS partsupp AS SELECT * FROM {read}('{dir}/partsupp_*.{ext}');
CREATE {object} IF NOT EXISTS region AS SELECT * FROM {read}('{dir}/region_*.{ext}');
CREATE {object} IF NOT EXISTS supplier AS SELECT * FROM {read}('{dir}/supplier_*.{ext}');

-- @engine datafusion
CREATE EXTERNAL TABLE IF NOT EXISTS customer STORED AS {format} LOCATION '{dir}/customer_*.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS lineitem STORED AS {format} LOCATION '{dir}/lineitem_*.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS nation STORED AS {format} LOCATION '{dir}/nation_*.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS orders STORED AS {format} LOCATION '{dir}/orders_*.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS part STORED AS {format} LOCATION '{dir}/part_*.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS partsupp STORED AS {format} LOCATION '{dir}/partsupp_*.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS region STORED AS {format} LOCATION '{dir}/region_*.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS supplier STORED AS {format} LOCATION '{dir}/supplier_*.{ext}';
