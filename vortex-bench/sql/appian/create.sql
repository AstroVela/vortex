-- Appian table registration, the companion of this benchmark's query files.
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
-- Each table is a single {table}.{ext} file, lowercased during data generation.
-- Statements are split on semicolons, so a comment must never contain one.

-- @engine duckdb
CREATE {object} IF NOT EXISTS addressview AS SELECT * FROM {read}('{dir}/addressview.{ext}');
CREATE {object} IF NOT EXISTS categoryview AS SELECT * FROM {read}('{dir}/categoryview.{ext}');
CREATE {object} IF NOT EXISTS creditcardview AS SELECT * FROM {read}('{dir}/creditcardview.{ext}');
CREATE {object} IF NOT EXISTS customerview AS SELECT * FROM {read}('{dir}/customerview.{ext}');
CREATE {object} IF NOT EXISTS orderitemnovelty_update AS SELECT * FROM {read}('{dir}/orderitemnovelty_update.{ext}');
CREATE {object} IF NOT EXISTS orderitemview AS SELECT * FROM {read}('{dir}/orderitemview.{ext}');
CREATE {object} IF NOT EXISTS orderview AS SELECT * FROM {read}('{dir}/orderview.{ext}');
CREATE {object} IF NOT EXISTS productview AS SELECT * FROM {read}('{dir}/productview.{ext}');
CREATE {object} IF NOT EXISTS taxrecordview AS SELECT * FROM {read}('{dir}/taxrecordview.{ext}');

-- @engine datafusion
CREATE EXTERNAL TABLE IF NOT EXISTS addressview STORED AS {format} LOCATION '{dir}/addressview.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS categoryview STORED AS {format} LOCATION '{dir}/categoryview.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS creditcardview STORED AS {format} LOCATION '{dir}/creditcardview.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS customerview STORED AS {format} LOCATION '{dir}/customerview.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS orderitemnovelty_update STORED AS {format} LOCATION '{dir}/orderitemnovelty_update.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS orderitemview STORED AS {format} LOCATION '{dir}/orderitemview.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS orderview STORED AS {format} LOCATION '{dir}/orderview.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS productview STORED AS {format} LOCATION '{dir}/productview.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS taxrecordview STORED AS {format} LOCATION '{dir}/taxrecordview.{ext}';
