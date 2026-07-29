-- TPC-DS table registration, the companion of this benchmark's query files.
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
-- Each table is a single {table}.{ext} file in the format directory.
-- Statements are split on semicolons, so a comment must never contain one.

-- @engine duckdb
CREATE {object} IF NOT EXISTS call_center AS SELECT * FROM {read}('{dir}/call_center.{ext}');
CREATE {object} IF NOT EXISTS catalog_page AS SELECT * FROM {read}('{dir}/catalog_page.{ext}');
CREATE {object} IF NOT EXISTS catalog_returns AS SELECT * FROM {read}('{dir}/catalog_returns.{ext}');
CREATE {object} IF NOT EXISTS catalog_sales AS SELECT * FROM {read}('{dir}/catalog_sales.{ext}');
CREATE {object} IF NOT EXISTS customer AS SELECT * FROM {read}('{dir}/customer.{ext}');
CREATE {object} IF NOT EXISTS customer_address AS SELECT * FROM {read}('{dir}/customer_address.{ext}');
CREATE {object} IF NOT EXISTS customer_demographics AS SELECT * FROM {read}('{dir}/customer_demographics.{ext}');
CREATE {object} IF NOT EXISTS date_dim AS SELECT * FROM {read}('{dir}/date_dim.{ext}');
CREATE {object} IF NOT EXISTS household_demographics AS SELECT * FROM {read}('{dir}/household_demographics.{ext}');
CREATE {object} IF NOT EXISTS income_band AS SELECT * FROM {read}('{dir}/income_band.{ext}');
CREATE {object} IF NOT EXISTS inventory AS SELECT * FROM {read}('{dir}/inventory.{ext}');
CREATE {object} IF NOT EXISTS item AS SELECT * FROM {read}('{dir}/item.{ext}');
CREATE {object} IF NOT EXISTS promotion AS SELECT * FROM {read}('{dir}/promotion.{ext}');
CREATE {object} IF NOT EXISTS reason AS SELECT * FROM {read}('{dir}/reason.{ext}');
CREATE {object} IF NOT EXISTS ship_mode AS SELECT * FROM {read}('{dir}/ship_mode.{ext}');
CREATE {object} IF NOT EXISTS store AS SELECT * FROM {read}('{dir}/store.{ext}');
CREATE {object} IF NOT EXISTS store_returns AS SELECT * FROM {read}('{dir}/store_returns.{ext}');
CREATE {object} IF NOT EXISTS store_sales AS SELECT * FROM {read}('{dir}/store_sales.{ext}');
CREATE {object} IF NOT EXISTS time_dim AS SELECT * FROM {read}('{dir}/time_dim.{ext}');
CREATE {object} IF NOT EXISTS warehouse AS SELECT * FROM {read}('{dir}/warehouse.{ext}');
CREATE {object} IF NOT EXISTS web_page AS SELECT * FROM {read}('{dir}/web_page.{ext}');
CREATE {object} IF NOT EXISTS web_returns AS SELECT * FROM {read}('{dir}/web_returns.{ext}');
CREATE {object} IF NOT EXISTS web_sales AS SELECT * FROM {read}('{dir}/web_sales.{ext}');
CREATE {object} IF NOT EXISTS web_site AS SELECT * FROM {read}('{dir}/web_site.{ext}');

-- @engine datafusion
CREATE EXTERNAL TABLE IF NOT EXISTS call_center STORED AS {format} LOCATION '{dir}/call_center.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS catalog_page STORED AS {format} LOCATION '{dir}/catalog_page.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS catalog_returns STORED AS {format} LOCATION '{dir}/catalog_returns.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS catalog_sales STORED AS {format} LOCATION '{dir}/catalog_sales.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS customer STORED AS {format} LOCATION '{dir}/customer.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS customer_address STORED AS {format} LOCATION '{dir}/customer_address.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS customer_demographics STORED AS {format} LOCATION '{dir}/customer_demographics.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS date_dim STORED AS {format} LOCATION '{dir}/date_dim.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS household_demographics STORED AS {format} LOCATION '{dir}/household_demographics.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS income_band STORED AS {format} LOCATION '{dir}/income_band.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS inventory STORED AS {format} LOCATION '{dir}/inventory.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS item STORED AS {format} LOCATION '{dir}/item.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS promotion STORED AS {format} LOCATION '{dir}/promotion.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS reason STORED AS {format} LOCATION '{dir}/reason.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS ship_mode STORED AS {format} LOCATION '{dir}/ship_mode.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS store STORED AS {format} LOCATION '{dir}/store.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS store_returns STORED AS {format} LOCATION '{dir}/store_returns.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS store_sales STORED AS {format} LOCATION '{dir}/store_sales.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS time_dim STORED AS {format} LOCATION '{dir}/time_dim.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS warehouse STORED AS {format} LOCATION '{dir}/warehouse.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS web_page STORED AS {format} LOCATION '{dir}/web_page.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS web_returns STORED AS {format} LOCATION '{dir}/web_returns.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS web_sales STORED AS {format} LOCATION '{dir}/web_sales.{ext}';
CREATE EXTERNAL TABLE IF NOT EXISTS web_site STORED AS {format} LOCATION '{dir}/web_site.{ext}';
