// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Star Schema Benchmark data generation.
//!
//! There is no Rust SSB generator, and SSB is *not* derivable from TPC-H output — its
//! cardinalities differ (`customer` is SF x 30k rather than SF x 150k, `supplier` SF x 2k rather
//! than SF x 10k, `part` is `200000 * floor(1 + log2(SF))`, and `dwdate` is a 2557-row calendar
//! that TPC-H has no analogue for). So we build the reference C `dbgen` from source, run it, and
//! convert its pipe-delimited `.tbl` output into Parquet with the `duckdb` CLI — the same
//! shell-out the Appian suite uses.
//!
//! ## Which `dbgen`
//!
//! SSB has no official upstream, only a tree of unsynchronized `dbgen` forks. We pin
//! [`eyalroz/ssb-dbgen`][fork], which unifies them and is the fork ClickHouse's own SSB docs
//! use. The alternatives are not interchangeable: `lemire/StarSchemaBenchmark`, for instance,
//! carries a `/*bug!*/`-annotated `gen_city()` that draws from random stream 98 when
//! `MAX_STREAM` is 47 — `UnifInt()` clamps the stream for the *value* but `dss_random()` then
//! increments `Seed[98].usage` out of bounds, corrupting an adjacent global. That is a `SIGBUS`
//! at SF 10 and, below the crash threshold, data whose contents depend on global layout. The
//! pinned fork fixes it properly (a real `P_CITY_SD` stream, `MAX_STREAM` raised to 49, and an
//! assert in `dss_random()`), and also emits the full 2557-day calendar rather than 2556.
//!
//! ## Prerequisites
//!
//! A C compiler, `cmake`, `git`, and the `duckdb` CLI (the last already required by Appian).
//!
//! [fork]: https://github.com/eyalroz/ssb-dbgen

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::bail;
use itertools::Itertools;
use tracing::info;

use crate::Format;
use crate::utils::file::data_dir;

/// Maintained fork of the SSB `dbgen` published alongside O'Neil, O'Neil & Chen,
/// *The Star Schema Benchmark, Revision 3* (2009). See the module docs for why this fork.
const DBGEN_REPO: &str = "https://github.com/eyalroz/ssb-dbgen.git";

/// Pinned so the generated data is reproducible across runs and machines.
const DBGEN_REV: &str = "ae1e254aa4d603d8ef1f44078e5abed011634b23";

/// One SSB table: the registered name, the `.tbl` `dbgen` writes it to, and its columns as
/// `(name, DuckDB type)` in file order.
pub struct Table {
    /// Name registered with the query engines, and the Parquet file stem.
    pub name: &'static str,
    /// Stem of the `.tbl` file `dbgen` emits. Only `dwdate` differs from [`Table::name`].
    tbl_stem: &'static str,
    columns: &'static [(&'static str, &'static str)],
}

impl Table {
    /// `read_csv` column spec, plus a trailing `dummy` for the line-terminating `|` that every
    /// `.tbl` row carries. The `dummy` is dropped again by [`Table::copy_stmt`].
    fn read_csv_columns(&self) -> String {
        self.columns
            .iter()
            .map(|(name, ty)| format!("'{name}':'{ty}'"))
            .chain(std::iter::once("'dummy':'VARCHAR'".to_string()))
            .join(",")
    }

    fn copy_stmt(&self, tbl_dir: &Path, parquet_dir: &Path) -> String {
        let tbl = tbl_dir.join(format!("{}.tbl", self.tbl_stem));
        let parquet = parquet_dir.join(format!("{}.parquet", self.name));
        format!(
            "COPY (SELECT * EXCLUDE (dummy) FROM read_csv('{}', delim='|', header=false, \
             columns={{{}}})) TO '{}' (FORMAT PARQUET);\n",
            tbl.display(),
            self.read_csv_columns(),
            parquet.display(),
        )
    }
}

/// The five SSB tables, typed to match the reference DDL in the SSB paper. Every numeric column
/// is a 32-bit `INTEGER`: at SF 100 the widest values are `lo_orderkey` (6e8) and
/// `lo_ordtotalprice` (~5e7), both comfortably inside `i32`.
pub const TABLES: &[Table] = &[
    Table {
        name: "customer",
        tbl_stem: "customer",
        columns: &[
            ("c_custkey", "INTEGER"),
            ("c_name", "VARCHAR"),
            ("c_address", "VARCHAR"),
            ("c_city", "VARCHAR"),
            ("c_nation", "VARCHAR"),
            ("c_region", "VARCHAR"),
            ("c_phone", "VARCHAR"),
            ("c_mktsegment", "VARCHAR"),
        ],
    },
    Table {
        name: "supplier",
        tbl_stem: "supplier",
        columns: &[
            ("s_suppkey", "INTEGER"),
            ("s_name", "VARCHAR"),
            ("s_address", "VARCHAR"),
            ("s_city", "VARCHAR"),
            ("s_nation", "VARCHAR"),
            ("s_region", "VARCHAR"),
            ("s_phone", "VARCHAR"),
        ],
    },
    Table {
        name: "part",
        tbl_stem: "part",
        columns: &[
            ("p_partkey", "INTEGER"),
            ("p_name", "VARCHAR"),
            ("p_mfgr", "VARCHAR"),
            ("p_category", "VARCHAR"),
            ("p_brand1", "VARCHAR"),
            ("p_color", "VARCHAR"),
            ("p_type", "VARCHAR"),
            ("p_size", "INTEGER"),
            ("p_container", "VARCHAR"),
        ],
    },
    // `date` is a reserved word in both DataFusion's and DuckDB's parsers, so the table is
    // registered as `dwdate` — the name the reference SSB load scripts use for the same reason.
    Table {
        name: "dwdate",
        tbl_stem: "date",
        columns: &[
            ("d_datekey", "INTEGER"),
            ("d_date", "VARCHAR"),
            ("d_dayofweek", "VARCHAR"),
            ("d_month", "VARCHAR"),
            ("d_year", "INTEGER"),
            ("d_yearmonthnum", "INTEGER"),
            ("d_yearmonth", "VARCHAR"),
            ("d_daynuminweek", "INTEGER"),
            ("d_daynuminmonth", "INTEGER"),
            ("d_daynuminyear", "INTEGER"),
            ("d_monthnuminyear", "INTEGER"),
            ("d_weeknuminyear", "INTEGER"),
            ("d_sellingseason", "VARCHAR"),
            ("d_lastdayinweekfl", "INTEGER"),
            ("d_lastdayinmonthfl", "INTEGER"),
            ("d_holidayfl", "INTEGER"),
            ("d_weekdayfl", "INTEGER"),
        ],
    },
    Table {
        name: "lineorder",
        tbl_stem: "lineorder",
        columns: &[
            ("lo_orderkey", "INTEGER"),
            ("lo_linenumber", "INTEGER"),
            ("lo_custkey", "INTEGER"),
            ("lo_partkey", "INTEGER"),
            ("lo_suppkey", "INTEGER"),
            ("lo_orderdate", "INTEGER"),
            ("lo_orderpriority", "VARCHAR"),
            ("lo_shippriority", "VARCHAR"),
            ("lo_quantity", "INTEGER"),
            ("lo_extendedprice", "INTEGER"),
            ("lo_ordtotalprice", "INTEGER"),
            ("lo_discount", "INTEGER"),
            ("lo_revenue", "INTEGER"),
            ("lo_supplycost", "INTEGER"),
            ("lo_tax", "INTEGER"),
            ("lo_commitdate", "INTEGER"),
            ("lo_shipmode", "VARCHAR"),
        ],
    },
];

/// Generate the SSB Parquet base data for `scale_factor` under `base_dir/parquet/`.
///
/// Idempotent: returns immediately once every table's Parquet is in place. The `.tbl`
/// intermediates are deleted after conversion (SF 10 alone is ~6.5 GB of text).
pub fn generate_tables(scale_factor: &str, base_dir: &Path) -> anyhow::Result<()> {
    let parquet_dir = base_dir.join(Format::Parquet.name());
    fs::create_dir_all(&parquet_dir)?;

    if TABLES
        .iter()
        .all(|t| parquet_dir.join(format!("{}.parquet", t.name)).exists())
    {
        info!(
            "ssb: {} Parquet shards already present in {}",
            TABLES.len(),
            parquet_dir.display(),
        );
        return Ok(());
    }

    let tbl_dir = base_dir.join("tbl");
    write_tbl_files(scale_factor, &tbl_dir)?;
    convert_tbl_to_parquet(&tbl_dir, &parquet_dir)?;

    fs::remove_dir_all(&tbl_dir)?;
    info!(
        "ssb base data generated in {} ({} Parquet shards)",
        parquet_dir.display(),
        TABLES.len(),
    );
    Ok(())
}

/// Run `dbgen` into `tbl_dir`, then verify it actually produced every table.
fn write_tbl_files(scale_factor: &str, tbl_dir: &Path) -> anyhow::Result<()> {
    let dbgen = build_dbgen()?;
    fs::create_dir_all(tbl_dir)?;

    // `dbgen` reads `dists.dss` from `DSS_CONFIG` and writes its `.tbl` files into `DSS_PATH`,
    // so neither needs to be the process working directory. Omitting `-T` generates every
    // table (this fork has no `-T a`).
    info!(scale_factor, "ssb: generating .tbl files with dbgen");
    run(Command::new(&dbgen)
        .env("DSS_CONFIG", dbgen_dir())
        .env("DSS_PATH", tbl_dir)
        .args(["-s", scale_factor, "-f"]))?;

    // `dbgen` exits 0 even when it rejects its arguments, so check its output rather than
    // its status.
    for table in TABLES {
        let tbl = tbl_dir.join(format!("{}.tbl", table.tbl_stem));
        if !tbl.exists() {
            bail!("ssb: dbgen did not produce {}", tbl.display());
        }
    }
    Ok(())
}

/// Convert every `.tbl` into its Parquet counterpart in one `duckdb` invocation.
fn convert_tbl_to_parquet(tbl_dir: &Path, parquet_dir: &Path) -> anyhow::Result<()> {
    info!("ssb: converting .tbl files to Parquet");
    let script = TABLES
        .iter()
        .map(|t| t.copy_stmt(tbl_dir, parquet_dir))
        .join("");
    run(Command::new("duckdb").arg("-c").arg(&script))
}

/// Checkout of the pinned upstream generator, shared by every scale factor.
fn dbgen_dir() -> PathBuf {
    data_dir().join("ssb-dbgen")
}

/// Clone (once) and build the pinned `dbgen`, returning the path to the binary. Idempotent.
fn build_dbgen() -> anyhow::Result<PathBuf> {
    let dir = dbgen_dir();
    let build_dir = dir.join("build");
    let binary = build_dir.join("dbgen");
    if binary.exists() {
        return Ok(binary);
    }

    if !dir.join(".git").exists() {
        if let Some(parent) = dir.parent() {
            fs::create_dir_all(parent)?;
        }
        info!("ssb: cloning {DBGEN_REPO} at {DBGEN_REV}");
        run(Command::new("git").args(["clone", DBGEN_REPO]).arg(&dir))?;
    }
    run(Command::new("git")
        .arg("-C")
        .arg(&dir)
        .args(["checkout", "--quiet", DBGEN_REV]))?;

    info!("ssb: building dbgen");
    run(Command::new("cmake")
        .arg("-S")
        .arg(&dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release"))?;
    run(Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .args(["--target", "dbgen"]))?;

    if !binary.exists() {
        bail!("ssb: cmake succeeded but {} is missing", binary.display());
    }
    Ok(binary)
}

fn run(command: &mut Command) -> anyhow::Result<()> {
    let program = format!("{:?}", command.get_program());
    let output = command
        .output()
        .with_context(|| format!("ssb: failed to spawn {program}"))?;
    if !output.status.success() {
        bail!(
            "ssb: {program} failed ({}): stdout={:?} stderr={:?}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::TABLES;
    use crate::datasets::SSB_TABLES;

    /// `datasets::SSB_TABLES` drives table registration while [`TABLES`] drives generation; a
    /// divergence would silently register a table nothing writes (or vice versa).
    #[test]
    fn table_lists_agree() {
        let generated = TABLES.iter().map(|t| t.name).collect::<Vec<_>>();
        assert_eq!(generated.as_slice(), SSB_TABLES);
    }
}
