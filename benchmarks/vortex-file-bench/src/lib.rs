// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File-level benchmarks for Vortex and other columnar formats.
//!
//! Both suites in this crate measure a whole file rather than a query: how big it is, how long
//! it takes to write and read back, and how long it takes to fetch individual rows out of it.
//! Neither one loads a query engine, which is what separates them from the SQL benchmarks.
//!
//! - [`compress`] writes each dataset to every format under test and reads it back, reporting
//!   encode time, decode time, on-disk size, and cross-format ratios.
//! - [`random_access`] fetches scattered rows by index, reporting take latency per access
//!   pattern and file-handle mode.

use std::path::PathBuf;

use clap::Args;
use vortex_bench::LogFormat;
use vortex_bench::display::DisplayFormat;

pub mod compress;
pub mod random_access;

/// Options shared by every suite in this crate.
///
/// Flattened into each subcommand rather than declared globally so that flag order matches the
/// single-suite binaries this crate replaced: `vortex-file-bench <suite> -d gh-json -o out.json`.
#[derive(Args, Debug, Clone)]
pub struct CommonArgs {
    /// Output format: `table` for humans, `gh-json` for machine-readable JSON.
    #[arg(short, long, default_value_t, value_enum)]
    pub display_format: DisplayFormat,

    /// Write the primary result output to this path instead of the default.
    #[arg(short, long)]
    pub output_path: Option<PathBuf>,

    /// Additionally write benchmark ingest JSONL records to this path.
    #[arg(long = "ingest-jsonl")]
    pub ingest_output: Option<PathBuf>,

    /// Enable verbose (debug-level) logging.
    #[arg(short, long)]
    pub verbose: bool,

    /// Enable span tracing output.
    #[arg(long)]
    pub tracing: bool,

    /// Format for the primary stderr log sink. `text` is the default human-readable format;
    /// `json` emits one JSON object per event, suitable for piping into `jq`.
    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    pub log_format: LogFormat,
}
