// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark data generation binary.
//!
//! Stage 1 of the benchmark pipeline (see [`vortex_bench::pipeline`]), run ahead of time so the
//! benchmark binaries find their data already present. The same [`vortex_bench::generate_data`]
//! call runs inside those binaries, so running this first is an optimization, not a requirement.

use clap::Parser;
use clap::value_parser;
use vortex_bench::BenchmarkArg;
use vortex_bench::Format;
use vortex_bench::LogFormat;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::create_benchmark;
use vortex_bench::generate_data;
use vortex_bench::pipeline::generate_duckdb_database;
use vortex_bench::setup_logging_and_tracing_with_format;

#[derive(Parser)]
#[command(name = "bench-data-gen")]
#[command(about = "Generate benchmark data for all requested formats")]
struct Args {
    #[arg(value_enum)]
    benchmark: BenchmarkArg,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long)]
    tracing: bool,

    /// Format for the primary stderr log sink. `text` is the default human-readable format;
    /// `json` emits one JSON object per event, suitable for piping into `jq`.
    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    log_format: LogFormat,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Format))]
    formats: Vec<Format>,

    #[arg(long = "opt", value_delimiter = ',', value_parser = value_parser!(Opt))]
    options: Vec<Opt>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let opts = Opts::from(args.options);

    setup_logging_and_tracing_with_format(args.verbose, args.tracing, args.log_format)?;

    let benchmark = create_benchmark(args.benchmark, &opts)?;

    generate_data(&*benchmark, args.formats.iter().copied()).await?;

    // DuckDB's catalog is the only one that persists across processes, so its tables are loaded
    // here rather than at registration time.
    if args.formats.contains(&Format::OnDiskDuckDB) && benchmark.data_url().scheme() == "file" {
        let base_path = benchmark
            .data_url()
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", benchmark.data_url()))?;
        generate_duckdb_database(&*benchmark, &base_path)?;
    }

    Ok(())
}
