// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use clap::Parser;
use clap::Subcommand;
use vortex_bench::setup_logging_and_tracing_with_format;
use vortex_file_bench::CommonArgs;
use vortex_file_bench::compress;
use vortex_file_bench::random_access;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    suite: Suite,
}

#[derive(Subcommand, Debug)]
enum Suite {
    /// Compression and decompression timing, on-disk size, and cross-format ratios.
    Compress(Box<compress::CompressArgs>),
    /// Point-lookup latency for scattered row indices.
    RandomAccess(Box<random_access::RandomAccessArgs>),
}

impl Suite {
    fn common(&self) -> &CommonArgs {
        match self {
            Suite::Compress(args) => &args.common,
            Suite::RandomAccess(args) => &args.common,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let common = args.suite.common();
    setup_logging_and_tracing_with_format(common.verbose, common.tracing, common.log_format)?;

    match args.suite {
        Suite::Compress(args) => compress::run(*args).await,
        Suite::RandomAccess(args) => random_access::run(*args).await,
    }
}
