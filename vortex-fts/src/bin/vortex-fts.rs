// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Command-line interface for Vortex full-text indexes.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use clap::Subcommand;
use vortex::VortexSessionDefault;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_fts::SearchIndex;
use vortex_fts::build_index;

#[derive(Parser)]
#[command(
    about = "Tantivy full-text search for immutable local Vortex files",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build a new sidecar index; the destination must not exist.
    Build {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        index: PathBuf,
        /// Comma-separated top-level UTF-8 columns.
        #[arg(long, value_delimiter = ',', required = true)]
        columns: Vec<String>,
    },
    /// Search and print JSON containing total_hits and BM25-ranked row IDs.
    Search {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        query: String,
        /// Maximum hits (0 returns only the count; maximum 10000).
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Validate an index and print its metadata as JSON.
    Inspect {
        #[arg(long)]
        index: PathBuf,
    },
}

async fn run(cli: Cli) -> VortexResult<()> {
    match cli.command {
        Command::Build {
            source,
            index,
            columns,
        } => {
            let session = VortexSession::default();
            let metadata = build_index(&session, source, index, &columns).await?;
            println!(
                "{}",
                serde_json::to_string_pretty(&metadata)
                    .map_err(|err| vortex_err!(External: err))?
            );
        }
        Command::Search {
            source,
            index,
            query,
            limit,
        } => {
            let index = SearchIndex::open(index)?;
            let results = index.search(source, &query, limit)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&results).map_err(|err| vortex_err!(External: err))?
            );
        }
        Command::Inspect { index } => {
            let index = SearchIndex::open(index)?;
            println!(
                "{}",
                serde_json::to_string_pretty(index.metadata())
                    .map_err(|err| vortex_err!(External: err))?
            );
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}
