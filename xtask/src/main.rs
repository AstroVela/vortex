// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod generate_editions_docs;
mod generate_fbs;
mod generate_proto;

use clap::Parser;

use crate::generate_editions_docs::generate_editions_docs;
use crate::generate_fbs::generate_fbs;
use crate::generate_proto::generate_proto;

#[derive(clap::Parser)]
struct Xtask {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
#[expect(
    clippy::enum_variant_names,
    reason = "variants mirror the generate-* subcommand names"
)]
enum Commands {
    /// Subcommand to regenerate flatbuffers language bindings for the Rust project.
    #[command(name = "generate-fbs")]
    GenerateFlatbuffers,
    /// Subcommand to regenerate protobuf language bindings for the Rust project.
    #[command(name = "generate-proto")]
    GenerateProto,
    /// Subcommand to regenerate the edition registry in `docs/specs/editions.md` from the
    /// edition declarations in the `vortex` crate.
    #[command(name = "generate-editions-docs")]
    GenerateEditionsDocs {
        /// Cargo profile to build the manifest emitter with (defaults to dev).
        #[arg(long)]
        profile: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Xtask::parse();
    match cli.command {
        Commands::GenerateFlatbuffers => generate_fbs()?,
        Commands::GenerateProto => generate_proto()?,
        Commands::GenerateEditionsDocs { profile } => generate_editions_docs(profile)?,
    }
    Ok(())
}
