// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use tokio::process::Command;
use tracing::info;

use crate::Format;
use crate::IdempotentPath;

/// Python package used to run the TPC-H generator.
///
/// `tpchgen-rs` v3 depends on Arrow 59 through `tpchgen-arrow`. Running the CLI behind a Parquet
/// boundary keeps that Arrow dependency out of the Vortex Rust workspace.
const TPCHGEN_CLI_PACKAGE: &str = "tpchgen-cli==3.0.0";
const TPCHGEN_CLI_BIN: &str = "tpchgen-cli";

/// Configuration for TPC-H data generation.
#[derive(Debug, Clone)]
pub struct TpchGenOptions {
    /// Scale factor (0.01, 0.1, 1, 10, 100, 1000).
    pub scale_factor: String,
    /// Output directory.
    pub output_dir: PathBuf,
    /// Output format.
    pub format: Format,
}

impl Default for TpchGenOptions {
    fn default() -> Self {
        Self {
            scale_factor: "1.0".to_string(),
            output_dir: "tpch".to_data_path(),
            format: Format::Parquet,
        }
    }
}

impl TpchGenOptions {
    pub fn new(scale_factor: String, output_dir: impl AsRef<Path>) -> Self {
        Self {
            scale_factor,
            output_dir: output_dir.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    pub fn with_format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }
}

/// Generate all TPC-H tables for a single scale factor.
///
/// This always generates Parquet files. Other benchmark formats are produced by reading these
/// Parquet files through the workspace Arrow version and converting them in `data-gen`.
pub async fn generate_tpch_tables(options: TpchGenOptions) -> Result<()> {
    if !matches!(options.format, Format::Parquet | Format::OnDiskDuckDB) {
        anyhow::bail!(
            "TPC-H generation only creates Parquet base data; use data-gen conversion for {}",
            options.format
        );
    }

    fs::create_dir_all(&options.output_dir)?;
    let parquet_dir = options.output_dir.join(Format::Parquet.name());
    fs::create_dir_all(&parquet_dir)?;

    let tables = [
        "nation", "region", "part", "supplier", "customer", "partsupp", "orders", "lineitem",
    ];

    for table_name in tables {
        info!(
            scale_factor = options.scale_factor,
            format = %Format::Parquet,
            table = table_name,
            "Generating TPC-H table",
        );
        generate_table_file(table_name, &options, &parquet_dir).await?;
    }

    Ok(())
}

/// Generate one Parquet file for a specific table.
async fn generate_table_file(
    table_name: &str,
    options: &TpchGenOptions,
    parquet_dir: &Path,
) -> Result<()> {
    let output_file = parquet_dir.join(format!("{table_name}_0.parquet"));
    if output_file.exists() {
        return Ok(());
    }

    let scratch_dir = parquet_dir.join(format!(".{table_name}-tpchgen-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&scratch_dir)?;

    let result = async {
        generate_table_with_cli(table_name, &options.scale_factor, &scratch_dir).await?;
        let generated_file = generated_file_path(table_name, &scratch_dir)?;
        fs::rename(&generated_file, &output_file).with_context(|| {
            format!(
                "failed to move generated TPC-H file from {} to {}",
                generated_file.display(),
                output_file.display()
            )
        })
    }
    .await;

    fs::remove_dir_all(&scratch_dir).ok();
    result
}

async fn generate_table_with_cli(
    table_name: &str,
    scale_factor: &str,
    output_dir: &Path,
) -> Result<()> {
    let mut command = tpchgen_command();
    command
        .arg("parquet")
        .arg("-s")
        .arg(scale_factor)
        .arg("--tables")
        .arg(table_name)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--no-progress")
        .arg("--quiet");

    let output = command
        .output()
        .await
        .with_context(|| format!("failed to spawn {TPCHGEN_CLI_BIN} via uvx"))?;

    if !output.status.success() {
        let status = output.status;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "{TPCHGEN_CLI_BIN} failed while generating TPC-H table {table_name}: {status}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }

    Ok(())
}

fn tpchgen_command() -> Command {
    if let Some(cli) = env::var_os("TPCHGEN_CLI") {
        Command::new(cli)
    } else {
        let mut command = Command::new("uvx");
        command.args(["--from", TPCHGEN_CLI_PACKAGE, TPCHGEN_CLI_BIN]);
        command
    }
}

fn generated_file_path(table_name: &str, scratch_dir: &Path) -> Result<PathBuf> {
    let single_file = scratch_dir.join(format!("{table_name}.parquet"));
    if single_file.exists() {
        return Ok(single_file);
    }

    anyhow::bail!(
        "expected {TPCHGEN_CLI_BIN} to write {}",
        single_file.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_file_path_matches_cli_layout() -> Result<()> {
        let scratch_dir = test_scratch_dir()?;
        let expected = scratch_dir.join("nation.parquet");
        fs::write(&expected, [])?;

        assert_eq!(generated_file_path("nation", &scratch_dir)?, expected);

        fs::remove_dir_all(scratch_dir)?;
        Ok(())
    }

    fn test_scratch_dir() -> Result<PathBuf> {
        let path = std::env::temp_dir().join(format!(
            "vortex-bench-tpchgen-test-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&path)?;
        Ok(path)
    }
}
