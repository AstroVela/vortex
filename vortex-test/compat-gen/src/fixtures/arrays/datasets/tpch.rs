// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use arrow_array::RecordBatch;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_arrow::FromArrowArray;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::fixtures::DatasetFixture;

const SCALE_FACTOR: &str = "0.01";
const BATCH_SIZE: usize = 65_536;
const TPCHGEN_CLI_PACKAGE: &str = "tpchgen-cli==3.0.0";
const TPCHGEN_CLI_BIN: &str = "tpchgen-cli";

fn cached_tpch_parquet(table: &str) -> VortexResult<PathBuf> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = crate_dir.join("data").join("tpch");
    let dest = data_dir.join(format!("{table}.parquet"));

    if dest.exists() {
        return Ok(dest);
    }

    fs::create_dir_all(&data_dir).map_err(|e| vortex_err!("failed to create data dir: {e}"))?;

    let scratch = tempfile::Builder::new()
        .prefix(&format!("{table}-"))
        .tempdir_in(&data_dir)
        .map_err(|e| vortex_err!("failed to create TPC-H scratch dir: {e}"))?;

    generate_tpch_parquet(table, scratch.path())?;

    let generated = scratch.path().join(format!("{table}.parquet"));
    if !generated.exists() {
        return Err(vortex_err!(
            "expected {TPCHGEN_CLI_BIN} to write {}",
            generated.display()
        ));
    }

    fs::rename(&generated, &dest).map_err(|e| {
        vortex_err!(
            "failed to move generated TPC-H parquet from {} to {}: {e}",
            generated.display(),
            dest.display()
        )
    })?;

    Ok(dest)
}

fn generate_tpch_parquet(table: &str, output_dir: &Path) -> VortexResult<()> {
    let output = tpchgen_command()
        .arg("parquet")
        .arg("-s")
        .arg(SCALE_FACTOR)
        .arg("--tables")
        .arg(table)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--no-progress")
        .arg("--quiet")
        .output()
        .map_err(|e| vortex_err!("failed to spawn {TPCHGEN_CLI_BIN} via uvx: {e}"))?;

    if !output.status.success() {
        let status = output.status;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(vortex_err!(
            "{TPCHGEN_CLI_BIN} failed while generating TPC-H table {table}: {status}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        ));
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

fn collect_parquet_as_vortex(path: PathBuf) -> VortexResult<ArrayRef> {
    let file_bytes = fs::read(&path)
        .map_err(|e| vortex_err!("failed to read cached parquet at {}: {e}", path.display()))?;
    let bytes = Bytes::from(file_bytes);

    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .map_err(|e| vortex_err!("failed to open parquet: {e}"))?
        .with_batch_size(BATCH_SIZE)
        .build()
        .map_err(|e| vortex_err!("failed to build parquet reader: {e}"))?;

    let batches: Vec<RecordBatch> = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| vortex_err!("failed to read parquet batches: {e}"))?;

    Ok(ChunkedArray::from_iter(
        batches
            .into_iter()
            .map(|batch| ArrayRef::from_arrow(batch, false))
            .collect::<VortexResult<Vec<_>>>()?,
    )
    .into_array())
}

struct TpchLineitemFixture;

impl DatasetFixture for TpchLineitemFixture {
    fn name(&self) -> &str {
        "tpch_lineitem"
    }

    fn description(&self) -> &str {
        "TPC-H lineitem table at scale factor 0.01 with decimals, dates, and strings"
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        collect_parquet_as_vortex(cached_tpch_parquet("lineitem")?)
    }
}

struct TpchOrdersFixture;

impl DatasetFixture for TpchOrdersFixture {
    fn name(&self) -> &str {
        "tpch_orders"
    }

    fn description(&self) -> &str {
        "TPC-H orders table at scale factor 0.01 with decimals, dates, and strings"
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        collect_parquet_as_vortex(cached_tpch_parquet("orders")?)
    }
}

pub fn fixtures() -> Vec<Box<dyn DatasetFixture>> {
    vec![Box::new(TpchLineitemFixture), Box::new(TpchOrdersFixture)]
}
