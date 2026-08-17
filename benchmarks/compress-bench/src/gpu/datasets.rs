// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Large full-table datasets for opt-in GPU decompression runs.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use async_trait::async_trait;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex_bench::Benchmark;
use vortex_bench::Format;
use vortex_bench::clickbench::ClickBenchBenchmark;
use vortex_bench::clickbench::Flavor;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::datasets::Dataset;
use vortex_bench::fineweb::FinewebBenchmark;
use vortex_bench::tpch::benchmark::TpcHBenchmark;

/// Minimum Parquet size for an SF10 TPC-H table included in the large-table suite.
const MIN_TPCH_TABLE_SIZE: u64 = 100 * 1024 * 1024;

/// A prepared full table represented as one compression-benchmark dataset.
pub struct GpuTableDataset {
    name: String,
    v3_dataset: String,
    v3_variant: String,
    parquet_path: PathBuf,
}

impl GpuTableDataset {
    fn new(
        name: impl Into<String>,
        v3_dataset: impl Into<String>,
        v3_variant: impl Into<String>,
        parquet_path: PathBuf,
    ) -> Self {
        Self {
            name: name.into(),
            v3_dataset: v3_dataset.into(),
            v3_variant: v3_variant.into(),
            parquet_path,
        }
    }
}

#[async_trait]
impl Dataset for GpuTableDataset {
    fn name(&self) -> &str {
        &self.name
    }

    fn v3_dataset_dims(&self) -> (&str, Option<&str>) {
        (&self.v3_dataset, Some(&self.v3_variant))
    }

    async fn to_vortex_array(&self, _ctx: &mut ExecutionCtx) -> anyhow::Result<ArrayRef> {
        Ok(parquet_to_vortex_chunks(self.parquet_path.clone())
            .await?
            .into_array())
    }

    async fn to_parquet_path(&self) -> anyhow::Result<PathBuf> {
        Ok(self.parquet_path.clone())
    }
}

/// Prepare and return the large datasets used only by explicit local GPU runs.
pub async fn large_gpu_datasets() -> anyhow::Result<Vec<GpuTableDataset>> {
    let fineweb = FinewebBenchmark::with_remote_data_dir(None)?;
    fineweb.generate_base_data().await?;
    let fineweb_path = local_format_dir(&fineweb, Format::Parquet)?.join("sample.parquet");
    ensure_file(&fineweb_path)?;

    let clickbench = ClickBenchBenchmark::new(Flavor::Single, None, None)?;
    clickbench.generate_base_data().await?;
    let clickbench_path = local_format_dir(&clickbench, Format::Parquet)?.join("hits.parquet");
    ensure_file(&clickbench_path)?;

    let tpch = TpcHBenchmark::new("10.0".to_string(), None)?;
    tpch.generate_base_data().await?;
    let tpch_dir = local_format_dir(&tpch, Format::Parquet)?;

    let mut datasets = vec![
        GpuTableDataset::new(
            "FineWeb all columns",
            "fineweb",
            "all-columns",
            fineweb_path,
        ),
        GpuTableDataset::new(
            "ClickBench all columns",
            "clickbench",
            "all-columns",
            clickbench_path,
        ),
    ];

    for table in tpch.table_specs() {
        let path = tpch_dir.join(format!("{}.parquet", table.name));
        let size = std::fs::metadata(&path)
            .with_context(|| format!("reading metadata for {}", path.display()))?
            .len();
        if !is_large_tpch_table(size) {
            tracing::info!(table = table.name, size, "skipping small TPC-H SF10 table");
            continue;
        }

        datasets.push(GpuTableDataset::new(
            format!("TPC-H SF10 {} all columns", table.name),
            "tpch",
            format!("sf10-{}-all-columns", table.name),
            path,
        ));
    }

    Ok(datasets)
}

fn local_format_dir(benchmark: &dyn Benchmark, format: Format) -> anyhow::Result<PathBuf> {
    benchmark
        .format_path(format, benchmark.data_url())?
        .to_file_path()
        .map_err(|()| anyhow::anyhow!("benchmark data URL is not local: {}", benchmark.data_url()))
}

fn ensure_file(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.is_file(),
        "benchmark data file is missing: {}",
        path.display()
    );
    Ok(())
}

fn is_large_tpch_table(size: u64) -> bool {
    size > MIN_TPCH_TABLE_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tpch_size_filter_is_strictly_over_100_mib() {
        assert!(!is_large_tpch_table(MIN_TPCH_TABLE_SIZE));
        assert!(is_large_tpch_table(MIN_TPCH_TABLE_SIZE + 1));
    }
}
