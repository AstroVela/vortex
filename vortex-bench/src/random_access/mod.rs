// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use anyhow::anyhow;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use url::Url;
use vortex::array::ArrayRef;

use crate::Format;
use crate::data_dir;

pub mod take;

// Re-export implementations
pub use take::ParquetRandomAccessor;
pub use take::VortexRandomAccessor;

/// Generate the data path for a random-access benchmark dataset file.
///
/// Returns a path like `random_access/{dataset}/{dataset}.{ext}`
/// (or `{dataset}-compact.{ext}` for [`Format::VortexCompact`]).
pub fn data_path(dataset: &str, format: Format) -> String {
    let ext = format.ext();
    match format {
        Format::VortexCompact => format!("random_access/{dataset}/{dataset}-compact.{ext}"),
        _ => format!("random_access/{dataset}/{dataset}.{ext}"),
    }
}

/// A remote directory holding the same layout as the local benchmark data directory.
///
/// Random access datasets are always materialized locally first, then uploaded verbatim, so a
/// remote object key is just the local path relative to [`data_dir`] appended to the URL path.
#[derive(Clone, Debug)]
pub struct RemoteDataDir {
    url: Url,
    store: Arc<dyn ObjectStore>,
}

impl RemoteDataDir {
    /// Build an object store for `url` (e.g. `s3://bucket/prefix/`) from the ambient environment.
    pub fn try_new(url: Url) -> Result<Self> {
        let store: Arc<dyn ObjectStore> = match url.scheme() {
            "s3" => {
                let bucket = url
                    .host_str()
                    .ok_or_else(|| anyhow!("remote data dir has no bucket: {url}"))?;
                Arc::new(
                    AmazonS3Builder::from_env()
                        .with_bucket_name(bucket)
                        .build()?,
                )
            }
            other => return Err(anyhow!("unsupported remote data dir scheme: {other}")),
        };
        Ok(Self { url, store })
    }

    /// The object store backing this directory.
    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// The object key of `local_path`, mirroring its location under the local data directory.
    pub fn key(&self, local_path: &Path) -> Result<String> {
        let relative = local_path.strip_prefix(data_dir()).map_err(|_| {
            anyhow!(
                "{} is not inside the benchmark data directory",
                local_path.display()
            )
        })?;
        let relative = relative
            .to_str()
            .ok_or_else(|| anyhow!("non-UTF-8 data path: {}", local_path.display()))?;
        let prefix = self
            .url
            .path()
            .trim_start_matches('/')
            .trim_end_matches('/');
        Ok(if prefix.is_empty() {
            relative.to_string()
        } else {
            format!("{prefix}/{relative}")
        })
    }

    /// The fully qualified URL of `local_path` in this remote directory.
    pub fn uri(&self, local_path: &Path) -> Result<String> {
        let scheme = self.url.scheme();
        let host = self
            .url
            .host_str()
            .ok_or_else(|| anyhow!("remote data dir has no bucket: {}", self.url))?;
        Ok(format!("{scheme}://{host}/{}", self.key(local_path)?))
    }
}

/// Trait for a benchmark dataset that knows how to prepare data files.
#[async_trait]
pub trait BenchDataset: Send + Sync {
    /// A descriptive name for this dataset (used in benchmark output and CLI).
    fn name(&self) -> &str;

    /// The total number of rows in this dataset.
    fn row_count(&self) -> u64;

    /// Prepare the data file for the given format and return its path.
    ///
    /// This writes the file if it doesn't already exist.
    async fn path(&self, format: Format) -> Result<PathBuf>;
}

pub enum RandomAccessorRet {
    RecordBatch(RecordBatch),
    ArrayRef(ArrayRef),
}

/// Trait for format-specific random access (take) operations.
///
/// Implementations handle reading specific rows by index from a data source.
/// Accessors are constructed in a ready-to-use state with metadata already parsed.
#[async_trait]
pub trait RandomAccessor: Send + Sync {
    /// A descriptive name for this accessor (used in benchmark output).
    fn name(&self) -> &str;

    /// The format this accessor handles.
    fn format(&self) -> Format;

    /// Take rows at the given indices, returning the handle.
    async fn take(&self, indices: &[u64]) -> Result<RandomAccessorRet>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remote(url: &str) -> Result<RemoteDataDir> {
        // `from_env` needs no credentials to construct the client.
        RemoteDataDir::try_new(Url::parse(url)?)
    }

    #[test]
    fn key_mirrors_the_local_data_dir_layout() -> Result<()> {
        let local = data_dir().join("random_access/taxi/taxi.vortex");

        assert_eq!(
            remote("s3://bucket/prefix/")?.key(&local)?,
            "prefix/random_access/taxi/taxi.vortex"
        );
        assert_eq!(
            remote("s3://bucket/")?.key(&local)?,
            "random_access/taxi/taxi.vortex"
        );
        assert_eq!(
            remote("s3://bucket/prefix/")?.uri(&local)?,
            "s3://bucket/prefix/random_access/taxi/taxi.vortex"
        );
        Ok(())
    }

    #[test]
    fn key_rejects_paths_outside_the_data_dir() -> Result<()> {
        assert!(
            remote("s3://bucket/prefix/")?
                .key(Path::new("/tmp/taxi.vortex"))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn unsupported_scheme_is_rejected() -> Result<()> {
        assert!(RemoteDataDir::try_new(Url::parse("gs://bucket/prefix/")?).is_err());
        Ok(())
    }
}
