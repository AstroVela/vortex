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
use object_store::path::Path as ObjectStorePath;
use object_store::registry::ObjectStoreRegistry;
use parquet::file::properties::WriterProperties;
use url::Url;
use vortex::array::ArrayRef;
use vortex::cloud::Registry;

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

/// Approximate byte budget for one row group in the synthetic random-access datasets.
const TARGET_ROW_GROUP_BYTES: usize = 128 * 1024 * 1024;

/// Row groups are sized in whole multiples of this many rows.
///
/// Parquet is converted to Vortex by streaming Arrow batches of this many rows, so keeping row
/// groups a whole multiple of it leaves the derived Vortex files byte-identical: all but the final
/// row group hold an exact number of batches, so the chunk boundaries do not move and only the
/// Parquet layout changes.
///
/// [`crate::conversions::parquet_to_vortex_chunks`] sets this explicitly rather than relying on
/// `parquet`'s default batch size, so that an upstream change to that default cannot silently
/// invalidate the guarantee above.
pub(crate) const PARQUET_READ_BATCH_SIZE: usize = 1024;

/// Bounds on the row group size, in units of [`PARQUET_READ_BATCH_SIZE`] rows.
///
/// The upper bound matters more than the byte budget for narrow rows: without it a million-row
/// dataset lands in a single row group, and a point lookup then has to read the entire file.
const MIN_BATCHES_PER_ROW_GROUP: usize = 8;
const MAX_BATCHES_PER_ROW_GROUP: usize = 64;

/// Rows per data page.
///
/// Finer pages than the 20k-row default give the page index enough resolution to be useful for
/// point lookups, at the cost of a slightly larger index.
const DATA_PAGE_ROWS: usize = PARQUET_READ_BATCH_SIZE;

/// Parquet writer properties for a synthetic random-access dataset of `approx_row_bytes` per row.
///
/// The defaults are wrong for this suite: the max row group row count defaults to 1Mi rows, which is more
/// than every dataset here holds, so each file ends up as one row group. Readers select row groups
/// before rows, so a single-row-group file forces a point lookup to fetch and decode the whole
/// file — cheap from page cache, ruinous over an object store.
pub fn random_access_writer_properties(approx_row_bytes: usize) -> WriterProperties {
    let batches = (TARGET_ROW_GROUP_BYTES / approx_row_bytes / PARQUET_READ_BATCH_SIZE)
        .clamp(MIN_BATCHES_PER_ROW_GROUP, MAX_BATCHES_PER_ROW_GROUP);

    WriterProperties::builder()
        .set_max_row_group_row_count(Some(batches * PARQUET_READ_BATCH_SIZE))
        .set_data_page_row_count_limit(DATA_PAGE_ROWS)
        .build()
}

/// A remote directory holding the same layout as the local benchmark data directory.
///
/// URLs are resolved through Vortex's own [`Registry`], so every scheme the registry serves
/// (`s3://`, `gs://`, `az://`, `hf://`, ...) works here, configured from the ambient environment
/// exactly as it is for the Python, Java, and DataFusion bindings.
#[derive(Clone, Debug)]
pub struct RemoteDataDir {
    /// The directory URL, always with a trailing slash so that [`Url::join`] appends to it.
    url: Url,
    /// Shared so that every dataset resolved through this directory reuses one client.
    registry: Arc<Registry>,
}

impl RemoteDataDir {
    /// A remote data directory at `url` (e.g. `s3://bucket/prefix/`).
    pub fn try_new(mut url: Url) -> Result<Self> {
        // `Url::join` replaces the last path segment unless the base ends in a slash, so a
        // `--remote-data-dir s3://bucket/prefix` would otherwise silently drop `prefix`.
        if !url.path().ends_with('/') {
            url.set_path(&format!("{}/", url.path()));
        }

        let registry = Arc::new(Registry::new());
        // Resolve once up front so an unsupported scheme or unusable configuration is reported
        // before any data is generated, rather than on the first take.
        registry.resolve(&url)?;

        Ok(Self { url, registry })
    }

    /// The object store serving `local_path`, along with that file's key within it.
    pub fn resolve(&self, local_path: &Path) -> Result<(Arc<dyn ObjectStore>, ObjectStorePath)> {
        Ok(self.registry.resolve(&self.url_of(local_path)?)?)
    }

    /// The fully qualified URL of `local_path` in this remote directory.
    ///
    /// Used for readers that resolve URLs themselves rather than taking an [`ObjectStore`].
    pub fn uri(&self, local_path: &Path) -> Result<String> {
        Ok(self.url_of(local_path)?.to_string())
    }

    /// The URL of `local_path` within this directory, mirroring its local location.
    ///
    /// Random access datasets are always materialized locally first, then uploaded verbatim, so
    /// the remote URL is the path relative to [`data_dir`] resolved against the directory URL.
    fn url_of(&self, local_path: &Path) -> Result<Url> {
        let relative = local_path.strip_prefix(data_dir()).map_err(|_| {
            anyhow!(
                "{} is not inside the benchmark data directory",
                local_path.display()
            )
        })?;
        let relative = relative
            .to_str()
            .ok_or_else(|| anyhow!("non-UTF-8 data path: {}", local_path.display()))?;
        Ok(self.url.join(relative)?)
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
    use rstest::rstest;

    use super::*;
    // The real row widths of the three synthetic random-access datasets, so that the cases below
    // cannot drift from the values the writers are actually configured with.
    use crate::datasets::feature_vectors::APPROX_ROW_BYTES as FEATURE_VECTORS_ROW_BYTES;
    use crate::datasets::nested_lists::APPROX_ROW_BYTES as NESTED_LISTS_ROW_BYTES;
    use crate::datasets::nested_structs::APPROX_ROW_BYTES as NESTED_STRUCTS_ROW_BYTES;

    /// Building a store needs no credentials, so these run in any environment.
    fn remote(url: &str) -> Result<RemoteDataDir> {
        RemoteDataDir::try_new(Url::parse(url)?)
    }

    #[rstest]
    // A trailing slash on the directory URL is optional.
    #[case::prefix("s3://bucket/prefix/")]
    #[case::prefix_without_trailing_slash("s3://bucket/prefix")]
    fn keys_mirror_the_local_data_dir_layout(#[case] url: &str) -> Result<()> {
        let local = data_dir().join("random_access/taxi/taxi.vortex");
        let (_store, key) = remote(url)?.resolve(&local)?;

        assert_eq!(key.as_ref(), "prefix/random_access/taxi/taxi.vortex");
        assert_eq!(
            remote(url)?.uri(&local)?,
            "s3://bucket/prefix/random_access/taxi/taxi.vortex"
        );
        Ok(())
    }

    #[test]
    fn a_bucket_root_needs_no_prefix() -> Result<()> {
        let local = data_dir().join("random_access/taxi/taxi.vortex");
        let (_store, key) = remote("s3://bucket/")?.resolve(&local)?;

        assert_eq!(key.as_ref(), "random_access/taxi/taxi.vortex");
        Ok(())
    }

    #[test]
    fn keys_reject_paths_outside_the_data_dir() -> Result<()> {
        assert!(
            remote("s3://bucket/prefix/")?
                .resolve(Path::new("/tmp/taxi.vortex"))
                .is_err()
        );
        Ok(())
    }

    /// Schemes other than `s3://` work too. Only those whose store needs no configuration beyond
    /// the URL are exercised here; `az://`, for one, requires an account name in the environment.
    #[test]
    fn other_registry_schemes_are_accepted() -> Result<()> {
        remote("gs://bucket/prefix/")?;
        Ok(())
    }

    #[test]
    fn unknown_schemes_are_rejected() -> Result<()> {
        assert!(remote("nosuchscheme://bucket/prefix/").is_err());
        Ok(())
    }

    /// Rows per row group, treating "unlimited" as the whole file.
    fn rows_per_row_group(props: &WriterProperties) -> usize {
        props.max_row_group_row_count().unwrap_or(usize::MAX)
    }

    #[rstest]
    #[case::feature_vectors(FEATURE_VECTORS_ROW_BYTES)]
    #[case::nested_lists(NESTED_LISTS_ROW_BYTES)]
    #[case::nested_structs(NESTED_STRUCTS_ROW_BYTES)]
    fn row_groups_split_a_million_row_dataset(#[case] approx_row_bytes: usize) {
        let props = random_access_writer_properties(approx_row_bytes);
        let rows_per_group = rows_per_row_group(&props);

        // The whole point: a million-row dataset must not land in a single row group.
        assert!(
            rows_per_group < 1_000_000,
            "{approx_row_bytes} byte rows produced one row group of {rows_per_group}"
        );
        // Row group boundaries stay aligned to the Arrow batches the Vortex conversion reads,
        // so the derived Vortex files are unaffected by this layout.
        assert_eq!(rows_per_group % PARQUET_READ_BATCH_SIZE, 0);
        assert!(rows_per_group * approx_row_bytes <= TARGET_ROW_GROUP_BYTES);
    }

    #[test]
    fn wide_rows_get_smaller_row_groups_than_narrow_rows() {
        let wide = rows_per_row_group(&random_access_writer_properties(FEATURE_VECTORS_ROW_BYTES));
        let narrow = rows_per_row_group(&random_access_writer_properties(NESTED_STRUCTS_ROW_BYTES));
        assert!(
            wide < narrow,
            "wide {wide} should be smaller than narrow {narrow}"
        );
    }
}
