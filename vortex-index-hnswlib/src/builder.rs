// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use serde::Deserialize;
use serde::Serialize;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_index::IndexBuildRequest;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexSource;
use vortex_index::IndexStore;
use vortex_session::VortexSession;

use crate::HnswBundle;
use crate::bundle::validate_metadata;
use crate::bundle::validate_rows;
use crate::ffi::Native;
use crate::import_bundle;

/// Required versioned JSON for static, non-null Float32 squared-L2 construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HnswBuildOptions {
    /// Configuration version, currently 1.
    pub format_version: u32,
    /// Components per vector, 1..=4096.
    pub dimension: u32,
    /// Graph degree parameter, 2..=64.
    pub m: u32,
    /// Construction width, m..=4096.
    pub ef_construction: u32,
    /// Upstream random seed.
    pub seed: u32,
    /// Native construction threads, currently must be 1; not a query setting or CPU quota.
    pub threads: u32,
}

impl HnswBuildOptions {
    pub(crate) fn validate(&self) -> VortexResult<()> {
        self.bundle(1).validate()?;
        if self.format_version != 1 || self.seed > i32::MAX as u32 {
            vortex_bail!("Invalid hnswlib build version or seed");
        }
        if self.threads != 1 {
            vortex_bail!("hnswlib construction requires threads=1");
        }
        Ok(())
    }
    pub(crate) fn bundle(&self, rows: u32) -> HnswBundle {
        HnswBundle {
            dimension: self.dimension,
            rows,
            m: self.m,
            ef_construction: self.ef_construction,
        }
    }
}

/// Construction bounds on buffered input and output, not total native RSS or disk quota.
#[derive(Debug, Clone, Copy)]
pub struct HnswBuildLimits {
    /// Maximum physical rows in the requested coverage.
    pub max_rows: u32,
    /// Maximum flattened Float32 input bytes, default 512 MiB.
    pub max_vector_bytes: u64,
    /// Maximum closed native index size, checked before validation/import.
    pub max_native_artifact_bytes: u64,
}

impl Default for HnswBuildLimits {
    fn default() -> Self {
        Self {
            max_rows: 1_000_000,
            max_vector_bytes: 512 * 1024 * 1024,
            max_native_artifact_bytes: 1024 * 1024 * 1024,
        }
    }
}

/// Initial-only construction from a pinned source; sealing/publication remain the owner's job.
#[derive(Debug)]
pub struct HnswIndexBuilder {
    scratch_root: PathBuf,
    session: VortexSession,
    limits: HnswBuildLimits,
}

fn validate_scratch(root: &Path) -> VortexResult<()> {
    if !root.is_absolute() {
        vortex_bail!("hnswlib scratch root must be absolute");
    }
    let mut path = PathBuf::new();
    for component in root.components() {
        if !matches!(component, Component::RootDir | Component::Normal(_)) {
            vortex_bail!("hnswlib scratch root contains parent components");
        }
        path.push(component);
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            vortex_bail!("hnswlib scratch requires real directories");
        }
    }
    Ok(())
}

impl HnswIndexBuilder {
    /// Configure an existing absolute, symlink-free scratch directory and decoder session.
    pub fn try_new(
        scratch_root: PathBuf,
        session: VortexSession,
        limits: HnswBuildLimits,
    ) -> VortexResult<Self> {
        validate_scratch(&scratch_root)?;
        if !(1..=i32::MAX as u32).contains(&limits.max_rows)
            || limits.max_vector_bytes == 0
            || limits.max_vector_bytes > i32::MAX as u64
            || limits.max_native_artifact_bytes == 0
        {
            vortex_bail!("Invalid hnswlib build limits");
        }
        Ok(Self {
            scratch_root,
            session,
            limits,
        })
    }
}

#[async_trait]
impl IndexBuilder for HnswIndexBuilder {
    async fn build(
        &self,
        request: IndexBuildRequest,
        source: Arc<dyn IndexSource>,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<IndexMetadata> {
        let metadata = request.metadata;
        metadata.validate_for(source.snapshot())?;
        validate_metadata(&metadata)?;
        if !metadata.artifacts.is_empty() || store.as_local_files().is_none() {
            vortex_bail!("hnswlib build requires empty artifacts and a local-file store");
        }
        if request.backend_options.len() > 4096 {
            vortex_bail!("hnswlib build options exceed 4096 bytes");
        }
        let options: HnswBuildOptions = serde_json::from_slice(&request.backend_options)
            .map_err(|err| vortex_err!("hnswlib build options: {err}"))?;
        options.validate()?;
        let count = metadata
            .snapshot
            .files
            .iter()
            .filter(|file| metadata.covers(file.id))
            .try_fold(0u64, |total, file| total.checked_add(file.row_count))
            .ok_or_else(|| vortex_err!("hnswlib coverage count overflow"))?;
        if count == 0 || count > u64::from(self.limits.max_rows) {
            vortex_bail!("hnswlib build exceeds row limit or has empty coverage");
        }
        let count = u32::try_from(count)?;
        let bundle = options.bundle(count);
        let vector_bytes = u64::from(count) * u64::from(bundle.dimension) * 4;
        if vector_bytes > self.limits.max_vector_bytes {
            vortex_bail!("hnswlib build exceeds vector byte limit");
        }
        let field_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            bundle.dimension,
            Nullability::NonNullable,
        );
        let expected = DType::Struct(
            StructFields::new([metadata.fields[0].as_str()].into(), vec![field_dtype]),
            Nullability::NonNullable,
        );
        let mut rows = Vec::new();
        let mut vectors = Vec::new();
        rows.try_reserve_exact(count as usize)
            .map_err(|err| vortex_err!("hnswlib row allocation: {err}"))?;
        vectors
            .try_reserve_exact(usize::try_from(vector_bytes)? / 4)
            .map_err(|err| vortex_err!("hnswlib vector allocation: {err}"))?;
        let mut stream = source.scan(&metadata.covered_files, &metadata.fields)?;
        let mut ctx = self.session.create_execution_ctx();
        while let Some(batch) = stream.try_next().await? {
            if batch.rows.len() != batch.data.len()
                || batch.rows.len() > count as usize - rows.len()
                || batch.data.dtype() != &expected
            {
                vortex_bail!(
                    "hnswlib source requires aligned, non-nullable fixed-size Float32 vectors"
                );
            }
            for row in &batch.rows {
                metadata.snapshot.validate_row(*row)?;
                if !metadata.covers(row.file_id) {
                    vortex_bail!("hnswlib source returned an uncovered file");
                }
            }
            let data = batch.data.execute::<StructArray>(&mut ctx)?;
            let lists = data
                .unmasked_field(0)
                .clone()
                .execute::<FixedSizeListArray>(&mut ctx)?;
            let values = lists
                .elements()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)?;
            let values = values.as_slice::<f32>();
            if values.len() != batch.rows.len() * bundle.dimension as usize
                || values.iter().any(|value| !value.is_finite())
            {
                vortex_bail!(
                    "hnswlib source contains invalid vector lengths or non-finite components"
                );
            }
            rows.extend(batch.rows);
            vectors.extend_from_slice(values);
        }
        metadata.validate_for(source.snapshot())?;
        validate_rows(&metadata, bundle, &rows)?;
        validate_scratch(&self.scratch_root)?;
        let scratch = tempfile::Builder::new()
            .prefix(".hnswlib-build-")
            .permissions(fs::Permissions::from_mode(0o700))
            .tempdir_in(&self.scratch_root)?;
        let path = scratch.path().join("index.bin");
        Native::build(&path, &vectors, bundle, &options)?;
        if fs::metadata(&path)?.len() > self.limits.max_native_artifact_bytes {
            vortex_bail!("hnswlib build exceeds native artifact byte limit");
        }
        drop(Native::open(&path, bundle)?);
        metadata.validate_for(source.snapshot())?;
        import_bundle(metadata, store.as_ref(), &path, bundle, &rows).await
    }
}
