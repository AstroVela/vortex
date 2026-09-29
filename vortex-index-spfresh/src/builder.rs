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

use crate::SpFreshBundle;
use crate::bundle::NATIVE_FILES;
use crate::bundle::validate_metadata;
use crate::bundle::validate_rows;
use crate::ffi::Native;
use crate::import_bundle;
use crate::validate;

/// Versioned JSON in [`IndexBuildRequest::backend_options`]. All fields are required.
///
/// Builds only non-nullable `FixedSizeList<Float32>` vectors with squared L2.
/// Other metrics, quantization, filters and incremental mutation are unsupported.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpFreshBuildOptions {
    /// Build configuration version, currently 1 (separate from the artifact version).
    pub format_version: u32,
    /// Number of Float32 components per vector, 1..=4096.
    pub dimension: u32,
    /// Randomly selected BKT heads, at least 32 and strictly fewer than covered rows.
    pub head_count: u32,
    /// Maximum posting size, in 4096-byte pages, 1..=4096.
    ///
    /// A configuration that loses row coverage through native posting truncation
    /// fails validation; increase the head count or this limit and build afresh.
    pub posting_page_limit: u32,
    /// Maximum posting replicas per non-head vector, 1..=8.
    pub replicas: u32,
}

impl SpFreshBuildOptions {
    pub(crate) fn validate(&self, rows: u32) -> VortexResult<()> {
        if self.format_version != 1
            || rows < 64
            || !(1..=4096).contains(&self.dimension)
            || self.head_count < 32
            || self.head_count >= rows
            || !(1..=4096).contains(&self.posting_page_limit)
            || !(1..=8).contains(&self.replicas)
        {
            vortex_bail!("Unsupported SPFresh static build options (requires at least 64 rows)");
        }
        Ok(())
    }
}

/// Bounds on buffered input and produced native artifacts, not total native RSS or disk quota.
#[derive(Debug, Clone, Copy)]
pub struct SpFreshBuildLimits {
    /// Maximum physical rows in the requested coverage; row mapping also uses memory.
    pub max_rows: u32,
    /// Maximum flattened Float32 input bytes. Vectors are materialized before native build.
    pub max_vector_bytes: u64,
    /// Maximum total size of the six native output files, checked before validation/import.
    ///
    /// This is not a limit on temporary native files during construction.
    pub max_native_artifact_bytes: u64,
}

impl Default for SpFreshBuildLimits {
    fn default() -> Self {
        Self {
            max_rows: 1_000_000,
            max_vector_bytes: 256 * 1024 * 1024,
            max_native_artifact_bytes: 1024 * 1024 * 1024,
        }
    }
}

/// Initial static construction from a pinned [`IndexSource`], with no catalog publication.
///
/// Scans every physical row in the declared coverage and retains scan order as
/// native ID order. NULLs, non-finite values, incomplete coverage and schema changes
/// fail explicitly. Native work is blocking and serialized with searches; callers
/// must run it on a blocking worker. Input and row mappings are held in memory.
#[derive(Debug)]
pub struct SpFreshIndexBuilder {
    scratch_root: PathBuf,
    session: VortexSession,
    limits: SpFreshBuildLimits,
}

impl SpFreshIndexBuilder {
    /// Configure an existing, absolute, symlink-free owner-managed scratch directory.
    ///
    /// The session must support the encodings returned by the source. Scratch must
    /// not be modified by another process during build. Drop cleans private files on
    /// normal/error returns; crash-left directories remain the owner's responsibility.
    pub fn try_new(
        scratch_root: PathBuf,
        session: VortexSession,
        limits: SpFreshBuildLimits,
    ) -> VortexResult<Self> {
        validate_scratch(&scratch_root)?;
        if !(64..=i32::MAX as u32).contains(&limits.max_rows)
            || limits.max_vector_bytes == 0
            || limits.max_vector_bytes > i32::MAX as u64
            || limits.max_native_artifact_bytes == 0
        {
            vortex_bail!("Invalid SPFresh build limits");
        }
        Ok(Self {
            scratch_root,
            session,
            limits,
        })
    }
}

fn validate_scratch(root: &Path) -> VortexResult<()> {
    if !root.is_absolute() {
        vortex_bail!("SPFresh build scratch must be absolute");
    }
    let mut path = PathBuf::new();
    for component in root.components() {
        if !matches!(component, Component::RootDir | Component::Normal(_)) {
            vortex_bail!("SPFresh build scratch must not contain parent components");
        }
        path.push(component);
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            vortex_bail!("SPFresh build scratch must contain only real directories");
        }
    }
    Ok(())
}

#[async_trait]
impl IndexBuilder for SpFreshIndexBuilder {
    /// Return durable artifact metadata after native validation and a successful reopen.
    ///
    /// This does not seal the store or publish a generation. The owner must seal,
    /// reopen through its provider, then publish against the expected snapshot.
    /// Scan/native failures write no store artifacts. An import failure can leave
    /// unreferenced artifacts; retry in a fresh private generation, never overwrite.
    async fn build(
        &self,
        request: IndexBuildRequest,
        source: Arc<dyn IndexSource>,
        store: Arc<dyn IndexStore>,
    ) -> VortexResult<IndexMetadata> {
        let metadata = request.metadata;
        metadata.validate_for(source.snapshot())?;
        validate_metadata(&metadata)?;
        if !metadata.artifacts.is_empty() {
            vortex_bail!("SPFresh build requires an empty artifact inventory");
        }
        if store.as_local_files().is_none() {
            vortex_bail!("SPFresh build requires a local-file store");
        }
        if request.backend_options.len() > 4096 {
            vortex_bail!("SPFresh build options exceed 4096 bytes");
        }
        let options: SpFreshBuildOptions = serde_json::from_slice(&request.backend_options)
            .map_err(|err| vortex_err!("SPFresh build options: {err}"))?;
        let count = metadata
            .snapshot
            .files
            .iter()
            .filter(|file| metadata.covers(file.id))
            .try_fold(0u64, |total, file| total.checked_add(file.row_count))
            .ok_or_else(|| vortex_err!("SPFresh build coverage count overflow"))?;
        if count > u64::from(self.limits.max_rows) {
            vortex_bail!("SPFresh build exceeds row limit");
        }
        let count = u32::try_from(count)?;
        options.validate(count)?;
        let vector_bytes = u64::from(count) * u64::from(options.dimension) * 4;
        if vector_bytes > self.limits.max_vector_bytes {
            vortex_bail!("SPFresh build exceeds vector byte limit");
        }
        validate_scratch(&self.scratch_root)?;
        let scratch = tempfile::Builder::new()
            .prefix(".spfresh-build-")
            .permissions(fs::Permissions::from_mode(0o700))
            .tempdir_in(&self.scratch_root)?;
        let bundle = SpFreshBundle {
            dimension: options.dimension,
            rows: count,
            posting_page_limit: options.posting_page_limit,
        };
        let field_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
            options.dimension,
            Nullability::NonNullable,
        );
        let expected = DType::Struct(
            StructFields::new([metadata.fields[0].as_str()].into(), vec![field_dtype]),
            Nullability::NonNullable,
        );
        let mut rows = Vec::new();
        let mut vectors = Vec::new();
        rows.try_reserve_exact(count as usize)
            .map_err(|err| vortex_err!("SPFresh row allocation: {err}"))?;
        vectors
            .try_reserve_exact(usize::try_from(vector_bytes)? / 4)
            .map_err(|err| vortex_err!("SPFresh vector allocation: {err}"))?;
        let mut stream = source.scan(&metadata.covered_files, &metadata.fields)?;
        let mut ctx = self.session.create_execution_ctx();
        while let Some(batch) = stream.try_next().await? {
            if batch.rows.len() != batch.data.len()
                || batch.rows.len() > count as usize - rows.len()
                || batch.data.dtype() != &expected
            {
                vortex_bail!(
                    "SPFresh source batch has invalid length or non-nullable Float32 vector schema"
                );
            }
            for row in &batch.rows {
                metadata.snapshot.validate_row(*row)?;
                if !metadata.covers(row.file_id) {
                    vortex_bail!("SPFresh source returned an uncovered file");
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
            if values.len() != batch.rows.len() * options.dimension as usize
                || values.iter().any(|value| !value.is_finite())
            {
                vortex_bail!("SPFresh source contains invalid or non-finite vectors");
            }
            rows.extend(batch.rows);
            vectors.extend_from_slice(values);
        }
        metadata.validate_for(source.snapshot())?;
        validate_rows(&metadata, bundle, &rows)?;
        let root = scratch.path().join("native");
        Native::build(&root, &vectors, bundle, &options)?;
        let mut bytes = 0u64;
        for name in NATIVE_FILES {
            bytes = bytes
                .checked_add(fs::metadata(root.join(name))?.len())
                .ok_or_else(|| vortex_err!("SPFresh output byte count overflow"))?;
        }
        if bytes > self.limits.max_native_artifact_bytes {
            vortex_bail!("SPFresh build exceeds native artifact byte limit");
        }
        validate::native_files(&root, bundle)?;
        drop(Native::open(&root, bundle)?);
        metadata.validate_for(source.snapshot())?;
        import_bundle(metadata, store.as_ref(), &root, bundle, &rows).await
    }
}
