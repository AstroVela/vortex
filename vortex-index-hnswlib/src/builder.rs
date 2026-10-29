// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
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
use vortex_index::Index;
use vortex_index::IndexBuildRequest;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexSource;
use vortex_index::IndexStore;
use vortex_index::attributes::AttributeBlockKind;
use vortex_index::attributes::AttributeBlockRequest;
use vortex_session::VortexSession;

use crate::HnswBundle;
use crate::HnswPayload;
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
    /// Additional degree per attribute block, 1..=64; zero disables construction.
    pub payload_m: u32,
    /// Attribute fields to group independently. Must be empty exactly when payload_m=0.
    pub payload_fields: BTreeMap<String, AttributeBlockKind>,
    /// Omit groups smaller than this many physical rows (at least 2).
    pub payload_min_rows: u32,
    /// Target numeric window size; equal values are never split.
    pub payload_target_rows: u32,
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
        if self.payload_m > 64
            || self.payload_m > self.ef_construction
            || (self.payload_m == 0) != self.payload_fields.is_empty()
            || self.payload_min_rows < 2
            || self.payload_target_rows < self.payload_min_rows
        {
            vortex_bail!("Invalid payload construction options");
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
    /// Maximum combined native index and payload bytes, checked before import.
    pub max_native_artifact_bytes: u64,
    /// Maximum directed supplementary edges after merging blocks.
    pub max_payload_edges: u64,
    /// Maximum supplementary degree of one node across all blocks.
    pub max_payload_degree: u32,
    /// Maximum rows in one block, including equal-value groups.
    pub max_payload_block_rows: u32,
    /// Maximum input postings and output memberships for the block plan.
    pub max_payload_memberships: u64,
    /// Maximum distinct terms per field and total blocks.
    pub max_payload_blocks: u32,
}

impl Default for HnswBuildLimits {
    fn default() -> Self {
        Self {
            max_rows: 1_000_000,
            max_vector_bytes: 512 * 1024 * 1024,
            max_native_artifact_bytes: 1024 * 1024 * 1024,
            max_payload_edges: 32_000_000,
            max_payload_degree: 256,
            max_payload_block_rows: 100_000,
            max_payload_memberships: 8_000_000,
            max_payload_blocks: 65_536,
        }
    }
}

/// Initial-only construction from a pinned source; sealing/publication remain the owner's job.
#[derive(Debug)]
pub struct HnswIndexBuilder {
    scratch_root: PathBuf,
    session: VortexSession,
    limits: HnswBuildLimits,
    attributes: Option<Arc<dyn Index>>,
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
            || limits.max_payload_edges == 0
            || limits.max_payload_degree == 0
            || limits.max_payload_block_rows < 2
            || limits.max_payload_memberships < u64::from(limits.max_payload_block_rows)
            || limits.max_payload_blocks == 0
        {
            vortex_bail!("Invalid hnswlib build limits");
        }
        Ok(Self {
            scratch_root,
            session,
            limits,
            attributes: None,
        })
    }

    /// Bind an already opened scalar index for construction. Build validates its
    /// snapshot and complete requested file coverage before reading vectors.
    pub fn with_attributes(mut self, attributes: Arc<dyn Index>) -> VortexResult<Self> {
        if attributes.as_attributes().is_none() {
            vortex_bail!("Index has no attribute capability");
        }
        self.attributes = Some(attributes);
        Ok(self)
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
        let plan = if options.payload_m == 0 {
            None
        } else {
            let attributes = self
                .attributes
                .as_ref()
                .ok_or_else(|| vortex_err!("Payload construction requires an attribute index"))?;
            if attributes.metadata().snapshot != metadata.snapshot
                || metadata
                    .covered_files
                    .iter()
                    .any(|id| !attributes.metadata().covers(*id))
            {
                vortex_bail!("Payload attribute snapshot does not match vector snapshot");
            }
            let request = AttributeBlockRequest {
                fields: options.payload_fields.clone(),
                covered_files: metadata.covered_files.clone(),
                min_rows: options.payload_min_rows,
                target_rows: options.payload_target_rows,
                max_block_rows: self.limits.max_payload_block_rows,
                max_blocks: self.limits.max_payload_blocks,
                max_memberships: self.limits.max_payload_memberships,
            };
            let plan = attributes
                .as_attributes()
                .ok_or_else(|| vortex_err!("Missing attributes"))?
                .blocks(&request)
                .await?;
            plan.validate()?;
            if plan.snapshot != metadata.snapshot || plan.request != request {
                vortex_bail!("Attribute block plan binding mismatch");
            }
            Some(plan)
        };
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
        let mut native = Native::open(&path, bundle)?;
        let payload = if let Some(plan) = &plan {
            let labels: BTreeMap<_, _> = rows.iter().copied().zip(0u32..).collect();
            native.begin_payload(options.payload_m, options.ef_construction, self.limits)?;
            for block in &plan.blocks {
                let block_labels = block
                    .rows
                    .iter()
                    .map(|row| {
                        labels
                            .get(row)
                            .copied()
                            .ok_or_else(|| vortex_err!("Payload row outside vector coverage"))
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                native.add_payload_block(&block_labels)?;
            }
            native.finish_payload(self.limits.max_payload_edges)?
        } else {
            HnswPayload::empty(bundle.rows)
        };
        drop(native);
        let artifact_bytes = fs::metadata(&path)?
            .len()
            .checked_add(payload.serialized_size()? as u64)
            .ok_or_else(|| vortex_err!("hnswlib artifact size overflow"))?;
        if artifact_bytes > self.limits.max_native_artifact_bytes {
            vortex_bail!("hnswlib base and payload exceed artifact byte limit");
        }
        metadata.validate_for(source.snapshot())?;
        import_bundle(metadata, store.as_ref(), &path, bundle, &rows, &payload).await
    }
}
