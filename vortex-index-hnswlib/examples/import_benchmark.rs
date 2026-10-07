// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Import a frozen native benchmark graph onto an existing full-coverage Vortex fixture.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use futures::executor::block_on;
use serde::Deserialize;
use serde::Serialize;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::RowAddress;
use vortex_index::Snapshot;
use vortex_index::store::LocalGeneration;
use vortex_index::store::LocalIndexStore;
use vortex_index::store::LocalStoreLimits;
use vortex_index_hnswlib::HNSWLIB_ID;
use vortex_index_hnswlib::HnswBundle;
use vortex_index_hnswlib::HnswLimits;
use vortex_index_hnswlib::HnswProvider;
use vortex_index_hnswlib::import_bundle;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    format_version: u32,
    snapshot: Snapshot,
    generation: LocalGeneration,
    dtype: serde_json::Value,
}

fn main() -> VortexResult<()> {
    let args = env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let [original, native, bundle, output] = args.as_slice() else {
        vortex_bail!(
            "Usage: import_benchmark ORIGINAL_REFERENCE NATIVE_INDEX BUNDLE_JSON NEW_ROOT"
        );
    };
    if output.exists() || !output.is_absolute() || !native.is_absolute() {
        vortex_bail!("Require a new absolute output root and native index path");
    }
    let mut reference: Reference =
        serde_json::from_slice(&fs::read(original)?).map_err(|err| vortex_err!("{err}"))?;
    if reference.format_version != 1 {
        vortex_bail!("Unsupported reference version");
    }
    let bundle: HnswBundle =
        serde_json::from_slice(&fs::read(bundle)?).map_err(|err| vortex_err!("{err}"))?;
    let rows = reference
        .snapshot
        .files
        .iter()
        .flat_map(|file| {
            (0..file.row_count).map(move |row_offset| RowAddress {
                file_id: file.id,
                row_offset,
            })
        })
        .collect::<Vec<_>>();
    fs::create_dir(output)?;
    let store = LocalIndexStore::create(
        output,
        "hnsw-benchmark-v1",
        LocalStoreLimits {
            max_artifact_bytes: 1024 * 1024 * 1024,
            max_manifest_bytes: 16 * 1024 * 1024,
        },
    )?;
    let metadata = IndexMetadata {
        format_version: 1,
        name: "embedding".into(),
        generation: "hnsw-benchmark-v1".into(),
        backend: HNSWLIB_ID.into(),
        backend_version: 1,
        fields: vec!["embedding".into()],
        covered_files: reference
            .snapshot
            .files
            .iter()
            .map(|file| file.id)
            .collect(),
        snapshot: reference.snapshot.clone(),
        artifacts: Vec::new(),
    };
    let metadata = block_on(import_bundle(metadata, &store, native, bundle, &rows))?;
    reference.generation = store.seal(&metadata)?;
    drop(block_on(
        HnswProvider::try_new(output.clone(), HnswLimits::default())?
            .open(&metadata, Arc::new(store)),
    )?);
    fs::write(
        output.join("index.json"),
        serde_json::to_vec_pretty(&reference).map_err(|err| vortex_err!("{err}"))?,
    )?;
    Ok(())
}
