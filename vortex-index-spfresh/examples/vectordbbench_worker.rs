// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Persistent local query worker for VectorDBBench. Timings include JSON and socket IO.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::BufRead;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::num::NonZeroUsize;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use futures::executor::block_on;
use serde::Deserialize;
use serde_json::json;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_index::IndexProvider;
use vortex_index::RowFilter;
use vortex_index::SearchMode;
use vortex_index::Snapshot;
use vortex_index::VectorSearchOptions;
use vortex_index::store::LocalGeneration;
use vortex_index::store::LocalIndexStore;
use vortex_index::store::LocalStoreLimits;
use vortex_index_spfresh::SpFreshLimits;
use vortex_index_spfresh::SpFreshProvider;
use vortex_index_spfresh::rabitq::RaBitQErrorBound;

#[derive(Deserialize)]
struct Reference {
    format_version: u32,
    snapshot: Snapshot,
    generation: LocalGeneration,
    #[serde(default)]
    limits: SpFreshLimits,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    query: Vec<f32>,
    k: NonZeroUsize,
    max_check: u32,
    internal_results: u32,
    search_pages: u32,
    #[serde(default)]
    rabitq: Option<RaBitQErrorBound>,
    #[serde(default)]
    rerank: Option<u32>,
}

fn main() -> VortexResult<()> {
    if env::var_os("VORTEX_SPFRESH_TRACE").is_some() {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .without_time()
            .with_writer(std::io::stderr)
            .init();
    }
    let args = env::args_os()
        .map(PathBuf::from)
        .skip(1)
        .collect::<Vec<_>>();
    let [reference_path, socket_path] = args.as_slice() else {
        vortex_bail!("Usage: vectordbbench_worker INDEX.json SOCKET");
    };
    let mut stream = BufWriter::new(UnixStream::connect(socket_path)?);
    let reference: Reference = serde_json::from_slice(&fs::read(reference_path)?)
        .map_err(|err| vortex_err!("Reference: {err}"))?;
    if reference.format_version != 1 {
        vortex_bail!("Unsupported reference format");
    }
    let root = reference_path
        .parent()
        .ok_or_else(|| vortex_err!("Missing reference directory"))?;
    let scratch = tempfile::Builder::new()
        .prefix("vectordbbench-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(root)?;
    let (store, metadata) = LocalIndexStore::open(
        root,
        &reference.generation,
        &reference.snapshot,
        LocalStoreLimits {
            max_artifact_bytes: usize::try_from(reference.limits.max_materialized_bytes)?,
            max_manifest_bytes: 16 * 1024 * 1024,
        },
    )?;
    let provider = SpFreshProvider::try_new(scratch.path().to_owned(), reference.limits)?;
    let index = block_on(provider.open(&metadata, Arc::new(store)))?;
    let vector = index
        .as_vector()
        .ok_or_else(|| vortex_err!("Missing vector capability"))?;
    let mut offset = 0u64;
    let offsets = reference
        .snapshot
        .files
        .iter()
        .map(|file| {
            let start = offset;
            offset += file.row_count;
            (file.id, (start, file.row_count))
        })
        .collect::<BTreeMap<_, _>>();
    let filter = RowFilter::try_new(reference.snapshot, None, BTreeSet::new())?;
    writeln!(stream, "{{\"ready\":true,\"rows\":{offset}}}")?;
    stream.flush()?;
    let mut reader = BufReader::new(stream.get_ref().try_clone()?);
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let result = (|| -> VortexResult<Vec<u64>> {
            let query: Query =
                serde_json::from_str(&line).map_err(|err| vortex_err!("Query: {err}"))?;
            let mut backend_options = json!({
                "max_check": query.max_check,
                "internal_results": query.internal_results,
                "search_pages": query.search_pages,
            });
            if let Some(rabitq) = query.rabitq {
                backend_options["rabitq"] =
                    serde_json::to_value(rabitq).map_err(|err| vortex_err!("Options: {err}"))?;
            }
            if let Some(rerank) = query.rerank {
                backend_options["rerank"] = json!(rerank);
            }
            let options = VectorSearchOptions {
                k: query.k,
                mode: SearchMode::Approximate,
                backend_options: Bytes::from(
                    serde_json::to_vec(&backend_options)
                        .map_err(|err| vortex_err!("Options: {err}"))?,
                ),
            };
            block_on(vector.search(&query.query, &options, &filter))?
                .into_iter()
                .map(|hit| {
                    let (start, rows) = offsets
                        .get(&hit.row.file_id)
                        .ok_or_else(|| vortex_err!("Unknown result file"))?;
                    if hit.row.row_offset >= *rows {
                        vortex_bail!("Result row out of bounds");
                    }
                    Ok(start + hit.row.row_offset)
                })
                .collect()
        })();
        let response = match result {
            Ok(ids) => json!({"positions": ids}),
            Err(err) => json!({"error": err.to_string()}),
        };
        serde_json::to_writer(&mut stream, &response)
            .map_err(|err| vortex_err!("Response: {err}"))?;
        writeln!(stream)?;
        stream.flush()?;
    }
    Ok(())
}
