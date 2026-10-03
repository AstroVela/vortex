// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Open a qualified generation once and time individual Provider searches without SQL.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::num::NonZeroUsize;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use futures::executor::block_on;
use serde::Deserialize;
use serde::Serialize;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_index::DistanceMetric;
use vortex_index::IndexProvider;
use vortex_index::IndexStore;
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::SearchMode;
use vortex_index::Snapshot;
use vortex_index::VectorSearchOptions;
use vortex_index::store::LocalGeneration;
use vortex_index::store::LocalIndexStore;
use vortex_index::store::LocalStoreLimits;
use vortex_index_spfresh::SPFRESH_REVISION;
use vortex_index_spfresh::SpFreshBundle;
use vortex_index_spfresh::SpFreshLimits;
use vortex_index_spfresh::SpFreshProvider;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    format_version: u32,
    reference: PathBuf,
    queries: Vec<Vec<f32>>,
    k: NonZeroUsize,
    max_check: u32,
    internal_results: u32,
    search_pages: u32,
    warmup_rounds: u32,
    rounds: u32,
}

impl Config {
    fn validate(&self) -> VortexResult<()> {
        if self.format_version != 1
            || !self.reference.is_absolute()
            || self.rounds == 0
            || self.rounds > 100
            || self.warmup_rounds > 100
            || self.queries.is_empty()
            || self.queries.len() > 256
            || self.k.get() > 4096
            || self.internal_results < u32::try_from(self.k.get())?
            || self.internal_results > 4096
            || self.max_check < self.internal_results
            || self.max_check > 1_048_576
            || self.search_pages == 0
            || self.search_pages > 4096
            || u64::from(self.search_pages) * u64::from(self.internal_results) * 4096
                > 256 * 1024 * 1024
        {
            vortex_bail!("Invalid Provider benchmark configuration");
        }
        let dimension = self.queries[0].len();
        if dimension == 0
            || dimension > 4096
            || self.queries.iter().any(|query| {
                query.len() != dimension || query.iter().any(|component| !component.is_finite())
            })
        {
            vortex_bail!("Invalid Provider benchmark query shape or components");
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    format_version: u32,
    snapshot: Snapshot,
    generation: LocalGeneration,
    #[serde(rename = "dtype")]
    _dtype: serde_json::Value,
}

#[derive(Deserialize)]
struct Descriptor {
    bundle: SpFreshBundle,
}

#[derive(Serialize)]
struct Hit {
    row: RowAddress,
    distance: f32,
}

#[derive(Serialize)]
struct Sample {
    round: u32,
    query: usize,
    warmup: bool,
    latency_ms: f64,
    hits: Vec<Hit>,
}

#[derive(Serialize)]
struct Report {
    format_version: u32,
    revision: &'static str,
    open_ms: f64,
    close_ms: f64,
    samples: Vec<Sample>,
}

fn main() -> VortexResult<()> {
    let args = env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let [config_path, output] = args.as_slice() else {
        vortex_bail!("Usage: bench_provider CONFIG.json OUTPUT.json");
    };
    if output.exists() {
        vortex_bail!("Provider benchmark output already exists");
    }
    let config: Config = serde_json::from_slice(&fs::read(config_path)?)
        .map_err(|err| vortex_err!("Benchmark config: {err}"))?;
    config.validate()?;
    let reference: Reference = serde_json::from_slice(&fs::read(&config.reference)?)
        .map_err(|err| vortex_err!("Benchmark reference: {err}"))?;
    if reference.format_version != 1 {
        vortex_bail!("Unsupported reference format");
    }
    let root = config
        .reference
        .parent()
        .ok_or_else(|| vortex_err!("Reference has no parent"))?;
    let scratch = tempfile::Builder::new()
        .prefix("provider-benchmark-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(root)?;
    let start = Instant::now();
    let (store, metadata) = LocalIndexStore::open(
        root,
        &reference.generation,
        &reference.snapshot,
        LocalStoreLimits {
            max_artifact_bytes: 1024 * 1024 * 1024,
            max_manifest_bytes: 16 * 1024 * 1024,
        },
    )?;
    let descriptor_artifact = metadata
        .artifacts
        .iter()
        .find(|artifact| artifact.path == "spfresh/bundle.json")
        .ok_or_else(|| vortex_err!("Missing SPFresh descriptor"))?;
    let descriptor: Descriptor =
        serde_json::from_slice(&block_on(store.read(descriptor_artifact))?)
            .map_err(|err| vortex_err!("Benchmark bundle: {err}"))?;
    if descriptor.bundle.posting_page_limit != config.search_pages {
        vortex_bail!("Benchmark page limit does not match the bundle");
    }
    // Match SQL's empty-options path when measuring its default configuration.
    let backend_options = if config.max_check == 4096
        && config.internal_results == u32::try_from(config.k.get())?.max(64)
    {
        Bytes::new()
    } else {
        Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "max_check": config.max_check,
                "internal_results": config.internal_results,
                "search_pages": config.search_pages,
            }))
            .map_err(|err| vortex_err!("Query options: {err}"))?,
        )
    };
    let options = VectorSearchOptions {
        k: config.k,
        mode: SearchMode::Approximate,
        backend_options,
    };
    let provider = SpFreshProvider::try_new(scratch.path().to_owned(), SpFreshLimits::default())?;
    let index = block_on(provider.open(&metadata, Arc::new(store)))?;
    let open_ms = start.elapsed().as_secs_f64() * 1000.0;
    let vector = index
        .as_vector()
        .ok_or_else(|| vortex_err!("Missing vector capability"))?;
    if vector.spec().dimension.get() != config.queries[0].len()
        || vector.spec().metric != DistanceMetric::SquaredL2
    {
        vortex_bail!("Benchmark vector specification mismatch");
    }
    let filter = RowFilter::try_new(reference.snapshot, None, BTreeSet::new())?;
    let mut samples = Vec::new();
    for round in 0..config.warmup_rounds + config.rounds {
        for (query_id, query) in config.queries.iter().enumerate() {
            let start = Instant::now();
            let hits = block_on(vector.search(query, &options, &filter))?;
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            samples.push(Sample {
                round,
                query: query_id,
                warmup: round < config.warmup_rounds,
                latency_ms,
                hits: hits
                    .into_iter()
                    .map(|hit| Hit {
                        row: hit.row,
                        distance: hit.distance,
                    })
                    .collect(),
            });
        }
    }
    let start = Instant::now();
    drop(index);
    drop(scratch);
    let report = Report {
        format_version: 1,
        revision: SPFRESH_REVISION,
        open_ms,
        close_ms: start.elapsed().as_secs_f64() * 1000.0,
        samples,
    };
    fs::write(
        output,
        serde_json::to_vec(&report).map_err(|err| vortex_err!("Benchmark report: {err}"))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VortexResult<Config> {
        serde_json::from_value(serde_json::json!({
            "format_version": 1, "reference": "/tmp/index.json", "queries": [[0.0, 1.0]],
            "k": 10, "max_check": 4096, "internal_results": 64, "search_pages": 256,
            "warmup_rounds": 1, "rounds": 3,
        }))
        .map_err(|err| vortex_err!("{err}"))
    }

    #[test]
    fn test_config_validates() -> VortexResult<()> {
        config()?.validate()
    }

    #[test]
    fn test_invalid_configuration_rejected() -> VortexResult<()> {
        let mut invalid = config()?;
        invalid.rounds = 0;
        assert!(invalid.validate().is_err());
        invalid = config()?;
        invalid.queries.push(vec![0.0]);
        assert!(invalid.validate().is_err());
        invalid = config()?;
        invalid.queries[0][0] = f32::NAN;
        assert!(invalid.validate().is_err());
        invalid = config()?;
        invalid.search_pages = 4096;
        assert!(invalid.validate().is_err());
        invalid = config()?;
        invalid.reference = "index.json".into();
        assert!(invalid.validate().is_err());
        Ok(())
    }
}
