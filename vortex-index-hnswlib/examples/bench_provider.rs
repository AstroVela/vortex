// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Time individual ANN queries on one opened immutable generation, without SQL or take.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io;
use std::io::Write;
use std::num::NonZeroUsize;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process;
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
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::SearchMode;
use vortex_index::Snapshot;
use vortex_index::VectorSearchOptions;
use vortex_index::store::LocalGeneration;
use vortex_index::store::LocalIndexStore;
use vortex_index::store::LocalStoreLimits;
use vortex_index_hnswlib::HNSWLIB_REVISION;
use vortex_index_hnswlib::HnswLimits;
use vortex_index_hnswlib::HnswProvider;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    format_version: u32,
    reference: PathBuf,
    queries: Vec<Vec<f32>>,
    k: NonZeroUsize,
    ef: u32,
    warmup_rounds: u32,
    rounds: u32,
}

impl Config {
    fn validate(&self) -> VortexResult<()> {
        if self.format_version != 1
            || !self.reference.is_absolute()
            || !(1..=100).contains(&self.rounds)
            || self.warmup_rounds > 100
            || self.queries.is_empty()
            || self.queries.len() > 10_000
            || self.k.get() > 4096
            || self.ef < u32::try_from(self.k.get())?
            || self.ef > 1_048_576
        {
            vortex_bail!("Invalid Provider benchmark config");
        }
        let dimension = self.queries[0].len();
        if !(1..=4096).contains(&dimension)
            || self.queries.iter().any(|query| {
                query.len() != dimension || query.iter().any(|value| !value.is_finite())
            })
        {
            vortex_bail!("Invalid Provider benchmark vectors");
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

fn barrier(enabled: bool, phase: &str) -> VortexResult<()> {
    if enabled {
        println!("SIFT_BENCH_PHASE {} {phase}", process::id());
        io::stdout().flush()?;
        let mut acknowledgment = String::new();
        io::stdin().read_line(&mut acknowledgment)?;
        if acknowledgment.trim() != "continue" {
            vortex_bail!("Missing phase acknowledgment");
        }
    }
    Ok(())
}

fn main() -> VortexResult<()> {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let [config_path, output, tail @ ..] = args.as_slice() else {
        vortex_bail!("Usage: bench_provider CONFIG.json OUTPUT.json [--barriers]");
    };
    let barriers = match tail {
        [] => false,
        [flag] if flag == "--barriers" => true,
        _ => vortex_bail!("Unknown benchmark arguments"),
    };
    let output = PathBuf::from(output);
    if output.exists() {
        vortex_bail!("Benchmark output already exists");
    }
    let config: Config =
        serde_json::from_slice(&fs::read(config_path)?).map_err(|err| vortex_err!("{err}"))?;
    config.validate()?;
    let reference: Reference = serde_json::from_slice(&fs::read(&config.reference)?)
        .map_err(|err| vortex_err!("{err}"))?;
    if reference.format_version != 1 {
        vortex_bail!("Unsupported reference version");
    }
    let root = config
        .reference
        .parent()
        .ok_or_else(|| vortex_err!("Missing reference parent"))?;
    let scratch = tempfile::Builder::new()
        .prefix(".hnsw-provider-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in(root)?;
    barrier(barriers, "before_open")?;
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
    let index = block_on(
        HnswProvider::try_new(scratch.path().to_owned(), HnswLimits::default())?
            .open(&metadata, Arc::new(store)),
    )?;
    let open_ms = start.elapsed().as_secs_f64() * 1000.0;
    let vector = index
        .as_vector()
        .ok_or_else(|| vortex_err!("Missing vector capability"))?;
    if vector.spec().dimension.get() != config.queries[0].len()
        || vector.spec().metric != DistanceMetric::SquaredL2
    {
        vortex_bail!("Vector spec mismatch");
    }
    let options = VectorSearchOptions {
        k: config.k,
        mode: SearchMode::Approximate,
        backend_options: Bytes::from(
            serde_json::to_vec(&serde_json::json!({"ef": config.ef}))
                .map_err(|err| vortex_err!("{err}"))?,
        ),
    };
    let filter = RowFilter::try_new(reference.snapshot, None, BTreeSet::new())?;
    barrier(barriers, "ready")?;
    let mut samples = Vec::new();
    for round in 0..config.warmup_rounds + config.rounds {
        barrier(barriers, &format!("before_round_{round}"))?;
        for (query_id, query) in config.queries.iter().enumerate() {
            if round == 0 && query_id == 0 {
                barrier(barriers, "before_first_search")?;
            }
            let start = Instant::now();
            let hits = block_on(vector.search(query, &options, &filter))?;
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            if round == 0 && query_id == 0 {
                barrier(barriers, "after_first_search")?;
            }
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
        barrier(barriers, &format!("after_round_{round}"))?;
    }
    let start = Instant::now();
    drop(index);
    drop(scratch);
    let close_ms = start.elapsed().as_secs_f64() * 1000.0;
    barrier(barriers, "after_close")?;
    fs::write(
        output,
        serde_json::to_vec(&Report {
            format_version: 1,
            revision: HNSWLIB_REVISION,
            open_ms,
            close_ms,
            samples,
        })
        .map_err(|err| vortex_err!("{err}"))?,
    )?;
    Ok(())
}
