// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::File;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use arrow_array::Array as ArrowArray;
use arrow_array::Date32Array;
use arrow_array::Decimal128Array;
use arrow_array::Float32Array;
use arrow_array::Float64Array;
use arrow_array::Int64Array;
use arrow_array::StringViewArray;
use arrow_array::cast::AsArray;
use arrow_cast::cast;
use arrow_schema::DataType;
use divan::Bencher;
use futures::FutureExt;
use futures::TryStreamExt;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::and;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutRef;
use vortex_layout::layouts::self_paced::SelfPacedLayoutStrategy;
use vortex_layout::plan::exec::Conjunct;
use vortex_layout::plan::exec::FieldId;
use vortex_layout::plan::exec::Predicate;
use vortex_layout::plan::exec::RetentionPolicy;
use vortex_layout::plan::exec::RunOptions;
use vortex_layout::plan::exec::ScanQuery;
use vortex_layout::plan::exec::SchedulePolicy;
use vortex_layout::plan::exec::SourcePlan;
use vortex_layout::plan::exec::SpeculativeIoConfig;
use vortex_layout::plan::exec::SpeculativeReadPolicy;
use vortex_layout::plan::exec::run_self_paced;
use vortex_layout::plan::exec::run_self_paced_ranges;
use vortex_layout::plan::exec::stable_output_hash;
use vortex_layout::plan::exec::write_execution_trace;
use vortex_layout::scan::scan_builder::ScanBuilder;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

fn main() {
    if let Ok(iterations) = std::env::var("VORTEX_SELF_PACED_COMPARE_ITERATIONS") {
        match std::env::var("VORTEX_SELF_PACED_COMPARE_WORKLOAD") {
            Ok(workload) if workload.starts_with("fineweb_") => {
                LazyLock::force(&FINEWEB_FIXTURE);
            }
            Ok(workload) if workload.starts_with("clickbench_") => {
                LazyLock::force(&CLICKBENCH_FIXTURE);
            }
            Ok(workload) if workload.starts_with("tpch_") => {
                LazyLock::force(&TPCH_FIXTURE);
            }
            Ok(_) | Err(_) => force_all_fixtures(),
        }
        RUNTIME
            .block_on(compare_cases(iterations.parse::<usize>().unwrap()))
            .unwrap();
        return;
    }
    if let Ok(requested) = std::env::var("VORTEX_SELF_PACED_PROFILE") {
        if requested.starts_with("fineweb-") {
            LazyLock::force(&FINEWEB_FIXTURE);
        } else if requested.starts_with("tpch-") {
            LazyLock::force(&TPCH_FIXTURE);
        } else if requested.starts_with("q40-") || requested.starts_with("q42-") {
            LazyLock::force(&CLICKBENCH_FIXTURE);
        } else {
            LazyLock::force(&FIXTURE);
        }
        let iterations = std::env::var("VORTEX_SELF_PACED_PROFILE_ITERATIONS")
            .map_or(Ok(500), |value| value.parse::<usize>())
            .unwrap();
        RUNTIME
            .block_on(profile_case(&requested, iterations))
            .unwrap();
        return;
    }
    if let Ok(trace_case) = std::env::var("VORTEX_SELF_PACED_TRACE") {
        if trace_case.starts_with("fineweb-") {
            LazyLock::force(&FINEWEB_FIXTURE);
        } else if trace_case.starts_with("clickbench-") {
            LazyLock::force(&CLICKBENCH_FIXTURE);
        } else if trace_case.starts_with("q6-")
            || trace_case.starts_with("q1-")
            || trace_case.starts_with("v1-friendly-")
        {
            LazyLock::force(&TPCH_FIXTURE);
        } else {
            LazyLock::force(&FIXTURE);
        }
        RUNTIME.block_on(trace_cases(&trace_case)).unwrap();
        return;
    }
    force_all_fixtures();
    RUNTIME.block_on(validate_outputs()).unwrap();
    divan::main();
}

fn force_all_fixtures() {
    LazyLock::force(&FIXTURE);
    LazyLock::force(&CLICKBENCH_FIXTURE);
    LazyLock::force(&TPCH_FIXTURE);
    LazyLock::force(&FINEWEB_FIXTURE);
}

const CHUNKS: usize = 4;
const ROWS_PER_CHUNK: usize = 262_144;
const TPCH_ROWS_PER_CHUNK: usize = 524_288;
const TRACE_CASES: [(&str, usize, usize, usize); 3] = [
    ("1", 1, 4_096, 1),
    ("50", 50, 16_384, 16),
    ("95", 95, 65_536, 16),
];

static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(16)
        .enable_all()
        .build()
        .unwrap()
});

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let _guard = RUNTIME.enter();
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_tokio();
    vortex_file::register_default_encodings(&session);
    enable_self_paced_file_encoding(&session);
    session
});

const SELF_PACED_FILE_EDITION: EditionId = EditionId::new("self-paced-file", 2026, 8, 0);

fn enable_self_paced_file_encoding(session: &VortexSession) {
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: SELF_PACED_FILE_EDITION,
            min_vortex_version: None,
        })
        .unwrap();
    editions
        .declare_inclusion(EditionInclusion::new(
            "vortex.primitive",
            SELF_PACED_FILE_EDITION,
        ))
        .unwrap();
    session.enable_edition(SELF_PACED_FILE_EDITION).unwrap();
}
static SPECULATIVE_IO_CONFIG: LazyLock<SpeculativeIoConfig> =
    LazyLock::new(benchmark_speculative_io_config);

struct Fixture {
    layout: LayoutRef,
    plan: SourcePlan,
    natural_splits: Arc<[u64]>,
    source: Arc<dyn SegmentSource>,
    serialized_bytes: usize,
    serialized_hash: u64,
    logical_array: ArrayRef,
    physical_splits: Option<PhysicalSplitCatalog>,
    physical_field_sources: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct PhysicalSplitCatalog {
    files: Vec<PhysicalSplitFile>,
}

#[derive(Deserialize)]
struct PhysicalSplitFile {
    row_count: u64,
    fields: BTreeMap<String, Vec<u64>>,
}

static FIXTURE: LazyLock<Fixture> = LazyLock::new(|| RUNTIME.block_on(build_fixture()).unwrap());
static CLICKBENCH_FIXTURE: LazyLock<Fixture> =
    LazyLock::new(|| RUNTIME.block_on(build_clickbench_fixture()).unwrap());
static TPCH_FIXTURE: LazyLock<Fixture> =
    LazyLock::new(|| RUNTIME.block_on(build_tpch_fixture()).unwrap());
static FINEWEB_FIXTURE: LazyLock<Fixture> =
    LazyLock::new(|| RUNTIME.block_on(build_fineweb_fixture()).unwrap());

fn clickbench_fixture() -> &'static Fixture {
    &CLICKBENCH_FIXTURE
}

fn tpch_fixture() -> &'static Fixture {
    &TPCH_FIXTURE
}

fn fineweb_fixture() -> &'static Fixture {
    &FINEWEB_FIXTURE
}

#[derive(Default)]
struct SourceCounts {
    requests: AtomicUsize,
    bytes: AtomicUsize,
    outstanding: AtomicUsize,
    peak_outstanding: AtomicUsize,
    requests_by_segment: parking_lot::Mutex<BTreeMap<vortex_layout::segments::SegmentId, usize>>,
}

#[derive(Clone, Copy, Debug)]
struct SourceSummary {
    requests: usize,
    bytes: usize,
    unique_segments: usize,
    segment_requests_min: usize,
    segment_requests_max: usize,
}

impl SourceCounts {
    fn summary(&self) -> SourceSummary {
        let requests = self.requests_by_segment.lock();
        let min = requests.values().copied().min().unwrap_or(0);
        let max = requests.values().copied().max().unwrap_or(0);
        SourceSummary {
            requests: self.requests.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            unique_segments: requests.len(),
            segment_requests_min: min,
            segment_requests_max: max,
        }
    }
}

struct CountingSource {
    inner: Arc<dyn SegmentSource>,
    counts: Arc<SourceCounts>,
    track_segments: bool,
}

impl CountingSource {
    fn new(inner: Arc<dyn SegmentSource>, track_segments: bool) -> (Arc<Self>, Arc<SourceCounts>) {
        let counts = Arc::new(SourceCounts::default());
        (
            Arc::new(Self {
                inner,
                counts: Arc::<SourceCounts>::clone(&counts),
                track_segments,
            }),
            counts,
        )
    }
}

impl SegmentSource for CountingSource {
    fn estimated_size(&self, id: vortex_layout::segments::SegmentId) -> Option<usize> {
        self.inner.estimated_size(id)
    }

    fn request(&self, id: vortex_layout::segments::SegmentId) -> SegmentFuture {
        self.counts.requests.fetch_add(1, Ordering::Relaxed);
        if self.track_segments {
            *self
                .counts
                .requests_by_segment
                .lock()
                .entry(id)
                .or_default() += 1;
        }
        let outstanding = self.counts.outstanding.fetch_add(1, Ordering::Relaxed) + 1;
        self.counts
            .peak_outstanding
            .fetch_max(outstanding, Ordering::Relaxed);
        let future = self.inner.request(id);
        let counts = Arc::<SourceCounts>::clone(&self.counts);
        async move {
            let result = future.await;
            counts.outstanding.fetch_sub(1, Ordering::Relaxed);
            if let Ok(buffer) = &result {
                counts.bytes.fetch_add(buffer.len(), Ordering::Relaxed);
            }
            result
        }
        .boxed()
    }
}

async fn build_fixture() -> VortexResult<Fixture> {
    let names = FieldNames::from(["a", "b", "c", "d"]);
    let chunks = (0..CHUNKS)
        .map(|chunk| {
            let base = chunk * ROWS_PER_CHUNK;
            let fields = vec![
                Buffer::from_iter((0..ROWS_PER_CHUNK).map(|row| ((base + row) % 100) as i64))
                    .into_array(),
                Buffer::from_iter(
                    (0..ROWS_PER_CHUNK).map(|row| (((base + row) * 17) % 100) as i64),
                )
                .into_array(),
                Buffer::from_iter((0..ROWS_PER_CHUNK).map(|row| (base + row) as i64)).into_array(),
                Buffer::from_iter((0..ROWS_PER_CHUNK).map(|row| -(base as i64 + row as i64)))
                    .into_array(),
            ];
            StructArray::try_new(names.clone(), fields, ROWS_PER_CHUNK, Validity::NonNullable)
                .map(IntoArray::into_array)
        })
        .collect::<VortexResult<Vec<_>>>()?;
    build_serialized_fixture(chunks).await
}

async fn build_serialized_fixture(chunks: Vec<ArrayRef>) -> VortexResult<Fixture> {
    build_serialized_fixture_with_physical_splits(chunks, None, Vec::new()).await
}

async fn build_serialized_fixture_with_physical_splits(
    chunks: Vec<ArrayRef>,
    catalog_env: Option<&str>,
    physical_field_sources: Vec<Vec<String>>,
) -> VortexResult<Fixture> {
    let dtype = chunks[0].dtype().clone();
    let chunked_array = ChunkedArray::try_new(chunks, dtype)?.into_array();
    let logical_array = chunked_array.clone();
    let mut serialized = ByteBufferMut::empty();
    SESSION
        .write_options()
        .with_strategy(Arc::new(SelfPacedLayoutStrategy::default()))
        .with_file_statistics(Vec::new())
        .write(&mut serialized, chunked_array.to_array_stream())
        .await?;
    let serialized = serialized.freeze();
    let serialized_bytes = serialized.len();
    let serialized_hash = serialized
        .as_ref()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        });
    let file = SESSION.open_options().open_buffer(serialized)?;
    let layout = Arc::clone(file.footer().layout());
    let source = file.segment_source();
    let plan = SourcePlan::try_from_layout(&layout)?;
    let natural_splits = std::iter::once(0)
        .chain(plan.chunks.iter().map(|chunk| chunk.root_coverage.end))
        .collect::<Vec<_>>()
        .into();
    Ok(Fixture {
        layout,
        plan,
        natural_splits,
        source,
        serialized_bytes,
        serialized_hash,
        logical_array,
        physical_splits: catalog_env
            .and_then(std::env::var_os)
            .map(|path| {
                let path = PathBuf::from(path);
                let file = File::open(&path).map_err(|error| {
                    vortex_error::vortex_err!(
                        "failed to open physical split catalog {}: {error}",
                        path.display()
                    )
                })?;
                serde_json::from_reader(file).map_err(|error| {
                    vortex_error::vortex_err!(
                        "failed to parse physical split catalog {}: {error}",
                        path.display()
                    )
                })
            })
            .transpose()?,
        physical_field_sources,
    })
}

async fn rechunk_fixture_at_natural_splits(
    fixture: &Fixture,
    natural_splits: Arc<[u64]>,
) -> VortexResult<Fixture> {
    let chunks = natural_splits
        .windows(2)
        .map(|bounds| {
            fixture
                .logical_array
                .slice(usize::try_from(bounds[0])?..usize::try_from(bounds[1])?)
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let rechunked = build_serialized_fixture(chunks).await?;
    if rechunked.natural_splits.as_ref() != natural_splits.as_ref() {
        vortex_error::vortex_bail!(
            "rechunked fixture boundaries do not match the real natural splits"
        );
    }
    Ok(rechunked)
}

const CLICKBENCH_PARQUET_COLUMNS: [&str; 24] = [
    "EventTime",
    "UserID",
    "CounterID",
    "RegionID",
    "IsMobile",
    "ResponseEndTiming",
    "SendTiming",
    "EventDate",
    "WatchID",
    "AdvEngineID",
    "ResolutionWidth",
    "SearchEngineID",
    "TraficSourceID",
    "SearchEngineID",
    "RefererHash",
    "URLHash",
    "IsRefresh",
    "RefererHash",
    "URLHash",
    "WindowClientWidth",
    "WindowClientHeight",
    "DontCountHits",
    "IsLink",
    "IsDownload",
];

async fn build_clickbench_fixture() -> VortexResult<Fixture> {
    let parquet_dir = std::env::var_os("VORTEX_CLICKBENCH_PARQUET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/vortex-ssd/data/clickbench_partitioned/parquet"));
    let file_count = std::env::var("VORTEX_CLICKBENCH_MAX_FILES")
        .map_or(Ok(10), |value| value.parse::<usize>())
        .map_err(|error| {
            vortex_error::vortex_err!("invalid VORTEX_CLICKBENCH_MAX_FILES: {error}")
        })?;
    let mut chunks = Vec::with_capacity(file_count);
    for shard in 0..file_count {
        let path = parquet_dir.join(format!("hits_{shard}.parquet"));
        let file = File::open(&path).map_err(|error| {
            vortex_error::vortex_err!("failed to open {}: {error}", path.display())
        })?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
            vortex_error::vortex_err!("failed to inspect {}: {error}", path.display())
        })?;
        let mut projection = CLICKBENCH_PARQUET_COLUMNS
            .iter()
            .map(|name| {
                builder.schema().index_of(name).map_err(|error| {
                    vortex_error::vortex_err!("{name} in {}: {error}", path.display())
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        projection.sort_unstable();
        projection.dedup();
        let projection = ProjectionMask::roots(builder.parquet_schema(), projection);
        builder = builder.with_projection(projection).with_batch_size(131_072);

        let mut fields = (0..CLICKBENCH_FIELD_NAMES.len())
            .map(|_| Vec::<i64>::new())
            .collect::<Vec<_>>();
        for batch in builder.build().map_err(|error| {
            vortex_error::vortex_err!("failed to read {}: {error}", path.display())
        })? {
            let batch = batch.map_err(|error| {
                vortex_error::vortex_err!("failed to decode {}: {error}", path.display())
            })?;
            for (values, source_name) in fields.iter_mut().zip(CLICKBENCH_PARQUET_COLUMNS) {
                let source = batch.column_by_name(source_name).ok_or_else(|| {
                    vortex_error::vortex_err!("missing {source_name} in {}", path.display())
                })?;
                let casted = cast(source, &DataType::Int64).map_err(|error| {
                    vortex_error::vortex_err!(
                        "cannot cast {source_name} from {:?} in {} to i64: {error}",
                        source.data_type(),
                        path.display()
                    )
                })?;
                let casted = casted
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        vortex_error::vortex_err!("cast of {source_name} did not produce i64")
                    })?;
                if casted.null_count() != 0 {
                    vortex_error::vortex_bail!(
                        "{source_name} in {} contains nulls",
                        path.display()
                    );
                }
                values.extend_from_slice(casted.values());
            }
        }
        let row_count = fields.first().map_or(0, Vec::len);
        let arrays = fields
            .into_iter()
            .map(|values| Buffer::from_iter(values).into_array())
            .collect::<Vec<_>>();
        chunks.push(
            StructArray::try_new(
                FieldNames::from(CLICKBENCH_FIELD_NAMES),
                arrays,
                row_count,
                Validity::NonNullable,
            )?
            .into_array(),
        );
    }
    eprintln!(
        "clickbench_fixture files={file_count} chunks={} rows={}",
        chunks.len(),
        chunks.iter().map(|chunk| chunk.len()).sum::<usize>(),
    );
    build_serialized_fixture_with_physical_splits(
        chunks,
        Some("VORTEX_CLICKBENCH_SPLIT_CATALOG"),
        CLICKBENCH_PARQUET_COLUMNS
            .iter()
            .map(|name| vec![(*name).to_string()])
            .collect(),
    )
    .await
}

const FINEWEB_SOURCE_COLUMNS: [&str; 7] = [
    "dump",
    "date",
    "url",
    "text",
    "language",
    "language_score",
    "file_path",
];

const FINEWEB_FIELD_NAMES: [&str; 14] = [
    "dump_hash",
    "date_ym",
    "url_hash",
    "text_hash",
    "language_hash",
    "language_score_ppm",
    "file_path_hash",
    "url_google",
    "text_google",
    "google_any",
    "text_vortex",
    "url_espn",
    "file_path_old",
    "text_len",
];

fn fineweb_parquet_paths() -> VortexResult<Vec<PathBuf>> {
    let input = std::env::var_os("VORTEX_FINEWEB_PARQUET")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/vortex-ssd/data/fineweb/parquet/sample.parquet"));
    let mut paths = if input.is_dir() {
        std::fs::read_dir(&input)
            .map_err(|error| {
                vortex_error::vortex_err!("failed to list {}: {error}", input.display())
            })?
            .map(|entry| {
                entry
                    .map(|entry| entry.path())
                    .map_err(|error| vortex_error::vortex_err!("failed to list FineWeb: {error}"))
            })
            .filter_map(|path| match path {
                Ok(path)
                    if path
                        .extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("parquet")) =>
                {
                    Some(Ok(path))
                }
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<VortexResult<Vec<_>>>()?
    } else {
        vec![input]
    };
    paths.sort_unstable();
    if let Some(max_files) = std::env::var("VORTEX_FINEWEB_MAX_FILES")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| vortex_error::vortex_err!("invalid VORTEX_FINEWEB_MAX_FILES: {error}"))?
    {
        paths.truncate(max_files);
    }
    if paths.is_empty() {
        vortex_error::vortex_bail!("FineWeb input contains no Parquet files");
    }
    Ok(paths)
}

async fn build_fineweb_fixture() -> VortexResult<Fixture> {
    let paths = fineweb_parquet_paths()?;
    let file_count = paths.len();
    let mut chunks = Vec::new();
    for path in paths {
        let file = File::open(&path).map_err(|error| {
            vortex_error::vortex_err!("failed to open {}: {error}", path.display())
        })?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
            vortex_error::vortex_err!("failed to inspect {}: {error}", path.display())
        })?;
        let projection = FINEWEB_SOURCE_COLUMNS
            .iter()
            .map(|name| {
                builder.schema().index_of(name).map_err(|error| {
                    vortex_error::vortex_err!("{name} in {}: {error}", path.display())
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let projection = ProjectionMask::roots(builder.parquet_schema(), projection);
        builder = builder.with_projection(projection).with_batch_size(100_000);

        for batch in builder.build().map_err(|error| {
            vortex_error::vortex_err!("failed to read {}: {error}", path.display())
        })? {
            let batch = batch.map_err(|error| {
                vortex_error::vortex_err!("failed to decode {}: {error}", path.display())
            })?;
            let dump = batch.column_by_name("dump").unwrap();
            let date = batch.column_by_name("date").unwrap();
            let url = batch.column_by_name("url").unwrap();
            let text = batch.column_by_name("text").unwrap();
            let language = batch.column_by_name("language").unwrap();
            let language_score = batch.column_by_name("language_score").unwrap();
            let file_path = batch.column_by_name("file_path").unwrap();
            let mut fields = (0..FINEWEB_FIELD_NAMES.len())
                .map(|_| Vec::with_capacity(batch.num_rows()))
                .collect::<Vec<Vec<i64>>>();

            for row in 0..batch.num_rows() {
                let dump = arrow_string_value(dump.as_ref(), row).unwrap_or_default();
                let date = arrow_string_value(date.as_ref(), row).unwrap_or_default();
                let url = arrow_string_value(url.as_ref(), row).unwrap_or_default();
                let text = arrow_string_value(text.as_ref(), row).unwrap_or_default();
                let language = arrow_string_value(language.as_ref(), row).unwrap_or_default();
                let file_path = arrow_string_value(file_path.as_ref(), row).unwrap_or_default();
                let url_google = url.contains("google");
                let text_google = text.contains("Google");
                fields[0].push(stable_string_hash(dump));
                fields[1].push(date_year_month(date));
                fields[2].push(stable_string_hash(url));
                fields[3].push(stable_string_hash(text));
                fields[4].push(stable_string_hash(language));
                fields[5].push(language_score_ppm(language_score.as_ref(), row));
                fields[6].push(stable_string_hash(file_path));
                fields[7].push(i64::from(url_google));
                fields[8].push(i64::from(text_google));
                fields[9].push(i64::from(
                    url.contains(".google.") || text.contains(" Google "),
                ));
                fields[10].push(i64::from(text.contains(" vortex ")));
                fields[11].push(i64::from(url.contains("espn")));
                fields[12].push(i64::from(file_path.contains("/CC-MAIN-2014-")));
                fields[13].push(i64::try_from(text.len()).unwrap_or(i64::MAX));
            }
            let arrays = fields
                .into_iter()
                .map(|values| Buffer::from_iter(values).into_array())
                .collect::<Vec<_>>();
            chunks.push(
                StructArray::try_new(
                    FieldNames::from(FINEWEB_FIELD_NAMES),
                    arrays,
                    batch.num_rows(),
                    Validity::NonNullable,
                )?
                .into_array(),
            );
        }
    }
    eprintln!(
        "fineweb_fixture files={file_count} chunks={} rows={}",
        chunks.len(),
        chunks.iter().map(|chunk| chunk.len()).sum::<usize>(),
    );
    let physical_field_sources = [
        &["dump"][..],
        &["date"],
        &["url"],
        &["text"],
        &["language"],
        &["language_score"],
        &["file_path"],
        &["url"],
        &["text"],
        &["url", "text"],
        &["text"],
        &["url"],
        &["file_path"],
        &["text"],
    ]
    .into_iter()
    .map(|names| names.iter().map(|name| (*name).to_string()).collect())
    .collect();
    build_serialized_fixture_with_physical_splits(
        chunks,
        Some("VORTEX_FINEWEB_SPLIT_CATALOG"),
        physical_field_sources,
    )
    .await
}

fn arrow_string_value(array: &dyn ArrowArray, index: usize) -> Option<&str> {
    if array.is_null(index) {
        return None;
    }
    match array.data_type() {
        DataType::Utf8 => Some(array.as_string::<i32>().value(index)),
        DataType::LargeUtf8 => Some(array.as_string::<i64>().value(index)),
        DataType::Utf8View => Some(
            array
                .as_any()
                .downcast_ref::<StringViewArray>()?
                .value(index),
        ),
        _ => None,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "FineWeb language scores are finite values in [0, 1] scaled to integer ppm"
)]
fn language_score_ppm(array: &dyn ArrowArray, index: usize) -> i64 {
    if array.is_null(index) {
        return 0;
    }
    let score = match array.data_type() {
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|array| array.value(index)),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|array| f64::from(array.value(index))),
        _ => None,
    };
    score.map_or(0, |score| (score * 1_000_000.0) as i64)
}

fn stable_string_hash(value: &str) -> i64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        }) as i64
}

fn date_year_month(value: &str) -> i64 {
    value
        .bytes()
        .filter(u8::is_ascii_digit)
        .take(6)
        .fold(0i64, |number, digit| number * 10 + i64::from(digit - b'0'))
}

async fn build_tpch_fixture() -> VortexResult<Fixture> {
    let Some(path) = std::env::var_os("VORTEX_TPCH_LINEITEM_PARQUET").map(PathBuf::from) else {
        return build_synthetic_tpch_fixture(TPCH_ROWS_PER_CHUNK).await;
    };
    let file = File::open(&path)
        .map_err(|error| vortex_error::vortex_err!("failed to open {}: {error}", path.display()))?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
        vortex_error::vortex_err!("failed to inspect {}: {error}", path.display())
    })?;
    const COLUMNS: [(&str, i128); 8] = [
        ("l_orderkey", 1),
        ("l_partkey", 1),
        ("l_suppkey", 1),
        ("l_quantity", 100),
        ("l_extendedprice", 1),
        ("l_discount", 1),
        ("l_tax", 1),
        ("l_shipdate", 1),
    ];
    let projection = COLUMNS
        .iter()
        .map(|(name, _)| {
            builder
                .schema()
                .index_of(name)
                .map_err(|error| vortex_error::vortex_err!("{name} in {}: {error}", path.display()))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), projection);
    builder = builder
        .with_projection(projection)
        .with_batch_size(TPCH_ROWS_PER_CHUNK);
    let names = FieldNames::from([
        "orderkey",
        "partkey",
        "suppkey",
        "quantity",
        "extendedprice",
        "discount",
        "tax",
        "shipdate",
    ]);
    let mut chunks = Vec::new();
    for batch in builder
        .build()
        .map_err(|error| vortex_error::vortex_err!("failed to read {}: {error}", path.display()))?
    {
        let batch = batch.map_err(|error| {
            vortex_error::vortex_err!("failed to decode {}: {error}", path.display())
        })?;
        let arrays = COLUMNS
            .iter()
            .map(|(name, divisor)| {
                tpch_i64_values(
                    batch.column_by_name(name).ok_or_else(|| {
                        vortex_error::vortex_err!("missing {name} in {}", path.display())
                    })?,
                    *divisor,
                )
                .map(|values| Buffer::from_iter(values).into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        chunks.push(
            StructArray::try_new(
                names.clone(),
                arrays,
                batch.num_rows(),
                Validity::NonNullable,
            )?
            .into_array(),
        );
    }
    eprintln!(
        "tpch_fixture source={} chunks={} rows={}",
        path.display(),
        chunks.len(),
        chunks.iter().map(|chunk| chunk.len()).sum::<usize>(),
    );
    build_serialized_fixture_with_physical_splits(
        chunks,
        Some("VORTEX_TPCH_SPLIT_CATALOG"),
        COLUMNS
            .iter()
            .map(|(name, _)| vec![(*name).to_string()])
            .collect(),
    )
    .await
}

fn tpch_i64_values(array: &dyn ArrowArray, divisor: i128) -> VortexResult<Vec<i64>> {
    if array.null_count() != 0 {
        vortex_error::vortex_bail!("TPC-H input contains nulls");
    }
    if let Some(array) = array.as_any().downcast_ref::<Decimal128Array>() {
        return array
            .values()
            .iter()
            .map(|value| Ok(i64::try_from(*value / divisor)?))
            .collect();
    }
    if let Some(array) = array.as_any().downcast_ref::<Date32Array>() {
        return Ok(array
            .values()
            .iter()
            .map(|value| i64::from(*value))
            .collect());
    }
    let casted = cast(array, &DataType::Int64).map_err(|error| {
        vortex_error::vortex_err!("cannot cast {:?} to i64: {error}", array.data_type())
    })?;
    let casted = casted
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| vortex_error::vortex_err!("TPC-H cast did not produce i64"))?;
    casted
        .values()
        .iter()
        .map(|value| Ok(i64::try_from(i128::from(*value) / divisor)?))
        .collect()
}

async fn build_synthetic_tpch_fixture(rows_per_chunk: usize) -> VortexResult<Fixture> {
    let names = FieldNames::from([
        "orderkey",
        "partkey",
        "suppkey",
        "quantity",
        "extendedprice",
        "discount",
        "tax",
        "shipdate",
    ]);
    let chunks = (0..CHUNKS)
        .map(|chunk| {
            let base = chunk * rows_per_chunk;
            let fields = vec![
                Buffer::from_iter((0..rows_per_chunk).map(|row| (base + row) as i64)).into_array(),
                Buffer::from_iter(
                    (0..rows_per_chunk).map(|row| (((base + row) * 13) % 200_000) as i64),
                )
                .into_array(),
                Buffer::from_iter(
                    (0..rows_per_chunk).map(|row| (((base + row) * 29) % 10_000) as i64),
                )
                .into_array(),
                Buffer::from_iter(
                    (0..rows_per_chunk).map(|row| (((base + row) * 7) % 50 + 1) as i64),
                )
                .into_array(),
                Buffer::from_iter(
                    (0..rows_per_chunk).map(|row| (((base + row) * 101) % 10_000_000 + 100) as i64),
                )
                .into_array(),
                Buffer::from_iter((0..rows_per_chunk).map(|row| (((base + row) * 17) % 11) as i64))
                    .into_array(),
                Buffer::from_iter((0..rows_per_chunk).map(|row| (((base + row) * 19) % 9) as i64))
                    .into_array(),
                Buffer::from_iter(
                    (0..rows_per_chunk).map(|row| (((base + row) * 23) % 2_500) as i64),
                )
                .into_array(),
            ];
            StructArray::try_new(names.clone(), fields, rows_per_chunk, Validity::NonNullable)
                .map(IntoArray::into_array)
        })
        .collect::<VortexResult<Vec<_>>>()?;
    build_serialized_fixture(chunks).await
}

fn query(
    selectivity: usize,
) -> (
    ScanQuery,
    vortex_array::expr::Expression,
    vortex_array::expr::Expression,
) {
    let query = ScanQuery {
        conjuncts: vec![
            Conjunct {
                field: FieldId(0),
                predicate: Predicate::LessThan(selectivity as i64),
            },
            Conjunct {
                field: FieldId(1),
                predicate: Predicate::LessThan(100),
            },
        ],
        projection: vec![FieldId(2), FieldId(3)],
    };
    let filter = and(
        lt(get_item("a", root()), lit(selectivity as i64)),
        lt(get_item("b", root()), lit(100i64)),
    );
    let projection = select(["c", "d"], root());
    (query, filter, projection)
}

fn clickbench_selective_query() -> (
    ScanQuery,
    vortex_array::expr::Expression,
    vortex_array::expr::Expression,
) {
    let query = ScanQuery {
        conjuncts: vec![
            Conjunct {
                field: FieldId(2),
                predicate: Predicate::Equal(42),
            },
            Conjunct {
                field: FieldId(4),
                predicate: Predicate::Equal(0),
            },
        ],
        projection: vec![FieldId(1), FieldId(5), FieldId(6)],
    };
    let filter = and(
        eq(get_item("counter_id", root()), lit(42i64)),
        eq(get_item("is_mobile", root()), lit(0i64)),
    );
    let projection = select(["user_id", "response_time", "bytes"], root());
    (query, filter, projection)
}

fn clickbench_dashboard_query() -> (
    ScanQuery,
    vortex_array::expr::Expression,
    vortex_array::expr::Expression,
) {
    let query = ScanQuery {
        conjuncts: vec![
            Conjunct {
                field: FieldId(3),
                predicate: Predicate::LessThan(5),
            },
            Conjunct {
                field: FieldId(5),
                predicate: Predicate::GreaterThan(500),
            },
        ],
        projection: vec![FieldId(2), FieldId(1), FieldId(6), FieldId(7)],
    };
    let filter = and(
        lt(get_item("region_id", root()), lit(5i64)),
        gt(get_item("response_time", root()), lit(500i64)),
    );
    let projection = select(["counter_id", "user_id", "bytes", "event_date"], root());
    (query, filter, projection)
}

const CLICKBENCH_FIELD_NAMES: [&str; 24] = [
    "event_time",
    "user_id",
    "counter_id",
    "region_id",
    "is_mobile",
    "response_time",
    "bytes",
    "event_date",
    "watch_id",
    "adv_engine_id",
    "resolution_width",
    "search_phrase_id",
    "traffic_source_id",
    "search_engine_id",
    "referer_id",
    "url_id",
    "is_refresh",
    "referer_hash",
    "url_hash",
    "window_client_width",
    "window_client_height",
    "dont_count_hits",
    "is_link",
    "is_download",
];
const JULY_2013_START_US: i64 = 1_372_636_800_000_000;
const AUGUST_2013_START_US: i64 = 1_375_315_200_000_000;
const JULY_14_2013_START_US: i64 = 1_373_760_000_000_000;
const JULY_16_2013_START_US: i64 = 1_373_932_800_000_000;

/// Restricted scan-input analogues of zero-based ClickBench Q0-Q9 and Q39-Q42.
/// Aggregation, grouping, ordering, strings, and disjunction remain outside this experiment.
fn clickbench_suite_query(
    query_id: usize,
) -> (
    ScanQuery,
    vortex_array::expr::Expression,
    vortex_array::expr::Expression,
) {
    let all_rows = || Conjunct {
        field: FieldId(8),
        predicate: Predicate::GreaterThan(-1),
    };
    let date_range = || {
        vec![
            Conjunct {
                field: FieldId(7),
                predicate: Predicate::GreaterThan(JULY_2013_START_US - 1),
            },
            Conjunct {
                field: FieldId(7),
                predicate: Predicate::LessThan(AUGUST_2013_START_US),
            },
        ]
    };
    let query = match query_id {
        0 => ScanQuery {
            conjuncts: vec![all_rows()],
            projection: vec![FieldId(8)],
        },
        1 | 7 => ScanQuery {
            conjuncts: vec![Conjunct {
                field: FieldId(9),
                predicate: Predicate::GreaterThan(0),
            }],
            projection: vec![FieldId(9)],
        },
        2 => ScanQuery {
            conjuncts: vec![all_rows()],
            projection: vec![FieldId(9), FieldId(10)],
        },
        3 | 4 => ScanQuery {
            conjuncts: vec![all_rows()],
            projection: vec![FieldId(1)],
        },
        5 => ScanQuery {
            conjuncts: vec![all_rows()],
            projection: vec![FieldId(11)],
        },
        6 => ScanQuery {
            conjuncts: vec![all_rows()],
            projection: vec![FieldId(7)],
        },
        8 => ScanQuery {
            conjuncts: vec![all_rows()],
            projection: vec![FieldId(3), FieldId(1)],
        },
        9 => ScanQuery {
            conjuncts: vec![all_rows()],
            projection: vec![FieldId(3), FieldId(9), FieldId(10), FieldId(1)],
        },
        39 => {
            let mut conjuncts = vec![Conjunct {
                field: FieldId(2),
                predicate: Predicate::Equal(62),
            }];
            conjuncts.extend(date_range());
            conjuncts.push(Conjunct {
                field: FieldId(16),
                predicate: Predicate::Equal(0),
            });
            ScanQuery {
                conjuncts,
                projection: vec![
                    FieldId(12),
                    FieldId(13),
                    FieldId(9),
                    FieldId(14),
                    FieldId(15),
                ],
            }
        }
        40 => {
            let mut conjuncts = vec![Conjunct {
                field: FieldId(2),
                predicate: Predicate::Equal(62),
            }];
            conjuncts.extend(date_range());
            conjuncts.extend([
                Conjunct {
                    field: FieldId(16),
                    predicate: Predicate::Equal(0),
                },
                Conjunct {
                    field: FieldId(12),
                    predicate: Predicate::Equal(5),
                },
                Conjunct {
                    field: FieldId(17),
                    predicate: Predicate::Equal(3_594_120_000_172_545_465),
                },
            ]);
            ScanQuery {
                conjuncts,
                projection: vec![FieldId(18), FieldId(7)],
            }
        }
        41 => {
            let mut conjuncts = vec![Conjunct {
                field: FieldId(2),
                predicate: Predicate::Equal(62),
            }];
            conjuncts.extend(date_range());
            conjuncts.extend([
                Conjunct {
                    field: FieldId(16),
                    predicate: Predicate::Equal(0),
                },
                Conjunct {
                    field: FieldId(21),
                    predicate: Predicate::Equal(0),
                },
                Conjunct {
                    field: FieldId(18),
                    predicate: Predicate::Equal(2_868_770_270_353_813_622),
                },
            ]);
            ScanQuery {
                conjuncts,
                projection: vec![FieldId(19), FieldId(20)],
            }
        }
        42 => {
            let mut conjuncts = vec![Conjunct {
                field: FieldId(2),
                predicate: Predicate::Equal(62),
            }];
            conjuncts.extend([
                Conjunct {
                    field: FieldId(7),
                    predicate: Predicate::GreaterThan(JULY_14_2013_START_US - 1),
                },
                Conjunct {
                    field: FieldId(7),
                    predicate: Predicate::LessThan(JULY_16_2013_START_US),
                },
                Conjunct {
                    field: FieldId(16),
                    predicate: Predicate::Equal(0),
                },
                Conjunct {
                    field: FieldId(21),
                    predicate: Predicate::Equal(0),
                },
            ]);
            ScanQuery {
                conjuncts,
                projection: vec![FieldId(0)],
            }
        }
        _ => unreachable!("unsupported ClickBench scan-shape query"),
    };
    let Some(filter) = query
        .conjuncts
        .iter()
        .map(|conjunct| {
            let field = get_item(CLICKBENCH_FIELD_NAMES[conjunct.field.0], root());
            match conjunct.predicate {
                Predicate::Equal(value) => eq(field, lit(value)),
                Predicate::LessThan(value) => lt(field, lit(value)),
                Predicate::GreaterThan(value) => gt(field, lit(value)),
                Predicate::RangeExclusive { lower, upper } => {
                    and(gt(field.clone(), lit(lower)), lt(field, lit(upper)))
                }
            }
        })
        .reduce(and)
    else {
        unreachable!("ClickBench scan shapes always have a predicate");
    };
    let projection = select(
        FieldNames::from(
            query
                .projection
                .iter()
                .map(|field| Arc::<str>::from(CLICKBENCH_FIELD_NAMES[field.0]))
                .collect::<Vec<_>>(),
        ),
        root(),
    );
    (query, filter, projection)
}

macro_rules! clickbench_query_factory {
    ($name:ident, $query_id:literal) => {
        fn $name() -> (
            ScanQuery,
            vortex_array::expr::Expression,
            vortex_array::expr::Expression,
        ) {
            clickbench_suite_query($query_id)
        }
    };
}

clickbench_query_factory!(clickbench_q00_query, 0);
clickbench_query_factory!(clickbench_q01_query, 1);
clickbench_query_factory!(clickbench_q02_query, 2);
clickbench_query_factory!(clickbench_q03_query, 3);
clickbench_query_factory!(clickbench_q04_query, 4);
clickbench_query_factory!(clickbench_q05_query, 5);
clickbench_query_factory!(clickbench_q06_query, 6);
clickbench_query_factory!(clickbench_q07_query, 7);
clickbench_query_factory!(clickbench_q08_query, 8);
clickbench_query_factory!(clickbench_q09_query, 9);
clickbench_query_factory!(clickbench_q39_query, 39);
clickbench_query_factory!(clickbench_q40_query, 40);
clickbench_query_factory!(clickbench_q41_query, 41);
clickbench_query_factory!(clickbench_q42_query, 42);

const CLICKBENCH_SUITE_QUERY_IDS: [usize; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 39, 40, 41, 42];

fn clickbench_suite_query_factory(query_id: usize) -> QueryFactory {
    match query_id {
        0 => clickbench_q00_query,
        1 => clickbench_q01_query,
        2 => clickbench_q02_query,
        3 => clickbench_q03_query,
        4 => clickbench_q04_query,
        5 => clickbench_q05_query,
        6 => clickbench_q06_query,
        7 => clickbench_q07_query,
        8 => clickbench_q08_query,
        9 => clickbench_q09_query,
        39 => clickbench_q39_query,
        40 => clickbench_q40_query,
        41 => clickbench_q41_query,
        42 => clickbench_q42_query,
        _ => unreachable!("unsupported ClickBench scan-shape query"),
    }
}

fn fineweb_suite_query(
    query_id: usize,
) -> (
    ScanQuery,
    vortex_array::expr::Expression,
    vortex_array::expr::Expression,
) {
    let query = match query_id {
        0 => ScanQuery {
            conjuncts: vec![Conjunct {
                field: FieldId(13),
                predicate: Predicate::GreaterThan(-1),
            }],
            projection: vec![FieldId(0)],
        },
        1 => ScanQuery {
            conjuncts: vec![Conjunct {
                field: FieldId(0),
                predicate: Predicate::Equal(stable_string_hash("CC-MAIN-2016-30")),
            }],
            projection: (0..FINEWEB_FIELD_NAMES.len()).map(FieldId).collect(),
        },
        2 => ScanQuery {
            conjuncts: vec![Conjunct {
                field: FieldId(1),
                predicate: Predicate::Equal(202_010),
            }],
            projection: vec![FieldId(2), FieldId(3), FieldId(13)],
        },
        3 => ScanQuery {
            conjuncts: vec![
                Conjunct {
                    field: FieldId(7),
                    predicate: Predicate::Equal(1),
                },
                Conjunct {
                    field: FieldId(8),
                    predicate: Predicate::Equal(1),
                },
            ],
            projection: vec![FieldId(2), FieldId(3), FieldId(13)],
        },
        4 => ScanQuery {
            conjuncts: vec![Conjunct {
                field: FieldId(9),
                predicate: Predicate::Equal(1),
            }],
            projection: vec![FieldId(2), FieldId(3), FieldId(13)],
        },
        5 => ScanQuery {
            conjuncts: vec![Conjunct {
                field: FieldId(10),
                predicate: Predicate::Equal(1),
            }],
            projection: vec![FieldId(2), FieldId(3), FieldId(13)],
        },
        6 => ScanQuery {
            conjuncts: vec![
                Conjunct {
                    field: FieldId(11),
                    predicate: Predicate::Equal(1),
                },
                Conjunct {
                    field: FieldId(4),
                    predicate: Predicate::Equal(stable_string_hash("en")),
                },
                Conjunct {
                    field: FieldId(5),
                    predicate: Predicate::GreaterThan(920_000),
                },
            ],
            projection: vec![FieldId(2), FieldId(3), FieldId(13)],
        },
        7 => ScanQuery {
            conjuncts: vec![Conjunct {
                field: FieldId(11),
                predicate: Predicate::Equal(1),
            }],
            projection: vec![FieldId(2), FieldId(3), FieldId(13)],
        },
        8 => ScanQuery {
            conjuncts: vec![Conjunct {
                field: FieldId(12),
                predicate: Predicate::Equal(1),
            }],
            projection: vec![FieldId(2), FieldId(3), FieldId(13)],
        },
        _ => unreachable!("unsupported FineWeb scan-shape query"),
    };
    let filter = query
        .conjuncts
        .iter()
        .map(|conjunct| {
            let field = get_item(FINEWEB_FIELD_NAMES[conjunct.field.0], root());
            match conjunct.predicate {
                Predicate::Equal(value) => eq(field, lit(value)),
                Predicate::LessThan(value) => lt(field, lit(value)),
                Predicate::GreaterThan(value) => gt(field, lit(value)),
                Predicate::RangeExclusive { lower, upper } => {
                    and(gt(field.clone(), lit(lower)), lt(field, lit(upper)))
                }
            }
        })
        .reduce(and)
        .unwrap();
    let projection = select(
        FieldNames::from(
            query
                .projection
                .iter()
                .map(|field| Arc::<str>::from(FINEWEB_FIELD_NAMES[field.0]))
                .collect::<Vec<_>>(),
        ),
        root(),
    );
    (query, filter, projection)
}

macro_rules! fineweb_query_factory {
    ($name:ident, $query_id:literal) => {
        fn $name() -> (
            ScanQuery,
            vortex_array::expr::Expression,
            vortex_array::expr::Expression,
        ) {
            fineweb_suite_query($query_id)
        }
    };
}

fineweb_query_factory!(fineweb_q00_query, 0);
fineweb_query_factory!(fineweb_q01_query, 1);
fineweb_query_factory!(fineweb_q02_query, 2);
fineweb_query_factory!(fineweb_q03_query, 3);
fineweb_query_factory!(fineweb_q04_query, 4);
fineweb_query_factory!(fineweb_q05_query, 5);
fineweb_query_factory!(fineweb_q06_query, 6);
fineweb_query_factory!(fineweb_q07_query, 7);
fineweb_query_factory!(fineweb_q08_query, 8);

const FINEWEB_QUERY_IDS: [usize; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];

fn fineweb_query_factory(query_id: usize) -> QueryFactory {
    match query_id {
        0 => fineweb_q00_query,
        1 => fineweb_q01_query,
        2 => fineweb_q02_query,
        3 => fineweb_q03_query,
        4 => fineweb_q04_query,
        5 => fineweb_q05_query,
        6 => fineweb_q06_query,
        7 => fineweb_q07_query,
        8 => fineweb_q08_query,
        _ => unreachable!("unsupported FineWeb scan-shape query"),
    }
}

fn fineweb_query_id(value: &str, prefix: &str, suffix: &str) -> Option<usize> {
    let query_id = value
        .strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse::<usize>()
        .ok()?;
    FINEWEB_QUERY_IDS.contains(&query_id).then_some(query_id)
}

fn tpch_q6_scan_query() -> (
    ScanQuery,
    vortex_array::expr::Expression,
    vortex_array::expr::Expression,
) {
    let (shipdate_start, shipdate_end) =
        if std::env::var_os("VORTEX_TPCH_LINEITEM_PARQUET").is_some() {
            (8_765, 9_131)
        } else {
            (1_000, 1_365)
        };
    let query = ScanQuery {
        conjuncts: vec![
            Conjunct {
                field: FieldId(7),
                predicate: Predicate::GreaterThan(shipdate_start),
            },
            Conjunct {
                field: FieldId(7),
                predicate: Predicate::LessThan(shipdate_end),
            },
            Conjunct {
                field: FieldId(5),
                predicate: Predicate::GreaterThan(4),
            },
            Conjunct {
                field: FieldId(5),
                predicate: Predicate::LessThan(8),
            },
            Conjunct {
                field: FieldId(3),
                predicate: Predicate::LessThan(24),
            },
        ],
        projection: vec![FieldId(4), FieldId(5)],
    };
    let filter = and(
        and(
            gt(get_item("shipdate", root()), lit(shipdate_start)),
            lt(get_item("shipdate", root()), lit(shipdate_end)),
        ),
        and(
            and(
                gt(get_item("discount", root()), lit(4i64)),
                lt(get_item("discount", root()), lit(8i64)),
            ),
            lt(get_item("quantity", root()), lit(24i64)),
        ),
    );
    let projection = select(["extendedprice", "discount"], root());
    (query, filter, projection)
}

fn tpch_q1_scan_query() -> (
    ScanQuery,
    vortex_array::expr::Expression,
    vortex_array::expr::Expression,
) {
    let shipdate_end = if std::env::var_os("VORTEX_TPCH_LINEITEM_PARQUET").is_some() {
        10_472
    } else {
        2_300
    };
    let query = ScanQuery {
        conjuncts: vec![Conjunct {
            field: FieldId(7),
            predicate: Predicate::LessThan(shipdate_end),
        }],
        projection: vec![FieldId(3), FieldId(4), FieldId(5), FieldId(6)],
    };
    let filter = lt(get_item("shipdate", root()), lit(shipdate_end));
    let projection = select(["quantity", "extendedprice", "discount", "tax"], root());
    (query, filter, projection)
}

fn tpch_v1_friendly_query() -> (
    ScanQuery,
    vortex_array::expr::Expression,
    vortex_array::expr::Expression,
) {
    let query = ScanQuery {
        conjuncts: vec![Conjunct {
            field: FieldId(3),
            predicate: Predicate::GreaterThan(0),
        }],
        projection: vec![FieldId(3)],
    };
    let filter = gt(get_item("quantity", root()), lit(0i64));
    let projection = select(["quantity"], root());
    (query, filter, projection)
}

async fn run_v1(
    selectivity: usize,
    concurrency: usize,
    diagnostics: bool,
) -> VortexResult<((usize, u64), SourceSummary)> {
    run_v1_fixture(&FIXTURE, query(selectivity), concurrency, diagnostics).await
}

async fn run_v1_fixture(
    fixture: &Fixture,
    query: (
        ScanQuery,
        vortex_array::expr::Expression,
        vortex_array::expr::Expression,
    ),
    concurrency: usize,
    diagnostics: bool,
) -> VortexResult<((usize, u64), SourceSummary)> {
    let source_inner = Arc::clone(&fixture.source);
    let (source, counts) = CountingSource::new(source_inner, diagnostics);
    let reader = fixture.layout.new_reader(
        "self-paced-v1".into(),
        source,
        &SESSION,
        &LayoutReaderContext::default(),
    )?;
    let (_, filter, projection) = query;
    let filter = filter.bind(reader.dtype())?;
    let projection = projection.bind(reader.dtype())?;
    let arrays = ScanBuilder::new(SESSION.clone(), reader)
        .with_filter(filter)
        .with_projection(projection)
        // V1 is the unmodified scan model: preserve its natural layout chunks. In particular,
        // never feed the self-paced experiment's morsel size into this path.
        .with_natural_splits(Arc::clone(&fixture.natural_splits))
        .with_concurrency(concurrency)
        .into_array_stream()?
        .try_collect::<Vec<_>>()
        .await?;
    let hash = if diagnostics {
        stable_array_hash(&arrays)?
    } else {
        let rows = arrays.iter().map(|array| array.len()).sum();
        std::hint::black_box(&arrays);
        (rows, 0)
    };
    Ok((hash, counts.summary()))
}

async fn run_v1_fixture_with_splits(
    fixture: &Fixture,
    query: (
        ScanQuery,
        vortex_array::expr::Expression,
        vortex_array::expr::Expression,
    ),
    natural_splits: Arc<[u64]>,
    concurrency: usize,
    diagnostics: bool,
) -> VortexResult<((usize, u64), SourceSummary)> {
    let source_inner = Arc::clone(&fixture.source);
    let (source, counts) = CountingSource::new(source_inner, diagnostics);
    let reader = fixture.layout.new_reader(
        "self-paced-v1-real-splits".into(),
        source,
        &SESSION,
        &LayoutReaderContext::default(),
    )?;
    let (_, filter, projection) = query;
    let arrays = ScanBuilder::new(SESSION.clone(), Arc::clone(&reader))
        .with_filter(filter.bind(reader.dtype())?)
        .with_projection(projection.bind(reader.dtype())?)
        .with_natural_splits(natural_splits)
        .with_concurrency(concurrency)
        .into_array_stream()?
        .try_collect::<Vec<_>>()
        .await?;
    let hash = if diagnostics {
        stable_array_hash(&arrays)?
    } else {
        let rows = arrays.iter().map(|array| array.len()).sum();
        std::hint::black_box(&arrays);
        (rows, 0)
    };
    Ok((hash, counts.summary()))
}

async fn run_candidate(
    selectivity: usize,
    morsel_rows: usize,
    concurrency: usize,
    diagnostics: bool,
) -> VortexResult<((usize, u64), SourceSummary)> {
    run_candidate_fixture(
        &FIXTURE,
        query(selectivity),
        morsel_rows,
        concurrency,
        diagnostics,
    )
    .await
}

async fn run_candidate_fixture(
    fixture: &Fixture,
    query: (
        ScanQuery,
        vortex_array::expr::Expression,
        vortex_array::expr::Expression,
    ),
    morsel_rows: usize,
    concurrency: usize,
    diagnostics: bool,
) -> VortexResult<((usize, u64), SourceSummary)> {
    let source_inner = Arc::clone(&fixture.source);
    let (source, counts) = CountingSource::new(source_inner, diagnostics);
    let (query, ..) = query;
    let policy = benchmark_schedule_policy(concurrency);
    let result = run_self_paced(
        &fixture.plan,
        query,
        morsel_rows,
        source,
        &SESSION,
        RunOptions {
            policy,
            transition_budget: 32,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency,
            collect_trace: false,
            speculative_io: *SPECULATIVE_IO_CONFIG,
        },
    )
    .await?;
    let hash = if diagnostics {
        stable_output_hash(&result.batches, &SESSION)?
    } else {
        let rows = result.batches.iter().map(|batch| batch.array.len()).sum();
        std::hint::black_box(&result.batches);
        (rows, 0)
    };
    Ok((hash, counts.summary()))
}

async fn run_candidate_fixture_with_ranges(
    fixture: &Fixture,
    query: (
        ScanQuery,
        vortex_array::expr::Expression,
        vortex_array::expr::Expression,
    ),
    morsel_ranges: &[Range<u64>],
    concurrency: usize,
    diagnostics: bool,
) -> VortexResult<((usize, u64), SourceSummary)> {
    let source_inner = Arc::clone(&fixture.source);
    let (source, counts) = CountingSource::new(source_inner, diagnostics);
    let (query, ..) = query;
    let result = run_self_paced_ranges(
        &fixture.plan,
        query,
        morsel_ranges,
        source,
        &SESSION,
        RunOptions {
            policy: benchmark_schedule_policy(concurrency),
            transition_budget: 32,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency,
            collect_trace: false,
            speculative_io: *SPECULATIVE_IO_CONFIG,
        },
    )
    .await?;
    let hash = if diagnostics {
        stable_output_hash(&result.batches, &SESSION)?
    } else {
        let rows = result.batches.iter().map(|batch| batch.array.len()).sum();
        std::hint::black_box(&result.batches);
        (rows, 0)
    };
    Ok((hash, counts.summary()))
}

fn benchmark_schedule_policy(concurrency: usize) -> SchedulePolicy {
    match std::env::var("VORTEX_SELF_PACED_POLICY").as_deref() {
        Ok("all-ready") => SchedulePolicy::AllReady,
        Ok("projection-prefetch") => SchedulePolicy::ProjectionPrefetch,
        Ok("predicate-first") => SchedulePolicy::PredicateFirst,
        Ok("legacy-adaptive") => SchedulePolicy::LegacyAdaptivePredicates { concurrency },
        _ => SchedulePolicy::AdaptivePredicates { concurrency },
    }
}

fn benchmark_speculative_io_config() -> SpeculativeIoConfig {
    let mode =
        std::env::var("VORTEX_SELF_PACED_SPECULATIVE_IO").unwrap_or_else(|_| "off".to_owned());
    if mode == "off" {
        return SpeculativeIoConfig::disabled();
    }
    let max_in_flight_bytes = std::env::var("VORTEX_SELF_PACED_SPECULATIVE_IO_MAX_BYTES")
        .map_or(64 << 20, |value| value.parse().unwrap());
    let unknown_read_bytes = std::env::var("VORTEX_SELF_PACED_SPECULATIVE_IO_UNKNOWN_BYTES")
        .map_or(8 << 20, |value| value.parse().unwrap());
    let minimum_expected_rows = std::env::var("VORTEX_SELF_PACED_SPECULATIVE_IO_MIN_ROWS")
        .map_or(1, |value| value.parse().unwrap());
    let adaptive = SpeculativeReadPolicy::Adaptive {
        minimum_expected_rows,
    };
    let eager = mode.ends_with("-eager") || mode == "eager";
    let selected = |enabled: bool| {
        if !enabled {
            SpeculativeReadPolicy::Disabled
        } else if eager {
            SpeculativeReadPolicy::Eager
        } else {
            adaptive
        }
    };
    let predicate = selected(matches!(
        mode.as_str(),
        "predicate" | "predicate-eager" | "adaptive" | "eager"
    ));
    let projection = selected(matches!(
        mode.as_str(),
        "projection" | "projection-eager" | "adaptive" | "eager"
    ));
    assert!(
        predicate != SpeculativeReadPolicy::Disabled
            || projection != SpeculativeReadPolicy::Disabled,
        "unsupported VORTEX_SELF_PACED_SPECULATIVE_IO mode {mode}"
    );
    SpeculativeIoConfig {
        predicate,
        projection,
        max_in_flight_bytes,
        unknown_read_bytes,
    }
}

async fn trace_cases(requested: &str) -> VortexResult<()> {
    if let Some((query_id, morsel_rows)) = [
        (fineweb_query_id(requested, "fineweb-q", "-65k"), 65_536),
        (fineweb_query_id(requested, "fineweb-q", "-128k"), 131_072),
    ]
    .into_iter()
    .find_map(|(query_id, morsel_rows)| query_id.map(|query_id| (query_id, morsel_rows)))
    {
        return trace_workload(
            &format!("fineweb_q{query_id:02}_scan_analogue"),
            fineweb_fixture(),
            fineweb_query_factory(query_id),
            morsel_rows,
        )
        .await;
    }
    if matches!(
        requested,
        "clickbench-q40-65k"
            | "clickbench-q40-128k"
            | "clickbench-q41-65k"
            | "clickbench-q41-128k"
            | "clickbench-q42-65k"
            | "clickbench-q42-128k"
    ) {
        let morsel_rows = if requested.ends_with("65k") {
            65_536
        } else {
            131_072
        };
        let query_id = if requested.contains("q40") {
            40
        } else if requested.contains("q41") {
            41
        } else {
            42
        };
        return trace_workload(
            &format!("clickbench_q{query_id}"),
            clickbench_fixture(),
            clickbench_suite_query_factory(query_id),
            morsel_rows,
        )
        .await;
    }
    if matches!(requested, "q6-65k" | "q6-128k") {
        let morsel_rows = if requested == "q6-65k" {
            65_536
        } else {
            131_072
        };
        return trace_workload(
            "tpch_q6_scan",
            tpch_fixture(),
            tpch_q6_scan_query,
            morsel_rows,
        )
        .await;
    }
    if requested == "q1-128k" {
        return trace_workload("tpch_q1_scan", tpch_fixture(), tpch_q1_scan_query, 131_072).await;
    }
    if matches!(requested, "v1-friendly-65k" | "v1-friendly-128k") {
        let morsel_rows = if requested == "v1-friendly-65k" {
            65_536
        } else {
            131_072
        };
        return trace_workload(
            "tpch_v1_friendly",
            tpch_fixture(),
            tpch_v1_friendly_query,
            morsel_rows,
        )
        .await;
    }
    let mut matched = false;
    for (name, selectivity, morsel_rows, concurrency) in TRACE_CASES {
        if requested != "all" && requested != name {
            continue;
        }
        matched = true;
        let source_inner = Arc::clone(&FIXTURE.source);
        let (source, counts) = CountingSource::new(source_inner, true);
        let (query, ..) = query(selectivity);
        let result = run_self_paced(
            &FIXTURE.plan,
            query,
            morsel_rows,
            source,
            &SESSION,
            RunOptions {
                policy: SchedulePolicy::PredicateFirst,
                transition_budget: 32,
                retention: RetentionPolicy::RetainUntilDead,
                concurrency,
                collect_trace: true,
                speculative_io: *SPECULATIVE_IO_CONFIG,
            },
        )
        .await?;
        let (rows, hash) = stable_output_hash(&result.batches, &SESSION)?;
        println!(
            "trace_begin selectivity={selectivity} morsel_rows={morsel_rows} concurrency={concurrency} rows={rows} hash={hash:#x}"
        );
        write_execution_trace(std::io::stdout().lock(), &result.trace).map_err(|error| {
            vortex_error::vortex_err!("failed to write self-paced trace: {error}")
        })?;
        let source = counts.summary();
        println!(
            "trace_end requests={} unique_segments={} segment_requests_min={} segment_requests_max={} bytes={} advance_calls={} transitions={} tasks_offered={} tasks_claimed={} tasks_completed={} scheduler_passes={} scheduler_tasks_considered={} scheduler_tasks_admitted={} completion_batches={} completions_drained={} max_completion_batch={} speculative_io_offered={} speculative_io_admitted={} speculative_io_unknown_size={} speculative_io_estimated_bytes_offered={} speculative_io_estimated_bytes_admitted={} speculative_io_completed_bytes={} speculative_io_useful_bytes={} speculative_io_wasted_bytes={} speculative_predicate_io_offered={} speculative_projection_io_offered={} demand_combinations={} inline_demand_combinations={} demand_direct_adoptions={} demand_noop_adoptions={} adaptive_launches={} adaptive_waits={} predicate_reorders={} demand_initial={} demand_final={} trace_events={}",
            source.requests,
            source.unique_segments,
            source.segment_requests_min,
            source.segment_requests_max,
            source.bytes,
            result.metrics.advance_calls,
            result.metrics.transitions,
            result.metrics.tasks_offered,
            result.metrics.tasks_claimed,
            result.metrics.tasks_completed,
            result.metrics.scheduler_passes,
            result.metrics.scheduler_tasks_considered,
            result.metrics.scheduler_tasks_admitted,
            result.metrics.completion_batches,
            result.metrics.completions_drained,
            result.metrics.max_completion_batch,
            result.metrics.speculative_io_offered,
            result.metrics.speculative_io_admitted,
            result.metrics.speculative_io_unknown_size,
            result.metrics.speculative_io_estimated_bytes_offered,
            result.metrics.speculative_io_estimated_bytes_admitted,
            result.metrics.speculative_io_completed_bytes,
            result.metrics.speculative_io_useful_bytes,
            result.metrics.speculative_io_wasted_bytes,
            result.metrics.speculative_predicate_io_offered,
            result.metrics.speculative_projection_io_offered,
            result.metrics.demand_combinations,
            result.metrics.inline_demand_combinations,
            result.metrics.demand_direct_adoptions,
            result.metrics.demand_noop_adoptions,
            result.metrics.adaptive_predicate_launches,
            result.metrics.adaptive_predicate_waits,
            result.metrics.predicate_reorders,
            result.metrics.demand_rows_initial,
            result.metrics.demand_rows_current,
            result.trace.len(),
        );
    }
    if !matched {
        vortex_error::vortex_bail!("unsupported VORTEX_SELF_PACED_TRACE case");
    }
    Ok(())
}

async fn trace_workload(
    name: &str,
    fixture: &Fixture,
    query: QueryFactory,
    morsel_rows: usize,
) -> VortexResult<()> {
    std::hint::black_box(run_candidate_fixture(fixture, query(), morsel_rows, 16, false).await?);
    let (v1_output, v1_source) = run_v1_fixture(fixture, query(), 16, true).await?;
    let source_inner = Arc::clone(&fixture.source);
    let (source, counts) = CountingSource::new(source_inner, true);
    let (query, ..) = query();
    let policy = benchmark_schedule_policy(16);
    let result = run_self_paced(
        &fixture.plan,
        query,
        morsel_rows,
        source,
        &SESSION,
        RunOptions {
            policy,
            transition_budget: 32,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: 16,
            collect_trace: true,
            speculative_io: *SPECULATIVE_IO_CONFIG,
        },
    )
    .await?;
    let (rows, hash) = stable_output_hash(&result.batches, &SESSION)?;
    println!(
        "trace_begin workload={name} morsel_rows={morsel_rows} concurrency=16 rows={rows} hash={hash:#x}"
    );
    println!(
        "v1_summary rows={} hash={:#x} requests={} unique_segments={} segment_requests_min={} segment_requests_max={} bytes={}",
        v1_output.0,
        v1_output.1,
        v1_source.requests,
        v1_source.unique_segments,
        v1_source.segment_requests_min,
        v1_source.segment_requests_max,
        v1_source.bytes,
    );
    write_execution_trace(std::io::stdout().lock(), &result.trace)
        .map_err(|error| vortex_error::vortex_err!("failed to write self-paced trace: {error}"))?;
    let source = counts.summary();
    println!(
        "trace_end requests={} unique_segments={} segment_requests_min={} segment_requests_max={} bytes={} advance_calls={} transitions={} nodes_inspected={} completion_wake_candidates_inspected={} tasks_offered={} tasks_claimed={} tasks_completed={} scheduler_passes={} scheduler_tasks_considered={} scheduler_tasks_admitted={} completion_batches={} completions_drained={} max_completion_batch={} speculative_io_offered={} speculative_io_admitted={} speculative_io_unknown_size={} speculative_io_estimated_bytes_offered={} speculative_io_estimated_bytes_admitted={} speculative_io_completed_bytes={} speculative_io_useful_bytes={} speculative_io_wasted_bytes={} speculative_predicate_io_offered={} speculative_projection_io_offered={} demand_combinations={} inline_demand_combinations={} demand_direct_adoptions={} demand_noop_adoptions={} adaptive_launches={} adaptive_waits={} predicate_reorders={} demand_initial={} demand_final={} resource_nodes={} morsel_slots={} trace_events={}",
        source.requests,
        source.unique_segments,
        source.segment_requests_min,
        source.segment_requests_max,
        source.bytes,
        result.metrics.advance_calls,
        result.metrics.transitions,
        result.metrics.nodes_inspected,
        result.metrics.completion_wake_candidates_inspected,
        result.metrics.tasks_offered,
        result.metrics.tasks_claimed,
        result.metrics.tasks_completed,
        result.metrics.scheduler_passes,
        result.metrics.scheduler_tasks_considered,
        result.metrics.scheduler_tasks_admitted,
        result.metrics.completion_batches,
        result.metrics.completions_drained,
        result.metrics.max_completion_batch,
        result.metrics.speculative_io_offered,
        result.metrics.speculative_io_admitted,
        result.metrics.speculative_io_unknown_size,
        result.metrics.speculative_io_estimated_bytes_offered,
        result.metrics.speculative_io_estimated_bytes_admitted,
        result.metrics.speculative_io_completed_bytes,
        result.metrics.speculative_io_useful_bytes,
        result.metrics.speculative_io_wasted_bytes,
        result.metrics.speculative_predicate_io_offered,
        result.metrics.speculative_projection_io_offered,
        result.metrics.demand_combinations,
        result.metrics.inline_demand_combinations,
        result.metrics.demand_direct_adoptions,
        result.metrics.demand_noop_adoptions,
        result.metrics.adaptive_predicate_launches,
        result.metrics.adaptive_predicate_waits,
        result.metrics.predicate_reorders,
        result.metrics.demand_rows_initial,
        result.metrics.demand_rows_current,
        result.metrics.resource_nodes,
        result.metrics.morsel_slots,
        result.trace.len(),
    );
    Ok(())
}

async fn profile_case(requested: &str, iterations: usize) -> VortexResult<()> {
    if matches!(
        requested,
        "tpch-q6-v1-128k" | "tpch-q6-self-128k" | "tpch-q1-v1-128k" | "tpch-q1-self-128k"
    ) {
        let fixture = tpch_fixture();
        let query = if requested.starts_with("tpch-q1-") {
            tpch_q1_scan_query
        } else {
            tpch_q6_scan_query
        };
        println!("profile_begin workload={requested} iterations={iterations}");
        for _ in 0..iterations {
            if requested.contains("-v1-") {
                std::hint::black_box(run_v1_fixture(fixture, query(), 16, false).await?);
            } else {
                std::hint::black_box(
                    run_candidate_fixture(fixture, query(), 131_072, 16, false).await?,
                );
            }
        }
        println!("profile_end workload={requested} iterations={iterations}");
        return Ok(());
    }
    if let Some((query_id, engine, morsel_rows)) = ["65k", "128k"]
        .into_iter()
        .flat_map(|size| ["v1", "self"].map(|engine| (size, engine)))
        .find_map(|(size, engine)| {
            fineweb_query_id(requested, "fineweb-q", &format!("-{engine}-{size}")).map(|query_id| {
                (
                    query_id,
                    engine,
                    if size == "65k" { 65_536 } else { 131_072 },
                )
            })
        })
    {
        let fixture = fineweb_fixture();
        let query = fineweb_query_factory(query_id);
        println!("profile_begin workload={requested} iterations={iterations}");
        for _ in 0..iterations {
            if engine == "v1" {
                std::hint::black_box(run_v1_fixture(fixture, query(), 16, false).await?);
            } else {
                std::hint::black_box(
                    run_candidate_fixture(fixture, query(), morsel_rows, 16, false).await?,
                );
            }
        }
        println!("profile_end workload={requested} iterations={iterations}");
        return Ok(());
    }
    if matches!(
        requested,
        "q40-v1-65k" | "q40-self-65k" | "q42-v1-128k" | "q42-self-128k"
    ) {
        let fixture = clickbench_fixture();
        println!("profile_begin workload={requested} iterations={iterations}");
        for _ in 0..iterations {
            let query = if requested.starts_with("q40") {
                clickbench_q40_query
            } else {
                clickbench_q42_query
            };
            if requested.contains("-v1-") {
                std::hint::black_box(run_v1_fixture(fixture, query(), 16, false).await?);
            } else {
                let morsel_rows = if requested.ends_with("65k") {
                    65_536
                } else {
                    131_072
                };
                std::hint::black_box(
                    run_candidate_fixture(fixture, query(), morsel_rows, 16, false).await?,
                );
            }
        }
        println!("profile_end workload={requested} iterations={iterations}");
        return Ok(());
    }
    let Some((_, selectivity, morsel_rows, concurrency)) = TRACE_CASES
        .into_iter()
        .find(|(name, ..)| *name == requested)
    else {
        vortex_error::vortex_bail!("VORTEX_SELF_PACED_PROFILE must be one of: 1, 50, 95");
    };
    println!(
        "profile_begin selectivity={selectivity} morsel_rows={morsel_rows} concurrency={concurrency} iterations={iterations}"
    );
    for _ in 0..iterations {
        std::hint::black_box(run_candidate(selectivity, morsel_rows, concurrency, false).await?);
    }
    println!("profile_end iterations={iterations}");
    Ok(())
}

async fn compare_cases(iterations: usize) -> VortexResult<()> {
    if iterations == 0 {
        vortex_error::vortex_bail!("comparison iterations must be non-zero");
    }
    if let Ok(workload) = std::env::var("VORTEX_SELF_PACED_COMPARE_WORKLOAD") {
        if workload == "fineweb_all" {
            return compare_fineweb_workloads(iterations).await;
        }
        if let Some(query_id) = fineweb_query_id(&workload, "fineweb_q", "") {
            return compare_workload(
                &workload,
                fineweb_fixture,
                fineweb_query_factory(query_id),
                iterations,
            )
            .await;
        }
        if workload == "clickbench_all" {
            return compare_clickbench_workloads(iterations).await;
        }
        if workload == "tpch_all" {
            return compare_tpch_workloads(iterations).await;
        }
        if workload.starts_with("tpch_") {
            let query = match workload.as_str() {
                "tpch_q6" => tpch_q6_scan_query,
                "tpch_q1" => tpch_q1_scan_query,
                "tpch_v1_friendly" => tpch_v1_friendly_query,
                _ => vortex_error::vortex_bail!("unsupported comparison workload {workload}"),
            };
            return compare_workload(&workload, tpch_fixture, query, iterations).await;
        }
        let query_id = match workload.as_str() {
            "clickbench_q40" => 40,
            "clickbench_q41" => 41,
            "clickbench_q42" => 42,
            _ => vortex_error::vortex_bail!("unsupported comparison workload {workload}"),
        };
        return compare_workload(
            &workload,
            clickbench_fixture,
            clickbench_suite_query_factory(query_id),
            iterations,
        )
        .await;
    }
    for selectivity in [1, 50, 95] {
        for morsel_rows in [65_536, 131_072] {
            compare_synthetic_case(selectivity, morsel_rows, iterations).await?;
        }
    }
    compare_clickbench_workloads(iterations).await?;
    compare_tpch_workloads(iterations).await?;
    compare_fineweb_workloads(iterations).await?;
    Ok(())
}

async fn compare_clickbench_workloads(iterations: usize) -> VortexResult<()> {
    compare_workload(
        "clickbench_selective",
        clickbench_fixture,
        clickbench_selective_query,
        iterations,
    )
    .await?;
    compare_workload(
        "clickbench_dashboard",
        clickbench_fixture,
        clickbench_dashboard_query,
        iterations,
    )
    .await?;
    for query_id in CLICKBENCH_SUITE_QUERY_IDS {
        compare_workload(
            &format!("clickbench_q{query_id:02}"),
            clickbench_fixture,
            clickbench_suite_query_factory(query_id),
            iterations,
        )
        .await?;
    }
    Ok(())
}

async fn compare_tpch_workloads(iterations: usize) -> VortexResult<()> {
    compare_workload("tpch_q6_scan", tpch_fixture, tpch_q6_scan_query, iterations).await?;
    compare_workload("tpch_q1_scan", tpch_fixture, tpch_q1_scan_query, iterations).await?;
    compare_workload(
        "tpch_v1_friendly",
        tpch_fixture,
        tpch_v1_friendly_query,
        iterations,
    )
    .await?;
    Ok(())
}

async fn compare_fineweb_workloads(iterations: usize) -> VortexResult<()> {
    for query_id in FINEWEB_QUERY_IDS {
        compare_workload(
            &format!("fineweb_q{query_id:02}_scan_analogue"),
            fineweb_fixture,
            fineweb_query_factory(query_id),
            iterations,
        )
        .await?;
    }
    Ok(())
}

async fn compare_synthetic_case(
    selectivity: usize,
    morsel_rows: usize,
    iterations: usize,
) -> VortexResult<()> {
    let v1_warmup = run_v1(selectivity, 16, true).await?;
    let candidate_warmup = run_candidate(selectivity, morsel_rows, 16, true).await?;
    if v1_warmup.0 != candidate_warmup.0 {
        vortex_error::vortex_bail!(
            "synthetic output mismatch at selectivity={selectivity} morsel_rows={morsel_rows}"
        );
    }
    let mut v1_times = Vec::with_capacity(iterations);
    let mut candidate_times = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let started = Instant::now();
        if iteration % 2 == 0 {
            std::hint::black_box(run_v1(selectivity, 16, false).await?);
            v1_times.push(started.elapsed());
            let started = Instant::now();
            std::hint::black_box(run_candidate(selectivity, morsel_rows, 16, false).await?);
            candidate_times.push(started.elapsed());
        } else {
            std::hint::black_box(run_candidate(selectivity, morsel_rows, 16, false).await?);
            candidate_times.push(started.elapsed());
            let started = Instant::now();
            std::hint::black_box(run_v1(selectivity, 16, false).await?);
            v1_times.push(started.elapsed());
        }
    }
    print_comparison(
        &format!("synthetic_{selectivity}"),
        morsel_rows,
        iterations,
        &mut v1_times,
        &mut candidate_times,
    );
    Ok(())
}

type QueryBundle = (
    ScanQuery,
    vortex_array::expr::Expression,
    vortex_array::expr::Expression,
);
type QueryFactory = fn() -> QueryBundle;
type FixtureFactory = fn() -> &'static Fixture;

struct SplitMergePlan {
    natural_splits: Arc<[u64]>,
    morsel_ranges: Vec<Range<u64>>,
    natural_split_count: usize,
    concurrency: usize,
}

fn split_merge_plan(
    fixture: &Fixture,
    query: &ScanQuery,
    merge_factor: usize,
) -> VortexResult<SplitMergePlan> {
    if merge_factor == 0 {
        vortex_error::vortex_bail!("split merge factor must be non-zero");
    }
    let catalog = fixture.physical_splits.as_ref().ok_or_else(|| {
        vortex_error::vortex_err!("physical split catalog is required for real-split comparison")
    })?;
    let used_fields = query
        .conjuncts
        .iter()
        .map(|conjunct| conjunct.field)
        .chain(query.projection.iter().copied())
        .collect::<BTreeSet<_>>();
    let source_fields = used_fields
        .iter()
        .flat_map(|field| {
            fixture
                .physical_field_sources
                .get(field.0)
                .into_iter()
                .flatten()
        })
        .collect::<BTreeSet<_>>();
    if source_fields.is_empty() {
        vortex_error::vortex_bail!("query maps to no physical source fields");
    }

    let mut natural_splits = vec![0];
    let mut morsel_ranges = Vec::new();
    let mut natural_split_count = 0;
    let mut file_offset = 0u64;
    for file in &catalog.files {
        let mut boundaries = BTreeSet::from([0, file.row_count]);
        for field in &source_fields {
            let field_boundaries = file.fields.get(field.as_str()).ok_or_else(|| {
                vortex_error::vortex_err!("physical split catalog is missing field {field}")
            })?;
            boundaries.extend(field_boundaries.iter().copied());
        }
        let boundaries = boundaries.into_iter().collect::<Vec<_>>();
        if boundaries.first() != Some(&0) || boundaries.last() != Some(&file.row_count) {
            vortex_error::vortex_bail!("physical split boundaries do not cover a catalog file");
        }
        natural_split_count += boundaries.len() - 1;
        natural_splits.extend(
            boundaries
                .iter()
                .skip(1)
                .map(|boundary| file_offset + boundary),
        );
        for split_start in (0..boundaries.len() - 1).step_by(merge_factor) {
            let split_end = (split_start + merge_factor).min(boundaries.len() - 1);
            morsel_ranges
                .push(file_offset + boundaries[split_start]..file_offset + boundaries[split_end]);
        }
        file_offset += file.row_count;
    }
    if file_offset != fixture.plan.row_count {
        vortex_error::vortex_bail!(
            "physical split catalog has {file_offset} rows but fixture has {}",
            fixture.plan.row_count
        );
    }
    let concurrency = 16.min(morsel_ranges.len()).max(1);
    Ok(SplitMergePlan {
        natural_splits: natural_splits.into(),
        morsel_ranges,
        natural_split_count,
        concurrency,
    })
}

async fn compare_workload(
    name: &str,
    fixture: FixtureFactory,
    query: QueryFactory,
    iterations: usize,
) -> VortexResult<()> {
    let source_fixture = fixture();
    // Materialize one canonical query bundle for the whole comparison. Both execution paths
    // receive clones of this exact object rather than independently rebuilding the query.
    let query = query();
    let scan_query = &query.0;
    let first_plan = split_merge_plan(source_fixture, scan_query, 16)?;
    let fixture =
        rechunk_fixture_at_natural_splits(source_fixture, Arc::clone(&first_plan.natural_splits))
            .await?;
    eprintln!(
        "comparison_fixture workload={name} rows={} natural_splits={} serialized_bytes={} serialized_hash={:#x}",
        fixture.plan.row_count,
        first_plan.natural_split_count,
        fixture.serialized_bytes,
        fixture.serialized_hash,
    );
    if std::env::var_os("VORTEX_SELF_PACED_COMPARE_TRACE").is_some() {
        return trace_split_workload(name, &fixture, query, &first_plan).await;
    }
    for merge_factor in [16, 32] {
        let split_plan = split_merge_plan(source_fixture, scan_query, merge_factor)?;
        let v1_warmup = run_v1_fixture_with_splits(
            &fixture,
            query.clone(),
            Arc::clone(&split_plan.natural_splits),
            split_plan.concurrency,
            true,
        )
        .await?;
        let candidate_warmup = run_candidate_fixture_with_ranges(
            &fixture,
            query.clone(),
            &split_plan.morsel_ranges,
            split_plan.concurrency,
            true,
        )
        .await?;
        if !logical_outputs_match(v1_warmup.0, candidate_warmup.0) {
            vortex_error::vortex_bail!(
                "{name} output mismatch at split_merge={merge_factor}: v1={:?} self_paced={:?}",
                v1_warmup.0,
                candidate_warmup.0,
            );
        }
        let mut v1_times = Vec::with_capacity(iterations);
        let mut candidate_times = Vec::with_capacity(iterations);
        for iteration in 0..iterations {
            let started = Instant::now();
            if iteration % 2 == 0 {
                std::hint::black_box(
                    run_v1_fixture_with_splits(
                        &fixture,
                        query.clone(),
                        Arc::clone(&split_plan.natural_splits),
                        split_plan.concurrency,
                        false,
                    )
                    .await?,
                );
                v1_times.push(started.elapsed());
                let started = Instant::now();
                std::hint::black_box(
                    run_candidate_fixture_with_ranges(
                        &fixture,
                        query.clone(),
                        &split_plan.morsel_ranges,
                        split_plan.concurrency,
                        false,
                    )
                    .await?,
                );
                candidate_times.push(started.elapsed());
            } else {
                std::hint::black_box(
                    run_candidate_fixture_with_ranges(
                        &fixture,
                        query.clone(),
                        &split_plan.morsel_ranges,
                        split_plan.concurrency,
                        false,
                    )
                    .await?,
                );
                candidate_times.push(started.elapsed());
                let started = Instant::now();
                std::hint::black_box(
                    run_v1_fixture_with_splits(
                        &fixture,
                        query.clone(),
                        Arc::clone(&split_plan.natural_splits),
                        split_plan.concurrency,
                        false,
                    )
                    .await?,
                );
                v1_times.push(started.elapsed());
            }
        }
        print_split_comparison(
            name,
            merge_factor,
            &split_plan,
            iterations,
            &mut v1_times,
            &mut candidate_times,
        );
    }
    Ok(())
}

async fn trace_split_workload(
    name: &str,
    fixture: &Fixture,
    query: QueryBundle,
    split_plan: &SplitMergePlan,
) -> VortexResult<()> {
    let (v1_output, v1_source) = run_v1_fixture_with_splits(
        fixture,
        query.clone(),
        Arc::clone(&split_plan.natural_splits),
        split_plan.concurrency,
        true,
    )
    .await?;
    let source_inner = Arc::clone(&fixture.source);
    let (source, counts) = CountingSource::new(source_inner, true);
    let (scan_query, ..) = query;
    let result = run_self_paced_ranges(
        &fixture.plan,
        scan_query,
        &split_plan.morsel_ranges,
        source,
        &SESSION,
        RunOptions {
            policy: benchmark_schedule_policy(split_plan.concurrency),
            transition_budget: 32,
            retention: RetentionPolicy::RetainUntilDead,
            concurrency: split_plan.concurrency,
            collect_trace: true,
            speculative_io: *SPECULATIVE_IO_CONFIG,
        },
    )
    .await?;
    let output = stable_output_hash(&result.batches, &SESSION)?;
    if !logical_outputs_match(v1_output, output) {
        vortex_error::vortex_bail!(
            "{name} traced output mismatch: v1={v1_output:?} self_paced={output:?}"
        );
    }
    println!(
        "trace_begin workload={name} split_merge=16 natural_splits={} morsels={} concurrency={} rows={} hash={:#x}",
        split_plan.natural_split_count,
        split_plan.morsel_ranges.len(),
        split_plan.concurrency,
        output.0,
        output.1,
    );
    println!(
        "v1_summary rows={} hash={:#x} requests={} unique_segments={} segment_requests_min={} segment_requests_max={} bytes={}",
        v1_output.0,
        v1_output.1,
        v1_source.requests,
        v1_source.unique_segments,
        v1_source.segment_requests_min,
        v1_source.segment_requests_max,
        v1_source.bytes,
    );
    write_execution_trace(std::io::stdout().lock(), &result.trace)
        .map_err(|error| vortex_error::vortex_err!("failed to write self-paced trace: {error}"))?;
    let source = counts.summary();
    println!(
        "trace_end requests={} unique_segments={} segment_requests_min={} segment_requests_max={} bytes={} advance_calls={} transitions={} nodes_inspected={} completion_wake_candidates_inspected={} tasks_offered={} tasks_claimed={} tasks_completed={} scheduler_passes={} scheduler_tasks_considered={} scheduler_tasks_admitted={} completion_batches={} completions_drained={} max_completion_batch={} demand_combinations={} inline_demand_combinations={} demand_direct_adoptions={} demand_noop_adoptions={} adaptive_launches={} adaptive_waits={} predicate_reorders={} demand_initial={} demand_final={} resource_nodes={} morsel_slots={} trace_events={}",
        source.requests,
        source.unique_segments,
        source.segment_requests_min,
        source.segment_requests_max,
        source.bytes,
        result.metrics.advance_calls,
        result.metrics.transitions,
        result.metrics.nodes_inspected,
        result.metrics.completion_wake_candidates_inspected,
        result.metrics.tasks_offered,
        result.metrics.tasks_claimed,
        result.metrics.tasks_completed,
        result.metrics.scheduler_passes,
        result.metrics.scheduler_tasks_considered,
        result.metrics.scheduler_tasks_admitted,
        result.metrics.completion_batches,
        result.metrics.completions_drained,
        result.metrics.max_completion_batch,
        result.metrics.demand_combinations,
        result.metrics.inline_demand_combinations,
        result.metrics.demand_direct_adoptions,
        result.metrics.demand_noop_adoptions,
        result.metrics.adaptive_predicate_launches,
        result.metrics.adaptive_predicate_waits,
        result.metrics.predicate_reorders,
        result.metrics.demand_rows_initial,
        result.metrics.demand_rows_current,
        result.metrics.resource_nodes,
        result.metrics.morsel_slots,
        result.trace.len(),
    );
    Ok(())
}

fn print_split_comparison(
    workload: &str,
    merge_factor: usize,
    split_plan: &SplitMergePlan,
    iterations: usize,
    v1_times: &mut [Duration],
    candidate_times: &mut [Duration],
) {
    let v1 = median(v1_times);
    let candidate = median(candidate_times);
    println!(
        "compare workload={workload} split_merge={merge_factor} natural_splits={} morsels={} concurrency={} iterations={iterations} v1_ms={:.3} self_paced_ms={:.3} ratio={:.3}",
        split_plan.natural_split_count,
        split_plan.morsel_ranges.len(),
        split_plan.concurrency,
        v1.as_secs_f64() * 1_000.0,
        candidate.as_secs_f64() * 1_000.0,
        candidate.as_secs_f64() / v1.as_secs_f64(),
    );
}

fn print_comparison(
    workload: &str,
    morsel_rows: usize,
    iterations: usize,
    v1_times: &mut [Duration],
    candidate_times: &mut [Duration],
) {
    let v1 = median(v1_times);
    let candidate = median(candidate_times);
    println!(
        "compare workload={workload} self_paced_morsel_rows={morsel_rows} concurrency=16 iterations={iterations} v1_ms={:.3} self_paced_ms={:.3} ratio={:.3}",
        v1.as_secs_f64() * 1_000.0,
        candidate.as_secs_f64() * 1_000.0,
        candidate.as_secs_f64() / v1.as_secs_f64(),
    );
}

fn logical_outputs_match(lhs: (usize, u64), rhs: (usize, u64)) -> bool {
    lhs.0 == rhs.0 && (lhs.0 == 0 || lhs.1 == rhs.1)
}

fn median(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn stable_array_hash(arrays: &[ArrayRef]) -> VortexResult<(usize, u64)> {
    let mut field_hashes = Vec::new();
    let mut rows = 0usize;
    let mut ctx = SESSION.create_execution_ctx();
    for array in arrays {
        let array = if array.is::<Struct>() {
            array.as_::<Struct>().into_owned()
        } else {
            array.clone().execute::<StructArray>(&mut ctx)?
        };
        rows += array.len();
        let fields = array.iter_unmasked_fields().collect::<Vec<_>>();
        if field_hashes.is_empty() {
            field_hashes.resize(fields.len(), 0xcbf29ce484222325u64);
        } else if field_hashes.len() != fields.len() {
            vortex_error::vortex_bail!(
                "output field count changed from {} to {} between batches",
                field_hashes.len(),
                fields.len()
            );
        }
        for (field_idx, field) in fields.into_iter().enumerate() {
            let values = if field.is::<Primitive>() {
                field.as_::<Primitive>().into_owned()
            } else {
                field.clone().execute::<PrimitiveArray>(&mut ctx)?
            };
            for value in values.as_slice::<i64>() {
                field_hashes[field_idx] ^= u64::from_le_bytes(value.to_le_bytes());
                field_hashes[field_idx] = field_hashes[field_idx].wrapping_mul(0x100000001b3);
            }
        }
    }
    let hash = field_hashes
        .into_iter()
        .fold(0xcbf29ce484222325u64, |hash, field_hash| {
            (hash ^ field_hash).wrapping_mul(0x100000001b3)
        });
    Ok((rows, hash))
}

async fn validate_outputs() -> VortexResult<()> {
    for (selectivity, morsel_rows, concurrency) in [
        (1, 65_536, 16),
        (1, 131_072, 16),
        (50, 65_536, 16),
        (50, 131_072, 16),
        (95, 65_536, 16),
        (95, 131_072, 16),
    ] {
        let v1 = run_v1(selectivity, concurrency, true).await?;
        let candidate = run_candidate(selectivity, morsel_rows, concurrency, true).await?;
        if !logical_outputs_match(v1.0, candidate.0) {
            vortex_error::vortex_bail!(
                "self-paced output differs from V1 at {selectivity}% selectivity: V1 rows={} hash={:#x}, self-paced rows={} hash={:#x}",
                v1.0.0,
                v1.0.1,
                candidate.0.0,
                candidate.0.1,
            );
        }
        eprintln!(
            "parity selectivity={selectivity}% morsel_rows={morsel_rows} concurrency={concurrency} rows={} hash={:#x} v1_requests={} v1_bytes={} self_paced_requests={} self_paced_bytes={}",
            v1.0.0, v1.0.1, v1.1.requests, v1.1.bytes, candidate.1.requests, candidate.1.bytes,
        );
        eprintln!(
            "reuse v1_unique_segments={} v1_segment_requests={}..{} self_paced_unique_segments={} self_paced_segment_requests={}..{}",
            v1.1.unique_segments,
            v1.1.segment_requests_min,
            v1.1.segment_requests_max,
            candidate.1.unique_segments,
            candidate.1.segment_requests_min,
            candidate.1.segment_requests_max,
        );
    }
    validate_workload(
        "clickbench_selective",
        clickbench_fixture,
        clickbench_selective_query,
    )
    .await?;
    validate_workload(
        "clickbench_dashboard",
        clickbench_fixture,
        clickbench_dashboard_query,
    )
    .await?;
    for query_id in CLICKBENCH_SUITE_QUERY_IDS {
        validate_workload(
            &format!("clickbench_q{query_id:02}"),
            clickbench_fixture,
            clickbench_suite_query_factory(query_id),
        )
        .await?;
    }
    for query_id in FINEWEB_QUERY_IDS {
        validate_workload(
            &format!("fineweb_q{query_id:02}_scan_analogue"),
            fineweb_fixture,
            fineweb_query_factory(query_id),
        )
        .await?;
    }
    validate_workload("tpch_q6_scan", tpch_fixture, tpch_q6_scan_query).await?;
    validate_workload("tpch_q1_scan", tpch_fixture, tpch_q1_scan_query).await?;
    validate_workload("tpch_v1_friendly", tpch_fixture, tpch_v1_friendly_query).await?;
    Ok(())
}

async fn validate_workload(
    name: &str,
    fixture: FixtureFactory,
    query: QueryFactory,
) -> VortexResult<()> {
    for morsel_rows in [65_536, 131_072] {
        let fixture = fixture();
        let v1 = run_v1_fixture(fixture, query(), 16, true).await?;
        let candidate = run_candidate_fixture(fixture, query(), morsel_rows, 16, true).await?;
        if !logical_outputs_match(v1.0, candidate.0) {
            vortex_error::vortex_bail!(
                "{name} output mismatch at morsel_rows={morsel_rows}: v1={:?} self_paced={:?}",
                v1.0,
                candidate.0,
            );
        }
        eprintln!(
            "parity workload={name} morsel_rows={morsel_rows} concurrency=16 rows={} hash={:#x} v1_requests={} self_paced_requests={}",
            v1.0.0, v1.0.1, v1.1.requests, candidate.1.requests,
        );
    }
    Ok(())
}

#[divan::bench(args = [(1usize, 16usize), (50, 16), (95, 16)])]
fn layout_reader_v1(bencher: Bencher, args: (usize, usize)) {
    let (selectivity, concurrency) = args;
    bencher.bench(|| {
        divan::black_box(
            RUNTIME
                .block_on(run_v1(selectivity, concurrency, false))
                .unwrap(),
        )
    });
}

#[divan::bench(args = [
    (1usize, 65_536usize, 16usize),
    (1, 131_072, 16),
    (50, 65_536, 16),
    (50, 131_072, 16),
    (95, 65_536, 16),
    (95, 131_072, 16),
])]
fn self_paced(bencher: Bencher, args: (usize, usize, usize)) {
    let (selectivity, morsel_rows, concurrency) = args;
    bencher.bench(|| {
        divan::black_box(
            RUNTIME
                .block_on(run_candidate(selectivity, morsel_rows, concurrency, false))
                .unwrap(),
        )
    });
}

fn bench_v1_fixture(bencher: Bencher, fixture: &Fixture, query: QueryFactory) {
    bencher.bench(|| {
        divan::black_box(
            RUNTIME
                .block_on(run_v1_fixture(fixture, query(), 16, false))
                .unwrap(),
        )
    });
}

fn bench_self_paced_fixture(
    bencher: Bencher,
    fixture: &Fixture,
    query: QueryFactory,
    morsel_rows: usize,
) {
    bencher.bench(|| {
        divan::black_box(
            RUNTIME
                .block_on(run_candidate_fixture(
                    fixture,
                    query(),
                    morsel_rows,
                    16,
                    false,
                ))
                .unwrap(),
        )
    });
}

#[divan::bench]
fn clickbench_selective_v1(bencher: Bencher) {
    bench_v1_fixture(bencher, clickbench_fixture(), clickbench_selective_query);
}

#[divan::bench(args = [65_536usize, 131_072])]
fn clickbench_selective_self_paced(bencher: Bencher, morsel_rows: usize) {
    bench_self_paced_fixture(
        bencher,
        clickbench_fixture(),
        clickbench_selective_query,
        morsel_rows,
    );
}

#[divan::bench]
fn clickbench_dashboard_v1(bencher: Bencher) {
    bench_v1_fixture(bencher, clickbench_fixture(), clickbench_dashboard_query);
}

#[divan::bench(args = [65_536usize, 131_072])]
fn clickbench_dashboard_self_paced(bencher: Bencher, morsel_rows: usize) {
    bench_self_paced_fixture(
        bencher,
        clickbench_fixture(),
        clickbench_dashboard_query,
        morsel_rows,
    );
}

#[divan::bench(args = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 39, 40, 41, 42])]
fn clickbench_suite_v1(bencher: Bencher, query_id: usize) {
    bench_v1_fixture(
        bencher,
        clickbench_fixture(),
        clickbench_suite_query_factory(query_id),
    );
}

#[divan::bench(args = [
    (0usize, 65_536usize), (0, 131_072),
    (1, 65_536), (1, 131_072),
    (2, 65_536), (2, 131_072),
    (3, 65_536), (3, 131_072),
    (4, 65_536), (4, 131_072),
    (5, 65_536), (5, 131_072),
    (6, 65_536), (6, 131_072),
    (7, 65_536), (7, 131_072),
    (8, 65_536), (8, 131_072),
    (9, 65_536), (9, 131_072),
    (39, 65_536), (39, 131_072),
    (40, 65_536), (40, 131_072),
    (41, 65_536), (41, 131_072),
    (42, 65_536), (42, 131_072),
])]
fn clickbench_suite_self_paced(bencher: Bencher, args: (usize, usize)) {
    let (query_id, morsel_rows) = args;
    bench_self_paced_fixture(
        bencher,
        clickbench_fixture(),
        clickbench_suite_query_factory(query_id),
        morsel_rows,
    );
}

#[divan::bench]
fn tpch_q6_scan_v1(bencher: Bencher) {
    bench_v1_fixture(bencher, tpch_fixture(), tpch_q6_scan_query);
}

#[divan::bench(args = [65_536usize, 131_072])]
fn tpch_q6_scan_self_paced(bencher: Bencher, morsel_rows: usize) {
    bench_self_paced_fixture(bencher, tpch_fixture(), tpch_q6_scan_query, morsel_rows);
}

#[divan::bench]
fn tpch_q1_scan_v1(bencher: Bencher) {
    bench_v1_fixture(bencher, tpch_fixture(), tpch_q1_scan_query);
}

#[divan::bench(args = [65_536usize, 131_072])]
fn tpch_q1_scan_self_paced(bencher: Bencher, morsel_rows: usize) {
    bench_self_paced_fixture(bencher, tpch_fixture(), tpch_q1_scan_query, morsel_rows);
}

#[divan::bench]
fn tpch_v1_friendly_v1(bencher: Bencher) {
    bench_v1_fixture(bencher, tpch_fixture(), tpch_v1_friendly_query);
}

#[divan::bench(args = [65_536usize, 131_072])]
fn tpch_v1_friendly_self_paced(bencher: Bencher, morsel_rows: usize) {
    bench_self_paced_fixture(bencher, tpch_fixture(), tpch_v1_friendly_query, morsel_rows);
}
