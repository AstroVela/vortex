// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Audit the physical natural-split boundaries of real Vortex files for the `self_paced_vs_v1`
//! comparison benchmark.
//!
//! `VORTEX_SPLIT_AUDIT_MODE` selects the dataset:
//!
//! - `fineweb` (default): every Parquet shard in `VORTEX_FINEWEB_PARQUET`, writing the seven raw
//!   source columns (strings and the language score) with the default write strategy.
//! - `tpch`: the single lineitem Parquet at `VORTEX_TPCH_LINEITEM_PARQUET`, writing the eight
//!   benchmark columns converted to the restricted executor's `i64` domain.
//! - `clickbench`: `hits_0.parquet..hits_N.parquet` under `VORTEX_CLICKBENCH_PARQUET_DIR`
//!   (`VORTEX_CLICKBENCH_MAX_FILES`, default 100), writing the benchmark columns cast to `i64`.
//!
//! Each mode reopens the written bytes and collects the per-field natural split boundaries that
//! `SplitBy::natural_splits` reports before any scheduler subdivision, emitting the physical
//! split catalog consumed through `VORTEX_*_SPLIT_CATALOG`.

#![expect(clippy::unwrap_used)]

use std::fs::File;
use std::path::PathBuf;
use std::sync::LazyLock;

use arrow_array::Array as ArrowArray;
use arrow_array::Float32Array;
use arrow_array::Float64Array;
use arrow_array::StringViewArray;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use futures::stream;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::session::ArraySessionExt;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::validity::Validity;
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
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

const SOURCE_COLUMNS: [&str; 7] = [
    "dump",
    "date",
    "url",
    "text",
    "language",
    "language_score",
    "file_path",
];

static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
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
    enable_all_registered_array_encodings(&session);
    session
});

const AUDIT_EDITION: EditionId = EditionId::new("fineweb-split-audit", 2026, 8, 0);

fn enable_all_registered_array_encodings(session: &VortexSession) {
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: AUDIT_EDITION,
            min_vortex_version: None,
        })
        .unwrap();
    let ids = session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect::<Vec<_>>());
    for id in ids {
        editions
            .declare_inclusion(EditionInclusion::new(&id, AUDIT_EDITION))
            .unwrap();
    }
    session.enable_edition(AUDIT_EDITION).unwrap();
}

fn main() {
    RUNTIME.block_on(run()).unwrap();
}

fn parquet_paths() -> VortexResult<Vec<PathBuf>> {
    let input = std::env::var_os("VORTEX_FINEWEB_PARQUET")
        .map(PathBuf::from)
        .ok_or_else(|| vortex_error::vortex_err!("VORTEX_FINEWEB_PARQUET is required"))?;
    let mut paths = std::fs::read_dir(&input)
        .map_err(|error| vortex_error::vortex_err!("failed to list {}: {error}", input.display()))?
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
        .collect::<VortexResult<Vec<_>>>()?;
    paths.sort_unstable();
    if paths.is_empty() {
        vortex_error::vortex_bail!("FineWeb input contains no Parquet files");
    }
    Ok(paths)
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

fn string_column(array: &dyn ArrowArray) -> VarBinViewArray {
    VarBinViewArray::from_iter_nullable_str(
        (0..array.len()).map(|index| arrow_string_value(array, index)),
    )
}

fn score_column(array: &dyn ArrowArray) -> VortexResult<PrimitiveArray> {
    let values = (0..array.len()).map(|index| {
        if array.is_null(index) {
            return None;
        }
        match array.data_type() {
            DataType::Float64 => array
                .as_any()
                .downcast_ref::<Float64Array>()
                .map(|array| array.value(index)),
            DataType::Float32 => array
                .as_any()
                .downcast_ref::<Float32Array>()
                .map(|array| f64::from(array.value(index))),
            _ => None,
        }
    });
    Ok(PrimitiveArray::from_option_iter(values))
}

fn batch_to_struct(batch: &arrow_array::RecordBatch) -> VortexResult<ArrayRef> {
    let arrays = SOURCE_COLUMNS
        .iter()
        .map(|name| {
            let column = batch
                .column_by_name(name)
                .ok_or_else(|| vortex_error::vortex_err!("missing column {name}"))?;
            Ok(if *name == "language_score" {
                score_column(column.as_ref())?.into_array()
            } else {
                string_column(column.as_ref()).into_array()
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(StructArray::try_new(
        FieldNames::from(SOURCE_COLUMNS),
        arrays,
        batch.num_rows(),
        Validity::NonNullable,
    )?
    .into_array())
}

/// Write the chunk stream with the default strategy, reopen it, and report the row count plus
/// each named field's natural split boundaries.
async fn audit_written_file(
    dtype: vortex_array::dtype::DType,
    chunks: impl futures::Stream<Item = VortexResult<ArrayRef>> + Send + 'static,
    fields: &[&str],
    label: &str,
) -> VortexResult<serde_json::Value> {
    let mut serialized = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut serialized, ArrayStreamAdapter::new(dtype, chunks))
        .await?;
    let serialized = serialized.freeze();
    eprintln!("audit file={label} vortex_bytes={}", serialized.len());
    let vortex_file = SESSION.open_options().open_buffer(serialized)?;
    let layout = vortex_file.footer().layout();
    let reader = layout.new_reader(
        "split-audit".into(),
        vortex_file.segment_source(),
        &SESSION,
        &LayoutReaderContext::default(),
    )?;
    let row_count = reader.row_count();
    let mut field_boundaries = serde_json::Map::new();
    for field in fields {
        let boundaries = SplitBy::natural_splits(
            reader.as_ref(),
            &(0..row_count),
            &[FieldMask::Prefix(FieldPath::from_name(*field))],
        )?;
        eprintln!(
            "audit file={label} field={field} boundaries={}",
            boundaries.len(),
        );
        field_boundaries.insert(
            (*field).to_string(),
            serde_json::Value::from(boundaries.to_vec()),
        );
    }
    Ok(serde_json::json!({
        "row_count": row_count,
        "fields": serde_json::Value::Object(field_boundaries),
    }))
}

fn write_catalog(output: &PathBuf, files: Vec<serde_json::Value>) -> VortexResult<()> {
    let catalog = serde_json::json!({ "files": files });
    let encoded = serde_json::to_vec(&catalog)
        .map_err(|error| vortex_error::vortex_err!("failed to encode catalog: {error}"))?;
    std::fs::write(output, encoded).map_err(|error| {
        vortex_error::vortex_err!("failed to write {}: {error}", output.display())
    })?;
    eprintln!("audit catalog written to {}", output.display());
    Ok(())
}

const TPCH_COLUMNS: [(&str, i128); 8] = [
    ("l_orderkey", 1),
    ("l_partkey", 1),
    ("l_suppkey", 1),
    ("l_quantity", 100),
    ("l_extendedprice", 1),
    ("l_discount", 1),
    ("l_tax", 1),
    ("l_shipdate", 1),
];

fn tpch_i64_values(array: &dyn ArrowArray, divisor: i128) -> VortexResult<Vec<i64>> {
    if array.null_count() != 0 {
        vortex_error::vortex_bail!("TPC-H input contains nulls");
    }
    if let Some(array) = array
        .as_any()
        .downcast_ref::<arrow_array::Decimal128Array>()
    {
        return array
            .values()
            .iter()
            .map(|value| Ok(i64::try_from(*value / divisor)?))
            .collect();
    }
    if let Some(array) = array.as_any().downcast_ref::<arrow_array::Date32Array>() {
        return Ok(array
            .values()
            .iter()
            .map(|value| i64::from(*value))
            .collect());
    }
    let casted = arrow_cast::cast(array, &DataType::Int64)
        .map_err(|error| vortex_error::vortex_err!("cannot cast to i64: {error}"))?;
    let casted = casted
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .ok_or_else(|| vortex_error::vortex_err!("cast did not produce i64"))?;
    Ok(casted.values().to_vec())
}

fn i64_batch_to_struct(
    batch: &arrow_array::RecordBatch,
    columns: &[(&'static str, i128)],
    names: &[&'static str],
) -> VortexResult<ArrayRef> {
    let arrays = columns
        .iter()
        .map(|(name, divisor)| {
            let column = batch
                .column_by_name(name)
                .ok_or_else(|| vortex_error::vortex_err!("missing column {name}"))?;
            Ok(
                vortex_buffer::Buffer::from_iter(tpch_i64_values(column.as_ref(), *divisor)?)
                    .into_array(),
            )
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(StructArray::try_new(
        FieldNames::from(
            names
                .iter()
                .map(|name| std::sync::Arc::<str>::from(*name))
                .collect::<Vec<_>>(),
        ),
        arrays,
        batch.num_rows(),
        Validity::NonNullable,
    )?
    .into_array())
}

fn i64_parquet_stream(
    path: &PathBuf,
    columns: &'static [(&'static str, i128)],
    names: &'static [&'static str],
    batch_size: usize,
) -> VortexResult<(
    vortex_array::dtype::DType,
    impl futures::Stream<Item = VortexResult<ArrayRef>> + Send + 'static,
)> {
    let file = File::open(path)
        .map_err(|error| vortex_error::vortex_err!("failed to open {}: {error}", path.display()))?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
        vortex_error::vortex_err!("failed to inspect {}: {error}", path.display())
    })?;
    let mut projection = columns
        .iter()
        .map(|(name, _)| {
            builder
                .schema()
                .index_of(name)
                .map_err(|error| vortex_error::vortex_err!("{name} in {}: {error}", path.display()))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    projection.sort_unstable();
    projection.dedup();
    let projection = ProjectionMask::roots(builder.parquet_schema(), projection);
    builder = builder
        .with_projection(projection)
        .with_batch_size(batch_size);
    let mut reader = builder
        .build()
        .map_err(|error| vortex_error::vortex_err!("failed to read {}: {error}", path.display()))?;
    let first = reader
        .next()
        .transpose()
        .map_err(|error| vortex_error::vortex_err!("failed to decode: {error}"))?
        .ok_or_else(|| vortex_error::vortex_err!("{} is empty", path.display()))?;
    let first = i64_batch_to_struct(&first, columns, names)?;
    let dtype = first.dtype().clone();
    let chunks = stream::iter(std::iter::once(Ok(first)).chain(reader.map(move |batch| {
        let batch =
            batch.map_err(|error| vortex_error::vortex_err!("failed to decode: {error}"))?;
        i64_batch_to_struct(&batch, columns, names)
    })));
    Ok((dtype, chunks))
}

async fn run_tpch(output: PathBuf) -> VortexResult<()> {
    let path = std::env::var_os("VORTEX_TPCH_LINEITEM_PARQUET")
        .map(PathBuf::from)
        .ok_or_else(|| vortex_error::vortex_err!("VORTEX_TPCH_LINEITEM_PARQUET is required"))?;
    static NAMES: [&str; 8] = [
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_shipdate",
    ];
    let (dtype, chunks) = i64_parquet_stream(&path, &TPCH_COLUMNS, &NAMES, 524_288)?;
    let entry = audit_written_file(dtype, chunks, &NAMES, &path.display().to_string()).await?;
    write_catalog(&output, vec![entry])
}

const CLICKBENCH_COLUMNS: [(&str, i128); 21] = [
    ("EventTime", 1),
    ("UserID", 1),
    ("CounterID", 1),
    ("RegionID", 1),
    ("IsMobile", 1),
    ("ResponseEndTiming", 1),
    ("SendTiming", 1),
    ("EventDate", 1),
    ("WatchID", 1),
    ("AdvEngineID", 1),
    ("ResolutionWidth", 1),
    ("SearchEngineID", 1),
    ("TraficSourceID", 1),
    ("RefererHash", 1),
    ("URLHash", 1),
    ("IsRefresh", 1),
    ("WindowClientWidth", 1),
    ("WindowClientHeight", 1),
    ("DontCountHits", 1),
    ("IsLink", 1),
    ("IsDownload", 1),
];

async fn run_clickbench(output: PathBuf) -> VortexResult<()> {
    let parquet_dir = std::env::var_os("VORTEX_CLICKBENCH_PARQUET_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| vortex_error::vortex_err!("VORTEX_CLICKBENCH_PARQUET_DIR is required"))?;
    let file_count = std::env::var("VORTEX_CLICKBENCH_MAX_FILES")
        .map_or(Ok(100), |value| value.parse::<usize>())
        .map_err(|error| {
            vortex_error::vortex_err!("invalid VORTEX_CLICKBENCH_MAX_FILES: {error}")
        })?;
    static NAMES: [&str; 21] = [
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
        "RefererHash",
        "URLHash",
        "IsRefresh",
        "WindowClientWidth",
        "WindowClientHeight",
        "DontCountHits",
        "IsLink",
        "IsDownload",
    ];
    let mut files = Vec::with_capacity(file_count);
    for shard in 0..file_count {
        let path = parquet_dir.join(format!("hits_{shard}.parquet"));
        let (dtype, chunks) = i64_parquet_stream(&path, &CLICKBENCH_COLUMNS, &NAMES, 131_072)?;
        files.push(audit_written_file(dtype, chunks, &NAMES, &path.display().to_string()).await?);
    }
    write_catalog(&output, files)
}

/// Audit the statpopgen gnomAD parquet: strings stay strings, QUAL stays floating point, and the
/// integer columns are written as integers, so the physical file mirrors a production write.
async fn run_statpopgen(output: PathBuf) -> VortexResult<()> {
    static COLUMNS: [&str; 10] = [
        "POS",
        "QUAL",
        "ID",
        "REF",
        "AN",
        "AN_raw",
        "gnomad_AN",
        "AN_ceu",
        "AN_yri_XX",
        "AN_fin_XX",
    ];
    let path = std::env::var_os("VORTEX_STATPOPGEN_PARQUET")
        .map(PathBuf::from)
        .ok_or_else(|| vortex_error::vortex_err!("VORTEX_STATPOPGEN_PARQUET is required"))?;
    let file = File::open(&path)
        .map_err(|error| vortex_error::vortex_err!("failed to open {}: {error}", path.display()))?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
        vortex_error::vortex_err!("failed to inspect {}: {error}", path.display())
    })?;
    let projection = COLUMNS
        .iter()
        .map(|name| {
            builder
                .schema()
                .index_of(name)
                .map_err(|error| vortex_error::vortex_err!("{name} in {}: {error}", path.display()))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let projection = ProjectionMask::roots(builder.parquet_schema(), projection);
    builder = builder.with_projection(projection).with_batch_size(100_000);
    let mut reader = builder
        .build()
        .map_err(|error| vortex_error::vortex_err!("failed to read {}: {error}", path.display()))?;

    let to_struct = |batch: &arrow_array::RecordBatch| -> VortexResult<ArrayRef> {
        let arrays = COLUMNS
            .iter()
            .map(|name| {
                let column = batch
                    .column_by_name(name)
                    .ok_or_else(|| vortex_error::vortex_err!("missing column {name}"))?;
                Ok(match *name {
                    "ID" | "REF" => string_column(column.as_ref()).into_array(),
                    "QUAL" => score_column(column.as_ref())?.into_array(),
                    _ => {
                        let casted = arrow_cast::cast(column.as_ref(), &DataType::Int64).map_err(
                            |error| vortex_error::vortex_err!("cannot cast {name} to i64: {error}"),
                        )?;
                        let casted = casted
                            .as_any()
                            .downcast_ref::<arrow_array::Int64Array>()
                            .ok_or_else(|| {
                                vortex_error::vortex_err!("cast of {name} did not produce i64")
                            })?;
                        vortex_buffer::Buffer::from_iter((0..casted.len()).map(|row| {
                            if casted.is_null(row) {
                                0
                            } else {
                                casted.value(row)
                            }
                        }))
                        .into_array()
                    }
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(StructArray::try_new(
            FieldNames::from(COLUMNS),
            arrays,
            batch.num_rows(),
            Validity::NonNullable,
        )?
        .into_array())
    };

    let first = reader
        .next()
        .transpose()
        .map_err(|error| vortex_error::vortex_err!("failed to decode: {error}"))?
        .ok_or_else(|| vortex_error::vortex_err!("{} is empty", path.display()))?;
    let first = to_struct(&first)?;
    let dtype = first.dtype().clone();
    let chunks = stream::iter(std::iter::once(Ok(first)).chain(reader.map(move |batch| {
        let batch =
            batch.map_err(|error| vortex_error::vortex_err!("failed to decode: {error}"))?;
        to_struct(&batch)
    })));
    let entry = audit_written_file(dtype, chunks, &COLUMNS, &path.display().to_string()).await?;
    write_catalog(&output, vec![entry])
}

async fn run() -> VortexResult<()> {
    let output = std::env::var_os("VORTEX_SPLIT_CATALOG_OUT")
        .map(PathBuf::from)
        .ok_or_else(|| vortex_error::vortex_err!("VORTEX_SPLIT_CATALOG_OUT is required"))?;
    match std::env::var("VORTEX_SPLIT_AUDIT_MODE").as_deref() {
        Ok("tpch") => return run_tpch(output).await,
        Ok("clickbench") => return run_clickbench(output).await,
        Ok("statpopgen") => return run_statpopgen(output).await,
        Ok("fineweb") | Err(_) => {}
        Ok(other) => vortex_error::vortex_bail!("unknown VORTEX_SPLIT_AUDIT_MODE {other}"),
    }
    let mut files = Vec::new();
    for path in parquet_paths()? {
        let file = File::open(&path).map_err(|error| {
            vortex_error::vortex_err!("failed to open {}: {error}", path.display())
        })?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
            vortex_error::vortex_err!("failed to inspect {}: {error}", path.display())
        })?;
        let projection = SOURCE_COLUMNS
            .iter()
            .map(|name| {
                builder.schema().index_of(name).map_err(|error| {
                    vortex_error::vortex_err!("{name} in {}: {error}", path.display())
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let projection = ProjectionMask::roots(builder.parquet_schema(), projection);
        builder = builder.with_projection(projection).with_batch_size(100_000);
        let mut reader = builder.build().map_err(|error| {
            vortex_error::vortex_err!("failed to read {}: {error}", path.display())
        })?;

        let first = reader
            .next()
            .transpose()
            .map_err(|error| {
                vortex_error::vortex_err!("failed to decode {}: {error}", path.display())
            })?
            .ok_or_else(|| vortex_error::vortex_err!("{} is empty", path.display()))?;
        let first = batch_to_struct(&first)?;
        let dtype = first.dtype().clone();
        let chunks = stream::iter(std::iter::once(Ok(first)).chain(reader.map(|batch| {
            let batch =
                batch.map_err(|error| vortex_error::vortex_err!("failed to decode: {error}"))?;
            batch_to_struct(&batch)
        })));

        let mut serialized = ByteBufferMut::empty();
        SESSION
            .write_options()
            .write(&mut serialized, ArrayStreamAdapter::new(dtype, chunks))
            .await?;
        let serialized = serialized.freeze();
        eprintln!(
            "audit file={} vortex_bytes={}",
            path.display(),
            serialized.len(),
        );

        let vortex_file = SESSION.open_options().open_buffer(serialized)?;
        let layout = vortex_file.footer().layout();
        let reader = layout.new_reader(
            "fineweb-split-audit".into(),
            vortex_file.segment_source(),
            &SESSION,
            &LayoutReaderContext::default(),
        )?;
        let row_count = reader.row_count();
        let mut fields = serde_json::Map::new();
        for field in SOURCE_COLUMNS {
            let boundaries = SplitBy::natural_splits(
                reader.as_ref(),
                &(0..row_count),
                &[FieldMask::Prefix(FieldPath::from_name(field))],
            )?;
            eprintln!(
                "audit file={} field={field} boundaries={}",
                path.display(),
                boundaries.len(),
            );
            fields.insert(
                field.to_string(),
                serde_json::Value::from(boundaries.to_vec()),
            );
        }
        files.push(serde_json::json!({
            "row_count": row_count,
            "fields": serde_json::Value::Object(fields),
        }));
    }
    let catalog = serde_json::json!({ "files": files });
    let encoded = serde_json::to_vec(&catalog)
        .map_err(|error| vortex_error::vortex_err!("failed to encode catalog: {error}"))?;
    std::fs::write(&output, encoded).map_err(|error| {
        vortex_error::vortex_err!("failed to write {}: {error}", output.display())
    })?;
    eprintln!("audit catalog written to {}", output.display());
    Ok(())
}
