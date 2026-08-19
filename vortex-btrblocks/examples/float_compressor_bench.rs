// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::hint::black_box;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_select::concat::concat;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_arrow::ArrowSessionExt;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::float::FloatQuantScheme;
use vortex_btrblocks::schemes::float::OrderedBlockResidualScheme;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

const DEFAULT_ROW_COUNT: usize = 2_000_000;
const CALIFORNIA_COLUMNS: [&str; 9] = [
    "longitude",
    "latitude",
    "housingMedianAge",
    "totalRooms",
    "totalBedrooms",
    "population",
    "households",
    "medianIncome",
    "medianHouseValue",
];

struct Column {
    name: String,
    primitive: PrimitiveArray,
    array: ArrayRef,
}

fn column(name: impl Into<String>, primitive: PrimitiveArray) -> Column {
    let array = primitive.clone().into_array();
    Column {
        name: name.into(),
        primitive,
        array,
    }
}

fn synthetic_datasets(row_count: usize) -> Vec<(String, Vec<Column>)> {
    let mut widened_rng = StdRng::seed_from_u64(1);
    let widened_f32 = PrimitiveArray::from_iter((0..row_count).map(|index| {
        let trend = (index % 10_000) as f32 * 0.001;
        f64::from(trend + widened_rng.random_range(-1.0_f32..1.0))
    }));

    let mut walk_rng = StdRng::seed_from_u64(2);
    let mut value = 1_000.0_f64;
    let random_walk = PrimitiveArray::from_iter((0..row_count).map(|_| {
        value += walk_rng.random_range(-0.01_f64..0.01);
        value
    }));

    let mut uniform_rng = StdRng::seed_from_u64(3);
    let uniform = PrimitiveArray::from_iter(
        (0..row_count).map(|_| uniform_rng.random_range(-1_000_000.0_f64..1_000_000.0)),
    );

    let integer_valued = PrimitiveArray::from_iter((0..row_count).map(|index| {
        let permuted = (index as u64).wrapping_mul(2_654_435_761) % 1_000_000;
        permuted as f64 - 500_000.0
    }));

    vec![
        (
            "synthetic-widened-f32".to_string(),
            vec![column("value", widened_f32)],
        ),
        (
            "synthetic-random-walk".to_string(),
            vec![column("value", random_walk)],
        ),
        (
            "synthetic-uniform".to_string(),
            vec![column("value", uniform)],
        ),
        (
            "synthetic-integer-valued".to_string(),
            vec![column("value", integer_valued)],
        ),
    ]
}

fn read_california(path: &Path, row_count: usize) -> VortexResult<Vec<Column>> {
    let reader = BufReader::new(File::open(path)?);
    let mut values = std::array::from_fn::<_, 9, _>(|_| Vec::<f32>::new());
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let fields = line.split(',').collect::<Vec<_>>();
        vortex_ensure!(
            fields.len() == CALIFORNIA_COLUMNS.len(),
            "line {} contains {} fields instead of {}",
            line_index + 1,
            fields.len(),
            CALIFORNIA_COLUMNS.len()
        );
        for (column_index, field) in fields.into_iter().enumerate() {
            values[column_index].push(field.parse::<f32>().map_err(|error| {
                vortex_err!(
                    "cannot parse line {} column {} as f32: {}",
                    line_index + 1,
                    column_index + 1,
                    error
                )
            })?);
        }
    }
    for values in &mut values {
        vortex_ensure!(!values.is_empty(), "California Housing input is empty");
        values.truncate(row_count);
        while values.len() < row_count {
            let copy_len = (row_count - values.len()).min(values.len());
            values.extend_from_within(..copy_len);
        }
    }
    Ok(CALIFORNIA_COLUMNS
        .into_iter()
        .zip(values)
        .map(|(name, values)| column(name, PrimitiveArray::from_iter(values)))
        .collect())
}

fn read_parquet_numeric(
    path: &Path,
    row_count: usize,
    session: &VortexSession,
) -> VortexResult<Vec<Column>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)
        .map_err(|error| vortex_err!("cannot read Parquet metadata: {error}"))?;
    let schema = Arc::clone(builder.schema());
    let selected = schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, field)| {
            matches!(
                field.data_type(),
                DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
                    | DataType::Float32
                    | DataType::Float64
            )
            .then_some(index)
        })
        .collect::<Vec<_>>();
    vortex_ensure!(
        !selected.is_empty(),
        "Parquet file contains no numeric columns"
    );
    let mask = ProjectionMask::roots(builder.parquet_schema(), selected.iter().copied());
    let reader = builder
        .with_projection(mask)
        .with_batch_size(65_536)
        .build()
        .map_err(|error| vortex_err!("cannot build Parquet reader: {error}"))?;
    let mut chunks = (0..selected.len())
        .map(|_| Vec::<ArrowArrayRef>::new())
        .collect::<Vec<_>>();
    let mut rows_read = 0usize;
    for batch in reader {
        let batch = batch.map_err(|error| vortex_err!("cannot read Parquet batch: {error}"))?;
        let batch_len = batch.num_rows().min(row_count - rows_read);
        for (column_chunks, array) in chunks.iter_mut().zip(batch.columns()) {
            column_chunks.push(array.slice(0, batch_len));
        }
        rows_read += batch_len;
        if rows_read == row_count {
            break;
        }
    }
    vortex_ensure!(rows_read > 0, "Parquet file contains no rows");

    selected
        .into_iter()
        .zip(chunks)
        .map(|(field_index, chunks)| {
            let field = &schema.fields()[field_index];
            let chunk_refs = chunks
                .iter()
                .map(|chunk| chunk.as_ref())
                .collect::<Vec<_>>();
            let combined = concat(&chunk_refs)
                .map_err(|error| vortex_err!("cannot concatenate {}: {error}", field.name()))?;
            let array = session.arrow().from_arrow_array(combined, field.as_ref())?;
            let primitive = array.execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
            Ok(column(field.name(), primitive))
        })
        .collect()
}

fn encoding_tree(array: &ArrayRef) -> String {
    let children = array.children();
    if children.is_empty() {
        return array.encoding_id().to_string();
    }
    let children = children
        .iter()
        .map(encoding_tree)
        .collect::<Vec<_>>()
        .join(",");
    format!("{}({children})", array.encoding_id())
}

fn encode_all(
    compressor: &BtrBlocksCompressor,
    columns: &[Column],
    session: &VortexSession,
) -> VortexResult<Vec<ArrayRef>> {
    columns
        .iter()
        .map(|column| compressor.compress(&column.array, &mut session.create_execution_ctx()))
        .collect()
}

fn decode_all(arrays: &[ArrayRef], session: &VortexSession) -> VortexResult<()> {
    for array in arrays {
        black_box(
            array
                .clone()
                .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?,
        );
    }
    Ok(())
}

fn percentile(durations: &mut [Duration], numerator: usize, denominator: usize) -> Duration {
    durations.sort_unstable();
    durations[durations.len() * numerator / denominator]
}

fn measure_dataset(
    dataset: &str,
    columns: &[Column],
    configs: &[(&str, BtrBlocksCompressor)],
    session: &VortexSession,
) -> VortexResult<()> {
    let input_bytes = columns
        .iter()
        .map(|column| column.primitive.nbytes())
        .sum::<u64>();
    let encoded = configs
        .iter()
        .map(|(name, compressor)| Ok((*name, encode_all(compressor, columns, session)?)))
        .collect::<VortexResult<Vec<_>>>()?;

    for (config, arrays) in &encoded {
        for (column, array) in columns.iter().zip(arrays) {
            assert_arrays_eq!(array, column.array, &mut session.create_execution_ctx());
            println!(
                "structure\t{dataset}\t{}\t{config}\t{}\t{}",
                column.name,
                encoding_tree(array),
                array.nbytes()
            );
        }
    }

    let encode_iterations = (128_000_000_u64 / input_bytes).clamp(3, 10) as usize;
    let mut encode_durations = (0..configs.len())
        .map(|_| Vec::with_capacity(encode_iterations))
        .collect::<Vec<_>>();
    for iteration in 0..encode_iterations {
        for offset in 0..configs.len() {
            let index = (iteration + offset) % configs.len();
            let start = Instant::now();
            black_box(encode_all(&configs[index].1, columns, session)?);
            encode_durations[index].push(start.elapsed());
        }
    }

    let decode_iterations = (512_000_000_u64 / input_bytes).clamp(5, 30) as usize;
    let mut decode_durations = (0..configs.len())
        .map(|_| Vec::with_capacity(decode_iterations))
        .collect::<Vec<_>>();
    for iteration in 0..decode_iterations {
        for offset in 0..configs.len() {
            let index = (iteration + offset) % configs.len();
            let start = Instant::now();
            black_box(decode_all(&encoded[index].1, session)?);
            decode_durations[index].push(start.elapsed());
        }
    }

    for (index, (config, arrays)) in encoded.iter().enumerate() {
        let encoded_bytes = arrays.iter().map(ArrayRef::nbytes).sum::<u64>();
        let encode_median = percentile(&mut encode_durations[index], 1, 2);
        let decode_median = percentile(&mut decode_durations[index], 1, 2);
        let encode_throughput = input_bytes as f64 / encode_median.as_secs_f64() / 1_000_000.0;
        let decode_throughput = input_bytes as f64 / decode_median.as_secs_f64() / 1_000_000.0;
        println!(
            "result\t{dataset}\t{config}\t{}\t{input_bytes}\t{encoded_bytes}\t{encode_throughput:.1}\t{decode_throughput:.1}",
            columns[0].primitive.len()
        );
    }
    Ok(())
}

fn compressors() -> Vec<(&'static str, BtrBlocksCompressor)> {
    let new_scheme_ids = [FloatQuantScheme.id(), OrderedBlockResidualScheme.id()];
    vec![
        (
            "prior-default",
            BtrBlocksCompressorBuilder::default()
                .exclude_schemes(new_scheme_ids)
                .build(),
        ),
        (
            "float-quant-only",
            BtrBlocksCompressorBuilder::default()
                .exclude_schemes([OrderedBlockResidualScheme.id()])
                .build(),
        ),
        (
            "ordered-block-residual-only",
            BtrBlocksCompressorBuilder::default()
                .exclude_schemes([FloatQuantScheme.id()])
                .build(),
        ),
        (
            "proposed-default",
            BtrBlocksCompressorBuilder::default().build(),
        ),
        (
            "prior-compact",
            BtrBlocksCompressorBuilder::default()
                .exclude_schemes(new_scheme_ids)
                .with_compact()
                .build(),
        ),
        (
            "compact",
            BtrBlocksCompressorBuilder::default().with_compact().build(),
        ),
    ]
}

fn main() -> VortexResult<()> {
    let row_count = std::env::var("VORTEX_BENCH_ROWS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| vortex_err!("invalid VORTEX_BENCH_ROWS: {error}"))
        })
        .transpose()?
        .unwrap_or(DEFAULT_ROW_COUNT);
    let session = array_session();
    let configs = compressors();

    println!("structure\tdataset\tcolumn\tconfig\tencoding\tbytes");
    println!("result\tdataset\tconfig\trows\tinput-bytes\tencoded-bytes\tencode-MB/s\tdecode-MB/s");
    if std::env::var_os("VORTEX_BENCH_SKIP_SYNTHETIC").is_none() {
        for (dataset, columns) in synthetic_datasets(row_count) {
            measure_dataset(&dataset, &columns, &configs, &session)?;
        }
    }
    for argument in std::env::args().skip(1) {
        let path = Path::new(&argument);
        let dataset = path
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or_else(|| vortex_err!("data path has no valid file name"))?;
        let columns = if path
            .extension()
            .is_some_and(|extension| extension == "parquet")
        {
            read_parquet_numeric(path, row_count, &session)?
        } else {
            read_california(path, row_count)?
        };
        measure_dataset(dataset, &columns, &configs, &session)?;
    }
    Ok(())
}
