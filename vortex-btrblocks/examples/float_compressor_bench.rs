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
use pco::ChunkConfig;
use pco::data_types::Number;
use pco::metadata::ChunkLatentVarMeta;
use pco::metadata::DeltaEncoding;
use pco::metadata::DynBins;
use pco::metadata::Mode;
use pco::wrapped::FileCompressor;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::ALPArraySlotsExt;
use vortex_alp::ALPFloat;
use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_block_residual::BlockResidual;
use vortex_block_residual::BlockResidualArraySlotsExt;
use vortex_block_residual::OrderedFloat;
use vortex_block_residual::OrderedFloatArraySlotsExt;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::float::FloatQuantScheme;
use vortex_btrblocks::schemes::float::OrderedBlockResidualScheme;
use vortex_btrblocks::schemes::integer::BlockResidualScheme;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

#[path = "support/range_packed.rs"]
mod range_packed;

use range_packed::RangePacked;
use range_packed::RangePackedCodec;

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
    primitive: Option<PrimitiveArray>,
    array: ArrayRef,
    input_bytes: u64,
    dtype_label: String,
}

fn column(name: impl Into<String>, primitive: PrimitiveArray) -> Column {
    let array = primitive.clone().into_array();
    let input_bytes = array.nbytes();
    let dtype_label = primitive.ptype().to_string();
    Column {
        name: name.into(),
        primitive: Some(primitive),
        array,
        input_bytes,
        dtype_label,
    }
}

fn array_column(name: impl Into<String>, array: ArrayRef) -> Column {
    Column {
        name: name.into(),
        primitive: None,
        input_bytes: array.nbytes(),
        dtype_label: array.dtype().to_string(),
        array,
    }
}

fn synthetic_datasets(row_count: usize) -> VortexResult<Vec<(String, Vec<Column>)>> {
    let mut widened_rng = StdRng::seed_from_u64(1);
    let widened_f32 = PrimitiveArray::from_iter((0..row_count).map(|index| {
        let trend = (index % 10_000) as f32 * 0.001;
        f64::from(trend + widened_rng.random_range(-1.0_f32..1.0))
    }));
    let nonzero_secondary =
        PrimitiveArray::from_iter(widened_f32.as_slice::<f64>().iter().enumerate().map(
            |(index, value)| {
                if index % 10 == 0 {
                    f64::from_bits(value.to_bits() | 1)
                } else {
                    *value
                }
            },
        ));
    let quantized_f32 = PrimitiveArray::from_iter((0_u32..).take(row_count).map(|index| {
        let mantissa = (index.wrapping_mul(7_919) & 0x7fff) << 8;
        f32::from_bits(0x3f80_0000 | mantissa)
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
    let block_local_integer_valued = PrimitiveArray::from_iter((0..row_count).map(|index| {
        let block = index / 1_024;
        let residual = index.wrapping_mul(2_654_435_761) % 1_024;
        (block * 1_000_000 + residual) as f64
    }));
    let block_local_ordered_float = PrimitiveArray::from_iter((0..row_count).map(|index| {
        let block = index / 1_024;
        let residual = index.wrapping_mul(2_654_435_761) % 1_024;
        let bits = 0x3ff0_0000_0000_0000_u64
            + u64::try_from(block).unwrap_or(u64::MAX) * 0x1_0000
            + u64::try_from(residual).unwrap_or(u64::MAX);
        f64::from_bits(bits)
    }));

    let patch_density_4 = PrimitiveArray::from_iter((0..row_count).map(|index| {
        if index % 4 == 0 {
            u32::MAX - u32::try_from(index).unwrap_or(u32::MAX)
        } else {
            42
        }
    }));
    let patch_density_1 = PrimitiveArray::from_iter(
        (0..row_count).map(|index| u32::MAX - u32::try_from(index).unwrap_or(u32::MAX)),
    );
    let block_local_i16 = PrimitiveArray::from_iter((0..row_count).map(|index| {
        let block = (index / 1_024) % 128;
        let residual = index.wrapping_mul(2_654_435_761) % 128;
        i16::try_from(block * 128 + residual).unwrap_or(i16::MAX)
    }));
    let sparse_block_local = PrimitiveArray::from_iter((0..row_count).map(|index| {
        if index % 16 == 0 {
            let value_index = index / 16;
            let block = value_index / 1_024;
            let residual = value_index.wrapping_mul(2_654_435_761) % 1_024;
            u64::try_from(block).unwrap_or(u64::MAX) * 1_000_000_000_000
                + u64::try_from(residual).unwrap_or(u64::MAX)
        } else {
            42
        }
    }));
    let runend_block_local = PrimitiveArray::from_iter((0..row_count).map(|index| {
        let value_index = index / 16;
        let block = value_index / 1_024;
        let residual = value_index.wrapping_mul(2_654_435_761) % 1_024;
        u64::try_from(block).unwrap_or(u64::MAX) * 1_000_000_000_000
            + u64::try_from(residual).unwrap_or(u64::MAX)
    }));

    let mut temporal_rng = StdRng::seed_from_u64(4);
    let mut timestamp = 1_700_000_000_000_000_i64;
    let timestamp_values = PrimitiveArray::from_iter((0..row_count).map(|_| {
        timestamp += temporal_rng.random_range(1_000_i64..1_000_000);
        timestamp
    }));
    let temporal = TemporalArray::new_timestamp(
        timestamp_values.into_array(),
        TimeUnit::Microseconds,
        Some(Arc::from("UTC")),
    )
    .into_array();

    let list_elements = PrimitiveArray::from_iter((0..row_count).map(|index| {
        let block = index / 1_024;
        let residual = index.wrapping_mul(2_654_435_761) % 1_024;
        u64::try_from(block).unwrap_or(u64::MAX) * 1_000_000_000_000
            + u64::try_from(residual).unwrap_or(u64::MAX)
    }));
    let mut list_offsets = Vec::with_capacity(row_count.div_ceil(8) + 1);
    list_offsets.push(0_u32);
    let mut list_offset = 0usize;
    while list_offset < row_count {
        list_offset = (list_offset + 8).min(row_count);
        list_offsets.push(u32::try_from(list_offset).unwrap_or(u32::MAX));
    }
    let list = ListArray::try_new(
        list_elements.into_array(),
        PrimitiveArray::from_iter(list_offsets).into_array(),
        Validity::NonNullable,
    )?
    .into_array();

    let string_count = row_count.min(500_000);
    let strings = (0..string_count)
        .map(|index| {
            format!(
                "user{:06}@example{}.com",
                index.wrapping_mul(2_654_435_761) % 1_000_000,
                index % 100
            )
        })
        .collect::<Vec<_>>();
    let fsst_strings =
        VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array();

    Ok(vec![
        (
            "synthetic-widened-f32".to_string(),
            vec![column("value", widened_f32)],
        ),
        (
            "synthetic-nonzero-secondary".to_string(),
            vec![column("value", nonzero_secondary)],
        ),
        (
            "synthetic-quantized-f32".to_string(),
            vec![column("value", quantized_f32)],
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
        (
            "synthetic-alp-block-local".to_string(),
            vec![column("value", block_local_integer_valued)],
        ),
        (
            "synthetic-ordered-float-block-local".to_string(),
            vec![column("value", block_local_ordered_float)],
        ),
        (
            "synthetic-patch-density-4".to_string(),
            vec![column("value", patch_density_4)],
        ),
        (
            "synthetic-patch-density-1".to_string(),
            vec![column("value", patch_density_1)],
        ),
        (
            "synthetic-block-local-i16".to_string(),
            vec![column("value", block_local_i16)],
        ),
        (
            "synthetic-sparse-block-local".to_string(),
            vec![column("value", sparse_block_local)],
        ),
        (
            "synthetic-runend-block-local".to_string(),
            vec![column("value", runend_block_local)],
        ),
        (
            "synthetic-temporal-parent".to_string(),
            vec![array_column("value", temporal)],
        ),
        (
            "synthetic-list-parent".to_string(),
            vec![array_column("value", list)],
        ),
        (
            "synthetic-fsst-parent".to_string(),
            vec![array_column("value", fsst_strings)],
        ),
    ])
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
    let repeat_short_input = std::env::var_os("VORTEX_BENCH_REPEAT_SHORT").is_some();
    for values in &mut values {
        vortex_ensure!(!values.is_empty(), "California Housing input is empty");
        values.truncate(row_count);
        if repeat_short_input {
            while values.len() < row_count {
                let copy_len = (row_count - values.len()).min(values.len());
                values.extend_from_within(..copy_len);
            }
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
                    | DataType::Timestamp(_, _)
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
            let array = if matches!(field.data_type(), DataType::Timestamp(_, _)) {
                array.cast(DType::Primitive(PType::I64, array.dtype().nullability()))?
            } else {
                array
            };
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

fn profile_block_residual_array(
    dataset: &str,
    column: &str,
    config: &str,
    path: &str,
    array: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<()> {
    if let Some(residuals) = array.as_typed::<BlockResidual>() {
        let mut ctx = session.create_execution_ctx();
        let residual_widths = residuals
            .residual_widths()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let high_widths = residuals
            .high_widths()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let blocks = residual_widths.len();
        let average_residual_width = residual_widths
            .as_slice::<u8>()
            .iter()
            .map(|&width| f64::from(width))
            .sum::<f64>()
            / blocks as f64;
        let average_high_width = high_widths
            .as_slice::<u8>()
            .iter()
            .map(|&width| f64::from(width))
            .sum::<f64>()
            / blocks as f64;
        let patch_starts = residuals
            .patch_starts()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let patch_starts = patch_starts.as_slice::<u32>();
        let mut maximum_patch_density = 0.0_f64;
        let mut blocks_above_one_eighth = 0usize;
        let mut blocks_at_one_quarter = 0usize;
        for (block_index, starts) in patch_starts.windows(2).enumerate() {
            let patch_count = usize::try_from(starts[1] - starts[0])?;
            let block_start = block_index * 1_024;
            let block_len = (array.len() - block_start).min(1_024);
            maximum_patch_density =
                maximum_patch_density.max(patch_count as f64 / block_len as f64);
            blocks_above_one_eighth += usize::from(patch_count * 8 > block_len);
            blocks_at_one_quarter += usize::from(patch_count * 4 >= block_len);
        }
        println!(
            "block-residual-profile\t{dataset}\t{column}\t{config}\t{path}\t{}\t{}\t{}\t{blocks}\t{average_residual_width:.3}\t{average_high_width:.3}\t{maximum_patch_density:.3}\t{blocks_above_one_eighth}\t{blocks_at_one_quarter}\t{}",
            array.dtype().as_ptype(),
            array.len(),
            residuals.patch_positions().len(),
            array.nbytes(),
        );
    }
    for (child_index, child) in array.children().iter().enumerate() {
        profile_block_residual_array(
            dataset,
            column,
            config,
            &format!("{path}/{child_index}"),
            child,
            session,
        )?;
    }
    Ok(())
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
                .execute::<RecursiveCanonical>(&mut session.create_execution_ctx())?,
        );
    }
    Ok(())
}

fn percentile(durations: &mut [Duration], numerator: usize, denominator: usize) -> Duration {
    durations.sort_unstable();
    durations[durations.len() * numerator / denominator]
}

struct BinProfile {
    count: usize,
    ans_size_log: u32,
    max_offset_bits: u32,
    average_bits: f64,
}

fn bin_profile(meta: &ChunkLatentVarMeta) -> BinProfile {
    pco::match_latent_enum!(&meta.bins, DynBins<L>(bins) => {
        let total_weight = (1_u64 << meta.ans_size_log) as f64;
        let average_bits = bins
            .iter()
            .map(|bin| {
                let ans_bits = f64::from(meta.ans_size_log) - f64::from(bin.weight).log2();
                (ans_bits + f64::from(bin.offset_bits)) * f64::from(bin.weight) / total_weight
            })
            .sum();
        BinProfile {
            count: bins.len(),
            ans_size_log: meta.ans_size_log,
            max_offset_bits: bins
                .iter()
                .map(|bin| bin.offset_bits)
                .max()
                .unwrap_or_default(),
            average_bits,
        }
    })
}

fn optional_bin_profile(meta: Option<&ChunkLatentVarMeta>) -> BinProfile {
    meta.map(bin_profile).unwrap_or(BinProfile {
        count: 0,
        ans_size_log: 0,
        max_offset_bits: 0,
        average_bits: 0.0,
    })
}

fn mode_name(mode: &Mode) -> String {
    match mode {
        Mode::Classic => "classic".to_string(),
        Mode::IntMult(_) => "int-mult".to_string(),
        Mode::FloatMult(_) => "float-mult".to_string(),
        Mode::FloatQuant(k) => format!("float-quant-{k}"),
        Mode::Dict(_) => "dict".to_string(),
        _ => "unknown".to_string(),
    }
}

fn delta_name(delta: &DeltaEncoding) -> String {
    match delta {
        DeltaEncoding::NoOp => "none".to_string(),
        DeltaEncoding::Consecutive {
            order,
            secondary_uses_delta,
        } => format!("consecutive-{order}-secondary-{secondary_uses_delta}"),
        DeltaEncoding::Lookback {
            config,
            secondary_uses_delta,
        } => format!(
            "lookback-state-{}-window-{}-secondary-{secondary_uses_delta}",
            config.state_n_log, config.window_n_log
        ),
        DeltaEncoding::Conv1(config) => format!("conv1-quantization-{}", config.quantization),
        _ => "unknown".to_string(),
    }
}

fn profile_pco_values<T: Number>(
    dataset: &str,
    column: &str,
    path: &str,
    ptype: PType,
    values: &[T],
) -> VortexResult<()> {
    let file_compressor = FileCompressor::default();
    for (chunk_index, chunk) in values.chunks(pco::DEFAULT_MAX_PAGE_N).enumerate() {
        let compressor = file_compressor
            .chunk_compressor(chunk, &ChunkConfig::default())
            .map_err(|error| vortex_err!("cannot profile Pco chunk: {error}"))?;
        let meta = compressor.meta();
        let delta = optional_bin_profile(meta.per_latent_var.delta.as_ref());
        let primary = bin_profile(&meta.per_latent_var.primary);
        let secondary = optional_bin_profile(meta.per_latent_var.secondary.as_ref());
        println!(
            "pco-profile\t{dataset}\t{column}\t{path}\t{ptype}\t{chunk_index}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{:.3}",
            chunk.len(),
            mode_name(&meta.mode),
            delta_name(&meta.delta_encoding),
            delta.count,
            delta.max_offset_bits,
            delta.average_bits,
            primary.count,
            primary.ans_size_log,
            primary.max_offset_bits,
            primary.average_bits,
            secondary.count,
            secondary.ans_size_log,
            secondary.max_offset_bits,
            secondary.average_bits,
        );
    }
    Ok(())
}

fn reference_id_bits(reference_count: usize) -> usize {
    match reference_count {
        0 | 1 => 0,
        2 => 1,
        _ => 2,
    }
}

fn estimate_multi_reference_block(
    values: &[u64],
    requested_references: usize,
    bits: usize,
) -> usize {
    const BLOCK_LEN: usize = 1024;
    const METADATA_BYTES: usize = 12;
    const HIGH_PADDING_BYTES: usize = 15;

    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mut references = (0..requested_references)
        .map(|index| sorted[index * sorted.len() / requested_references])
        .collect::<Vec<_>>();
    references.dedup();

    let mut width_counts = [0_usize; 65];
    let mut maximum_width = 0_usize;
    for &value in values {
        let reference_index = references
            .partition_point(|&reference| reference <= value)
            .saturating_sub(1);
        let residual = value - references[reference_index];
        let width = u64::BITS as usize - residual.leading_zeros() as usize;
        width_counts[width] += 1;
        maximum_width = maximum_width.max(width);
    }

    let mut patch_count = values.len();
    let mut best_bits = usize::MAX;
    for residual_width in 0..=maximum_width {
        patch_count -= width_counts[residual_width];
        let high_width = if patch_count == 0 {
            0
        } else {
            maximum_width - residual_width
        };
        let cost_bits = residual_width * BLOCK_LEN
            + reference_id_bits(references.len()) * BLOCK_LEN
            + references.len() * bits
            + patch_count * (u16::BITS as usize + high_width)
            + METADATA_BYTES * 8
            + usize::from(patch_count > 0) * HIGH_PADDING_BYTES * 8;
        best_bits = best_bits.min(cost_bits);
    }
    best_bits.div_ceil(8)
}

fn estimate_multi_reference(values: &[u64], requested_references: usize, bits: usize) -> usize {
    values
        .chunks(1024)
        .map(|block| estimate_multi_reference_block(block, requested_references, bits))
        .sum()
}

fn estimate_bitmap_patches(values: &[u64], bits: usize) -> usize {
    const BLOCK_LEN: usize = 1_024;
    const METADATA_BYTES: usize = 12;
    const HIGH_PADDING_BYTES: usize = 15;

    values
        .chunks(BLOCK_LEN)
        .map(|block| {
            let base = block.iter().copied().min().unwrap_or_default();
            let mut width_counts = [0_usize; 65];
            let mut maximum_width = 0_usize;
            for value in block {
                let residual = value - base;
                let width = u64::BITS as usize - residual.leading_zeros() as usize;
                width_counts[width] += 1;
                maximum_width = maximum_width.max(width);
            }

            let mut patch_count = block.len();
            let mut best_bits = usize::MAX;
            for residual_width in 0..=maximum_width {
                patch_count -= width_counts[residual_width];
                let high_width = if patch_count == 0 {
                    0
                } else {
                    maximum_width - residual_width
                };
                let cost_bits = residual_width * BLOCK_LEN
                    + bits
                    + usize::from(patch_count > 0) * BLOCK_LEN
                    + patch_count * high_width
                    + METADATA_BYTES * 8
                    + usize::from(patch_count > 0) * HIGH_PADDING_BYTES * 8;
                best_bits = best_bits.min(cost_bits);
            }
            best_bits.div_ceil(8)
        })
        .sum()
}

fn estimate_mode_bitmap(values: &[u64], bits: usize) -> usize {
    const BLOCK_LEN: usize = 1_024;
    const METADATA_BYTES: usize = 8;
    const VALUE_PADDING_BYTES: usize = 15;

    values
        .chunks(BLOCK_LEN)
        .map(|block| {
            let mut counts = HashMap::<u64, usize>::new();
            for value in block {
                *counts.entry(*value).or_default() += 1;
            }
            let mode = counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(value, _)| value)
                .unwrap_or_default();
            let exceptions = block.iter().filter(|&&value| value != mode).count();
            let exception_width = block
                .iter()
                .filter(|&&value| value != mode)
                .map(|&value| u64::BITS as usize - value.leading_zeros() as usize)
                .max()
                .unwrap_or_default();
            let cost_bits = bits
                + BLOCK_LEN
                + exceptions * exception_width
                + METADATA_BYTES * 8
                + usize::from(exceptions > 0) * VALUE_PADDING_BYTES * 8;
            cost_bits.div_ceil(8)
        })
        .sum()
}

fn ordered_f32(value: f32) -> u64 {
    let bits = value.to_bits();
    u64::from(if bits & (1_u32 << 31) == 0 {
        bits ^ (1_u32 << 31)
    } else {
        !bits
    })
}

fn ordered_f64(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn ordered_values(values: &PrimitiveArray) -> Vec<u64> {
    match values.ptype() {
        PType::F32 => values
            .as_slice::<f32>()
            .iter()
            .copied()
            .map(ordered_f32)
            .collect(),
        PType::F64 => values
            .as_slice::<f64>()
            .iter()
            .copied()
            .map(ordered_f64)
            .collect(),
        PType::I16 => values
            .as_slice::<i16>()
            .iter()
            .map(|&value| u64::from((value as u16) ^ (1_u16 << 15)))
            .collect(),
        PType::I32 => values
            .as_slice::<i32>()
            .iter()
            .map(|&value| u64::from((value as u32) ^ (1_u32 << 31)))
            .collect(),
        PType::I64 => values
            .as_slice::<i64>()
            .iter()
            .map(|&value| (value as u64) ^ (1_u64 << 63))
            .collect(),
        PType::U16 => values
            .as_slice::<u16>()
            .iter()
            .map(|&value| u64::from(value))
            .collect(),
        PType::U32 => values
            .as_slice::<u32>()
            .iter()
            .map(|&value| u64::from(value))
            .collect(),
        PType::U64 => values.as_slice::<u64>().to_vec(),
        ptype => unreachable!("Pco does not support {ptype}"),
    }
}

fn profile_quotient_remainder(
    dataset: &str,
    column: &str,
    path: &str,
    ptype: PType,
    pco_bytes: u64,
    ordered: &[u64],
    session: &VortexSession,
) -> VortexResult<()> {
    const BASES: [u64; 9] = [2, 4, 5, 8, 10, 16, 32, 100, 1_000];

    let compressor = BtrBlocksCompressor::default();
    for base in BASES {
        let mut remainder_counts = HashMap::<u64, usize>::new();
        for value in ordered {
            *remainder_counts.entry(value % base).or_default() += 1;
        }
        let remainder_entropy = remainder_counts
            .values()
            .map(|&count| {
                let probability = count as f64 / ordered.len() as f64;
                -probability * probability.log2()
            })
            .sum::<f64>();
        let most_common_remainder_share =
            remainder_counts.values().copied().max().unwrap_or_default() as f64
                / ordered.len() as f64;
        let quotients = PrimitiveArray::from_iter(ordered.iter().map(|value| value / base));
        let remainders = PrimitiveArray::from_iter(ordered.iter().map(|value| value % base));
        let quotient_bitmap_bytes =
            estimate_bitmap_patches(quotients.as_slice::<u64>(), ptype.bit_width());
        let remainder_mode_bitmap_bytes =
            estimate_mode_bitmap(remainders.as_slice::<u64>(), ptype.bit_width());
        let quotient =
            compressor.compress(&quotients.into_array(), &mut session.create_execution_ctx())?;
        let remainder = compressor.compress(
            &remainders.into_array(),
            &mut session.create_execution_ctx(),
        )?;
        println!(
            "quotient-remainder-estimate\t{dataset}\t{column}\t{path}\t{ptype}\t{base}\t{pco_bytes}\t{remainder_entropy:.3}\t{most_common_remainder_share:.3}\t{}\t{}\t{}\t{quotient_bitmap_bytes}\t{remainder_mode_bitmap_bytes}\t{}\t{}\t{}",
            quotient.nbytes(),
            remainder.nbytes(),
            quotient.nbytes() + remainder.nbytes(),
            quotient_bitmap_bytes + remainder_mode_bitmap_bytes,
            encoding_tree(&quotient),
            encoding_tree(&remainder),
        );
    }
    Ok(())
}

fn profile_range_packed(
    dataset: &str,
    column: &str,
    path: &str,
    logical_bytes: usize,
    pco_bytes: u64,
    validity_bytes: u64,
    ordered: &[u64],
) -> VortexResult<()> {
    if ordered.is_empty() {
        return Ok(());
    }

    let mut encode_durations = Vec::with_capacity(5);
    let mut codec = RangePackedCodec::encode(ordered, 1_024)?;
    for _ in 0..5 {
        let start = Instant::now();
        codec = RangePackedCodec::encode(black_box(ordered), 1_024)?;
        encode_durations.push(start.elapsed());
    }
    let encode_median = percentile(&mut encode_durations, 1, 2);

    vortex_ensure!(codec.decode()? == ordered, "range packed decode differs");
    let mut decode_durations = Vec::with_capacity(20);
    for _ in 0..20 {
        let start = Instant::now();
        black_box(codec.decode()?);
        decode_durations.push(start.elapsed());
    }
    let decode_median = percentile(&mut decode_durations, 1, 2);

    let scalar_iterations = 200_000;
    let mut scalar_index = 0usize;
    let mut scalar_checksum = 0u64;
    let scalar_start = Instant::now();
    for _ in 0..scalar_iterations {
        scalar_index = scalar_index.wrapping_add(2_654_435_761) % ordered.len();
        scalar_checksum ^= black_box(codec.scalar_at(scalar_index)?);
    }
    black_box(scalar_checksum);
    let scalar_duration = scalar_start.elapsed();
    let input_bytes = ordered.len() * logical_bytes;
    let encode_throughput = input_bytes as f64 / encode_median.as_secs_f64() / 1_000_000.0;
    let decode_throughput = input_bytes as f64 / decode_median.as_secs_f64() / 1_000_000.0;
    let scalar_nanoseconds =
        scalar_duration.as_secs_f64() * 1_000_000_000.0 / scalar_iterations as f64;
    let encoded_bytes = u64::try_from(codec.encoded_size())? + validity_bytes;
    println!(
        "fixed-bin-checkpoint\t{dataset}\t{column}\t{path}\t{}\t{pco_bytes}\t{encoded_bytes}\t{}\t{}\t{}\t{encode_throughput:.1}\t{decode_throughput:.1}\t{scalar_nanoseconds:.1}",
        ordered.len(),
        codec.bin_count(),
        codec.max_offset_bits(),
        codec.offset_widths(),
    );
    Ok(())
}

fn fixed_bin_float_tree(
    compact: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<Option<ArrayRef>> {
    if compact.encoding_id().as_ref() == "vortex.alp" {
        let alp = compact.as_::<ALP>();
        if alp.encoded().encoding_id().as_ref() != "vortex.pco" {
            return Ok(None);
        }
        let primitive = alp
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
        let primitive = primitive
            .into_array()
            .cast(alp.encoded().dtype().clone())?
            .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
        let encoded =
            RangePacked::from_primitive(primitive.as_view(), &mut session.create_execution_ctx())?
                .into_array();
        return Ok(Some(
            ALP::try_new(encoded, alp.exponents(), alp.patches())?.into_array(),
        ));
    }

    if compact.encoding_id().as_ref() != "vortex.pco" || !compact.dtype().is_float() {
        return Ok(None);
    }
    let primitive = compact
        .clone()
        .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
    let primitive =
        fill_float_nulls_with_first_valid(primitive, &mut session.create_execution_ctx())?;
    let ordered = OrderedFloat::from_primitive(primitive.as_view())?;
    let encoded = RangePacked::from_primitive(
        ordered.encoded().as_::<Primitive>(),
        &mut session.create_execution_ctx(),
    )?;
    Ok(Some(
        OrderedFloat::try_new(encoded.into_array(), primitive.ptype())?.into_array(),
    ))
}

fn fill_float_nulls_with_first_valid(
    primitive: PrimitiveArray,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let validity = primitive.validity()?;
    let mask = validity.execute_mask(primitive.len(), ctx)?;
    if mask.all_true() {
        return Ok(primitive);
    }
    let first_valid = mask.first();
    Ok(match primitive.ptype() {
        PType::F32 => {
            let values = primitive.as_slice::<f32>();
            let fill = first_valid.map_or(0.0, |index| values[index]);
            PrimitiveArray::new::<f32>(
                values
                    .iter()
                    .zip(mask.iter())
                    .map(|(&value, valid)| if valid { value } else { fill })
                    .collect::<Vec<_>>(),
                validity,
            )
        }
        PType::F64 => {
            let values = primitive.as_slice::<f64>();
            let fill = first_valid.map_or(0.0, |index| values[index]);
            PrimitiveArray::new::<f64>(
                values
                    .iter()
                    .zip(mask.iter())
                    .map(|(&value, valid)| if valid { value } else { fill })
                    .collect::<Vec<_>>(),
                validity,
            )
        }
        ptype => return Err(vortex_err!("null fill does not support {ptype}")),
    })
}

fn measure_fixed_bin_tree(
    dataset: &str,
    column: &str,
    expected: &ArrayRef,
    compact: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<()> {
    let Some(encoded) = fixed_bin_float_tree(compact, session)? else {
        return Ok(());
    };
    vortex_ensure!(
        encoded.dtype() == expected.dtype(),
        "fixed-bin tree dtype {} differs from expected {}",
        encoded.dtype(),
        expected.dtype()
    );
    assert_arrays_eq!(encoded, expected, &mut session.create_execution_ctx());

    let mut decode_durations = Vec::with_capacity(20);
    for _ in 0..20 {
        let start = Instant::now();
        black_box(
            encoded
                .clone()
                .execute::<RecursiveCanonical>(&mut session.create_execution_ctx())?,
        );
        decode_durations.push(start.elapsed());
    }
    let decode_median = percentile(&mut decode_durations, 1, 2);
    let decode_throughput = expected.nbytes() as f64 / decode_median.as_secs_f64() / 1_000_000.0;

    let scalar_nanoseconds = measure_scalar_access(&encoded, session)?;
    println!(
        "fixed-bin-tree\t{dataset}\t{column}\t{}\t{}\t{decode_throughput:.1}\t{scalar_nanoseconds:.1}\t{}",
        encoded.len(),
        encoded.nbytes(),
        encoding_tree(&encoded),
    );

    let mut fused_durations = Vec::with_capacity(20);
    for _ in 0..20 {
        let start = Instant::now();
        black_box(decode_fixed_bin_float_tree(&encoded, session)?);
        fused_durations.push(start.elapsed());
    }
    let fused_median = percentile(&mut fused_durations, 1, 2);
    let fused_throughput = expected.nbytes() as f64 / fused_median.as_secs_f64() / 1_000_000.0;
    println!(
        "fixed-bin-tree-fused\t{dataset}\t{column}\t{}\t{}\t{fused_throughput:.1}\t{}",
        encoded.len(),
        encoded.nbytes(),
        encoding_tree(&encoded),
    );

    let mut encode_durations = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        black_box(encode_fixed_bin_float_tree(
            expected.as_::<Primitive>(),
            compact,
            session,
        )?);
        encode_durations.push(start.elapsed());
    }
    let encode_median = percentile(&mut encode_durations, 1, 2);
    let encode_throughput = expected.nbytes() as f64 / encode_median.as_secs_f64() / 1_000_000.0;
    println!(
        "fixed-bin-tree-encode\t{dataset}\t{column}\t{}\t{}\t{encode_throughput:.1}\t{}",
        encoded.len(),
        encoded.nbytes(),
        encoding_tree(&encoded),
    );
    Ok(())
}

fn encode_fixed_bin_float_tree(
    primitive: vortex_array::ArrayView<'_, Primitive>,
    compact: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    if compact.encoding_id().as_ref() == "vortex.alp" {
        let alp = alp_encode(primitive, None, &mut session.create_execution_ctx())?;
        let packed = RangePacked::from_primitive(
            alp.encoded().as_::<Primitive>(),
            &mut session.create_execution_ctx(),
        )?;
        return Ok(ALP::try_new(packed.into_array(), alp.exponents(), alp.patches())?.into_array());
    }

    let primitive = fill_float_nulls_with_first_valid(
        primitive.into_owned(),
        &mut session.create_execution_ctx(),
    )?;
    let ordered = OrderedFloat::from_primitive(primitive.as_view())?;
    let packed = RangePacked::from_primitive(
        ordered.encoded().as_::<Primitive>(),
        &mut session.create_execution_ctx(),
    )?;
    Ok(OrderedFloat::try_new(packed.into_array(), primitive.ptype())?.into_array())
}

fn measure_existing_tree(
    dataset: &str,
    column: &str,
    label: &str,
    expected: &ArrayRef,
    encoded: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<()> {
    assert_arrays_eq!(encoded, expected, &mut session.create_execution_ctx());
    let mut decode_durations = Vec::with_capacity(20);
    for _ in 0..20 {
        let start = Instant::now();
        black_box(
            encoded
                .clone()
                .execute::<RecursiveCanonical>(&mut session.create_execution_ctx())?,
        );
        decode_durations.push(start.elapsed());
    }
    let decode_median = percentile(&mut decode_durations, 1, 2);
    let decode_throughput = expected.nbytes() as f64 / decode_median.as_secs_f64() / 1_000_000.0;
    let scalar_nanoseconds = measure_scalar_access(encoded, session)?;
    println!(
        "tree-throughput\t{dataset}\t{column}\t{label}\t{}\t{}\t{decode_throughput:.1}\t{scalar_nanoseconds:.1}\t{}",
        encoded.len(),
        encoded.nbytes(),
        encoding_tree(encoded),
    );
    Ok(())
}

fn measure_scalar_access(encoded: &ArrayRef, session: &VortexSession) -> VortexResult<f64> {
    let scalar_iterations = 200_000;
    let mut scalar_index = 0usize;
    let mut scalar_checksum = 0u64;
    let mut scalar_ctx = session.create_execution_ctx();
    let scalar_start = Instant::now();
    for _ in 0..scalar_iterations {
        scalar_index = scalar_index.wrapping_add(2_654_435_761) % encoded.len();
        let scalar = encoded.execute_scalar(scalar_index, &mut scalar_ctx)?;
        let bits = if scalar.is_null() {
            u64::MAX
        } else {
            match encoded.dtype().as_ptype() {
                PType::F32 => u64::from(
                    scalar
                        .as_primitive()
                        .typed_value::<f32>()
                        .ok_or_else(|| vortex_err!("tree produced an invalid f32 scalar"))?
                        .to_bits(),
                ),
                PType::F64 => scalar
                    .as_primitive()
                    .typed_value::<f64>()
                    .ok_or_else(|| vortex_err!("tree produced an invalid f64 scalar"))?
                    .to_bits(),
                ptype => {
                    return Err(vortex_err!(
                        "tree scalar benchmark does not support {ptype}"
                    ));
                }
            }
        };
        scalar_checksum ^= black_box(bits);
    }
    black_box(scalar_checksum);
    Ok(scalar_start.elapsed().as_secs_f64() * 1_000_000_000.0 / scalar_iterations as f64)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the range-packed child retains the ALP integer word width"
)]
fn decode_fixed_bin_float_tree(
    encoded: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<PrimitiveArray> {
    if encoded.encoding_id().as_ref() == "vortex.ordered_float" {
        let ordered = encoded.as_::<OrderedFloat>();
        let packed = ordered.encoded().as_::<RangePacked>();
        let validity = ordered.encoded().validity()?;
        return Ok(match encoded.dtype().as_ptype() {
            PType::F32 => {
                let values = RangePacked::decode_mapped(
                    packed,
                    |ordered| f32::from_bits(unordered_u32(ordered as u32)),
                    0.0,
                )?;
                PrimitiveArray::new::<f32>(values, validity)
            }
            PType::F64 => {
                let values = RangePacked::decode_mapped(
                    packed,
                    |ordered| f64::from_bits(unordered_u64(ordered)),
                    0.0,
                )?;
                PrimitiveArray::new::<f64>(values, validity)
            }
            ptype => return Err(vortex_err!("fixed-bin float tree does not support {ptype}")),
        });
    }

    let alp = encoded.as_::<ALP>();
    let packed = alp.encoded().as_::<RangePacked>();
    let validity = alp.encoded().validity()?;
    let exponents = alp.exponents();
    let decoded = match encoded.dtype().as_ptype() {
        PType::F32 => {
            let values = RangePacked::decode_mapped(
                packed,
                |ordered| {
                    let encoded = ((ordered as u32) ^ (1_u32 << 31)) as i32;
                    <f32 as ALPFloat>::decode_single(encoded, exponents)
                },
                0.0,
            )?;
            PrimitiveArray::new::<f32>(values, validity)
        }
        PType::F64 => {
            let values = RangePacked::decode_mapped(
                packed,
                |ordered| {
                    let encoded = (ordered ^ (1_u64 << 63)) as i64;
                    <f64 as ALPFloat>::decode_single(encoded, exponents)
                },
                0.0,
            )?;
            PrimitiveArray::new::<f64>(values, validity)
        }
        ptype => return Err(vortex_err!("fixed-bin ALP tree does not support {ptype}")),
    };
    if let Some(patches) = alp.patches() {
        decoded.patch(&patches, &mut session.create_execution_ctx())
    } else {
        Ok(decoded)
    }
}

fn unordered_u32(value: u32) -> u32 {
    if value & (1_u32 << 31) == 0 {
        !value
    } else {
        value ^ (1_u32 << 31)
    }
}

fn unordered_u64(value: u64) -> u64 {
    if value & (1_u64 << 63) == 0 {
        !value
    } else {
        value ^ (1_u64 << 63)
    }
}

fn profile_pco_array(
    dataset: &str,
    column: &str,
    path: &str,
    array: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<()> {
    if array.encoding_id().as_ref() == "vortex.pco" {
        let mut ctx = session.create_execution_ctx();
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx)?;
        let primitive_array = primitive.clone().into_array();
        let mask = primitive_array
            .validity()?
            .execute_mask(primitive.len(), &mut ctx)?;
        let values = primitive_array
            .filter(mask)?
            .execute::<PrimitiveArray>(&mut ctx)?;
        let ordered = ordered_values(&values);
        let validity_bytes = array.children().iter().map(ArrayRef::nbytes).sum::<u64>();
        if std::env::var_os("VORTEX_BENCH_FIXED_BIN").is_some() {
            profile_range_packed(
                dataset,
                column,
                path,
                values.ptype().byte_width(),
                array.nbytes(),
                validity_bytes,
                &ordered,
            )?;
        }
        profile_quotient_remainder(
            dataset,
            column,
            path,
            values.ptype(),
            array.nbytes(),
            &ordered,
            session,
        )?;
        let bits = values.ptype().bit_width();
        let one_reference =
            u64::try_from(estimate_multi_reference(&ordered, 1, bits))? + validity_bytes;
        let two_references =
            u64::try_from(estimate_multi_reference(&ordered, 2, bits))? + validity_bytes;
        let four_references =
            u64::try_from(estimate_multi_reference(&ordered, 4, bits))? + validity_bytes;
        let bitmap_patches =
            u64::try_from(estimate_bitmap_patches(&ordered, bits))? + validity_bytes;
        println!(
            "multi-ref-estimate\t{dataset}\t{column}\t{path}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            values.ptype(),
            array.nbytes(),
            one_reference,
            two_references,
            four_references,
            one_reference.min(two_references).min(four_references),
            bitmap_patches,
        );
        match values.ptype() {
            PType::F32 => {
                profile_pco_values(dataset, column, path, PType::F32, values.as_slice::<f32>())?
            }
            PType::F64 => {
                profile_pco_values(dataset, column, path, PType::F64, values.as_slice::<f64>())?
            }
            PType::I16 => {
                profile_pco_values(dataset, column, path, PType::I16, values.as_slice::<i16>())?
            }
            PType::I32 => {
                profile_pco_values(dataset, column, path, PType::I32, values.as_slice::<i32>())?
            }
            PType::I64 => {
                profile_pco_values(dataset, column, path, PType::I64, values.as_slice::<i64>())?
            }
            PType::U16 => {
                profile_pco_values(dataset, column, path, PType::U16, values.as_slice::<u16>())?
            }
            PType::U32 => {
                profile_pco_values(dataset, column, path, PType::U32, values.as_slice::<u32>())?
            }
            PType::U64 => {
                profile_pco_values(dataset, column, path, PType::U64, values.as_slice::<u64>())?
            }
            ptype => return Err(vortex_err!("cannot profile Pco ptype {ptype}")),
        }
    }
    for (child_index, child) in array.children().iter().enumerate() {
        profile_pco_array(
            dataset,
            column,
            &format!("{path}/{child_index}"),
            child,
            session,
        )?;
    }
    Ok(())
}

fn measure_dataset(
    dataset: &str,
    columns: &[Column],
    configs: &[(&str, BtrBlocksCompressor)],
    session: &VortexSession,
) -> VortexResult<()> {
    let mut input_bytes = 0_u64;
    for column in columns {
        input_bytes += column.input_bytes;
    }
    let encoded = configs
        .iter()
        .map(|(name, compressor)| Ok((*name, encode_all(compressor, columns, session)?)))
        .collect::<VortexResult<Vec<_>>>()?;

    for (config, arrays) in &encoded {
        for (column, array) in columns.iter().zip(arrays) {
            assert_arrays_eq!(array, column.array, &mut session.create_execution_ctx());
            println!(
                "structure\t{dataset}\t{}\t{}\t{config}\t{}\t{}",
                column.name,
                column.dtype_label,
                encoding_tree(array),
                array.nbytes()
            );
            if matches!(
                *config,
                "integer-block-residual-only" | "ordered-block-residual-only"
            ) {
                profile_block_residual_array(
                    dataset,
                    &column.name,
                    config,
                    "root",
                    array,
                    session,
                )?;
            }
        }
    }

    let prior_default = encoded
        .iter()
        .find(|(config, _)| *config == "prior-default")
        .ok_or_else(|| vortex_err!("prior-default configuration is missing"))?;
    let prior_compact = encoded
        .iter()
        .find(|(config, _)| *config == "prior-compact")
        .ok_or_else(|| vortex_err!("prior-compact configuration is missing"))?;
    for (column_index, column) in columns.iter().enumerate() {
        let Some(primitive) = &column.primitive else {
            continue;
        };
        if !primitive.ptype().is_float() {
            continue;
        }
        let default_array = &prior_default.1[column_index];
        let compact_array = &prior_compact.1[column_index];
        let input_bytes = primitive.len() * primitive.ptype().byte_width();
        let default_bytes = default_array.nbytes();
        let compact_bytes = compact_array.nbytes();
        let compact_savings = if default_bytes == 0 {
            0.0
        } else {
            100.0 * (1.0 - compact_bytes as f64 / default_bytes as f64)
        };
        println!(
            "float-column\t{dataset}\t{}\t{}\t{}\t{input_bytes}\t{default_bytes}\t{compact_bytes}\t{compact_savings:.3}\t{}\t{}",
            column.name,
            primitive.ptype(),
            primitive.len(),
            encoding_tree(default_array),
            encoding_tree(compact_array),
        );
        if compact_bytes * 10 <= default_bytes * 9
            || std::env::var_os("VORTEX_BENCH_FIXED_BIN").is_some()
        {
            profile_pco_array(dataset, &column.name, "root", compact_array, session)?;
        }
        if std::env::var_os("VORTEX_BENCH_FIXED_BIN").is_some() {
            measure_existing_tree(
                dataset,
                &column.name,
                "default",
                &column.array,
                default_array,
                session,
            )?;
            measure_existing_tree(
                dataset,
                &column.name,
                "compact",
                &column.array,
                compact_array,
                session,
            )?;
            measure_fixed_bin_tree(dataset, &column.name, &column.array, compact_array, session)?;
        }
    }

    if std::env::var_os("VORTEX_BENCH_PROFILE_ONLY").is_some() {
        return Ok(());
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
            columns[0].array.len()
        );
    }
    Ok(())
}

fn compressors() -> Vec<(&'static str, BtrBlocksCompressor)> {
    let new_scheme_ids = [
        FloatQuantScheme.id(),
        OrderedBlockResidualScheme.id(),
        BlockResidualScheme.id(),
    ];
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
                .exclude_schemes([OrderedBlockResidualScheme.id(), BlockResidualScheme.id()])
                .build(),
        ),
        (
            "ordered-block-residual-only",
            BtrBlocksCompressorBuilder::default()
                .exclude_schemes([FloatQuantScheme.id(), BlockResidualScheme.id()])
                .build(),
        ),
        (
            "integer-block-residual-only",
            BtrBlocksCompressorBuilder::default()
                .exclude_schemes([FloatQuantScheme.id(), OrderedBlockResidualScheme.id()])
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

    println!("structure\tdataset\tcolumn\tptype\tconfig\tencoding\tbytes");
    println!(
        "float-column\tdataset\tcolumn\tptype\trows\tinput-bytes\tdefault-bytes\tcompact-bytes\tcompact-savings-pct\tdefault-encoding\tcompact-encoding"
    );
    println!(
        "pco-profile\tdataset\tcolumn\tpath\tptype\tchunk\trows\tmode\tdelta\tdelta-bins\tdelta-max-offset-bits\tdelta-average-bits\tprimary-bins\tprimary-ans-log\tprimary-max-offset-bits\tprimary-average-bits\tsecondary-bins\tsecondary-ans-log\tsecondary-max-offset-bits\tsecondary-average-bits"
    );
    println!(
        "multi-ref-estimate\tdataset\tcolumn\tpath\tptype\tpco-child-bytes\tone-reference-bytes\ttwo-reference-bytes\tfour-reference-bytes\tbest-reference-bytes\tbitmap-patch-bytes"
    );
    println!(
        "quotient-remainder-estimate\tdataset\tcolumn\tpath\tptype\tbase\tpco-child-bytes\tremainder-entropy\tmost-common-remainder-share\tquotient-bytes\tremainder-bytes\ttotal-bytes\tquotient-bitmap-bytes\tremainder-mode-bitmap-bytes\tbitmap-total-bytes\tquotient-encoding\tremainder-encoding"
    );
    println!(
        "fixed-bin-checkpoint\tdataset\tcolumn\tpath\trows\tpco-bytes\tfixed-bin-bytes\tbins\tmax-offset-bits\toffset-widths\tencode-MB/s\tdecode-MB/s\tscalar-ns"
    );
    println!("fixed-bin-tree\tdataset\tcolumn\trows\tbytes\tdecode-MB/s\tscalar-ns\tencoding");
    println!("fixed-bin-tree-fused\tdataset\tcolumn\trows\tbytes\tdecode-MB/s\tencoding");
    println!("fixed-bin-tree-encode\tdataset\tcolumn\trows\tbytes\tencode-MB/s\tencoding");
    println!(
        "tree-throughput\tdataset\tcolumn\tconfig\trows\tbytes\tdecode-MB/s\tscalar-ns\tencoding"
    );
    println!("result\tdataset\tconfig\trows\tinput-bytes\tencoded-bytes\tencode-MB/s\tdecode-MB/s");
    println!(
        "block-residual-profile\tdataset\tcolumn\tconfig\tpath\tptype\trows\tpatches\tblocks\taverage-residual-width\taverage-high-width\tmaximum-patch-density\tblocks-above-one-eighth\tblocks-at-one-quarter\tbytes"
    );
    if std::env::var_os("VORTEX_BENCH_SKIP_SYNTHETIC").is_none() {
        let synthetic_filter = std::env::var("VORTEX_BENCH_SYNTHETIC").ok();
        for (dataset, columns) in synthetic_datasets(row_count)? {
            if synthetic_filter
                .as_deref()
                .is_some_and(|filter| filter != dataset)
            {
                continue;
            }
            measure_dataset(&dataset, &columns, &configs, &session)?;
        }
    }
    for argument in std::env::args().skip(1) {
        let path = Path::new(&argument);
        let dataset = path
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or_else(|| vortex_err!("data path has no valid file name"))?;
        let mut columns = if path
            .extension()
            .is_some_and(|extension| extension == "parquet")
        {
            read_parquet_numeric(path, row_count, &session)?
        } else {
            read_california(path, row_count)?
        };
        if let Ok(column_filter) = std::env::var("VORTEX_BENCH_COLUMN") {
            columns.retain(|column| column.name == column_filter);
            vortex_ensure!(
                !columns.is_empty(),
                "dataset does not contain the requested column"
            );
        }
        measure_dataset(dataset, &columns, &configs, &session)?;
    }
    Ok(())
}
