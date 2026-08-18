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
use pco::DeltaSpec;
use pco::ModeSpec;
use pco::data_types::Number;
use pco::metadata::DeltaEncoding;
use pco::metadata::Mode;
use pco::wrapped::FileCompressor;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::PType;
use vortex_arrow::ArrowSessionExt;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::float::FloatMultScheme;
use vortex_btrblocks::schemes::float::FloatQuantScheme;
use vortex_btrblocks::schemes::float::OrderedBlockResidualScheme;
use vortex_btrblocks::schemes::range_entropy::RangeEntropyScheme;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_float_quant::FloatMult;
use vortex_float_quant::FloatMultArrayExt;
use vortex_float_quant::estimate_float_mult_constant_base;
use vortex_pco::Pco;
use vortex_range_entropy::BitSplitCodec;
use vortex_range_entropy::PatchedFoRCodec;
use vortex_range_entropy::RangeEntropyCodec;
use vortex_range_entropy::RangePackedCodec;
use vortex_range_entropy::RangeTwoLevelCodec;
use vortex_session::VortexSession;

const COLUMN_NAMES: [&str; 9] = [
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
const DEFAULT_ROW_LIMIT: usize = 2_000_000;

struct Column {
    name: String,
    primitive: PrimitiveArray,
    array: ArrayRef,
}

#[derive(Clone, Copy)]
struct FloatMultBackendSizes {
    bit_split: usize,
    block_residual: usize,
    range_entropy: usize,
    range_packed: usize,
    range_two_level: usize,
}

struct FloatMultLatents {
    primary: Vec<u64>,
    secondary: Vec<u64>,
}

enum FloatMultPrototype {
    F32 {
        base: f32,
        primary: ArrayRef,
        secondary: ArrayRef,
        backend_sizes: FloatMultBackendSizes,
        latents: FloatMultLatents,
    },
    F64 {
        base: f64,
        primary: ArrayRef,
        secondary: ArrayRef,
        backend_sizes: FloatMultBackendSizes,
        latents: FloatMultLatents,
    },
}

impl FloatMultPrototype {
    fn base(&self) -> f64 {
        match self {
            Self::F32 { base, .. } => f64::from(*base),
            Self::F64 { base, .. } => *base,
        }
    }

    fn encoded_size(&self) -> u64 {
        let (primary, secondary) = match self {
            Self::F32 {
                primary, secondary, ..
            }
            | Self::F64 {
                primary, secondary, ..
            } => (primary, secondary),
        };
        primary.nbytes() + secondary.nbytes() + 16
    }

    fn structure(&self) -> String {
        let (primary, secondary) = match self {
            Self::F32 {
                primary, secondary, ..
            }
            | Self::F64 {
                primary, secondary, ..
            } => (primary, secondary),
        };
        format!(
            "float-mult({},{})",
            encoding_tree(primary),
            encoding_tree(secondary)
        )
    }

    fn backend_sizes(&self) -> FloatMultBackendSizes {
        match self {
            Self::F32 { backend_sizes, .. } | Self::F64 { backend_sizes, .. } => *backend_sizes,
        }
    }

    fn latents(&self) -> &FloatMultLatents {
        match self {
            Self::F32 { latents, .. } | Self::F64 { latents, .. } => latents,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn reconstruct(&self, primary: &[u64], secondary: &[u64]) -> VortexResult<()> {
        vortex_ensure!(
            primary.len() == secondary.len(),
            "FloatMult latent lengths differ"
        );
        match self {
            Self::F32 { base, .. } => {
                let values = primary
                    .iter()
                    .copied()
                    .zip(secondary.iter().copied())
                    .map(|(primary, secondary)| {
                        join_float_mult_f32(primary as u32, secondary as u32, *base)
                    })
                    .collect::<Vec<_>>();
                black_box(values);
            }
            Self::F64 { base, .. } => {
                let values = primary
                    .iter()
                    .copied()
                    .zip(secondary.iter().copied())
                    .map(|(primary, secondary)| join_float_mult_f64(primary, secondary, *base))
                    .collect::<Vec<_>>();
                black_box(values);
            }
        }
        Ok(())
    }

    fn decode(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        match self {
            Self::F32 {
                base,
                primary,
                secondary,
                ..
            } => {
                let primary = primary
                    .clone()
                    .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
                let secondary = secondary
                    .clone()
                    .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
                let validity = primary.as_view().validity()?;
                Ok(PrimitiveArray::new(
                    Buffer::from(
                        primary
                            .as_slice::<u32>()
                            .iter()
                            .copied()
                            .zip(secondary.as_slice::<u32>().iter().copied())
                            .map(|(primary, secondary)| {
                                join_float_mult_f32(primary, secondary, *base)
                            })
                            .collect::<Vec<_>>(),
                    ),
                    validity,
                )
                .into_array())
            }
            Self::F64 {
                base,
                primary,
                secondary,
                ..
            } => {
                let primary = primary
                    .clone()
                    .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
                let secondary = secondary
                    .clone()
                    .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
                let validity = primary.as_view().validity()?;
                Ok(PrimitiveArray::new(
                    Buffer::from(
                        primary
                            .as_slice::<u64>()
                            .iter()
                            .copied()
                            .zip(secondary.as_slice::<u64>().iter().copied())
                            .map(|(primary, secondary)| {
                                join_float_mult_f64(primary, secondary, *base)
                            })
                            .collect::<Vec<_>>(),
                    ),
                    validity,
                )
                .into_array())
            }
        }
    }
}

#[derive(Clone, Copy)]
enum FloatMultLatentBackend {
    BitSplit,
    BlockResidual,
    RangeEntropy,
    RangePacked,
    RangeTwoLevel,
}

impl FloatMultLatentBackend {
    const ALL: [Self; 5] = [
        Self::BitSplit,
        Self::BlockResidual,
        Self::RangeEntropy,
        Self::RangePacked,
        Self::RangeTwoLevel,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::BitSplit => "bit-split",
            Self::BlockResidual => "block-residual",
            Self::RangeEntropy => "range-entropy",
            Self::RangePacked => "range-packed",
            Self::RangeTwoLevel => "range-two-level",
        }
    }

    fn encode(self, latents: &FloatMultLatents) -> VortexResult<FloatMultCodecPair> {
        Ok(match self {
            Self::BitSplit => FloatMultCodecPair::BitSplit(
                BitSplitCodec::encode(&latents.primary)?,
                BitSplitCodec::encode(&latents.secondary)?,
            ),
            Self::BlockResidual => FloatMultCodecPair::BlockResidual(
                PatchedFoRCodec::encode_single_base(&latents.primary)?,
                PatchedFoRCodec::encode_single_base(&latents.secondary)?,
            ),
            Self::RangeEntropy => FloatMultCodecPair::RangeEntropy(
                RangeEntropyCodec::encode(&latents.primary, 8_192)?,
                RangeEntropyCodec::encode(&latents.secondary, 8_192)?,
            ),
            Self::RangePacked => FloatMultCodecPair::RangePacked(
                RangePackedCodec::encode(&latents.primary, 8_192)?,
                RangePackedCodec::encode(&latents.secondary, 8_192)?,
            ),
            Self::RangeTwoLevel => FloatMultCodecPair::RangeTwoLevel(
                RangeTwoLevelCodec::encode(&latents.primary, 8_192)?,
                RangeTwoLevelCodec::encode(&latents.secondary, 8_192)?,
            ),
        })
    }
}

enum FloatMultCodecPair {
    BitSplit(BitSplitCodec, BitSplitCodec),
    BlockResidual(PatchedFoRCodec, PatchedFoRCodec),
    RangeEntropy(RangeEntropyCodec, RangeEntropyCodec),
    RangePacked(RangePackedCodec, RangePackedCodec),
    RangeTwoLevel(RangeTwoLevelCodec, RangeTwoLevelCodec),
}

impl FloatMultCodecPair {
    fn encoded_size(&self) -> usize {
        match self {
            Self::BitSplit(primary, secondary) => {
                primary.encoded_size() + secondary.encoded_size() + 16
            }
            Self::BlockResidual(primary, secondary) => {
                primary.encoded_size() + secondary.encoded_size() + 16
            }
            Self::RangeEntropy(primary, secondary) => {
                primary.encoded_size() + secondary.encoded_size() + 16
            }
            Self::RangePacked(primary, secondary) => {
                primary.encoded_size() + secondary.encoded_size() + 16
            }
            Self::RangeTwoLevel(primary, secondary) => {
                primary.encoded_size() + secondary.encoded_size() + 16
            }
        }
    }

    fn decode(&self) -> VortexResult<(Vec<u64>, Vec<u64>)> {
        match self {
            Self::BitSplit(primary, secondary) => Ok((primary.decode()?, secondary.decode()?)),
            Self::BlockResidual(primary, secondary) => Ok((primary.decode()?, secondary.decode()?)),
            Self::RangeEntropy(primary, secondary) => Ok((primary.decode()?, secondary.decode()?)),
            Self::RangePacked(primary, secondary) => Ok((primary.decode()?, secondary.decode()?)),
            Self::RangeTwoLevel(primary, secondary) => Ok((primary.decode()?, secondary.decode()?)),
        }
    }
}

enum Encoder<'a> {
    BtrBlocks(&'a BtrBlocksCompressor),
    Pco(&'a ChunkConfig),
}

impl Encoder<'_> {
    fn encode(&self, columns: &[Column], session: &VortexSession) -> VortexResult<Vec<ArrayRef>> {
        match self {
            Self::BtrBlocks(compressor) => columns
                .iter()
                .map(|column| {
                    compressor.compress(&column.array, &mut session.create_execution_ctx())
                })
                .collect(),
            Self::Pco(config) => columns
                .iter()
                .map(|column| {
                    Ok(Pco::from_primitive_with_config(
                        column.primitive.as_view(),
                        config,
                        pco::DEFAULT_MAX_PAGE_N,
                        &mut session.create_execution_ctx(),
                    )?
                    .into_array())
                })
                .collect(),
        }
    }
}

fn read_california_housing(path: &Path) -> VortexResult<Vec<Column>> {
    let reader = BufReader::new(File::open(path)?);
    let mut values = std::array::from_fn::<_, 9, _>(|_| Vec::<f32>::new());
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let fields = line.split(',').collect::<Vec<_>>();
        vortex_ensure!(
            fields.len() == COLUMN_NAMES.len(),
            "line {} contains {} fields instead of {}",
            line_index + 1,
            fields.len(),
            COLUMN_NAMES.len()
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

    Ok(COLUMN_NAMES
        .into_iter()
        .zip(values)
        .map(|(name, values)| {
            let primitive = PrimitiveArray::from_iter(values);
            let array = primitive.clone().into_array();
            Column {
                name: name.to_string(),
                primitive,
                array,
            }
        })
        .collect::<Vec<_>>())
}

fn read_parquet_numeric(
    path: &Path,
    row_limit: usize,
    session: &VortexSession,
) -> VortexResult<Vec<Column>> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
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
    let mut row_count = 0usize;
    for batch in reader {
        let batch = batch.map_err(|error| vortex_err!("cannot read Parquet batch: {error}"))?;
        let batch_len = batch.num_rows().min(row_limit - row_count);
        for (column_chunks, array) in chunks.iter_mut().zip(batch.columns()) {
            column_chunks.push(array.slice(0, batch_len));
        }
        row_count += batch_len;
        if row_count == row_limit {
            break;
        }
    }
    vortex_ensure!(row_count > 0, "Parquet file contains no rows");

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
            let array = primitive.clone().into_array();
            Ok(Column {
                name: field.name().clone(),
                primitive,
                array,
            })
        })
        .collect()
}

fn percentile(durations: &mut [Duration], numerator: usize, denominator: usize) -> Duration {
    durations.sort_unstable();
    durations[durations.len() * numerator / denominator]
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

fn describe_pco_values<T: Number>(values: &[T], config: &ChunkConfig) -> VortexResult<String> {
    let compressor = FileCompressor::default();
    let chunk = compressor
        .chunk_compressor(values, config)
        .map_err(|error| vortex_err!("cannot inspect Pco selection: {error}"))?;
    let mode = match &chunk.meta().mode {
        Mode::Classic => "classic".to_string(),
        Mode::IntMult(_) => "int-mult".to_string(),
        Mode::FloatMult(_) => "float-mult".to_string(),
        Mode::FloatQuant(bits) => format!("float-quant-{bits}"),
        Mode::Dict(_) => "dict".to_string(),
        _ => "unknown".to_string(),
    };
    let delta = match &chunk.meta().delta_encoding {
        DeltaEncoding::NoOp => "none".to_string(),
        DeltaEncoding::Consecutive { order, .. } => format!("consecutive-{order}"),
        DeltaEncoding::Lookback { .. } => "lookback".to_string(),
        DeltaEncoding::Conv1(_) => "conv1".to_string(),
        _ => "unknown".to_string(),
    };
    Ok(format!("{mode}\t{delta}"))
}

fn describe_pco(
    column: &Column,
    config: &ChunkConfig,
    session: &VortexSession,
) -> VortexResult<String> {
    let mask = column
        .primitive
        .as_view()
        .validity()?
        .execute_mask(column.primitive.len(), &mut session.create_execution_ctx())?;
    macro_rules! describe_typed {
        ($T:ty) => {{
            let values = column
                .primitive
                .as_slice::<$T>()
                .iter()
                .copied()
                .zip(mask.iter())
                .filter_map(|(value, valid)| valid.then_some(value))
                .collect::<Vec<_>>();
            describe_pco_values(&values, config)
        }};
    }
    match column.primitive.ptype() {
        PType::I16 => describe_typed!(i16),
        PType::I32 => describe_typed!(i32),
        PType::I64 => describe_typed!(i64),
        PType::U16 => describe_typed!(u16),
        PType::U32 => describe_typed!(u32),
        PType::U64 => describe_typed!(u64),
        PType::F32 => describe_typed!(f32),
        PType::F64 => describe_typed!(f64),
        ptype => Err(vortex_err!("unsupported numeric type {ptype}")),
    }
}

fn ordered_f32(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & (1_u32 << 31) == 0 {
        bits ^ (1_u32 << 31)
    } else {
        !bits
    }
}

fn from_ordered_f32(value: u32) -> f32 {
    if value & (1_u32 << 31) == 0 {
        f32::from_bits(!value)
    } else {
        f32::from_bits(value ^ (1_u32 << 31))
    }
}

fn ordered_f64(value: f64) -> u64 {
    let bits = value.to_bits();
    if bits & (1_u64 << 63) == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    }
}

fn from_ordered_f64(value: u64) -> f64 {
    if value & (1_u64 << 63) == 0 {
        f64::from_bits(!value)
    } else {
        f64::from_bits(value ^ (1_u64 << 63))
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the branch limits the float to the exact u32 integer range"
)]
fn int_float_to_u32(value: f32) -> u32 {
    let absolute = value.abs();
    let greatest_precise_integer = 1_u32 << f32::MANTISSA_DIGITS;
    let greatest_precise_float = greatest_precise_integer as f32;
    let absolute_integer = if absolute < greatest_precise_float {
        absolute as u32
    } else {
        greatest_precise_integer + (absolute.to_bits() - greatest_precise_float.to_bits())
    };
    if value.is_sign_positive() {
        (1_u32 << 31) + absolute_integer
    } else {
        (1_u32 << 31) - 1 - absolute_integer
    }
}

fn int_float_from_u32(value: u32) -> f32 {
    let middle = 1_u32 << 31;
    let (negative, absolute_integer) = if value >= middle {
        (false, value - middle)
    } else {
        (true, middle - 1 - value)
    };
    let greatest_precise_integer = 1_u32 << f32::MANTISSA_DIGITS;
    let absolute = if absolute_integer < greatest_precise_integer {
        absolute_integer as f32
    } else {
        f32::from_bits(
            (greatest_precise_integer as f32).to_bits()
                + (absolute_integer - greatest_precise_integer),
        )
    };
    if negative { -absolute } else { absolute }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the branch limits the float to the exact u64 integer range"
)]
fn int_float_to_u64(value: f64) -> u64 {
    let absolute = value.abs();
    let greatest_precise_integer = 1_u64 << f64::MANTISSA_DIGITS;
    let greatest_precise_float = greatest_precise_integer as f64;
    let absolute_integer = if absolute < greatest_precise_float {
        absolute as u64
    } else {
        greatest_precise_integer + (absolute.to_bits() - greatest_precise_float.to_bits())
    };
    if value.is_sign_positive() {
        (1_u64 << 63) + absolute_integer
    } else {
        (1_u64 << 63) - 1 - absolute_integer
    }
}

fn int_float_from_u64(value: u64) -> f64 {
    let middle = 1_u64 << 63;
    let (negative, absolute_integer) = if value >= middle {
        (false, value - middle)
    } else {
        (true, middle - 1 - value)
    };
    let greatest_precise_integer = 1_u64 << f64::MANTISSA_DIGITS;
    let absolute = if absolute_integer < greatest_precise_integer {
        absolute_integer as f64
    } else {
        f64::from_bits(
            (greatest_precise_integer as f64).to_bits()
                + (absolute_integer - greatest_precise_integer),
        )
    };
    if negative { -absolute } else { absolute }
}

fn split_float_mult_f32(value: f32, base: f32) -> (u32, u32) {
    let multiple = (value / base).round();
    let primary = int_float_to_u32(multiple);
    let secondary = ordered_f32(value)
        .wrapping_sub(ordered_f32(multiple * base))
        .wrapping_add(1_u32 << 31);
    (primary, secondary)
}

fn split_float_mult_f64(value: f64, base: f64) -> (u64, u64) {
    let multiple = (value / base).round();
    let primary = int_float_to_u64(multiple);
    let secondary = ordered_f64(value)
        .wrapping_sub(ordered_f64(multiple * base))
        .wrapping_add(1_u64 << 63);
    (primary, secondary)
}

fn join_float_mult_f32(primary: u32, secondary: u32, base: f32) -> f32 {
    let approximate = int_float_from_u32(primary) * base;
    from_ordered_f32(ordered_f32(approximate).wrapping_add(secondary.wrapping_add(1_u32 << 31)))
}

fn join_float_mult_f64(primary: u64, secondary: u64, base: f64) -> f64 {
    let approximate = int_float_from_u64(primary) * base;
    from_ordered_f64(ordered_f64(approximate).wrapping_add(secondary.wrapping_add(1_u64 << 63)))
}

fn float_mult_prototype(
    column: &Column,
    pco_config: &ChunkConfig,
    compressor: &BtrBlocksCompressor,
    session: &VortexSession,
) -> VortexResult<Option<FloatMultPrototype>> {
    let mask = column
        .primitive
        .as_view()
        .validity()?
        .execute_mask(column.primitive.len(), &mut session.create_execution_ctx())?;
    macro_rules! build {
        ($T:ty, $L:ty, $variant:ident, $ordered:ident, $split:ident) => {{
            let values = column.primitive.as_slice::<$T>();
            let valid_values = values
                .iter()
                .copied()
                .zip(mask.iter())
                .filter_map(|(value, valid)| valid.then_some(value))
                .collect::<Vec<_>>();
            let chunk = FileCompressor::default()
                .chunk_compressor(&valid_values, pco_config)
                .map_err(|error| vortex_err!("cannot inspect Pco selection: {error}"))?;
            let Mode::FloatMult(base) = &chunk.meta().mode else {
                return Ok(None);
            };
            let base = $ordered(
                *base
                    .downcast_ref::<$L>()
                    .ok_or_else(|| vortex_err!("Pco FloatMult base has the wrong latent type"))?,
            );
            let (primary, secondary): (Vec<$L>, Vec<$L>) = values
                .iter()
                .copied()
                .map(|value| $split(value, base))
                .unzip();
            let primary_u64 = primary.iter().copied().map(u64::from).collect::<Vec<_>>();
            let secondary_u64 = secondary.iter().copied().map(u64::from).collect::<Vec<_>>();
            let backend_sizes = FloatMultBackendSizes {
                bit_split: BitSplitCodec::encode(&primary_u64)?.encoded_size()
                    + BitSplitCodec::encode(&secondary_u64)?.encoded_size()
                    + 16,
                block_residual: PatchedFoRCodec::encode_single_base(&primary_u64)?.encoded_size()
                    + PatchedFoRCodec::encode_single_base(&secondary_u64)?.encoded_size()
                    + 16,
                range_entropy: RangeEntropyCodec::encode(&primary_u64, 8_192)?.encoded_size()
                    + RangeEntropyCodec::encode(&secondary_u64, 8_192)?.encoded_size()
                    + 16,
                range_packed: RangePackedCodec::encode(&primary_u64, 8_192)?.encoded_size()
                    + RangePackedCodec::encode(&secondary_u64, 8_192)?.encoded_size()
                    + 16,
                range_two_level: RangeTwoLevelCodec::encode(&primary_u64, 8_192)?.encoded_size()
                    + RangeTwoLevelCodec::encode(&secondary_u64, 8_192)?.encoded_size()
                    + 16,
            };
            let latents = FloatMultLatents {
                primary: primary_u64,
                secondary: secondary_u64,
            };
            let validity = column.primitive.as_view().validity()?;
            let primary = PrimitiveArray::new(Buffer::from(primary), validity.clone()).into_array();
            let secondary = PrimitiveArray::new(Buffer::from(secondary), validity).into_array();
            let primary = compressor.compress(&primary, &mut session.create_execution_ctx())?;
            let secondary = compressor.compress(&secondary, &mut session.create_execution_ctx())?;
            Ok(Some(FloatMultPrototype::$variant {
                base,
                primary,
                secondary,
                backend_sizes,
                latents,
            }))
        }};
    }
    match column.primitive.ptype() {
        PType::F32 => build!(f32, u32, F32, from_ordered_f32, split_float_mult_f32),
        PType::F64 => build!(f64, u64, F64, from_ordered_f64, split_float_mult_f64),
        _ => Ok(None),
    }
}

fn decode_all(arrays: &[ArrayRef], session: &VortexSession) -> VortexResult<()> {
    for array in arrays {
        let decoded = array
            .clone()
            .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?;
        black_box(decoded.nbytes());
    }
    Ok(())
}

fn measure_compressors(
    dataset: &str,
    configs: &[(&str, Encoder<'_>)],
    columns: &[Column],
    input_bytes: u64,
    session: &VortexSession,
) -> VortexResult<()> {
    let iterations = (200_000_000u64 / input_bytes).clamp(3, 100) as usize;
    for (_, encoder) in configs {
        black_box(encoder.encode(columns, session)?);
    }

    let mut durations = (0..configs.len())
        .map(|_| Vec::with_capacity(iterations))
        .collect::<Vec<_>>();
    for iteration in 0..iterations {
        for offset in 0..configs.len() {
            let index = (iteration + offset) % configs.len();
            let start = Instant::now();
            black_box(configs[index].1.encode(columns, session)?);
            durations[index].push(start.elapsed());
        }
    }

    for (index, (name, _)) in configs.iter().enumerate() {
        let p10 = percentile(&mut durations[index].clone(), 1, 10);
        let p90 = percentile(&mut durations[index].clone(), 9, 10);
        let median = percentile(&mut durations[index], 1, 2);
        let throughput = input_bytes as f64 / median.as_secs_f64() / 1_000_000.0;
        println!(
            "compress\t{dataset}\t{name}\t{input_bytes}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
            p10.as_secs_f64() * 1_000.0,
            median.as_secs_f64() * 1_000.0,
            p90.as_secs_f64() * 1_000.0,
        );
    }
    Ok(())
}

fn measure_decoders(
    dataset: &str,
    configs: &[(&str, Vec<ArrayRef>)],
    input_bytes: u64,
    session: &VortexSession,
) -> VortexResult<()> {
    let iterations = (1_000_000_000u64 / input_bytes).clamp(5, 200) as usize;
    for (_, arrays) in configs {
        black_box(decode_all(arrays, session)?);
    }

    let mut durations = (0..configs.len())
        .map(|_| Vec::with_capacity(iterations))
        .collect::<Vec<_>>();
    for iteration in 0..iterations {
        for offset in 0..configs.len() {
            let index = (iteration + offset) % configs.len();
            let start = Instant::now();
            black_box(decode_all(&configs[index].1, session)?);
            durations[index].push(start.elapsed());
        }
    }

    for (index, (name, arrays)) in configs.iter().enumerate() {
        let output_bytes = arrays.iter().map(ArrayRef::nbytes).sum::<u64>();
        let p10 = percentile(&mut durations[index].clone(), 1, 10);
        let p90 = percentile(&mut durations[index].clone(), 9, 10);
        let median = percentile(&mut durations[index], 1, 2);
        let throughput = input_bytes as f64 / median.as_secs_f64() / 1_000_000.0;
        println!(
            "decode\t{dataset}\t{name}\t{output_bytes}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
            p10.as_secs_f64() * 1_000.0,
            median.as_secs_f64() * 1_000.0,
            p90.as_secs_f64() * 1_000.0,
        );
    }
    Ok(())
}

fn measure_float_mult_decoder(
    dataset: &str,
    prototypes: &[(&Column, FloatMultPrototype)],
    session: &VortexSession,
) -> VortexResult<()> {
    if prototypes.is_empty() {
        return Ok(());
    }
    let input_bytes = prototypes
        .iter()
        .map(|(column, _)| column.primitive.nbytes())
        .sum::<u64>();
    let encoded_bytes = prototypes
        .iter()
        .map(|(_, prototype)| prototype.encoded_size())
        .sum::<u64>();
    let iterations = (1_000_000_000_u64 / input_bytes).clamp(5, 200) as usize;
    for (_, prototype) in prototypes {
        black_box(prototype.decode(session)?);
    }
    let mut durations = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        for (_, prototype) in prototypes {
            black_box(prototype.decode(session)?);
        }
        durations.push(start.elapsed());
    }
    let p10 = percentile(&mut durations.clone(), 1, 10);
    let p90 = percentile(&mut durations.clone(), 9, 10);
    let median = percentile(&mut durations, 1, 2);
    let throughput = input_bytes as f64 / median.as_secs_f64() / 1_000_000.0;
    println!(
        "decode\t{dataset}\tfloat-mult-prototype\t{encoded_bytes}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
        p10.as_secs_f64() * 1_000.0,
        median.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn measure_float_mult_backends(
    dataset: &str,
    prototypes: &[(&Column, FloatMultPrototype)],
) -> VortexResult<()> {
    if prototypes.is_empty() {
        return Ok(());
    }
    let input_bytes = prototypes
        .iter()
        .map(|(column, _)| column.primitive.nbytes())
        .sum::<u64>();
    let encoded = FloatMultLatentBackend::ALL
        .into_iter()
        .map(|backend| {
            let codecs = prototypes
                .iter()
                .map(|(_, prototype)| backend.encode(prototype.latents()))
                .collect::<VortexResult<Vec<_>>>()?;
            Ok((backend, codecs))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let decode_iterations = (1_000_000_000_u64 / input_bytes).clamp(5, 50) as usize;
    for (backend, codecs) in &encoded {
        for ((_, prototype), codec) in prototypes.iter().zip(codecs) {
            let (primary, secondary) = codec.decode()?;
            prototype.reconstruct(&primary, &secondary)?;
        }
        let mut durations = Vec::with_capacity(decode_iterations);
        for _ in 0..decode_iterations {
            let start = Instant::now();
            for ((_, prototype), codec) in prototypes.iter().zip(codecs) {
                let (primary, secondary) = codec.decode()?;
                prototype.reconstruct(&primary, &secondary)?;
            }
            durations.push(start.elapsed());
        }
        let encoded_bytes = codecs
            .iter()
            .map(FloatMultCodecPair::encoded_size)
            .sum::<usize>();
        let p10 = percentile(&mut durations.clone(), 1, 10);
        let p90 = percentile(&mut durations.clone(), 9, 10);
        let median = percentile(&mut durations, 1, 2);
        let throughput = input_bytes as f64 / median.as_secs_f64() / 1_000_000.0;
        println!(
            "float-mult-backend-decode\t{dataset}\t{}\t{encoded_bytes}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
            backend.name(),
            p10.as_secs_f64() * 1_000.0,
            median.as_secs_f64() * 1_000.0,
            p90.as_secs_f64() * 1_000.0,
        );
    }

    let encode_iterations = (200_000_000_u64 / input_bytes).clamp(3, 20) as usize;
    for backend in FloatMultLatentBackend::ALL {
        let mut durations = Vec::with_capacity(encode_iterations);
        for _ in 0..encode_iterations {
            let start = Instant::now();
            let encoded_bytes = prototypes
                .iter()
                .try_fold(0_usize, |size, (_, prototype)| {
                    Ok::<_, vortex_error::VortexError>(
                        size + backend.encode(prototype.latents())?.encoded_size(),
                    )
                })?;
            durations.push(start.elapsed());
            black_box(encoded_bytes);
        }
        let p10 = percentile(&mut durations.clone(), 1, 10);
        let p90 = percentile(&mut durations.clone(), 9, 10);
        let median = percentile(&mut durations, 1, 2);
        let throughput = input_bytes as f64 / median.as_secs_f64() / 1_000_000.0;
        println!(
            "float-mult-backend-encode\t{dataset}\t{}\t{input_bytes}\t{:.3}\t{:.3}\t{:.3}\t{throughput:.1}",
            backend.name(),
            p10.as_secs_f64() * 1_000.0,
            median.as_secs_f64() * 1_000.0,
            p90.as_secs_f64() * 1_000.0,
        );
    }
    Ok(())
}

fn main() -> VortexResult<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| vortex_err!("usage: float_quant_dataset <data path> [row limit]"))?;
    let row_limit = std::env::args()
        .nth(2)
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| vortex_err!("invalid row limit: {error}"))
        })
        .transpose()?
        .unwrap_or(DEFAULT_ROW_LIMIT);
    let path = Path::new(&path);
    let dataset = path
        .file_stem()
        .and_then(|name| name.to_str())
        .ok_or_else(|| vortex_err!("data path has no valid file name"))?;
    let session = array_session();
    let columns = if path
        .extension()
        .is_some_and(|extension| extension == "parquet")
    {
        read_parquet_numeric(path, row_limit, &session)?
    } else {
        read_california_housing(path)?
    };
    let input_bytes = columns
        .iter()
        .map(|column| column.primitive.nbytes())
        .sum::<u64>();
    let baseline = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([
            FloatMultScheme.id(),
            FloatQuantScheme.id(),
            OrderedBlockResidualScheme.id(),
        ])
        .build();
    let range_candidate = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([
            FloatMultScheme.id(),
            FloatQuantScheme.id(),
            OrderedBlockResidualScheme.id(),
        ])
        .with_new_scheme(&RangeEntropyScheme)
        .build();
    let default_without_block_residual = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([OrderedBlockResidualScheme.id()])
        .build();
    let default_without_float_mult = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([FloatMultScheme.id()])
        .build();
    let float_candidate = BtrBlocksCompressor::default();
    let stacked_candidate = BtrBlocksCompressorBuilder::default()
        .with_new_scheme(&RangeEntropyScheme)
        .build();
    let compact = BtrBlocksCompressorBuilder::default().with_compact().build();
    let pco_config = ChunkConfig::default()
        .with_mode_spec(ModeSpec::Auto)
        .with_delta_spec(DeltaSpec::Auto);
    let configs = [
        ("default-without-new-float", Encoder::BtrBlocks(&baseline)),
        (
            "default-without-new-float+range-entropy",
            Encoder::BtrBlocks(&range_candidate),
        ),
        (
            "default-without-block-residual",
            Encoder::BtrBlocks(&default_without_block_residual),
        ),
        (
            "default-without-float-mult",
            Encoder::BtrBlocks(&default_without_float_mult),
        ),
        ("default", Encoder::BtrBlocks(&float_candidate)),
        (
            "default+range-entropy",
            Encoder::BtrBlocks(&stacked_candidate),
        ),
        ("compact", Encoder::BtrBlocks(&compact)),
        ("pco-auto", Encoder::Pco(&pco_config)),
    ];
    let encoded = configs
        .iter()
        .map(|(name, encoder)| Ok((*name, encoder.encode(&columns, &session)?)))
        .collect::<VortexResult<Vec<_>>>()?;
    let float_mult_prototypes = columns
        .iter()
        .filter_map(|column| {
            float_mult_prototype(column, &pco_config, &baseline, &session)
                .transpose()
                .map(|result| result.map(|prototype| (column, prototype)))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    println!("pco-selection\tdataset\tcolumn\tmode\tdelta");
    for column in &columns {
        println!(
            "pco-selection\t{dataset}\t{}\t{}",
            column.name,
            describe_pco(column, &pco_config, &session)?
        );
        if let Some(base) = estimate_float_mult_constant_base(column.primitive.as_view()) {
            println!(
                "float-mult-analysis\t{dataset}\t{}\tconstant-base\t{base}",
                column.name
            );
        }
    }
    println!("structure\tdataset\tcolumn\tconfig\tencoding\tbytes");
    for (config_name, arrays) in &encoded {
        for (column, array) in columns.iter().zip(arrays) {
            let mut verify_ctx = session.create_execution_ctx();
            assert_arrays_eq!(array, column.array, &mut verify_ctx);
            println!(
                "structure\t{dataset}\t{}\t{config_name}\t{}\t{}",
                column.name,
                encoding_tree(array),
                array.nbytes()
            );
            if let Some(float_mult) = array.as_opt::<FloatMult>() {
                println!(
                    "float-mult-base\t{dataset}\t{}\t{config_name}\t{}",
                    column.name,
                    float_mult.base()
                );
            }
        }
    }
    for (column, prototype) in &float_mult_prototypes {
        let decoded = prototype.decode(&session)?;
        let mut verify_ctx = session.create_execution_ctx();
        assert_arrays_eq!(decoded, column.array, &mut verify_ctx);
        let backend_sizes = prototype.backend_sizes();
        println!(
            "structure\t{dataset}\t{}\tfloat-mult-prototype\t{}\t{}",
            column.name,
            prototype.structure(),
            prototype.encoded_size(),
        );
        println!(
            "float-mult-base\t{dataset}\t{}\tpco-prototype\t{}",
            column.name,
            prototype.base()
        );
        println!(
            "float-mult-backends\t{dataset}\t{}\tbit-split={}\tblock-residual={}\trange-entropy={}\trange-packed={}\trange-two-level={}",
            column.name,
            backend_sizes.bit_split,
            backend_sizes.block_residual,
            backend_sizes.range_entropy,
            backend_sizes.range_packed,
            backend_sizes.range_two_level,
        );
    }
    println!("operation\tdataset\tconfig\tbytes\tp10-ms\tmedian-ms\tp90-ms\tMB/s");
    measure_decoders(dataset, &encoded, input_bytes, &session)?;
    measure_float_mult_decoder(dataset, &float_mult_prototypes, &session)?;
    measure_float_mult_backends(dataset, &float_mult_prototypes)?;
    measure_compressors(dataset, &configs, &columns, input_bytes, &session)?;
    Ok(())
}
