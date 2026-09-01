// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Portable state for Vane's explicit distributed table-scan contract.
//!
//! This module deliberately serializes only owned data. Runtime readers and
//! filesystems are reconstructed from an explicit assigned file set when the
//! real worker execution context initializes the scan.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use futures::StreamExt;
use futures::TryStreamExt;
use prost::Message;
use vortex::dtype::DType;
use vortex::dtype::proto::dtype as pb_dtype;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::proto::ExprSerializeProtoExt;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::proto::expr as pb_expr;
use vortex::scan::DataSource;
use vortex_utils::aliases::hash_set::HashSet;
use vortex_utils::parallelism::get_available_parallelism;

use crate::RUNTIME;
use crate::SESSION;
use crate::convert::PushedAggregate;
use crate::duckdb::AggregatePushdownInputRef;
use crate::multi_file::BoundFile;
use crate::multi_file::build_bound_file_scan;
use crate::multi_file::build_bound_fragment_scan;
use crate::multi_file::open_bound_file;
use crate::multi_file::validate_bound_file;
use crate::projection::DuckdbField;
use crate::projection::extract_schema_from_dtype;
use crate::table_function::ColumnAggregate;
use crate::table_function::TableFunctionBind;
use crate::table_function::TableFunctionGlobal;
use crate::table_function::pushdown_projection_aggregates;

const PORTABLE_BIND_VERSION: u32 = 2;

#[derive(Clone, PartialEq, Message)]
struct ProjectionProto {
    #[prost(uint64, tag = "1")]
    field_index: u64,
    #[prost(bytes = "vec", tag = "2")]
    expression: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
struct AggregateProto {
    #[prost(uint64, tag = "1")]
    projection_id: u64,
    #[prost(uint32, tag = "2")]
    kind: u32,
}

#[derive(Clone, PartialEq, Message)]
struct FileProto {
    #[prost(string, tag = "1")]
    source_url: String,
    #[prost(string, tag = "2")]
    path: String,
    #[prost(uint64, optional, tag = "3")]
    size: Option<u64>,
    #[prost(string, optional, tag = "4")]
    e_tag: Option<String>,
    #[prost(string, optional, tag = "5")]
    version: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct PortableBindProto {
    #[prost(uint32, tag = "1")]
    version: u32,
    #[prost(bytes = "vec", tag = "2")]
    dtype: Vec<u8>,
    #[prost(message, repeated, tag = "3")]
    projections: Vec<ProjectionProto>,
    #[prost(bytes = "vec", repeated, tag = "4")]
    filters: Vec<Vec<u8>>,
    #[prost(message, repeated, tag = "5")]
    aggregates: Vec<AggregateProto>,
    #[prost(bool, tag = "6")]
    has_non_optional_filter: bool,
    #[prost(message, repeated, tag = "7")]
    files: Vec<FileProto>,
}

pub struct PortableDistributedBind {
    pub encoded: Vec<u8>,
    pub files: Vec<BoundFile>,
    pub column_fields: Vec<DuckdbField>,
    pub aggregate_scan: bool,
}

pub struct DistributedRuntimeGlobal {
    pub bind_data: TableFunctionBind,
    pub global_data: TableFunctionGlobal,
}

/// One independently reopenable row range within an immutable bound Vortex file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistributedFragment {
    /// Stable coordinator index of the bound file.
    pub file_index: usize,
    /// Inclusive root-coordinate row offset.
    pub row_start: u64,
    /// Exclusive root-coordinate row offset.
    pub row_end: u64,
    /// Proportional on-storage byte estimate for scheduling.
    pub estimated_bytes: u64,
}

/// Owned result of deterministic distributed fragment planning.
pub struct DistributedFragmentPlan {
    /// Canonically ordered fragments grouped by file index.
    pub fragments: Vec<DistributedFragment>,
}

struct NaturalFileFragments {
    file_index: usize,
    file_size: u64,
    row_count: u64,
    ranges: Vec<std::ops::Range<u64>>,
}

fn encode_expression(expression: &Expression) -> VortexResult<Vec<u8>> {
    Ok(expression.serialize_proto()?.encode_to_vec())
}

fn decode_expression(bytes: &[u8]) -> VortexResult<Expression> {
    let proto = pb_expr::Expr::decode(bytes)
        .map_err(|error| vortex_err!("Invalid distributed Vortex expression: {error}"))?;
    if proto.encode_to_vec() != bytes {
        vortex_bail!("Distributed Vortex expression is not canonically encoded");
    }
    Expression::from_proto(&proto, &SESSION)
}

fn encode_aggregate(aggregate: &ColumnAggregate) -> AggregateProto {
    match aggregate {
        ColumnAggregate::Real {
            projection_id,
            aggregate,
        } => AggregateProto {
            projection_id: *projection_id,
            kind: match aggregate {
                PushedAggregate::Min => 1,
                PushedAggregate::Max => 2,
                PushedAggregate::Sum => 3,
                PushedAggregate::Mean => 4,
                PushedAggregate::First => 5,
                PushedAggregate::Count => 6,
            },
        },
        ColumnAggregate::CountStar => AggregateProto {
            projection_id: 0,
            kind: 7,
        },
    }
}

fn decode_aggregate(aggregate: AggregateProto) -> VortexResult<ColumnAggregate> {
    let pushed = match aggregate.kind {
        1 => PushedAggregate::Min,
        2 => PushedAggregate::Max,
        3 => PushedAggregate::Sum,
        4 => PushedAggregate::Mean,
        5 => PushedAggregate::First,
        6 => PushedAggregate::Count,
        7 => {
            if aggregate.projection_id != 0 {
                vortex_bail!(
                    "Distributed Vortex count-star aggregate has an invalid projection id"
                );
            }
            return Ok(ColumnAggregate::CountStar);
        }
        kind => vortex_bail!("Unknown distributed Vortex aggregate kind: {kind}"),
    };
    Ok(ColumnAggregate::Real {
        projection_id: aggregate.projection_id,
        aggregate: pushed,
    })
}

fn decode_proto(bytes: &[u8]) -> VortexResult<PortableBindProto> {
    let proto = PortableBindProto::decode(bytes)
        .map_err(|error| vortex_err!("Invalid distributed Vortex bind data: {error}"))?;
    if proto.version != PORTABLE_BIND_VERSION {
        vortex_bail!(
            "Unsupported distributed Vortex bind version: {}",
            proto.version
        );
    }
    if proto.encode_to_vec() != bytes {
        vortex_bail!("Distributed Vortex bind data is not canonically encoded");
    }
    Ok(proto)
}

fn validate_natural_ranges(
    path: &str,
    row_count: u64,
    ranges: &[std::ops::Range<u64>],
) -> VortexResult<()> {
    if row_count == 0 {
        if !ranges.is_empty() {
            vortex_bail!("Empty Vortex file '{path}' produced non-empty scan fragments");
        }
        return Ok(());
    }
    if ranges.is_empty()
        || ranges[0].start != 0
        || ranges.last().is_none_or(|range| range.end != row_count)
        || ranges
            .iter()
            .any(|range| range.start >= range.end || range.end > row_count)
        || ranges.windows(2).any(|pair| pair[0].end != pair[1].start)
    {
        vortex_bail!(
            "Vortex file '{path}' produced scan fragments with a gap, overlap, or invalid bound"
        );
    }
    Ok(())
}

fn allocate_fragment_counts(files: &[NaturalFileFragments], target_count: usize) -> Vec<usize> {
    if files.is_empty() {
        return Vec::new();
    }
    let capacities = files
        .iter()
        .map(|file| file.ranges.len().max(1))
        .collect::<Vec<_>>();
    let maximum_count = capacities.iter().sum::<usize>();
    let desired_count = target_count.max(files.len()).min(maximum_count);
    let remaining = desired_count - files.len();
    let total_extra_capacity = maximum_count - files.len();
    let mut counts = vec![1; files.len()];
    if remaining == 0 || total_extra_capacity == 0 {
        return counts;
    }

    let mut remainders = Vec::with_capacity(files.len());
    let mut allocated = 0;
    for (file_index, &capacity) in capacities.iter().enumerate() {
        let extra_capacity = capacity - 1;
        let scaled = (remaining as u128) * (extra_capacity as u128);
        let extra = usize::try_from(scaled / (total_extra_capacity as u128))
            .vortex_expect("proportional fragment allocation must fit in usize");
        counts[file_index] += extra;
        allocated += extra;
        remainders.push((scaled % (total_extra_capacity as u128), file_index));
    }
    remainders
        .sort_unstable_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    for (_, file_index) in remainders.into_iter().take(remaining - allocated) {
        counts[file_index] += 1;
    }
    counts
}

fn coalesce_ranges(
    ranges: &[std::ops::Range<u64>],
    fragment_count: usize,
) -> Vec<std::ops::Range<u64>> {
    if ranges.is_empty() {
        return vec![0..0];
    }
    debug_assert!(fragment_count > 0 && fragment_count <= ranges.len());
    (0..fragment_count)
        .map(|fragment_index| {
            let start_index = usize::try_from(
                (fragment_index as u128) * (ranges.len() as u128) / (fragment_count as u128),
            )
            .vortex_expect("coalesced fragment start must fit in usize");
            let end_index = usize::try_from(
                ((fragment_index + 1) as u128) * (ranges.len() as u128) / (fragment_count as u128),
            )
            .vortex_expect("coalesced fragment end must fit in usize");
            ranges[start_index].start..ranges[end_index - 1].end
        })
        .collect()
}

fn estimate_fragment_bytes(
    file_size: u64,
    row_count: u64,
    row_range: &std::ops::Range<u64>,
) -> u64 {
    if row_count == 0 {
        return file_size;
    }
    let scaled_start = u128::from(file_size) * u128::from(row_range.start) / u128::from(row_count);
    let scaled_end = u128::from(file_size) * u128::from(row_range.end) / u128::from(row_count);
    u64::try_from(scaled_end - scaled_start)
        .vortex_expect("a proportional fragment estimate cannot exceed its u64 file size")
}

/// Reopen selected immutable files and plan canonical row-range fragments.
pub fn plan_fragments(
    bytes: &[u8],
    selected_file_indexes: &[u64],
    target_count: usize,
) -> VortexResult<DistributedFragmentPlan> {
    let decoded = decode_bind(bytes)?;
    let mut previous_file_index = None;
    let mut selected_files = Vec::with_capacity(selected_file_indexes.len());
    for &file_index in selected_file_indexes {
        let file_index = usize::try_from(file_index)?;
        if previous_file_index.is_some_and(|previous| previous >= file_index) {
            vortex_bail!("Distributed Vortex fragment files are not in canonical order");
        }
        previous_file_index = Some(file_index);
        let file = decoded.files.get(file_index).ok_or_else(|| {
            vortex_err!("Unknown distributed Vortex fragment file index: {file_index}")
        })?;
        selected_files.push((file_index, file.clone()));
    }
    let concurrency = get_available_parallelism()
        .unwrap_or(1)
        .min(selected_files.len().max(1));
    let files = RUNTIME.block_on(async move {
        futures::stream::iter(selected_files)
            .map(|(file_index, file)| async move {
                let vortex_file = open_bound_file(&file).await?;
                let row_count = vortex_file.row_count();
                let ranges = vortex_file.splits()?;
                validate_natural_ranges(&file.path, row_count, &ranges)?;
                Ok::<_, vortex::error::VortexError>(NaturalFileFragments {
                    file_index,
                    file_size: file.size,
                    row_count,
                    ranges,
                })
            })
            // `buffered` opens files concurrently while preserving canonical input order.
            .buffered(concurrency)
            .try_collect::<Vec<_>>()
            .await
    })?;

    let fragment_counts = allocate_fragment_counts(&files, target_count.max(1));
    let mut fragments = Vec::with_capacity(fragment_counts.iter().sum());
    for (file, fragment_count) in files.iter().zip(fragment_counts) {
        for row_range in coalesce_ranges(&file.ranges, fragment_count) {
            fragments.push(DistributedFragment {
                file_index: file.file_index,
                row_start: row_range.start,
                row_end: row_range.end,
                estimated_bytes: estimate_fragment_bytes(
                    file.file_size,
                    file.row_count,
                    &row_range,
                ),
            });
        }
    }
    Ok(DistributedFragmentPlan { fragments })
}

pub fn serialize_bind(bind_data: &TableFunctionBind) -> VortexResult<PortableDistributedBind> {
    let dtype = pb_dtype::DType::try_from(bind_data.data_source.dtype())?.encode_to_vec();
    let projections = bind_data
        .column_fields
        .iter()
        .enumerate()
        .filter_map(|(field_index, field)| {
            field.projection_expr.as_ref().map(|expression| {
                Ok(ProjectionProto {
                    field_index: u64::try_from(field_index)?,
                    expression: encode_expression(expression)?,
                })
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let filters = bind_data
        .filter_exprs
        .iter()
        .map(encode_expression)
        .collect::<VortexResult<Vec<_>>>()?;
    let files = bind_data.files.clone();
    for file in &files {
        validate_bound_file(file)?;
    }
    let proto = PortableBindProto {
        version: PORTABLE_BIND_VERSION,
        dtype,
        projections,
        filters,
        aggregates: bind_data.aggregates.iter().map(encode_aggregate).collect(),
        has_non_optional_filter: bind_data.has_non_optional_filter.load(Ordering::Relaxed),
        files: files
            .iter()
            .map(|file| FileProto {
                source_url: file.source_url.clone(),
                path: file.path.clone(),
                size: Some(file.size),
                e_tag: file.e_tag.clone(),
                version: file.version.clone(),
            })
            .collect(),
    };
    Ok(PortableDistributedBind {
        encoded: proto.encode_to_vec(),
        files,
        column_fields: bind_data.column_fields.clone(),
        aggregate_scan: !bind_data.aggregates.is_empty(),
    })
}

struct DecodedPortableBind {
    dtype: DType,
    column_fields: Vec<DuckdbField>,
    filter_exprs: Vec<Expression>,
    aggregates: Vec<ColumnAggregate>,
    has_non_optional_filter: bool,
    files: Vec<BoundFile>,
}

fn decode_bind(bytes: &[u8]) -> VortexResult<DecodedPortableBind> {
    let proto = decode_proto(bytes)?;
    let dtype_proto = pb_dtype::DType::decode(proto.dtype.as_slice())
        .map_err(|error| vortex_err!("Invalid distributed Vortex dtype: {error}"))?;
    if dtype_proto.encode_to_vec() != proto.dtype {
        vortex_bail!("Distributed Vortex dtype is not canonically encoded");
    }
    let dtype = DType::from_proto(&dtype_proto, &SESSION)?;
    let mut column_fields = extract_schema_from_dtype(&dtype)?;

    let mut projected_fields = HashSet::new();
    let mut previous_projection = None;
    for projection in proto.projections {
        let field_index = usize::try_from(projection.field_index)?;
        let field = column_fields.get_mut(field_index).ok_or_else(|| {
            vortex_err!("Distributed Vortex projection references unknown field {field_index}")
        })?;
        if !projected_fields.insert(field_index) {
            vortex_bail!(
                "Distributed Vortex bind contains duplicate projection field {field_index}"
            );
        }
        if previous_projection.is_some_and(|previous| previous >= field_index) {
            vortex_bail!("Distributed Vortex projections are not in canonical field order");
        }
        previous_projection = Some(field_index);
        field.projection_expr = Some(decode_expression(&projection.expression)?);
    }

    let filter_exprs = proto
        .filters
        .iter()
        .map(|filter| decode_expression(filter))
        .collect::<VortexResult<Vec<_>>>()?;
    let aggregates = proto
        .aggregates
        .into_iter()
        .map(decode_aggregate)
        .collect::<VortexResult<Vec<_>>>()?;
    for aggregate in &aggregates {
        if let ColumnAggregate::Real { projection_id, .. } = aggregate
            && usize::try_from(*projection_id)? >= column_fields.len()
        {
            vortex_bail!(
                "Distributed Vortex aggregate references unknown field {}",
                projection_id
            );
        }
    }

    let files = proto
        .files
        .into_iter()
        .map(|file| {
            let size = file.size.ok_or_else(|| {
                vortex_err!("Distributed Vortex file '{}' has no bound size", file.path)
            })?;
            let file = BoundFile {
                source_url: file.source_url,
                path: file.path,
                size,
                e_tag: file.e_tag,
                version: file.version,
            };
            validate_bound_file(&file)?;
            Ok(file)
        })
        .collect::<VortexResult<Vec<_>>>()?;
    if files.is_empty() {
        vortex_bail!("Distributed Vortex bind contains no bound files");
    }

    Ok(DecodedPortableBind {
        dtype,
        column_fields,
        filter_exprs,
        aggregates,
        has_non_optional_filter: proto.has_non_optional_filter,
        files,
    })
}

pub fn deserialize_bind(bytes: &[u8]) -> VortexResult<PortableDistributedBind> {
    let decoded = decode_bind(bytes)?;
    Ok(PortableDistributedBind {
        encoded: bytes.to_vec(),
        aggregate_scan: !decoded.aggregates.is_empty(),
        files: decoded.files,
        column_fields: decoded.column_fields,
    })
}

/// Apply aggregate pushdown to an owned coordinator bind without opening its files.
///
/// Vane serializes a bound logical plan before optimization, so the optimizer sees
/// portable bind data rather than the connection-scoped reader. Aggregate eligibility
/// only depends on the bound schema, and an empty deferred data source preserves that
/// schema while keeping planning free of worker I/O.
pub fn pushdown_serialized_projection_aggregates(
    bytes: &[u8],
    input: &AggregatePushdownInputRef,
) -> VortexResult<Option<PortableDistributedBind>> {
    let decoded = decode_bind(bytes)?;
    let data_source = build_bound_file_scan(&[], Some(decoded.dtype))?;
    let mut bind_data = TableFunctionBind {
        data_source: Arc::new(data_source),
        files: decoded.files,
        file_indexes: Vec::new(),
        filter_exprs: decoded.filter_exprs,
        column_fields: decoded.column_fields,
        has_non_optional_filter: AtomicBool::new(decoded.has_non_optional_filter),
        aggregates: decoded.aggregates,
    };
    if !pushdown_projection_aggregates(&mut bind_data, input)? {
        return Ok(None);
    }
    Ok(Some(serialize_bind(&bind_data)?))
}

pub fn deserialize_runtime_bind(
    bytes: &[u8],
    assigned_fragments: &[DistributedFragment],
) -> VortexResult<TableFunctionBind> {
    let decoded = decode_bind(bytes)?;

    let mut selected_files = Vec::with_capacity(assigned_fragments.len());
    let mut row_ranges = Vec::with_capacity(assigned_fragments.len());
    let mut file_indexes = Vec::with_capacity(assigned_fragments.len());
    let mut previous_fragment: Option<&DistributedFragment> = None;
    for fragment in assigned_fragments {
        if fragment.row_start > fragment.row_end || fragment.estimated_bytes == u64::MAX {
            vortex_bail!(
                "Distributed Vortex fragment has an invalid row range or byte estimate: {}..{}",
                fragment.row_start,
                fragment.row_end
            );
        }
        let index = fragment.file_index;
        if let Some(previous) = previous_fragment
            && (previous.file_index > index
                || (previous.file_index == index && previous.row_start >= fragment.row_start))
        {
            vortex_bail!("Distributed Vortex fragments are not in canonical order");
        }
        if let Some(previous) = previous_fragment
            && previous.file_index == index
            && previous.row_end > fragment.row_start
        {
            vortex_bail!("Distributed Vortex fragments overlap within file index {index}");
        }
        let file = decoded
            .files
            .get(index)
            .ok_or_else(|| vortex_err!("Unknown distributed Vortex file index: {index}"))?;
        if fragment.estimated_bytes > file.size {
            vortex_bail!(
                "Distributed Vortex fragment byte estimate {} exceeds file size {}",
                fragment.estimated_bytes,
                file.size
            );
        }
        selected_files.push(file.clone());
        row_ranges.push(fragment.row_start..fragment.row_end);
        file_indexes.push(index);
        previous_fragment = Some(fragment);
    }

    let data_source =
        build_bound_fragment_scan(&selected_files, &row_ranges, Some(decoded.dtype.clone()))?;
    if data_source.dtype() != &decoded.dtype {
        vortex_bail!(
            "Distributed Vortex file schema differs from the coordinator bind: expected {}, got {}",
            decoded.dtype,
            data_source.dtype()
        );
    }
    Ok(TableFunctionBind {
        data_source: Arc::new(data_source),
        files: selected_files,
        file_indexes,
        filter_exprs: decoded.filter_exprs,
        column_fields: decoded.column_fields,
        has_non_optional_filter: AtomicBool::new(decoded.has_non_optional_filter),
        aggregates: decoded.aggregates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn natural_file(
        file_index: usize,
        row_count: u64,
        ranges: Vec<std::ops::Range<u64>>,
    ) -> NaturalFileFragments {
        NaturalFileFragments {
            file_index,
            file_size: row_count * 10,
            row_count,
            ranges,
        }
    }

    #[test]
    fn fragment_counts_honor_target_and_capacity() {
        let files = vec![
            natural_file(0, 40, vec![0..10, 10..20, 20..30, 30..40]),
            natural_file(1, 20, vec![0..10, 10..20]),
        ];

        assert_eq!(allocate_fragment_counts(&files, 1), vec![1, 1]);
        assert_eq!(allocate_fragment_counts(&files, 4), vec![3, 1]);
        assert_eq!(allocate_fragment_counts(&files, 20), vec![4, 2]);
    }

    #[test]
    fn coalesced_fragments_tile_natural_ranges() {
        let ranges = vec![0..10, 10..20, 20..30, 30..40, 40..50];

        assert_eq!(coalesce_ranges(&ranges, 2), vec![0..20, 20..50]);
        assert_eq!(coalesce_ranges(&ranges, 3), vec![0..10, 10..30, 30..50]);
        assert_eq!(coalesce_ranges(&[], 1), vec![0..0]);
    }

    #[test]
    fn proportional_byte_estimates_sum_to_file_size() {
        let ranges = [0..3, 3..7, 7..10];
        let estimates = ranges
            .iter()
            .map(|range| estimate_fragment_bytes(101, 10, range))
            .collect::<Vec<_>>();

        assert_eq!(estimates, vec![30, 40, 31]);
        assert_eq!(estimates.iter().sum::<u64>(), 101);
    }

    #[test]
    fn natural_fragment_validation_rejects_gaps_and_overlap() {
        assert!(validate_natural_ranges("gap.vortex", 10, &[0..4, 5..10]).is_err());
        assert!(validate_natural_ranges("overlap.vortex", 10, &[0..6, 5..10]).is_err());
        assert!(validate_natural_ranges("reversed.vortex", 10, &[0..6, 6..5]).is_err());
        assert!(validate_natural_ranges("past-end.vortex", 10, &[0..11]).is_err());
        assert!(validate_natural_ranges("missing.vortex", 10, &[]).is_err());
        assert!(validate_natural_ranges("nonempty.vortex", 10, &[0..0, 0..10]).is_err());
        assert!(validate_natural_ranges("valid.vortex", 10, &[0..4, 4..10]).is_ok());
        assert!(validate_natural_ranges("empty.vortex", 0, &[]).is_ok());
        assert!(validate_natural_ranges("invalid-empty.vortex", 0, &[0..0]).is_err());
    }
}
