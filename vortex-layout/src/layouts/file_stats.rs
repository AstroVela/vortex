// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use futures::StreamExt;
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AccumulatorRef;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarTruncation;
use vortex_array::scalar::lower_bound;
use vortex_array::scalar::upper_bound;
use vortex_array::stats::StatsSet;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Wrap `stream` so that file-level statistics are accumulated as chunks flow through it.
///
/// Statistics are computed with the same aggregate functions that back zoned layout
/// statistics, so file-level values agree with zone-map semantics (e.g. NaN-skipping
/// min/max/sum). Variable-length min/max values are truncated to
/// `max_variable_length_statistics_size` bytes when the final statistics are extracted.
pub fn accumulate_stats(
    stream: SendableSequentialStream,
    stats: Arc<[Stat]>,
    g max_variable_length_statistics_size: usize,
    session: &VortexSession,
) -> VortexResult<(FileStatsAccumulator, SendableSequentialStream)> {
    let accumulator = FileStatsAccumulator::try_new(
        stream.dtype(),
        &stats,
        max_variable_length_statistics_size,
        session,
    )?;
    let stream = SequentialStreamAdapter::new(
        stream.dtype().clone(),
        stream.scan(accumulator.clone(), |acc, item| {
            future::ready(Some(acc.process(item)))
        }),
    )
    .sendable();
    Ok((accumulator, stream))
}

/// Accumulates write-time statistics for a single file column.
struct StatsAccumulator {
    accumulators: Vec<StatAccumulator>,
    max_variable_length_statistics_size: usize,
}

/// A running aggregate for a single statistic of a single column.
struct StatAccumulator {
    stat: Stat,
    accumulator: AccumulatorRef,
}

impl StatsAccumulator {
    fn try_new(
        dtype: &DType,
        stats: &[Stat],
        max_variable_length_statistics_size: usize,
    ) -> VortexResult<Self> {
        let mut accumulators = Vec::with_capacity(stats.len());
        if supports_file_stats(dtype) {
            for &stat in stats {
                let Some(aggregate_fn) = stat.aggregate_fn() else {
                    continue;
                };
                if aggregate_fn.state_dtype(dtype).is_none() {
                    continue;
                }
                accumulators.push(StatAccumulator {
                    stat,
                    accumulator: aggregate_fn.accumulator(dtype)?,
                });
            }
        }
        Ok(Self {
            accumulators,
            max_variable_length_statistics_size,
        })
    }

    fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        for acc in &mut self.accumulators {
            acc.accumulator.accumulate(array, ctx)?;
        }
        Ok(())
    }

    /// Returns the aggregated stats set for the column.
    fn as_stats_set(&self) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        for acc in &self.accumulators {
            let value = acc.accumulator.final_scalar()?;
            if value.is_null() {
                continue;
            }

            match acc.stat {
                Stat::Min | Stat::Max if is_varlen_dtype(value.dtype()) => {
                    // Bound the footer size by truncating variable-length min/max values. A
                    // truncated value is only a bound on the data, so it must be marked inexact.
                    let Some((bound, truncated)) = truncated_varlen_bound(
                        acc.stat,
                        value,
                        self.max_variable_length_statistics_size,
                    )?
                    else {
                        // No representable upper bound within the size limit.
                        continue;
                    };
                    if let Some(bound) = bound.into_value() {
                        stats_set.set(
                            acc.stat,
                            if truncated {
                                Precision::inexact(bound)
                            } else {
                                Precision::exact(bound)
                            },
                        );
                    }
                }
                stat => {
                    if let Some(value) = value.into_value() {
                        stats_set.set(stat, Precision::exact(value));
                    }
                }
            }
        }
        Ok(stats_set)
    }
}

fn supports_file_stats(dtype: &DType) -> bool {
    !matches!(dtype, DType::Variant(_))
}

fn is_varlen_dtype(dtype: &DType) -> bool {
    matches!(dtype, DType::Utf8(_) | DType::Binary(_))
}

/// Truncate a variable-length min/max scalar to at most `max_length` bytes.
///
/// Returns the bound and whether truncation occurred, or `None` if no bound of at most
/// `max_length` bytes exists (only possible for `Stat::Max`).
fn truncated_varlen_bound(
    stat: Stat,
    value: Scalar,
    max_length: usize,
) -> VortexResult<Option<(Scalar, bool)>> {
    let nullability = value.dtype().nullability();
    Ok(match (value.dtype().clone(), stat) {
        (DType::Utf8(_), Stat::Min) => {
            lower_bound(BufferString::from_scalar(value)?, max_length, nullability)
        }
        (DType::Utf8(_), Stat::Max) => {
            upper_bound(BufferString::from_scalar(value)?, max_length, nullability)
        }
        (DType::Binary(_), Stat::Min) => {
            lower_bound(ByteBuffer::from_scalar(value)?, max_length, nullability)
        }
        (DType::Binary(_), Stat::Max) => {
            upper_bound(ByteBuffer::from_scalar(value)?, max_length, nullability)
        }
        (dtype, stat) => vortex_panic!("unexpected varlen bound for {stat} of {dtype}"),
    })
}

/// An array stream processor that computes aggregate statistics for all fields.
///
/// Note: for now this only collects top-level struct fields.
#[derive(Clone)]
pub struct FileStatsAccumulator {
    accumulators: Arc<Mutex<Vec<StatsAccumulator>>>,
    ctx: Arc<Mutex<ExecutionCtx>>,
}

impl FileStatsAccumulator {
    fn try_new(
        dtype: &DType,
        stats: &[Stat],
        max_variable_length_statistics_size: usize,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let accumulators = match dtype.as_struct_fields_opt() {
            Some(struct_dtype) => {
                if dtype.nullability() == Nullability::Nullable {
                    // top level dtype could be nullable, but we don't support it yet
                    vortex_panic!(
                        "FileStatsAccumulator temporarily does not support nullable top-level structs, got: {}. Use Validity::NonNullable",
                        dtype
                    );
                }

                struct_dtype
                    .fields()
                    .map(|field_dtype| {
                        StatsAccumulator::try_new(
                            &field_dtype,
                            stats,
                            max_variable_length_statistics_size,
                        )
                    })
                    .try_collect()?
            }
            None => vec![StatsAccumulator::try_new(
                dtype,
                stats,
                max_variable_length_statistics_size,
            )?],
        };

        Ok(Self {
            accumulators: Arc::new(Mutex::new(accumulators)),
            ctx: Arc::new(Mutex::new(session.create_execution_ctx())),
        })
    }

    fn process(
        &self,
        chunk: VortexResult<(SequenceId, ArrayRef)>,
    ) -> VortexResult<(SequenceId, ArrayRef)> {
        let (sequence_id, chunk) = chunk?;
        let mut ctx = self.ctx.lock();
        if chunk.dtype().is_struct() {
            let struct_chunk = chunk.clone().execute::<StructArray>(&mut ctx)?;
            for (acc, field) in self
                .accumulators
                .lock()
                .iter_mut()
                .zip_eq(struct_chunk.iter_unmasked_fields())
            {
                acc.push_chunk(field, &mut ctx)?;
            }
        } else {
            self.accumulators.lock()[0].push_chunk(&chunk, &mut ctx)?;
        }
        Ok((sequence_id, chunk))
    }

    /// Returns the aggregated per-column statistics accumulated so far.
    pub fn stats_sets(&self) -> VortexResult<Vec<StatsSet>> {
        self.accumulators
            .lock()
            .iter()
            .map(StatsAccumulator::as_stats_set)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::scalar::ScalarValue;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use super::*;

    const MAX_LEN: usize = 12;

    fn utf8_chunk(values: impl IntoIterator<Item = &'static str>) -> ArrayRef {
        let values = values.into_iter().collect::<Vec<_>>();
        let mut builder =
            VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::NonNullable), values.len());
        for value in values {
            builder.append_value(value);
        }
        builder.finish()
    }

    fn stats_set_for(
        chunks: impl IntoIterator<Item = ArrayRef>,
        stats: &[Stat],
    ) -> VortexResult<StatsSet> {
        let mut ctx = array_session().create_execution_ctx();
        let mut chunks = chunks.into_iter().peekable();
        let dtype = chunks
            .peek()
            .vortex_expect("at least one chunk required")
            .dtype()
            .clone();
        let mut acc = StatsAccumulator::try_new(&dtype, stats, MAX_LEN)?;
        for chunk in chunks {
            acc.push_chunk(&chunk, &mut ctx)?;
        }
        acc.as_stats_set()
    }

    #[rstest]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    fn truncated_accumulated_stats_are_inexact(#[case] dtype: DType) {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = VarBinViewBuilder::with_capacity(dtype, 2);
        builder.append_value("Value to be truncated");
        builder.append_value("Another truncated value");
        let mut acc = StatsAccumulator::try_new(builder.dtype(), &[Stat::Max, Stat::Min], 12)
            .vortex_expect("new stats");
        acc.push_chunk(&builder.finish(), &mut ctx)
            .vortex_expect("push_chunk should succeed for test data");

        let stats = acc
            .as_stats_set()
            .vortex_expect("as_stats_set should succeed for test data");

        assert!(matches!(stats.get(Stat::Min), Precision::Inexact(_)));
        assert!(matches!(stats.get(Stat::Max), Precision::Inexact(_)));
    }

    #[test]
    fn truncated_min_max_are_inexact_bounds() -> VortexResult<()> {
        let stats_set = stats_set_for(
            [
                utf8_chunk(["Value to be truncated", "untruncated"]),
                utf8_chunk(["Another", "wait a minute"]),
            ],
            &[Stat::Min, Stat::Max],
        )?;

        // The min "Another" fits in MAX_LEN bytes and remains exact.
        assert_eq!(
            stats_set.get(Stat::Min),
            Precision::exact(ScalarValue::from("Another"))
        );
        // The max "wait a minute" exceeds MAX_LEN bytes, so the stored value is a truncated
        // upper bound and must be inexact.
        assert_eq!(
            stats_set.get(Stat::Max),
            Precision::inexact(ScalarValue::from("wait a minuu"))
        );
        Ok(())
    }

    #[test]
    fn untruncated_min_max_are_exact() -> VortexResult<()> {
        let stats_set = stats_set_for([utf8_chunk(["short", "values"])], &[Stat::Min, Stat::Max])?;

        assert_eq!(
            stats_set.get(Stat::Min),
            Precision::exact(ScalarValue::from("short"))
        );
        assert_eq!(
            stats_set.get(Stat::Max),
            Precision::exact(ScalarValue::from("values"))
        );
        Ok(())
    }

    #[test]
    fn unrepresentable_upper_bound_is_dropped() -> VortexResult<()> {
        let max_bytes = vec![0xffu8; MAX_LEN + 4];
        let mut builder =
            VarBinViewBuilder::with_capacity(DType::Binary(Nullability::NonNullable), 2);
        builder.append_value(max_bytes.as_slice());
        builder.append_value([0u8, 1].as_slice());

        let stats_set = stats_set_for([builder.finish()], &[Stat::Min, Stat::Max])?;

        // Incrementing a prefix of all-0xff bytes overflows, so no upper bound exists.
        assert!(stats_set.get(Stat::Max).is_absent());
        assert_eq!(
            stats_set.get(Stat::Min),
            Precision::exact(ScalarValue::from(ByteBuffer::copy_from([0u8, 1])))
        );
        Ok(())
    }

    #[test]
    fn min_max_skip_nans_and_count_them() -> VortexResult<()> {
        let stats_set = stats_set_for(
            [
                PrimitiveArray::from_iter([1.0f64, f64::NAN]).into_array(),
                PrimitiveArray::from_iter([-2.0f64, 4.0]).into_array(),
            ],
            &[Stat::Min, Stat::Max, Stat::NaNCount],
        )?;

        assert_eq!(
            stats_set.get(Stat::Min),
            Precision::exact(ScalarValue::from(-2.0f64))
        );
        assert_eq!(
            stats_set.get(Stat::Max),
            Precision::exact(ScalarValue::from(4.0f64))
        );
        assert_eq!(
            stats_set.get(Stat::NaNCount),
            Precision::exact(ScalarValue::from(1u64))
        );
        Ok(())
    }

    #[test]
    fn counts_and_sum_aggregate_across_chunks() -> VortexResult<()> {
        let stats_set = stats_set_for(
            [
                PrimitiveArray::from_option_iter([Some(1i64), None]).into_array(),
                PrimitiveArray::from_option_iter([None::<i64>, Some(3)]).into_array(),
            ],
            &[Stat::Sum, Stat::NullCount],
        )?;

        assert_eq!(
            stats_set.get(Stat::Sum),
            Precision::exact(ScalarValue::from(4i64))
        );
        assert_eq!(
            stats_set.get(Stat::NullCount),
            Precision::exact(ScalarValue::from(2u64))
        );
        Ok(())
    }

    #[test]
    fn all_null_column_has_no_min_max() -> VortexResult<()> {
        let stats_set = stats_set_for(
            [PrimitiveArray::from_option_iter([None::<i32>, None]).into_array()],
            &[Stat::Min, Stat::Max, Stat::NullCount],
        )?;

        assert!(stats_set.get(Stat::Min).is_absent());
        assert!(stats_set.get(Stat::Max).is_absent());
        assert_eq!(
            stats_set.get(Stat::NullCount),
            Precision::exact(ScalarValue::from(2u64))
        );
        Ok(())
    }

    #[rstest]
    #[case::is_sorted(Stat::IsSorted)]
    #[case::is_strict_sorted(Stat::IsStrictSorted)]
    #[case::is_constant(Stat::IsConstant)]
    fn non_aggregatable_stats_are_skipped(#[case] stat: Stat) -> VortexResult<()> {
        let stats_set = stats_set_for([buffer![0i32, 1, 2].into_array()], &[stat])?;
        assert!(stats_set.get(stat).is_absent());
        Ok(())
    }
}
