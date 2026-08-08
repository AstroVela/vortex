// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Nullable execution strategies derived from a concrete row dispatch.

use vortex_mask::Mask;

use crate::dtype::DType;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::SinkResult;

/// The execution policy and output dtype selected by a planning visit.
pub struct BatchPlan {
    /// The non-nullable dtype built by the selected output capability.
    pub output_dtype: DType,

    /// How this concrete dispatch executes nullable rows.
    pub policy: RowPolicy,
}

/// The nullable execution policy derived from one concrete dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowPolicy {
    /// Evaluate all rows and mask the result.
    Dense,

    /// Evaluate all rows, retrying only valid rows if a deferred error is raised.
    DenseWithRetry,

    /// Execute only valid rows, choosing between skip-invalid execution and filtering based on the
    /// mask and decode cost.
    ValidOnly {
        /// Relative per-row decode work that filtering would avoid.
        filtered_decode_cost: usize,
    },
}

impl RowPolicy {
    /// The policy for an infallible owned output.
    pub const fn for_owned_output<Args: ElementTuple>() -> Self {
        if Args::DENSE_SAFE && !Args::DECODE_FALLIBLE {
            Self::Dense
        } else {
            Self::ValidOnly {
                filtered_decode_cost: Args::FILTERED_DECODE_COST,
            }
        }
    }

    /// The policy for an owned output carrying batch-deferred failure evidence.
    pub const fn for_deferred_output<Args: ElementTuple>() -> Self {
        if Args::DENSE_SAFE && !Args::DECODE_FALLIBLE {
            Self::DenseWithRetry
        } else {
            Self::ValidOnly {
                filtered_decode_cost: Args::FILTERED_DECODE_COST,
            }
        }
    }

    /// The policy one concrete dispatch executes nullable rows under.
    ///
    /// This deliberately ignores [`OutputSink::SUPPORTS_SKIPPED_ROWS`]. Batch execution tries
    /// [`reduce_encoded`](crate::scalar_fn::RowFn::reduce_encoded) against the original arrays
    /// before it tries the sink or filters the inputs. Skipping that probe can change the result of
    /// an encoding-aware function.
    ///
    /// [`OutputSink::SUPPORTS_SKIPPED_ROWS`]: crate::scalar_fn::OutputSink::SUPPORTS_SKIPPED_ROWS
    pub const fn for_sink<Args: ElementTuple, ApplyResult: SinkResult>() -> Self {
        if Args::DENSE_SAFE && !Args::DECODE_FALLIBLE && !ApplyResult::FALLIBLE {
            if ApplyResult::DEFERRED {
                Self::DenseWithRetry
            } else {
                Self::Dense
            }
        } else {
            Self::ValidOnly {
                filtered_decode_cost: Args::FILTERED_DECODE_COST,
            }
        }
    }
}

/// Minimum surviving-row fractions for skipping when filtering avoids per-row decode work.
/// The thresholds distinguish one costly decode from multiple costly decodes.
const ONE_DECODE_SKIP_MIN_SURVIVING_FRACTION: f64 = 0.50;
const MULTI_DECODE_SKIP_MIN_SURVIVING_FRACTION: f64 = 0.85;

/// Whether skipping invalid rows should be preferred over filtering for a mixed mask.
pub(super) fn skipping_beats_filtering(filtered_decode_cost: usize, valid: &Mask) -> bool {
    if filtered_decode_cost == 0 {
        return true;
    }

    let minimum = if filtered_decode_cost == 1 {
        ONE_DECODE_SKIP_MIN_SURVIVING_FRACTION
    } else {
        MULTI_DECODE_SKIP_MIN_SURVIVING_FRACTION
    };

    valid.true_count() as f64 >= valid.len() as f64 * minimum
}
