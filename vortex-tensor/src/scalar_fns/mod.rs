// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar function expressions defined on tensor and tensor-like extension types.
//!
//! Each child module owns one expression. [`NormMode`] defines whether norm-based expressions
//! measure physical coordinates or trust normalized dtype and encoding claims.

use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

pub mod cosine_similarity;
pub mod inner_product;
pub mod l2_norm;
pub mod l2_normalize;

/// Controls whether norm-based functions may trust normalized-value claims.
///
/// [`Normalized`] encodings claim that their direction child is normalized, while [`UnitVector`]
/// dtypes claim that each non-null value is unit length. Ordinary tensors carry neither claim and
/// use their physical coordinates in both modes.
///
/// [`Normalized`]: crate::encodings::normalized::Normalized
/// [`UnitVector`]: crate::unit_vector::UnitVector
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NormMode {
    /// Compute physical direction norms instead of assuming that they are exactly one.
    Exact,

    /// Trust normalized encoding and dtype claims and omit their norm computation.
    ///
    /// Checked arrays satisfy the encoding's documented tolerance. Unchecked or lossy arrays do
    /// not carry an error bound, so this mode can produce values outside the mathematical range.
    AssumeNormalized,
}

impl NormMode {
    pub(crate) fn assumes_normalized(self) -> bool {
        matches!(self, Self::AssumeNormalized)
    }

    pub(crate) fn from_serialized(assume_normalized: Option<bool>) -> Self {
        match assume_normalized {
            Some(false) => Self::Exact,
            Some(true) | None => Self::AssumeNormalized,
        }
    }

    pub(crate) fn serialize(self) -> Vec<u8> {
        NormModeMetadata {
            assume_normalized: Some(self.assumes_normalized()),
        }
        .encode_to_vec()
    }

    pub(crate) fn deserialize(metadata: &[u8]) -> VortexResult<Self> {
        let metadata = NormModeMetadata::decode(metadata)
            .map_err(|error| vortex_err!("Failed to decode NormMode metadata: {error}"))?;

        Ok(Self::from_serialized(metadata.assume_normalized))
    }
}

impl Display for NormMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact => f.write_str("exact"),
            Self::AssumeNormalized => f.write_str("assume_normalized"),
        }
    }
}

#[derive(Clone, prost::Message)]
struct NormModeMetadata {
    /// Whether execution trusts normalized encoding evidence.
    #[prost(bool, optional, tag = "1")]
    assume_normalized: Option<bool>,
}
