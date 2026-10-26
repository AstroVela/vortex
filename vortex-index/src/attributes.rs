// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Backend-independent structured attribute predicates for vector prefiltering.
//!
//! ```
//! use bytes::Bytes;
//! use vortex_error::{VortexResult, vortex_err};
//! use vortex_index::{Index, RowFilter, SearchHit, VectorIndex, VectorSearchOptions};
//! use vortex_index::attributes::{AttributePredicate, AttributeValue};
//!
//! # async fn filtered_search(attributes: &dyn Index, vectors: &dyn VectorIndex,
//! # visibility: &RowFilter, query: &[f32], options: &VectorSearchOptions)
//! # -> VortexResult<Vec<SearchHit>> {
//! let allowed = attributes.as_attributes()
//!     .ok_or_else(|| vortex_err!("Attribute capability unavailable"))?
//!     .filter(&AttributePredicate::Eq(
//!         "category".into(), AttributeValue::String("tech".into()),
//!     ), visibility).await?;
//! vectors.search(query, options, &allowed).await
//! # }
//! ```

use std::cmp::Ordering;

use serde::Deserialize;
use serde::Serialize;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// An indexed scalar. Missing fields have no posting; null is not a scalar.
/// Values of different variants are distinct, with no implicit coercion.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AttributeValue {
    /// Boolean value.
    Bool(bool),
    /// Signed 64-bit integer.
    Int(i64),
    /// Unsigned 64-bit integer.
    UInt(u64),
    /// Finite Float64 value. Use `FloatValue::try_from` to validate a float.
    Float(FloatValue),
    /// UTF-8 string, ordered lexicographically by bytes.
    String(String),
}

/// Typed predicate evaluated by an attribute index before vector retrieval.
#[derive(Debug, Clone)]
pub enum AttributePredicate {
    /// Exact typed equality.
    Eq(String, AttributeValue),
    /// Equality to any listed value; an empty list matches nothing.
    In(String, Vec<AttributeValue>),
    /// Inclusive typed range. Bounds must have the same type and be ordered.
    Range(String, AttributeValue, AttributeValue),
    /// Field is present, regardless of value.
    Exists(String),
    /// Intersection; an empty conjunction matches every indexed row.
    And(Vec<Self>),
    /// Union; an empty disjunction matches nothing.
    Or(Vec<Self>),
}

/// Finite Float64 attribute with numeric ordering and a single representation of zero.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct FloatValue(f64);

impl TryFrom<f64> for FloatValue {
    type Error = VortexError;

    fn try_from(value: f64) -> VortexResult<Self> {
        if !value.is_finite() {
            vortex_bail!("Float attributes must be finite");
        }
        Ok(Self(if value == 0.0 { 0.0 } else { value }))
    }
}

impl From<FloatValue> for f64 {
    fn from(value: FloatValue) -> Self {
        value.0
    }
}

impl Eq for FloatValue {}
impl PartialOrd for FloatValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FloatValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}
