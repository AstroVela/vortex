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
use std::collections::BTreeMap;
use std::collections::BTreeSet;

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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

/// How one scalar field supplies construction blocks. Fields are independent;
/// implementations must not enumerate combinations of predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeBlockKind {
    /// One complete posting list per distinct value, including booleans and strings.
    Equality,
    /// Ordered numeric windows with approximately 50% overlap. Equal values are
    /// never split; actual window sizes can exceed the target size.
    NumericRange,
}

/// Bounded deterministic construction plan requested from a scalar index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttributeBlockRequest {
    /// Indexed fields and their grouping policy, in deterministic name order.
    pub fields: BTreeMap<String, AttributeBlockKind>,
    /// Only these snapshot files participate; sorted, unique and fully covered.
    pub covered_files: Vec<u64>,
    /// Smaller blocks are omitted because they provide little graph connectivity.
    pub min_rows: u32,
    /// Desired numeric window size; equality groups are never split.
    pub target_rows: u32,
    /// Reject an oversized block instead of truncating its postings.
    pub max_block_rows: u32,
    /// Maximum distinct terms per field and maximum output blocks overall.
    pub max_blocks: u32,
    /// Maximum input postings and maximum output row memberships, counted
    /// separately across all fields. Bounds buffered rows, not total RSS.
    pub max_memberships: u64,
}

impl AttributeBlockRequest {
    /// Check configuration and the physical coverage against a source snapshot.
    pub fn validate(&self, snapshot: &crate::Snapshot) -> VortexResult<()> {
        snapshot.validate()?;
        if self.fields.is_empty()
            || self.fields.len() > 16
            || self
                .fields
                .keys()
                .any(|name| name.is_empty() || name.len() > 1024)
            || self.covered_files.is_empty()
            || self.covered_files.windows(2).any(|pair| pair[0] >= pair[1])
            || self
                .covered_files
                .iter()
                .any(|id| snapshot.file(*id).is_none())
            || self.min_rows < 2
            || self.target_rows < self.min_rows
            || self.max_block_rows < self.target_rows
            || self.max_blocks == 0
            || self.max_memberships < u64::from(self.max_block_rows)
        {
            vortex_bail!("Invalid attribute block request");
        }
        Ok(())
    }
}

/// Complete physical rows for one equality group or inclusive numeric window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttributeBlock {
    /// Stable identity of the group within a plan (an Eq or Range predicate).
    pub predicate: AttributePredicate,
    /// Sorted, distinct physical addresses. Null values have no posting.
    pub rows: Vec<crate::RowAddress>,
}

/// Snapshot-bound block plan. Construction uses physical rows, without applying
/// query-time visibility or tombstones; those are mandatory at search time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttributeBlockPlan {
    /// Exact source snapshot used to enumerate all postings.
    pub snapshot: crate::Snapshot,
    /// Parameters and coverage that produced this plan.
    pub request: AttributeBlockRequest,
    /// Deterministic field/value order, with no repeated block identities.
    pub blocks: Vec<AttributeBlock>,
}

impl AttributeBlockPlan {
    /// Validate binding, unique block identities, row ordering and all output
    /// budgets before translating physical addresses into native node labels.
    pub fn validate(&self) -> VortexResult<()> {
        self.request.validate(&self.snapshot)?;
        if self.blocks.len() > self.request.max_blocks as usize {
            vortex_bail!("Attribute plan exceeds block limit");
        }
        let mut identities = BTreeSet::new();
        let mut memberships = 0u64;
        for block in &self.blocks {
            let (name, lower, upper, kind) = match &block.predicate {
                AttributePredicate::Eq(name, value) => {
                    (name, value, value, AttributeBlockKind::Equality)
                }
                AttributePredicate::Range(name, lower, upper)
                    if matches!(
                        lower,
                        AttributeValue::Int(_) | AttributeValue::UInt(_) | AttributeValue::Float(_)
                    ) && std::mem::discriminant(lower) == std::mem::discriminant(upper)
                        && lower <= upper =>
                {
                    (name, lower, upper, AttributeBlockKind::NumericRange)
                }
                _ => vortex_bail!("Attribute block must be an equality or numeric range"),
            };
            if self.request.fields.get(name) != Some(&kind)
                || !identities.insert((name, lower, upper))
                || block.rows.len() < self.request.min_rows as usize
                || block.rows.len() > self.request.max_block_rows as usize
                || block.rows.len() as u64 > self.request.max_memberships - memberships
                || block.rows.windows(2).any(|pair| pair[0] >= pair[1])
            {
                vortex_bail!("Invalid attribute block identity, rows or budget");
            }
            memberships += block.rows.len() as u64;
            for row in &block.rows {
                self.snapshot.validate_row(*row)?;
                if self
                    .request
                    .covered_files
                    .binary_search(&row.file_id)
                    .is_err()
                {
                    vortex_bail!("Attribute block row outside requested coverage");
                }
            }
        }
        Ok(())
    }
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
