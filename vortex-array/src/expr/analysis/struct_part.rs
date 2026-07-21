// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect;

use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::Expression;
use crate::expr::analysis::AnnotationFn;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::fns::root::Root;
use crate::scalar_fn::fns::select::Select;

/// A coordinate of a struct scope: either one of its fields, or — for a nullable struct —
/// the struct's own validity.
///
/// A nullable struct with `n` fields decomposes into `n + 1` coordinates. Annotating
/// expressions with `StructPart` instead of a bare [`FieldName`] lets the validity
/// coordinate be routed like any field, and cannot collide with a real field that happens
/// to be named "validity".
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StructPart {
    /// A top-level field of the struct scope.
    Field(FieldName),
    /// The validity of the struct scope itself, referenced as `is_not_null($)`.
    Validity,
}

impl Display for StructPart {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StructPart::Field(name) => write!(f, "{name}"),
            StructPart::Validity => write!(f, "$validity"),
        }
    }
}

impl From<StructPart> for FieldName {
    fn from(part: StructPart) -> Self {
        match part {
            StructPart::Field(name) => name,
            // NOTE: this name is only used in the synthetic scope that recombines
            // partitions. It is not distinguishable from a real field literally named
            // "$validity"; routing must use the `StructPart` annotation, never this name.
            StructPart::Validity => FieldName::from("$validity"),
        }
    }
}

/// Returns the free [`StructPart`] coordinates for an expression node.
///
/// This extends [`super::make_free_field_annotator`] with the validity coordinate of a
/// nullable scope:
///
/// - **`is_not_null(root())`**: Returns `[Validity]` — the canonical reference to the
///   scope's validity coordinate (it is also what [`crate::scalar_fn::fns::mask::Mask`]'s
///   validity derivation produces for the expanded root, see
///   [`crate::expr::transform::replace_root_scope`]).
/// - **[`Select`] / [`GetItem`] on [`Root`]**: Returns the referenced fields, as
///   `Field(..)`. These reference the field coordinates *without* the scope validity.
/// - **[`Root`]**: Returns every field, plus `Validity` if the scope is nullable
///   (conservative over-approximation: a bare `$` observes all coordinates).
/// - **Everything else**: Returns empty (annotations aggregate from children).
pub fn make_struct_part_annotator(
    scope: &StructFields,
    scope_nullability: Nullability,
) -> impl AnnotationFn<Annotation = StructPart> {
    move |expr: &Expression| {
        if expr.is::<IsNotNull>() && expr.child(0).is::<Root>() {
            if scope_nullability.is_nullable() {
                return vec![StructPart::Validity];
            }
            // A non-nullable scope has no validity coordinate; `is_not_null($)` is a
            // constant. Leave it unannotated so it does not force a partition.
            return vec![];
        } else if let Some(selection) = expr.as_opt::<Select>() {
            if expr.child(0).is::<Root>() {
                return selection
                    .normalize_to_included_fields(scope.names())
                    .vortex_expect("Select fields must be valid for scope")
                    .into_iter()
                    .map(StructPart::Field)
                    .collect();
            }
        } else if let Some(field_name) = expr.as_opt::<GetItem>() {
            if expr.child(0).is::<Root>() {
                return vec![StructPart::Field(field_name.clone())];
            }
        } else if expr.is::<Root>() {
            let mut parts: Vec<StructPart> = scope
                .names()
                .iter()
                .cloned()
                .map(StructPart::Field)
                .collect();
            if scope_nullability.is_nullable() {
                parts.push(StructPart::Validity);
            }
            return parts;
        }

        vec![]
    }
}
