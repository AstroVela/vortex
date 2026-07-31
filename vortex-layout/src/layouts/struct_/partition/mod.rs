// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Partitioning of expressions over the children of a [`StructLayout`](super::StructLayout).
//!
//! A struct layout stores `n + 1` children: the struct's own validity bitmap (only when the
//! struct is nullable) followed by one child per field. Expressions must therefore be partitioned
//! into `n + 1` slots, so that the struct's own nullability can be evaluated — and re-applied —
//! independently of its fields.
//!
//! ## How it works
//!
//! The splitting itself is [`vortex_array`]'s, driven through the [`Partitioner`] trait. This
//! module supplies the three struct-specific stages.
//!
//! 1. **Flattening.** Every access to the root scope is rewritten into an access of the *flat
//!    scope*: a non-nullable struct holding the validity slot plus one slot per field, mirroring
//!    the layout's children. The struct's nullability is made explicit in the expression itself,
//!    e.g. `$.a` over a nullable struct `{a, b}?` becomes `mask($.1, $.0)`. See
//!    [`StructPartitioner::flatten`].
//! 2. **Splitting.** The flattened expression is split into one sub-expression per slot, plus a
//!    root expression that re-assembles them. This mirrors the splitter in `vortex-array`.
//! 3. **Stepping down.** Each partition is rewritten from the flat scope into the scope of the
//!    child layout that will evaluate it, by replacing `$.<slot>` with `$`.
//!
//! Stage 1 is what makes stages 2 and 3 sound: afterwards, the *only* way an expression can touch
//! the root scope is `get_item(<slot>, root())`, so stepping down is a total rewrite rather than a
//! best-effort substitution.
//!
//! ## Naming the slots
//!
//! The flat scope addresses each child by its [`StructSlot`] index rendered as a field name, not
//! by the field's own name. Field names are arbitrary user data — any name reserved for validity
//! could be taken by a real field — whereas slot indices are assigned by the layout, so no user
//! field name ever appears in the flat scope and a clash is impossible by construction. It also
//! means a slot resolves to its child reader in `O(1)`, with no name lookup.

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::is_root;
use vortex_array::expr::lit;
use vortex_array::expr::mask;
use vortex_array::expr::not;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::transform::PartitionedExpr;
use vortex_array::expr::transform::Partitioner;
use vortex_array::expr::transform::partition_annotated;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

/// One child of a struct layout, identified by its logical slot index.
///
/// Slot 0 is the struct's own validity bitmap (present only when the struct is nullable) and slot
/// `i + 1` is field `i`. This is the same numbering as
/// [`slot_to_child`](crate::layouts::struct_::Struct::slot_to_child), so a slot resolves to its
/// child reader without any name lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StructSlot(usize);

impl StructSlot {
    /// The slot holding the struct's own validity.
    pub(crate) const VALIDITY: Self = Self(0);

    /// The slot holding field `index` of the struct.
    pub(crate) const fn field(index: usize) -> Self {
        Self(index + 1)
    }

    /// The logical slot index, as understood by the layout.
    pub(crate) const fn index(self) -> usize {
        self.0
    }

    /// Whether this is the validity slot.
    pub(crate) const fn is_validity(self) -> bool {
        self.0 == 0
    }
}

/// The slot's name in the flat scope. Slots are named by index, so a name can never clash with a
/// field of the struct.
impl Display for StructSlot {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The result of partitioning an expression over the children of a struct layout.
#[derive(Clone)]
pub(crate) enum StructPartitioned {
    /// The expression can be evaluated entirely by a single child of the struct layout.
    Single(StructSlot, Expression),
    /// The expression spans multiple children and must be re-assembled by the root expression.
    Multi(Arc<PartitionedExpr<StructSlot>>),
}

/// Partitions expressions over the children of a struct layout.
pub(crate) struct StructPartitioner {
    /// The dtype of the struct layout: the scope that incoming expressions are written against,
    /// before flattening.
    layout_dtype: DType,
    /// The scope that expressions are flattened into: a non-nullable struct with one field per
    /// child of the struct layout, named by [`StructSlot`] index.
    flat_scope: DType,
    /// Whether the struct is nullable, and therefore has a validity slot.
    nullable: bool,
    /// Field name to field index, built only for structs wide enough that the linear scan in
    /// [`StructFields::find`] would be worth avoiding.
    field_lookup: Option<HashMap<FieldName, usize>>,
}

impl StructPartitioner {
    /// Create a partitioner for a struct layout of the given `dtype`.
    pub(crate) fn new(dtype: &DType) -> VortexResult<Self> {
        let fields = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Struct layout dtype must be a struct, got {dtype}"))?;

        let nullable = dtype.is_nullable();

        let mut names: Vec<FieldName> = Vec::with_capacity(fields.nfields() + 1);
        let mut dtypes: Vec<DType> = Vec::with_capacity(fields.nfields() + 1);
        if nullable {
            names.push(slot_name(StructSlot::VALIDITY));
            dtypes.push(DType::Bool(Nullability::NonNullable));
        }
        names.extend((0..fields.nfields()).map(|idx| slot_name(StructSlot::field(idx))));
        dtypes.extend(fields.fields());

        // NOTE: This number is arbitrary and likely depends on the longest common prefix of the
        // field names.
        let field_lookup = (fields.nfields() > 80).then(|| {
            fields
                .names()
                .iter()
                .enumerate()
                .map(|(idx, name)| (name.clone(), idx))
                .collect()
        });

        Ok(Self {
            layout_dtype: dtype.clone(),
            flat_scope: DType::Struct(
                StructFields::new(names.into(), dtypes),
                Nullability::NonNullable,
            ),
            nullable,
            field_lookup,
        })
    }

    /// The slot addressed by a field of the flat scope, if it is one.
    fn name_slot(&self, name: &FieldName) -> Option<StructSlot> {
        let slot = StructSlot(name.as_ref().parse::<usize>().ok()?);
        self.contains(slot).then_some(slot)
    }

    /// Whether this layout actually has the given slot.
    fn contains(&self, slot: StructSlot) -> bool {
        if slot.is_validity() {
            self.nullable
        } else {
            slot.index() <= self.fields().nfields()
        }
    }

    /// The index of a field of the struct, by name.
    fn field_index(&self, name: &FieldName) -> Option<usize> {
        match &self.field_lookup {
            Some(lookup) => lookup.get(name).copied(),
            None => self.fields().find(name),
        }
    }

    /// An expression referencing a slot within the flat scope.
    fn slot_expr(&self, slot: StructSlot) -> Expression {
        col(slot_name(slot))
    }

    /// An expression referencing the struct's validity within the flat scope.
    fn validity_expr(&self) -> Option<Expression> {
        self.nullable.then(|| self.slot_expr(StructSlot::VALIDITY))
    }

    /// Partition `expr` (defined over [`Self::layout_dtype`]) over the children of the struct layout.
    pub(crate) fn partition(&self, expr: Expression) -> VortexResult<StructPartitioned> {
        let partitioned = partition_annotated(self, expr, |expr: &Expression| self.annotate(expr))?;

        // A single partition whose root is exactly the partition itself can be delegated straight
        // to the child reader, skipping re-assembly.
        if partitioned.partitions.len() == 1
            && partitioned.root == col(partitioned.partition_names[0].clone())
        {
            return Ok(StructPartitioned::Single(
                partitioned.partition_annotations[0],
                partitioned.partitions[0].clone(),
            ));
        }

        Ok(StructPartitioned::Multi(Arc::new(partitioned)))
    }

    /// Stage 1: rewrite every access of the root scope into an access of the flat scope.
    ///
    /// The catch-all rule replaces `root()` itself with a reconstruction of the struct from its
    /// children, which is always correct. The remaining rules exist so that expressions that only
    /// need *some* of the children don't end up reading all of them.
    fn flatten_scope(&self, expr: Expression) -> VortexResult<Expression> {
        let replaced = |value: Expression| {
            // The replacement re-introduces `get_item(.., root())` nodes addressing the flat
            // scope, so we must not recurse into it.
            Ok(Transformed {
                value,
                order: TraversalOrder::Skip,
                changed: true,
            })
        };

        Ok(expr
            .transform_down(|node| {
                // `$` — the struct itself. Rebuild it from all of its children.
                if is_root(&node) {
                    return replaced(self.reconstruct(0..self.fields().nfields()));
                }

                // `$.a` — a field of the struct. `get_item` intersects the struct's validity with
                // the field's, which the `mask` makes explicit.
                //
                // An unknown field name is left alone: the `root` rule then rebuilds the struct
                // underneath it, so the missing field is reported against the original scope.
                if let Some(field_name) = node.as_opt::<GetItem>()
                    && is_root(node.child(0))
                    && let Some(field_index) = self.field_index(field_name)
                {
                    let field = self.slot_expr(StructSlot::field(field_index));
                    return replaced(match self.validity_expr() {
                        Some(validity) => mask(field, validity),
                        None => field,
                    });
                }

                // `$.{a, b}` — a selection of fields. Unlike `get_item`, `select` keeps the
                // struct's nullability at the struct level rather than pushing it into the fields.
                if let Some(selection) = node.as_opt::<Select>()
                    && is_root(node.child(0))
                {
                    let included = selection.normalize_to_included_fields(self.fields().names())?;
                    let indices: Vec<usize> = included
                        .iter()
                        .map(|name| {
                            self.field_index(name).ok_or_else(|| {
                                vortex_err!("Field {name} not found in {}", self.layout_dtype)
                            })
                        })
                        .collect::<VortexResult<_>>()?;
                    return replaced(self.reconstruct(indices));
                }

                // `is_null($)` / `is_not_null($)` — the struct's own validity, which is exactly
                // what the validity child holds.
                if node.is::<IsNull>() && is_root(node.child(0)) {
                    return replaced(match self.validity_expr() {
                        Some(validity) => not(validity),
                        None => lit(false),
                    });
                }
                if node.is::<IsNotNull>() && is_root(node.child(0)) {
                    return replaced(match self.validity_expr() {
                        Some(validity) => validity,
                        None => lit(true),
                    });
                }

                Ok(Transformed::no(node))
            })?
            .into_inner())
    }

    /// Rebuild a (possibly nullable) struct of the given fields from the flat scope. The fields
    /// are addressed by slot, but keep their own names in the rebuilt struct.
    fn reconstruct(&self, field_indices: impl IntoIterator<Item = usize>) -> Expression {
        let packed = pack(
            field_indices.into_iter().map(|idx| {
                let name = self
                    .fields()
                    .field_name(idx)
                    .vortex_expect("Field index is in bounds")
                    .clone();
                (name, self.slot_expr(StructSlot::field(idx)))
            }),
            Nullability::NonNullable,
        );
        match self.validity_expr() {
            Some(validity) => mask(packed, validity),
            None => packed,
        }
    }

    fn fields(&self) -> &StructFields {
        self.layout_dtype
            .as_struct_fields_opt()
            .vortex_expect("Struct layout dtype must be a struct")
    }

    /// Annotate an expression node with the slots it accesses directly.
    ///
    /// After flattening, the only way to access the root scope is `get_item(<slot>, root())`, so
    /// a bare `root()` should be unreachable. We still annotate it with every slot so that it can
    /// never be mistaken for a single-slot expression.
    fn annotate(&self, expr: &Expression) -> Vec<StructSlot> {
        if let Some(field_name) = expr.as_opt::<GetItem>()
            && is_root(expr.child(0))
            && let Some(slot) = self.name_slot(field_name)
        {
            return vec![slot];
        }
        if is_root(expr) {
            return self
                .nullable
                .then_some(StructSlot::VALIDITY)
                .into_iter()
                .chain((0..self.fields().nfields()).map(StructSlot::field))
                .collect();
        }
        vec![]
    }
}

impl Partitioner for StructPartitioner {
    type Slot = StructSlot;

    fn scope(&self) -> &DType {
        &self.flat_scope
    }

    fn slot_name(&self, slot: &Self::Slot) -> FieldName {
        slot_name(*slot)
    }

    fn flatten(&self, expr: Expression) -> VortexResult<Expression> {
        // Simplifying here, rather than before flattening, keeps the slot accesses this stage
        // introduces from being split apart by the annotator any more finely than necessary.
        self.flatten_scope(expr)?
            .optimize_recursive(&self.flat_scope)
    }

    fn step_into(&self, expr: Expression, slot: &Self::Slot) -> VortexResult<Expression> {
        step_into(expr, *slot)
    }

    /// A slot referenced once is read directly by the root expression: projecting the whole
    /// struct asks each child for its own root rather than for a one-field pack of it.
    fn unwrap_single_sub_expression(&self) -> bool {
        true
    }

    /// The struct's validity is referenced once per field access, so sharing it matters.
    fn deduplicate_sub_expressions(&self) -> bool {
        true
    }
}

/// Stage 3: rewrite a partition from the flat scope into the scope of `slot`'s child layout, by
/// replacing `$.<slot>` with `$`.
fn step_into(expr: Expression, slot: StructSlot) -> VortexResult<Expression> {
    let slot_name = slot_name(slot);
    Ok(expr
        .transform_down(|node| {
            if let Some(field_name) = node.as_opt::<GetItem>()
                && is_root(node.child(0))
            {
                if *field_name != slot_name {
                    vortex_bail!(
                        "Partition for slot {slot_name} unexpectedly accesses slot {field_name}"
                    );
                }
                return Ok(Transformed {
                    value: root(),
                    order: TraversalOrder::Skip,
                    changed: true,
                });
            }
            if is_root(&node) {
                vortex_bail!("Partition for slot {slot_name} accesses the struct scope directly");
            }
            Ok(Transformed::no(node))
        })?
        .into_inner())
}

/// The slot's field name in the flat scope: its index, which no user field name can collide with.
fn slot_name(slot: StructSlot) -> FieldName {
    FieldName::from(slot.to_string())
}

#[cfg(test)]
mod tests;
