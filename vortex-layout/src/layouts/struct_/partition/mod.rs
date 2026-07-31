// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Partitioning of expressions over the children of a [`StructLayout`](super::StructLayout).
//!
//! A struct layout stores `n + 1` children: the struct's own validity bitmap (only when the
//! struct is nullable) followed by one child per field. Expressions must therefore be partitioned
//! into `n + 1` slots, so that the struct's own nullability can be evaluated — and re-applied —
//! independently of its fields.
//!
//! ## Why this is a copy
//!
//! This is a specialisation of [`vortex_array::expr::transform::partition`], which only knows how
//! to partition over the fields of a struct. Treating validity as just another partition requires
//! changes to every stage of the pipeline: the scope flattening, the annotator, the splitter, and
//! the "step down" into a child's scope. Rather than generalise the shared implementation (and
//! risk regressing its other callers) we keep a second copy here.
//!
//! TODO(joe): unify this with [`vortex_array::expr::transform::partition`].
//!
//! ## How it works
//!
//! Partitioning happens in three stages.
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
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::analysis::Annotations;
use vortex_array::expr::analysis::descendent_annotations;
use vortex_array::expr::col;
use vortex_array::expr::get_item;
use vortex_array::expr::is_root;
use vortex_array::expr::lit;
use vortex_array::expr::mask;
use vortex_array::expr::not;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::transform::PartitionedExpr;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::NodeRewriter;
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
    /// The dtype of the struct layout; the scope of any incoming expression.
    scope: DType,
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
            scope: dtype.clone(),
            flat_scope: DType::Struct(
                StructFields::new(names.into(), dtypes),
                Nullability::NonNullable,
            ),
            nullable,
            field_lookup,
        })
    }

    /// Each slot may hold several sub-expressions, so each one needs a unique name within the
    /// pack that makes up its partition.
    fn sub_expression_name(&self, slot: StructSlot, idx: usize) -> FieldName {
        FieldName::from(format!("{slot}_{idx}"))
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

    /// Partition `expr` (defined over [`Self::scope`]) over the children of the struct layout.
    pub(crate) fn partition(&self, expr: Expression) -> VortexResult<StructPartitioned> {
        let expr = self.flatten(expr)?.optimize_recursive(&self.flat_scope)?;

        let annotations = descendent_annotations(&expr, |expr: &Expression| self.annotate(expr));
        let partitioned = self.split(expr.clone(), annotations)?;

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
    fn flatten(&self, expr: Expression) -> VortexResult<Expression> {
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
                                vortex_err!("Field {name} not found in {}", self.scope)
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
        self.scope
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

    /// Stages 2 and 3: split the flattened expression into one partition per slot, then step each
    /// partition down into the scope of the child that will evaluate it.
    fn split(
        &self,
        expr: Expression,
        annotations: Annotations<'_, StructSlot>,
    ) -> VortexResult<PartitionedExpr<StructSlot>> {
        let mut splitter = SlotSplitter {
            partitioner: self,
            annotations: &annotations,
            slots: Vec::new(),
            sub_expressions: Vec::new(),
        };
        let root_expr = expr.rewrite(&mut splitter)?.value;

        let mut partitions = Vec::with_capacity(splitter.slots.len());
        let mut partition_names = Vec::with_capacity(splitter.slots.len());
        let mut partition_dtypes = Vec::with_capacity(splitter.slots.len());
        let mut single_slots = HashMap::new();

        for (&slot, exprs) in splitter.slots.iter().zip(splitter.sub_expressions.iter()) {
            // A slot with a single sub-expression doesn't need to be packed; the root expression
            // references it directly as `$.<slot>`.
            let partition = if let [only] = exprs.as_slice() {
                single_slots.insert(slot_name(slot), self.sub_expression_name(slot, 0));
                only.clone()
            } else {
                pack(
                    exprs
                        .iter()
                        .enumerate()
                        .map(|(idx, expr)| (self.sub_expression_name(slot, idx), expr.clone())),
                    Nullability::NonNullable,
                )
            };

            let partition = partition.optimize_recursive(&self.flat_scope)?;
            partition_dtypes.push(partition.return_dtype(&self.flat_scope)?);
            partition_names.push(slot_name(slot));
            partitions.push(step_into(partition, slot)?);
        }

        let partition_names = FieldNames::from(partition_names);
        let root_scope = DType::Struct(
            StructFields::new(partition_names.clone(), partition_dtypes.clone()),
            Nullability::NonNullable,
        );
        let root_expr = unpack_single(root_expr, &single_slots);

        Ok(PartitionedExpr {
            root: root_expr.optimize_recursive(&root_scope)?,
            partitions: partitions.into_boxed_slice(),
            partition_names,
            partition_dtypes: partition_dtypes.into_boxed_slice(),
            partition_annotations: splitter.slots.into_boxed_slice(),
        })
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

/// Splits an expression into sub-expressions, each of which accesses exactly one slot.
struct SlotSplitter<'a> {
    partitioner: &'a StructPartitioner,
    annotations: &'a Annotations<'a, StructSlot>,
    /// The slots encountered so far, in the order they were first encountered.
    slots: Vec<StructSlot>,
    /// The sub-expressions of each slot in [`Self::slots`], parallel to it.
    sub_expressions: Vec<Vec<Expression>>,
}

impl SlotSplitter<'_> {
    /// Record `expr` as a sub-expression of `slot`, returning the expression that reads its
    /// result back out of the root scope.
    fn push(&mut self, slot: StructSlot, expr: Expression) -> Expression {
        let slot_idx = self
            .slots
            .iter()
            .position(|s| *s == slot)
            .unwrap_or_else(|| {
                self.slots.push(slot);
                self.sub_expressions.push(Vec::new());
                self.slots.len() - 1
            });

        let sub_exprs = &mut self.sub_expressions[slot_idx];
        // Identical sub-expressions are evaluated once.
        let idx = sub_exprs
            .iter()
            .position(|e| e == &expr)
            .unwrap_or_else(|| {
                sub_exprs.push(expr);
                sub_exprs.len() - 1
            });

        // The partition is only packed if the slot has more than one sub-expression; this is
        // fixed up once splitting is complete and the counts are known.
        get_item(
            self.partitioner.sub_expression_name(slot, idx),
            col(slot_name(slot)),
        )
    }
}

impl NodeRewriter for SlotSplitter<'_> {
    type NodeTy = Expression;

    fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        match self.annotations.get(&node) {
            // If this expression only accesses a single slot, it becomes a partition.
            Some(slots) if slots.len() == 1 => {
                let slot = *slots.iter().next().vortex_expect("expected one slot");
                let value = self.push(slot, node.clone());
                Ok(Transformed {
                    value,
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            }

            // Otherwise, continue traversing.
            _ => Ok(Transformed::no(node)),
        }
    }
}

/// Rewrite the root expression's references to slots that ended up with a single sub-expression,
/// replacing `$.<slot>.<slot>_0` with `$.<slot>`.
fn unpack_single(
    root_expr: Expression,
    single_slots: &HashMap<FieldName, FieldName>,
) -> Expression {
    root_expr
        .transform_down(|node| {
            if let Some(field_name) = node.as_opt::<GetItem>()
                && let Some(slot_field) = node.child(0).as_opt::<GetItem>()
                && is_root(node.child(0).child(0))
                && let Some(expected) = single_slots.get(slot_field)
                && expected == field_name
            {
                return Ok(Transformed {
                    value: col(slot_field.clone()),
                    order: TraversalOrder::Skip,
                    changed: true,
                });
            }
            Ok(Transformed::no(node))
        })
        .vortex_expect("unpack_single is infallible")
        .into_inner()
}

#[cfg(test)]
mod tests;
