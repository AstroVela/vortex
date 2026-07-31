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
//!    e.g. `$.a` over a nullable struct becomes `mask($.a, $.__validity)`. See
//!    [`StructPartitioner::flatten`].
//! 2. **Splitting.** The flattened expression is split into one sub-expression per slot, plus a
//!    root expression that re-assembles them. This mirrors the splitter in `vortex-array`.
//! 3. **Stepping down.** Each partition is rewritten from the flat scope into the scope of the
//!    child layout that will evaluate it, by replacing `$.<slot>` with `$`.
//!
//! Stage 1 is what makes stages 2 and 3 sound: afterwards, the *only* way an expression can touch
//! the root scope is `get_item(<slot>, root())`, so stepping down is a total rewrite rather than a
//! best-effort substitution.

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

/// The prefix used to name the validity slot of a struct layout. A suffix is appended if needed
/// to avoid colliding with a field of the struct.
const VALIDITY_SLOT_PREFIX: &str = "__validity";

/// One child of a struct layout: either the struct's own validity, or one of its fields.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StructSlot {
    /// The struct's own validity bitmap, present only when the struct is nullable.
    Validity,
    /// A field of the struct.
    Field(FieldName),
}

/// A human-readable label for the slot, used in error messages. This is *not* the slot's name in
/// the flat scope — see [`StructPartitioner::slot_name`] for that.
impl Display for StructSlot {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StructSlot::Validity => write!(f, "<validity>"),
            StructSlot::Field(name) => write!(f, "{name}"),
        }
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
    /// child of the struct layout.
    flat_scope: DType,
    /// The name of the validity slot in [`Self::flat_scope`], if the struct is nullable.
    validity_name: Option<FieldName>,
}

impl StructPartitioner {
    /// Create a partitioner for a struct layout of the given `dtype`.
    pub(crate) fn new(dtype: &DType) -> VortexResult<Self> {
        let fields = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Struct layout dtype must be a struct, got {dtype}"))?;

        let validity_name = dtype
            .is_nullable()
            .then(|| unique_validity_name(fields.names()));

        let mut names: Vec<FieldName> = Vec::with_capacity(fields.nfields() + 1);
        let mut dtypes: Vec<DType> = Vec::with_capacity(fields.nfields() + 1);
        if let Some(validity_name) = &validity_name {
            names.push(validity_name.clone());
            dtypes.push(DType::Bool(Nullability::NonNullable));
        }
        names.extend(fields.names().iter().cloned());
        dtypes.extend(fields.fields());

        Ok(Self {
            scope: dtype.clone(),
            flat_scope: DType::Struct(
                StructFields::new(names.into(), dtypes),
                Nullability::NonNullable,
            ),
            validity_name,
        })
    }

    /// Each slot may hold several sub-expressions, so each one needs a unique name within the
    /// pack that makes up its partition.
    fn sub_expression_name(&self, slot: &StructSlot, idx: usize) -> FieldName {
        FieldName::from(format!("{}_{idx}", self.slot_name(slot)))
    }

    /// The flat scope's field name for a slot.
    fn slot_name(&self, slot: &StructSlot) -> FieldName {
        match slot {
            StructSlot::Validity => self
                .validity_name
                .clone()
                .vortex_expect("Validity slot only exists for nullable structs"),
            StructSlot::Field(name) => name.clone(),
        }
    }

    /// The slot addressed by a field of the flat scope.
    fn name_slot(&self, name: &FieldName) -> Option<StructSlot> {
        if self.validity_name.as_ref() == Some(name) {
            return Some(StructSlot::Validity);
        }
        self.scope
            .as_struct_fields_opt()
            .vortex_expect("Struct layout dtype must be a struct")
            .find(name)
            .map(|_| StructSlot::Field(name.clone()))
    }

    /// An expression referencing the struct's validity within the flat scope.
    fn validity_expr(&self) -> Option<Expression> {
        self.validity_name.as_ref().map(|name| col(name.clone()))
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
                partitioned.partition_annotations[0].clone(),
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
                    return replaced(self.reconstruct(self.field_names().iter().cloned()));
                }

                // `$.a` — a field of the struct. `get_item` intersects the struct's validity with
                // the field's, which the `mask` makes explicit.
                if let Some(field_name) = node.as_opt::<GetItem>()
                    && is_root(node.child(0))
                {
                    let field = col(field_name.clone());
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
                    let included = selection.normalize_to_included_fields(self.field_names())?;
                    return replaced(self.reconstruct(included.into_iter()));
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

    /// Rebuild a (possibly nullable) struct of the given fields from the flat scope.
    fn reconstruct(&self, names: impl Iterator<Item = FieldName>) -> Expression {
        let packed = pack(
            names.map(|name| (name.clone(), col(name))),
            Nullability::NonNullable,
        );
        match self.validity_expr() {
            Some(validity) => mask(packed, validity),
            None => packed,
        }
    }

    fn field_names(&self) -> &FieldNames {
        self.scope
            .as_struct_fields_opt()
            .vortex_expect("Struct layout dtype must be a struct")
            .names()
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
                .validity_name
                .iter()
                .map(|_| StructSlot::Validity)
                .chain(self.field_names().iter().cloned().map(StructSlot::Field))
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

        for (slot, exprs) in splitter.slots.iter().zip(splitter.sub_expressions.iter()) {
            // A slot with a single sub-expression doesn't need to be packed; the root expression
            // references it directly as `$.<slot>`.
            let partition = if let [only] = exprs.as_slice() {
                single_slots.insert(self.slot_name(slot), self.sub_expression_name(slot, 0));
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
            partition_names.push(self.slot_name(slot));
            partitions.push(self.step_into(partition, slot)?);
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

    /// Stage 3: rewrite a partition from the flat scope into the scope of `slot`'s child layout,
    /// by replacing `$.<slot>` with `$`.
    fn step_into(&self, expr: Expression, slot: &StructSlot) -> VortexResult<Expression> {
        let slot_name = self.slot_name(slot);
        Ok(expr
            .transform_down(|node| {
                if let Some(field_name) = node.as_opt::<GetItem>()
                    && is_root(node.child(0))
                {
                    if *field_name != slot_name {
                        vortex_bail!(
                            "Partition for slot {slot_name} unexpectedly accesses field \
                             {field_name}"
                        );
                    }
                    return Ok(Transformed {
                        value: root(),
                        order: TraversalOrder::Skip,
                        changed: true,
                    });
                }
                if is_root(&node) {
                    vortex_bail!(
                        "Partition for slot {slot_name} accesses the struct scope directly"
                    );
                }
                Ok(Transformed::no(node))
            })?
            .into_inner())
    }
}

/// Pick a name for the validity slot that cannot collide with a field of the struct.
fn unique_validity_name(field_names: &FieldNames) -> FieldName {
    let mut name = FieldName::from(VALIDITY_SLOT_PREFIX);
    while field_names.iter().any(|field| *field == name) {
        name = FieldName::from(format!("{name}_"));
    }
    name
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
    fn push(&mut self, slot: &StructSlot, expr: Expression) -> Expression {
        let slot_idx = self
            .slots
            .iter()
            .position(|s| s == slot)
            .unwrap_or_else(|| {
                self.slots.push(slot.clone());
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
            col(self.partitioner.slot_name(slot)),
        )
    }
}

impl NodeRewriter for SlotSplitter<'_> {
    type NodeTy = Expression;

    fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        match self.annotations.get(&node) {
            // If this expression only accesses a single slot, it becomes a partition.
            Some(slots) if slots.len() == 1 => {
                let slot = slots.iter().next().vortex_expect("expected one slot");
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
