// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Partitioning of expressions over a struct layout.
//!
//! A struct layout stores each field in its own child, plus a validity child when the struct is
//! nullable. To evaluate an expression the reader has to split it into per-child expressions that
//! the child readers can evaluate independently, and a root expression that recombines the
//! results.
//!
//! This is a struct-layout-specific replacement for
//! [`vortex_array::expr::transform::partition`], which only ever partitions over *fields* and so
//! silently drops the struct's own validity — see
//! <https://github.com/vortex-data/vortex/issues/1907>. Here every expression is partitioned over
//! the layout's `n + 1` slots (see [`StructSlot`]), which means `is_null(root())` reads the
//! validity child and nothing else, and `root().a` correctly reads the field *and* the validity.
//!
//! Which slots an expression reads, and how the reads recombine, is not hard-coded here: it comes
//! from the [`kernels`] registry, keyed by scalar function.
//!
//! # Algorithm
//!
//! 1. Annotate every node with the set of slots its subtree reads. A node that reads the scope
//!    directly is annotated from its kernel; everything else is the union over its children.
//! 2. Rewrite the tree top-down. A subtree that reads exactly one slot is lowered into that
//!    slot's scope and becomes a partition. A subtree that reads one field plus the validity is
//!    lowered into the field when every function between the two is strict, because `mask` then
//!    commutes upwards. Otherwise the node's kernel decomposition is emitted, or — for a node
//!    with no decomposition — its children are rewritten in place.
//! 3. Pack the sub-expressions registered against each slot into one partition per slot.

mod kernels;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

pub use kernels::SlotDecomposition;
pub use kernels::StructPartitionKernel;
pub use kernels::StructScope;
pub use kernels::StructSlot;
pub use kernels::WholeScope;
pub use kernels::struct_partition_kernel;
use kernels::sub_expr_name;
use kernels::sub_expr_ref;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::is_root;
use vortex_array::expr::mask;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::transform::PartitionedExpr;
use vortex_array::expr::transform::replace;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

/// An expression split across the children of a struct layout.
#[derive(Clone, Debug)]
pub enum StructPartitioned {
    /// The expression reads a single child, which can evaluate it in its entirety.
    Single(StructSlot, Expression),
    /// The expression reads several children and is recombined by the root expression.
    Multi(Arc<PartitionedExpr<StructSlot>>),
}

/// Partition `expr` over the children of a struct layout with dtype `dtype`.
///
/// `field_lookup`, when present, resolves field names in place of a linear scan over `dtype`.
///
/// The returned partitions are expressed in the scope of their child reader: a field partition
/// reads the raw, *unmasked* field values, and the validity partition reads the struct's validity
/// as a non-nullable boolean.
pub fn partition_struct_expr(
    expr: &Expression,
    dtype: &DType,
    field_lookup: Option<&HashMap<FieldName, usize>>,
) -> VortexResult<StructPartitioned> {
    let fields = dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("Expected a struct dtype, got {dtype}"))?;

    // Type-check and simplify the expression before splitting it up. Partitions are optimized
    // against their child's dtype, so this is also where a mistyped expression is rejected.
    let expr = expr.optimize_recursive(dtype)?;

    let scope = StructScope::new(fields, dtype.nullability(), field_lookup);
    let mut partitioner = Partitioner::new(scope);
    partitioner.annotate(&expr)?;
    let root_expr = partitioner.split(&expr)?;
    partitioner.finish(root_expr)
}

/// The partition that upper-bounds the root expression, if there is one.
///
/// Pruning may over-approximate — it must never discard a row the full expression would keep — so
/// a root expression that can only ever remove rows relative to one of its boolean partitions can
/// be pruned using that partition alone. This recovers predicate pruning for expressions over a
/// nullable struct, where the validity intersection otherwise forces a multi-partition split.
pub fn pruning_partition(partitioned: &PartitionedExpr<StructSlot>) -> Option<usize> {
    let mut expr = &partitioned.root;
    loop {
        if expr.is::<Mask>() {
            // `mask` only ever turns `true` into `null`, which pruning reads as "may be false".
            expr = expr.child(0);
            continue;
        }
        let idx = partitioned
            .partition_names
            .iter()
            .position(|name| expr == &col(name.clone()))?;
        return matches!(
            partitioned.partition_dtypes[idx],
            DType::Bool(Nullability::NonNullable)
        )
        .then_some(idx);
    }
}

/// The slots read by an expression subtree.
type SlotSet = BTreeSet<StructSlot>;

struct Partitioner<'a> {
    scope: StructScope<'a>,
    slots: HashMap<&'a Expression, SlotSet>,
    decompositions: HashMap<&'a Expression, Option<SlotDecomposition>>,
    sub_exprs: BTreeMap<StructSlot, Vec<Expression>>,
}

impl<'a> Partitioner<'a> {
    fn new(scope: StructScope<'a>) -> Self {
        Self {
            scope,
            slots: HashMap::new(),
            decompositions: HashMap::new(),
            sub_exprs: BTreeMap::new(),
        }
    }

    /// The decomposition of a node that reads the struct scope directly, if it has one.
    fn decomposition(&mut self, expr: &'a Expression) -> VortexResult<Option<SlotDecomposition>> {
        if let Some(decomposition) = self.decompositions.get(expr) {
            return Ok(decomposition.clone());
        }

        // Kernels only apply to a direct read of the scope: `root()` itself, or a function whose
        // first child is `root()`. Anything else recurses until it reaches one of those.
        let decomposition = if is_root(expr) || expr.children().first().is_some_and(is_root) {
            struct_partition_kernel(expr.scalar_fn().id()).decompose(expr, &self.scope)?
        } else {
            None
        };

        self.decompositions.insert(expr, decomposition.clone());
        Ok(decomposition)
    }

    /// Record the set of slots read by every node in the tree.
    fn annotate(&mut self, expr: &'a Expression) -> VortexResult<()> {
        if self.slots.contains_key(expr) {
            return Ok(());
        }

        let slots = match self.decomposition(expr)? {
            Some(decomposition) => decomposition.slots().collect(),
            None => {
                let mut slots = SlotSet::new();
                for child in expr.children().iter() {
                    self.annotate(child)?;
                    slots.extend(self.slots_of(child).iter().copied());
                }
                slots
            }
        };

        self.slots.insert(expr, slots);
        Ok(())
    }

    fn slots_of(&self, expr: &Expression) -> &SlotSet {
        self.slots
            .get(expr)
            .vortex_expect("every node is annotated before splitting")
    }

    /// Rewrite `expr` into an expression over the scope of the partitions.
    fn split(&mut self, expr: &'a Expression) -> VortexResult<Expression> {
        let decomposition = self.decomposition(expr)?;
        let slots = self.slots_of(expr).clone();

        // An expression that reads no slot is scope-independent. It still has to go through its
        // decomposition when it has one, since a kernel may answer a read from a constant — as
        // `is_null` does for a non-nullable struct.
        if slots.is_empty() {
            return Ok(match decomposition {
                Some(decomposition) => decomposition.combine(&[]),
                None => expr.clone(),
            });
        }

        if slots.len() == 1 {
            let slot = *slots.iter().next().vortex_expect("non-empty");
            if let Some(lowered) = self.lower(expr, slot, false)? {
                return Ok(self.register(slot, lowered));
            }
        } else if slots.len() == 2 && slots.contains(&StructSlot::Validity) {
            // One field plus the struct validity: if every function in between is strict then
            // the validity mask can be hoisted to the top, leaving the rest to push into the
            // field's child reader.
            let slot = *slots
                .iter()
                .find(|slot| **slot != StructSlot::Validity)
                .vortex_expect("two slots, one of which is validity");
            if let Some(lowered) = self.lower(expr, slot, true)? {
                let value = self.register(slot, lowered);
                let validity = self.register(StructSlot::Validity, root());
                return Ok(mask(value, validity));
            }
        }

        if let Some(decomposition) = decomposition {
            let parts: Vec<Expression> = decomposition
                .parts()
                .iter()
                .map(|(slot, part)| self.register(*slot, part.clone()))
                .collect();
            return Ok(decomposition.combine(&parts));
        }

        let mut children = Vec::with_capacity(expr.children().len());
        for child in expr.children().iter() {
            children.push(self.split(child)?);
        }
        expr.clone().with_children(children)
    }

    /// Rewrite `expr` into the scope of `slot`'s child reader.
    ///
    /// Returns `None` when some read within `expr` cannot be answered by `slot` alone, in which
    /// case the caller falls back to splitting the node.
    fn lower(
        &mut self,
        expr: &'a Expression,
        slot: StructSlot,
        hoist_validity: bool,
    ) -> VortexResult<Option<Expression>> {
        if let Some(decomposition) = self.decomposition(expr)? {
            return Ok(decomposition.lowerable(slot, hoist_validity).cloned());
        }

        // Sub-expressions that don't read the struct are carried down unchanged.
        if self.slots_of(expr).is_empty() {
            return Ok(Some(expr.clone()));
        }

        // `mask(x, v)` may only be hoisted above a function that maps null inputs to null outputs.
        if hoist_validity && !expr.signature().is_strict() {
            return Ok(None);
        }

        let mut children = Vec::with_capacity(expr.children().len());
        for child in expr.children().iter() {
            match self.lower(child, slot, hoist_validity)? {
                Some(child) => children.push(child),
                None => return Ok(None),
            }
        }
        expr.clone().with_children(children).map(Some)
    }

    /// Record a sub-expression to evaluate against `slot`, returning a reference to its result.
    fn register(&mut self, slot: StructSlot, expr: Expression) -> Expression {
        let sub_exprs = self.sub_exprs.entry(slot).or_default();
        let idx = sub_exprs
            .iter()
            .position(|existing| existing == &expr)
            .unwrap_or_else(|| {
                sub_exprs.push(expr);
                sub_exprs.len() - 1
            });
        sub_expr_ref(slot, idx)
    }

    fn finish(self, root_expr: Expression) -> VortexResult<StructPartitioned> {
        let Partitioner {
            scope, sub_exprs, ..
        } = self;

        let mut root_expr = root_expr;
        let mut slots = Vec::with_capacity(sub_exprs.len());
        let mut names = Vec::with_capacity(sub_exprs.len());
        let mut partitions = Vec::with_capacity(sub_exprs.len());
        let mut dtypes = Vec::with_capacity(sub_exprs.len());

        for (slot, exprs) in sub_exprs {
            let name = slot.partition_name();

            let partition = if exprs.len() == 1 {
                // A lone sub-expression needs no packing. Keeping it unwrapped preserves its
                // dtype, which is what lets a boolean partition be evaluated as a mask.
                root_expr = replace(root_expr, &sub_expr_ref(slot, 0), col(name.clone()));
                exprs.into_iter().next().vortex_expect("exactly one")
            } else {
                pack(
                    exprs
                        .into_iter()
                        .enumerate()
                        .map(|(idx, expr)| (sub_expr_name(idx), expr)),
                    Nullability::NonNullable,
                )
            };

            let slot_dtype = scope.slot_dtype(slot)?;
            let partition = partition.optimize_recursive(&slot_dtype)?;
            dtypes.push(partition.return_dtype(&slot_dtype)?);
            partitions.push(partition);
            names.push(name);
            slots.push(slot);
        }

        let partition_names: FieldNames = names.into_iter().collect();
        let root_scope = DType::Struct(
            StructFields::new(partition_names.clone(), dtypes.clone()),
            Nullability::NonNullable,
        );
        let root_expr = root_expr.optimize_recursive(&root_scope)?;

        // A root expression that is exactly the single partition is a pure delegation, which the
        // reader can hand straight to the child.
        if partitions.len() == 1 && root_expr == col(partition_names[0].clone()) {
            return Ok(StructPartitioned::Single(
                slots[0],
                partitions.into_iter().next().vortex_expect("exactly one"),
            ));
        }

        Ok(StructPartitioned::Multi(Arc::new(PartitionedExpr {
            root: root_expr,
            partitions: partitions.into_boxed_slice(),
            partition_names,
            partition_dtypes: dtypes.into_boxed_slice(),
            partition_annotations: slots.into_boxed_slice(),
        })))
    }
}
