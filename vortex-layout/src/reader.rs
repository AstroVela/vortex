// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::try_join;
use once_cell::sync::OnceCell;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutReaderContext;
use crate::children::LayoutChildren;
use crate::segments::SegmentSource;

/// Shared handle to a stateful layout reader.
pub type LayoutReaderRef = Arc<dyn LayoutReader>;

/// Shared handle to a heap-allocated physical scan plan.
pub type ScanPlanRef = Arc<dyn ScanPlan>;

/// A heap-allocated physical scan plan.
///
/// A source plan represents an instantiated layout. [`apply_expr`](Self::apply_expr) derives
/// another plan whose root value is the applied expression, and [`optimize`](Self::optimize)
/// rewrites that derived plan before execution. Execution therefore selects an already-bound plan
/// and supplies only its row range and mask.
pub trait ScanPlan: 'static + Send + Sync {
    /// Apply `expr` to this plan's root value and return the resulting plan.
    fn apply_expr(self: Arc<Self>, expr: Expression) -> VortexResult<ScanPlanRef>;

    /// Optimize this plan and return the resulting plan.
    fn optimize(self: Arc<Self>) -> VortexResult<ScanPlanRef>;

    /// Returns the name of the underlying layout reader for debugging.
    fn name(&self) -> &Arc<str>;

    /// Returns the dtype produced by this plan.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in this plan's row domain.
    fn row_count(&self) -> u64;

    /// Returns a mask where all false values are proven false for this plan.
    fn pruning_evaluation(&self, row_range: &Range<u64>, mask: Mask) -> VortexResult<MaskFuture>;

    /// Evaluates this boolean plan and intersects it with `mask`.
    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture>;

    /// Evaluates this plan over the selected rows.
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture>;
}

/// Compatibility source and expression plan backed by a V1 layout reader.
///
/// Applying an expression and optimizing it produce new heap-allocated plans. Execution delegates
/// the resulting expression to the established reader implementation. Layout-specific source plans
/// can replace this compatibility node without changing the split execution loop.
pub struct LayoutReaderScanPlan {
    reader: LayoutReaderRef,
    expr: Expression,
    dtype: DType,
}

impl LayoutReaderScanPlan {
    /// Create a source plan for `reader`.
    pub fn new(reader: LayoutReaderRef) -> Self {
        let dtype = reader.dtype().clone();
        Self {
            reader,
            expr: root(),
            dtype,
        }
    }

    /// Bind `expr` to `reader`.
    fn try_new(reader: LayoutReaderRef, expr: Expression) -> VortexResult<Self> {
        let dtype = expr.return_dtype(reader.dtype())?;
        Ok(Self {
            reader,
            expr,
            dtype,
        })
    }
}

impl ScanPlan for LayoutReaderScanPlan {
    fn apply_expr(self: Arc<Self>, expr: Expression) -> VortexResult<ScanPlanRef> {
        let expr = replace(expr, &root(), self.expr.clone());
        Ok(Arc::new(Self::try_new(Arc::clone(&self.reader), expr)?))
    }

    fn optimize(self: Arc<Self>) -> VortexResult<ScanPlanRef> {
        let expr = self.expr.optimize_recursive(self.reader.dtype())?;
        Ok(Arc::new(Self::try_new(Arc::clone(&self.reader), expr)?))
    }

    fn name(&self) -> &Arc<str> {
        self.reader.name()
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.reader.row_count()
    }

    fn pruning_evaluation(&self, row_range: &Range<u64>, mask: Mask) -> VortexResult<MaskFuture> {
        self.reader.pruning_evaluation(row_range, &self.expr, mask)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.reader.filter_evaluation(row_range, &self.expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        self.reader
            .projection_evaluation(row_range, &self.expr, mask)
    }
}

/// A row range used when registering natural scan splits.
///
/// Row range is relative to the reader that receives it. Offset is the offset
/// that the local row range needs to be shifted by to get the global row range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitRange {
    row_offset: u64,
    row_range: Range<u64>,
}

impl SplitRange {
    /// Constructs a split range, returning an error if the local row range is invalid.
    pub fn try_new(row_offset: u64, row_range: Range<u64>) -> VortexResult<Self> {
        if row_range.start > row_range.end {
            vortex_bail!("Invalid split range {:?}", row_range);
        }

        Ok(Self {
            row_offset,
            row_range,
        })
    }

    /// Constructs a split range for the root layout.
    pub fn root(row_range: Range<u64>) -> VortexResult<Self> {
        Self::try_new(0, row_range)
    }

    /// The root-layout row offset of this reader's local row zero.
    pub fn row_offset(&self) -> u64 {
        self.row_offset
    }

    /// The local row range within this reader.
    pub fn row_range(&self) -> &Range<u64> {
        &self.row_range
    }

    /// The length of the local row range.
    pub fn len(&self) -> u64 {
        self.row_range.end - self.row_range.start
    }

    /// Returns `true` if the local row range is empty.
    pub fn is_empty(&self) -> bool {
        self.row_range.is_empty()
    }

    /// Returns the equivalent row range in the root layout's coordinate space.
    pub fn root_row_range(&self) -> Range<u64> {
        self.row_offset + self.row_range.start..self.row_offset + self.row_range.end
    }

    /// Returns an error if the local row range is outside the given row count.
    pub fn check_bounds(&self, row_count: u64) -> VortexResult<()> {
        if self.row_range.end > row_count {
            vortex_bail!(
                "Split range {:?} is out of bounds for row count {}",
                self.row_range,
                row_count
            );
        }

        Ok(())
    }
}

/// A collection of root-coordinate row split points.
pub struct RowSplits(Vec<u64>);

impl RowSplits {
    /// Add a row boundary to the split set.
    pub fn push(&mut self, row: u64) {
        self.0.push(row);
    }

    /// Reserve space for additional row boundaries.
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    /// Create a new RowSplits with preallocated "capacity"
    pub(crate) fn new_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub(crate) fn into_sorted_deduped(mut self) -> Vec<u64> {
        self.0.sort_unstable();
        self.0.dedup();
        self.0.shrink_to_fit();
        self.0
    }
}

/// Stateful reader for a [`crate::Layout`].
///
/// A reader owns or references any state needed to evaluate many scan operations over the same
/// layout, such as child readers, decoded metadata, or segment caches. Scan planning calls
/// [`register_splits`](Self::register_splits); execution calls pruning, filter, and projection
/// evaluation for each selected row range.
pub trait LayoutReader: 'static + Send + Sync {
    /// Returns the name of the layout reader for debugging.
    fn name(&self) -> &Arc<str>;

    /// Returns this reader as [`Any`] for downcasting by specialized wrappers.
    fn as_any(&self) -> &dyn Any;

    /// Returns the un-projected dtype of the layout reader.
    fn dtype(&self) -> &DType;

    /// Returns the number of rows in the layout.
    fn row_count(&self) -> u64;

    /// Register natural split boundaries for this reader.
    ///
    /// `field_mask` contains the projected and filtered field paths needed by the scan.
    /// Implementations should add root-coordinate split boundaries to `splits`, constrained to
    /// `split_range`.
    // TODO(ngates): this is a temporary API until we make layout readers stream based.
    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()>;

    /// Returns a mask where all false values are proven to be false in the given expression.
    ///
    /// The returned mask **does not** need to have been intersected with the input mask.
    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture>;

    /// Refines the given mask, returning a mask equal in length to the input mask.
    ///
    /// It is recommended to defer awaiting the input mask for as long as possible (ideally, after
    /// all I/O is complete). This allows other conjuncts the opportunity to refine the mask as much
    /// as possible before it is used.
    ///
    /// ## Post-conditions
    ///
    /// The returned mask **MUST** have been intersected with the input mask.
    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture>;

    /// Evaluates an expression against an array.
    ///
    /// It is recommended to defer awaiting the input mask for as long as possible (ideally, after
    /// all I/O is complete). This allows other conjuncts the opportunity to refine the mask as much
    /// as possible before it is used.
    ///
    /// ## Post-conditions
    ///
    /// The returned array **MUST** have length equal to the true count of the input mask.
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture>;
}

/// Future resolving to a projected Vortex array.
pub type ArrayFuture = BoxFuture<'static, VortexResult<ArrayRef>>;

/// Helpers for futures that resolve to arrays.
pub trait ArrayFutureExt {
    /// Apply a row mask to the resolved array.
    fn masked(self, mask: MaskFuture) -> Self;
}

impl ArrayFutureExt for ArrayFuture {
    /// Returns a new `ArrayFuture` that masks the output with a mask
    fn masked(self, mask: MaskFuture) -> Self {
        Box::pin(async move {
            let (array, mask) = try_join!(self, mask)?;
            array.mask(mask.into_array())
        })
    }
}

/// Lazily constructs and caches child readers while preserving reader context.
pub struct LazyReaderChildren {
    children: Arc<dyn LayoutChildren>,
    dtypes: Vec<DType>,
    names: Vec<Arc<str>>,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    ctx: LayoutReaderContext,
    // TODO(ngates): we may want a hash map of some sort here?
    cache: Vec<OnceCell<LayoutReaderRef>>,
}

impl LazyReaderChildren {
    /// Create a lazy child-reader cache.
    ///
    /// `dtypes` and `names` must be aligned with the child indices exposed by `children`.
    pub fn new(
        children: Arc<dyn LayoutChildren>,
        dtypes: Vec<DType>,
        names: Vec<Arc<str>>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: LayoutReaderContext,
    ) -> Self {
        let nchildren = children.nchildren();
        let cache = (0..nchildren).map(|_| OnceCell::new()).collect();
        Self {
            children,
            dtypes,
            names,
            segment_source,
            session,
            ctx,
            cache,
        }
    }

    /// Return the child reader at `idx`, constructing it on first access.
    pub fn get(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        if idx >= self.cache.len() {
            vortex_bail!("Child index out of bounds: {} of {}", idx, self.cache.len());
        }

        self.cache[idx].get_or_try_init(|| {
            let dtype = &self.dtypes[idx];
            let child = self.children.child(idx, dtype)?;
            child.new_reader(
                Arc::clone(&self.names[idx]),
                Arc::clone(&self.segment_source),
                &self.session,
                &self.ctx,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::lit;

    use super::*;

    struct TestLayoutReader {
        name: Arc<str>,
        dtype: DType,
    }

    impl TestLayoutReader {
        fn new() -> Self {
            Self {
                name: Arc::from("test"),
                dtype: DType::Struct(
                    StructFields::from_iter([(
                        FieldName::from("a"),
                        DType::Primitive(PType::I32, Nullability::NonNullable),
                    )]),
                    Nullability::NonNullable,
                ),
            }
        }
    }

    impl LayoutReader for TestLayoutReader {
        fn name(&self) -> &Arc<str> {
            &self.name
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            1
        }

        fn register_splits(
            &self,
            _field_mask: &[FieldMask],
            _split_range: &SplitRange,
            _splits: &mut RowSplits,
        ) -> VortexResult<()> {
            unimplemented!("not needed for scan-plan construction")
        }

        fn pruning_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: Mask,
        ) -> VortexResult<MaskFuture> {
            unimplemented!("not needed for scan-plan construction")
        }

        fn filter_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<MaskFuture> {
            unimplemented!("not needed for scan-plan construction")
        }

        fn projection_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<ArrayFuture> {
            unimplemented!("not needed for scan-plan construction")
        }
    }

    #[test]
    fn scan_plan_applies_expressions_to_the_current_root() -> VortexResult<()> {
        let reader: LayoutReaderRef = Arc::new(TestLayoutReader::new());
        let source: ScanPlanRef = Arc::new(LayoutReaderScanPlan::new(reader));

        let field = Arc::clone(&source)
            .apply_expr(get_item("a", root()))?
            .optimize()?;
        assert_eq!(
            field.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        let predicate = field.apply_expr(eq(root(), lit(1_i32)))?.optimize()?;
        assert_eq!(predicate.dtype(), &DType::Bool(Nullability::NonNullable));

        Ok(())
    }
}
