// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lifting a strict kernel into a full [`ScalarFnVTable`].

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
#[cfg(any(test, feature = "_test-harness"))]
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::NullHandling;
use crate::scalar_fn::PersistableOptions;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::StrictScalarFnVTable;
use crate::scalar_fn::VecExecutionArgs;
use crate::validity::Validity;

/// Every [`StrictScalarFnVTable`] is a [`ScalarFnVTable`].
impl<V: StrictScalarFnVTable> ScalarFnVTable for V {
    type Options = V::Options;

    fn id(&self) -> ScalarFnId {
        StrictScalarFnVTable::id(self)
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        options.serialize()
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        Self::Options::deserialize(metadata, session)
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        StrictScalarFnVTable::arity(self, options)
    }

    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName {
        StrictScalarFnVTable::child_name(self, options, child_idx)
    }

    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        let element = self.return_element_dtype(options, args)?;
        let nullability =
            element.nullability() | Nullability::from(args.iter().any(DType::is_nullable));
        Ok(element.with_nullability(nullability))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let row_count = args.row_count();

        // Nullary functions have no input values that could be null.
        if args.num_inputs() == 0 {
            return self.execute_strict(options, args, ctx);
        }

        let inputs = (0..args.num_inputs())
            .map(|i| args.get(i))
            .collect::<VortexResult<Vec<_>>>()?;
        let arg_dtypes = inputs
            .iter()
            .map(|input| input.dtype().clone())
            .collect::<Vec<_>>();
        let result_dtype = self.return_dtype(options, &arg_dtypes)?;

        // Strictness: any null-constant input forces an all-null result without evaluating the
        // kernel.
        if inputs
            .iter()
            .any(|input| input.as_constant().is_some_and(|s| s.is_null()))
        {
            return Ok(all_null(result_dtype, row_count));
        }

        // All inputs constant (and non-null): evaluate a single row and broadcast. Reconciling the
        // row's dtype before reading the scalar keeps this path on the same kernel/declaration
        // agreement check as the dense and filter paths, rather than letting `cast` paper over a
        // disagreement.
        if row_count > 0 && inputs.iter().all(|input| input.as_constant().is_some()) {
            let one_row = inputs
                .iter()
                .map(|input| input.slice(0..1))
                .collect::<VortexResult<Vec<_>>>()?;
            let result = self.execute_strict(options, &VecExecutionArgs::new(one_row, 1), ctx)?;
            let result = with_return_dtype(result, result_dtype)?;
            let scalar = result.execute_scalar(0, ctx)?;
            return Ok(ConstantArray::new(scalar, row_count).into_array());
        }

        // A row of the output is valid iff the row is valid in every input. Conjoining validities
        // is lazy, so nothing is materialized unless the null handling below asks for it.
        let mut validity = Validity::NonNullable;
        for input in &inputs {
            validity = validity.and(input.validity()?)?;
        }

        match self.null_handling(options) {
            NullHandling::Dense => execute_dense(self, options, args, validity, result_dtype, ctx),
            NullHandling::Filter => {
                execute_filtered(self, options, args, &inputs, validity, result_dtype, ctx)
            }
        }
    }

    fn validity(
        &self,
        options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        StrictScalarFnVTable::validity(self, options, expression)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, options: &Self::Options) -> bool {
        StrictScalarFnVTable::is_fallible(self, options)
    }

    fn reduce(
        &self,
        options: &Self::Options,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        StrictScalarFnVTable::reduce(self, options, node, ctx)
    }
}

/// Run the kernel over every row, including the rows behind nulls, then mask its result.
///
/// This is the [`NullHandling::Dense`] path. `args` reaches the kernel untouched, so the inputs keep
/// their original encoding, and `validity` is handed to `mask` as an array rather than materialized
/// into a [`Mask`] first.
fn execute_dense<V: StrictScalarFnVTable>(
    vtable: &V,
    options: &V::Options,
    args: &dyn ExecutionArgs,
    validity: Validity,
    result_dtype: DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if StrictScalarFnVTable::is_fallible(vtable, options) {
        vortex_bail!(
            "{} is fallible and cannot use NullHandling::Dense: rows behind nulls hold arbitrary \
             values and must not raise errors",
            StrictScalarFnVTable::id(vtable),
        );
    }

    // Every row is null, so the kernel has nothing to contribute.
    if matches!(validity, Validity::AllInvalid) {
        return Ok(all_null(result_dtype, args.row_count()));
    }

    let values = vtable.execute_strict(options, args, ctx)?;

    match validity {
        Validity::NonNullable | Validity::AllValid => with_return_dtype(values, result_dtype),
        Validity::Array(valid) => with_return_dtype(values.mask(valid)?, result_dtype),
        // Handled by the guard above, before the kernel ran.
        Validity::AllInvalid => Ok(all_null(result_dtype, args.row_count())),
    }
}

/// The [`NullHandling::Filter`] path: materialize the conjoined validity once, take the all-true
/// and all-false shortcuts, and pick a null strategy per batch for a mixed mask.
///
/// Two strategies can execute a mixed mask, and neither is visible to the kernel:
///
/// - **Branch-and-skip** ([`execute_branched`]): hand the *unfiltered* inputs plus the mask to
///   [`StrictScalarFnVTable::execute_strict_branch`], which computes only the valid rows, then
///   mask the full-length result exactly as the dense path does. This skips the filter and the
///   scatter entirely, at the price of decoding full-length columns.
/// - **Filter** ([`filter_and_scatter`]): filter every input down to the conjoined-valid rows,
///   run the kernel over those, and scatter its results back into a null-padded output. Always
///   available, never encoding-preserving.
///
/// Branch-and-skip is preferred whenever [`branch_beats_filter`] says so, and the filter strategy
/// is also the fallback for kernels with no branch execution.
fn execute_filtered<V: StrictScalarFnVTable>(
    vtable: &V,
    options: &V::Options,
    args: &dyn ExecutionArgs,
    inputs: &[ArrayRef],
    validity: Validity,
    result_dtype: DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let row_count = args.row_count();
    let valid = validity.execute_mask(row_count, ctx)?;

    // Check all-true before all-false: an empty mask is both, and must not be treated as all-null (a
    // zero-length non-nullable execution keeps its non-nullable dtype).
    if valid.all_true() {
        let values = vtable.execute_strict(options, args, ctx)?;
        return with_return_dtype(values, result_dtype);
    }

    if valid.all_false() {
        return Ok(all_null(result_dtype, row_count));
    }

    if branch_beats_filter(vtable.decode_shrinks_when_filtered(options), &valid)
        && let Some(result) =
            execute_branched(vtable, options, args, &valid, result_dtype.clone(), ctx)?
    {
        return Ok(result);
    }

    filter_and_scatter(vtable, options, inputs, &valid, result_dtype, ctx)
}

/// The minimum surviving-row fraction (`true_count / len` of the conjoined mask) at which
/// branch-and-skip is still chosen for a function whose decode shrinks when filtered.
///
/// From the branch-and-skip measurements (65536 rows, divan fastest of 100 samples, two runs on a
/// shared 4-vCPU VM). A kernel with a *bulk* decode never lost under branch: `byte_length` over
/// the `Bytes` element ran 1.8-5.9x faster than filter at every null density from 1% to 90%, so
/// such kernels skip this check entirely. A kernel with a *per-row* decode (geo `contains`, which
/// arrow-exports and parses one geometry per row) pays that decode over the full column under
/// branch but only over the survivors under filter, so filter wins once validity is sparse:
///
/// - polygons CONTAINS constant point: branch won 1.07-1.18x at 1-50% nulls; filter won 1.38x at
///   90% nulls (10% of rows surviving).
/// - polygons CONTAINS points, independent nulls on both: branch won up to ~10% null density
///   (~81% surviving); filter won 1.2x at ~56% surviving, 1.9x at ~25%, 11.3x at ~1%.
///
/// The measured crossover sits between ~56% and ~81% surviving rows. 0.75 keeps the branch wins
/// at dense validity (the two-nullable-operand case included) and sends every batch where filter
/// measurably dominated to filter; the one concession is the single-operand 50%-null case, at
/// exactly 50% surviving, where branch's ~1.1x edge is given up for filter.
pub(super) const BRANCH_MIN_SURVIVING_FRACTION: f64 = 0.75;

/// Whether the branch-and-skip strategy should be preferred over filtering for the mixed mask
/// `valid`: always, unless a per-row decode would shrink under filter
/// (`decode_shrinks_when_filtered`) and fewer than [`BRANCH_MIN_SURVIVING_FRACTION`] of the rows
/// survive.
pub(super) fn branch_beats_filter(decode_shrinks_when_filtered: bool, valid: &Mask) -> bool {
    if !decode_shrinks_when_filtered {
        return true;
    }

    valid.true_count() as f64 >= valid.len() as f64 * BRANCH_MIN_SURVIVING_FRACTION
}

/// Try the branch-and-skip strategy for a mixed mask: the kernel computes only the rows set in
/// `valid` over the unfiltered inputs, and the full-length result is masked exactly as the dense
/// path masks. `Ok(None)` means the kernel has no branch execution for these inputs, and the
/// caller falls back to [`filter_and_scatter`].
fn execute_branched<V: StrictScalarFnVTable>(
    vtable: &V,
    options: &V::Options,
    args: &dyn ExecutionArgs,
    valid: &Mask,
    result_dtype: DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some(values) = vtable.execute_strict_branch(options, args, valid, ctx)? else {
        return Ok(None);
    };

    let mask = BoolArray::new(valid.to_bit_buffer(), Validity::NonNullable).into_array();
    with_return_dtype(values.mask(mask)?, result_dtype).map(Some)
}

/// The filter strategy for a mixed mask: filter every input down to the rows set in `valid`, run
/// the kernel over those, and scatter its results back into a null-padded output.
fn filter_and_scatter<V: StrictScalarFnVTable>(
    vtable: &V,
    options: &V::Options,
    inputs: &[ArrayRef],
    valid: &Mask,
    result_dtype: DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let filtered = inputs
        .iter()
        .map(|input| input.filter(valid.clone()))
        .collect::<VortexResult<Vec<_>>>()?;
    let values = vtable.execute_strict(
        options,
        &VecExecutionArgs::new(filtered, valid.true_count()),
        ctx,
    )?;

    with_return_dtype(scatter_valid(values, valid)?, result_dtype)
}

/// Which null strategy [`execute_strict_with_strategy`] forces for a mixed validity mask.
///
/// A test and benchmark seam: pinning a strategy is how the two are compared and how their
/// agreement is asserted. Production execution selects per batch inside the lifting and never
/// names one.
#[cfg(any(test, feature = "_test-harness"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullStrategy {
    /// Filter the inputs down to the conjoined-valid rows, run the kernel, and scatter back.
    Filter,

    /// Decode the unfiltered inputs null-tolerantly, compute only the conjoined-valid rows, and
    /// mask the full-length result.
    BranchAndSkip,
}

/// Execute `vtable` over `inputs` with a forced null strategy, bypassing the per-batch selection.
///
/// A test and benchmark seam only. It mirrors the [`NullHandling::Filter`] lifting (conjoined
/// validity, the all-true and all-false shortcuts, output dtype reconciliation) but takes the
/// strategy from the caller instead of the selection rule, and it skips the null-constant
/// and all-constant folds, so do not pass such inputs. Forcing [`NullStrategy::BranchAndSkip`] on
/// a kernel with no branch execution is an error rather than a silent fallback.
#[cfg(any(test, feature = "_test-harness"))]
pub fn execute_strict_with_strategy<V: StrictScalarFnVTable>(
    vtable: &V,
    options: &V::Options,
    inputs: Vec<ArrayRef>,
    row_count: usize,
    strategy: NullStrategy,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let arg_dtypes = inputs
        .iter()
        .map(|input| input.dtype().clone())
        .collect::<Vec<_>>();
    let result_dtype = ScalarFnVTable::return_dtype(vtable, options, &arg_dtypes)?;

    let mut validity = Validity::NonNullable;
    for input in &inputs {
        validity = validity.and(input.validity()?)?;
    }
    let valid = validity.execute_mask(row_count, ctx)?;

    let args = VecExecutionArgs::new(inputs.clone(), row_count);

    if valid.all_true() {
        let values = vtable.execute_strict(options, &args, ctx)?;
        return with_return_dtype(values, result_dtype);
    }

    if valid.all_false() {
        return Ok(all_null(result_dtype, row_count));
    }

    match strategy {
        NullStrategy::Filter => {
            filter_and_scatter(vtable, options, &inputs, &valid, result_dtype, ctx)
        }
        NullStrategy::BranchAndSkip => {
            execute_branched(vtable, options, &args, &valid, result_dtype, ctx)?.ok_or_else(|| {
                vortex_err!(
                    "{} has no branch-and-skip execution for these inputs",
                    StrictScalarFnVTable::id(vtable),
                )
            })
        }
    }
}

/// An all-null result of the function's declared return dtype.
fn all_null(dtype: DType, row_count: usize) -> ArrayRef {
    ConstantArray::new(Scalar::null(dtype), row_count).into_array()
}

/// Reconcile the kernel's output dtype with the function's declared return dtype.
///
/// The strict contract lets a kernel ignore nullability, so a nullability difference is cast away.
/// Any other difference means [`return_element_dtype`](StrictScalarFnVTable::return_element_dtype)
/// and the kernel disagree, which is a bug worth naming rather than silently casting away.
fn with_return_dtype(values: ArrayRef, result_dtype: DType) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        values.dtype().eq_ignore_nullability(&result_dtype),
        "strict kernel produced {} but the function declares {result_dtype}",
        values.dtype(),
    );

    if values.dtype() == &result_dtype {
        Ok(values)
    } else {
        values.cast(result_dtype)
    }
}

/// Scatter `values` (one per set bit of `valid`, in order) back to the positions of the set bits,
/// producing an array of length `valid.len()` that is null at every unset position.
fn scatter_valid(values: ArrayRef, valid: &Mask) -> VortexResult<ArrayRef> {
    vortex_ensure_eq!(
        values.len(),
        valid.true_count(),
        "strict kernel produced {} rows for {} filtered rows",
        values.len(),
        valid.true_count(),
    );

    let AllOr::Some(slices) = valid.slices() else {
        // The caller handles the all-true and all-false masks.
        vortex_bail!("scatter_valid requires a mixed mask");
    };

    // Gather indices: row i of the output reads values[rank(i)]. Rows behind nulls read index 0,
    // and any in-bounds index would do since they are masked out below (values is non-empty here).
    let mut indices = vec![0u64; valid.len()];
    let mut rank = 0u64;
    for &(start, end) in slices {
        for index in &mut indices[start..end] {
            *index = rank;
            rank += 1;
        }
    }
    let indices = PrimitiveArray::new(indices, Validity::NonNullable).into_array();

    let scattered = values.take(indices)?;
    let mask = BoolArray::new(valid.to_bit_buffer(), Validity::NonNullable).into_array();
    scattered.mask(mask)
}
