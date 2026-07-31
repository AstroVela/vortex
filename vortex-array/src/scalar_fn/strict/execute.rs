// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lifting a strict kernel into a full [`ScalarFnVTable`].

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
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

/// Filter the inputs down to the rows valid in every input, run the kernel over those, and scatter
/// its results back into a null-padded output.
///
/// This is the [`NullHandling::Filter`] path, always sound but never encoding-preserving: the kernel
/// sees filtered copies. Unlike [`execute_dense`] it needs the valid *positions*, so the conjoined
/// validity is materialized into a [`Mask`] here.
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

    let filtered = inputs
        .iter()
        .map(|input| input.filter(valid.clone()))
        .collect::<VortexResult<Vec<_>>>()?;
    let values = vtable.execute_strict(
        options,
        &VecExecutionArgs::new(filtered, valid.true_count()),
        ctx,
    )?;

    with_return_dtype(scatter_valid(values, &valid)?, result_dtype)
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
