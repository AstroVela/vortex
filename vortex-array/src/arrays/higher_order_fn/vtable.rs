// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::higher_order_fn::array::HigherOrderFnArrayExt;
use crate::arrays::higher_order_fn::array::HigherOrderFnData;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::higher_order_fn::HigherOrderFunctionId;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A lazy array produced by a higher-order function.
pub type HigherOrderFnArray = Array<HigherOrderFn>;

/// The Vortex array vtable for a registered higher-order function.
#[derive(Clone, Debug)]
pub struct HigherOrderFn {
    pub(super) id: HigherOrderFunctionId,
}

impl Display for HigherOrderFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.id.fmt(f)
    }
}

impl ArrayHash for HigherOrderFnData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.higher_order_fn.hash(state);
        self.arg_count.hash(state);
        self.lambdas.hash(state);
    }
}

impl ArrayEq for HigherOrderFnData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.higher_order_fn == other.higher_order_fn
            && self.arg_count == other.arg_count
            && self.lambdas == other.lambdas
    }
}

impl VTable for HigherOrderFn {
    type TypedArrayData = HigherOrderFnData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        self.id
    }

    fn validate(
        &self,
        data: &HigherOrderFnData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            data.higher_order_fn.id() == self.id,
            "HigherOrderFnArray data function does not match vtable"
        );
        vortex_ensure!(
            slots.len() >= data.arg_count,
            "HigherOrderFnArray has fewer slots than ordinary arguments"
        );
        vortex_ensure!(
            data.higher_order_fn.arity().matches(data.arg_count),
            "HigherOrderFnArray argument count does not match function arity"
        );
        vortex_ensure!(
            slots.iter().all(Option::is_some),
            "HigherOrderFnArray slots must not be empty"
        );
        vortex_ensure!(
            slots.iter().flatten().all(|slot| slot.len() == len),
            "HigherOrderFnArray slots must have the array length"
        );
        vortex_ensure!(
            data.lambdas.len() == data.higher_order_fn.lambda_arity(),
            "HigherOrderFnArray lambda count does not match function arity"
        );

        let capture_count = slots.len() - data.arg_count;
        for lambda in &data.lambdas {
            lambda.validate(capture_count)?;
        }

        let arg_dtypes = slots[..data.arg_count]
            .iter()
            .flatten()
            .map(|slot| slot.dtype().clone())
            .collect::<Vec<_>>();
        let lambdas = data
            .lambdas
            .iter()
            .map(|lambda| lambda.lambda().clone())
            .collect::<Vec<_>>();
        vortex_ensure!(
            data.higher_order_fn.return_dtype(&arg_dtypes, &lambdas)? == *dtype,
            "HigherOrderFnArray dtype does not match higher-order function return dtype"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("HigherOrderFnArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // Runtime closures carry array references and are intentionally not serializable.
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("Deserialization of HigherOrderFnArray metadata is not supported")
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        if idx < array.arg_count() {
            array.higher_order_fn().child_name(idx).to_string()
        } else {
            format!("capture[{}]", idx - array.arg_count())
        }
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let args = array.args();
        let captures = array.captures();
        let lambdas = array
            .lambdas()
            .iter()
            .map(|lambda| lambda.call(&captures))
            .collect::<Vec<_>>();
        array
            .higher_order_fn()
            .execute(&args, &lambdas, ctx)
            .map(ExecutionResult::done)
    }
}

impl OperationsVTable<HigherOrderFn> for HigherOrderFn {
    fn scalar_at(
        array: ArrayView<'_, HigherOrderFn>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<crate::scalar::Scalar> {
        array
            .array()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_array()
            .execute_scalar(index, ctx)
    }
}

impl ValidityVTable<HigherOrderFn> for HigherOrderFn {
    fn validity(array: ArrayView<'_, HigherOrderFn>) -> VortexResult<Validity> {
        let args = array.args();
        let lambdas = array
            .lambdas()
            .iter()
            .map(|lambda| lambda.lambda().clone())
            .collect::<Vec<_>>();
        array.higher_order_fn().validity(&args, &lambdas)
    }
}
