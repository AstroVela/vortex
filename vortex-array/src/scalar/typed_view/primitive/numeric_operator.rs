// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`NumericOperator`] enum for arithmetic operations on primitive scalars.

use std::fmt;

use prost::Message;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::scalar_fn::PersistableOptions;
use crate::scalar_fn::fns::operators::Operator;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Binary element-wise operations.
pub enum NumericOperator {
    /// Binary element-wise addition of two arrays or of two scalars.
    ///
    /// Errs at runtime if the sum would overflow or underflow.
    Add,
    /// Binary element-wise subtraction of two arrays or of two scalars.
    Sub,
    /// Binary element-wise multiplication of two arrays or of two scalars.
    Mul,
    /// Binary element-wise division of two arrays or of two scalars.
    Div,
}

impl fmt::Display for NumericOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl From<NumericOperator> for Operator {
    fn from(op: NumericOperator) -> Self {
        match op {
            NumericOperator::Add => Operator::Add,
            NumericOperator::Sub => Operator::Sub,
            NumericOperator::Mul => Operator::Mul,
            NumericOperator::Div => Operator::Div,
        }
    }
}

impl TryFrom<Operator> for NumericOperator {
    type Error = VortexError;

    fn try_from(op: Operator) -> Result<Self, Self::Error> {
        match op {
            Operator::Add => Ok(NumericOperator::Add),
            Operator::Sub => Ok(NumericOperator::Sub),
            Operator::Mul => Ok(NumericOperator::Mul),
            Operator::Div => Ok(NumericOperator::Div),
            _ => Err(vortex_err!(InvalidArgument: "{op} is not a numeric operator")),
        }
    }
}

/// A [`NumericOperator`] is the options type of the row function that executes the arithmetic
/// operators of `vortex.binary`. That function is internal to this crate and never registered, so
/// nothing serializes these options today. Encoding them as `vortex.binary` does keeps the two from
/// drifting if the row function is ever exposed on its own.
impl PersistableOptions for NumericOperator {
    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::BinaryOpts {
                op: Operator::from(*self).into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(metadata: &[u8], _session: &VortexSession) -> VortexResult<Self> {
        Self::try_from(Operator::try_from(pb::BinaryOpts::decode(metadata)?.op)?)
    }
}
