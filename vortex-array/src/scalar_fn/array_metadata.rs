// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array serde for scalar functions whose children's dtypes are not recoverable from the parent's
//! dtype.
//!
//! A [`ScalarFnArray`](crate::arrays::ScalarFnArray) persists its children as untyped arrays, so
//! deserialization has to be told each child's [`DType`]. Whenever the parent's dtype loses that
//! information (an extension-typed input behind a primitive output, or per-child nullability
//! collapsed into a union), the function has to persist the child dtypes itself. These helpers do
//! that once, for any arity, and carry the function's options along the way.

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use crate::dtype::DType;
use crate::dtype::proto::dtype as pb;
use crate::scalar_fn::PersistableOptions;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::StrictScalarFnVTable;
use crate::serde::ArrayChildren;

/// The persisted metadata of a scalar-fn array: its children's dtypes and its serialized options.
#[derive(Clone, prost::Message)]
struct ScalarFnArrayMetadata {
    #[prost(message, repeated, tag = "1")]
    child_dtypes: Vec<pb::DType>,
    #[prost(bytes = "vec", optional, tag = "2")]
    options: Option<Vec<u8>>,
}

/// Encodes the dtypes of the first `arity` children of `view`, plus its options, for
/// [`decode_children_and_options`].
pub fn encode_children_and_options<V: ScalarFnVTable>(
    view: &ScalarFnArrayView<V>,
    arity: usize,
) -> VortexResult<Vec<u8>>
where
    V::Options: PersistableOptions,
{
    let scalar_fn_array = view.as_::<ScalarFn>();
    let child_dtypes = (0..arity)
        .map(|i| scalar_fn_array.child_at(i).dtype().try_into())
        .collect::<VortexResult<Vec<_>>>()?;
    let options = view.options.serialize()?;

    Ok(ScalarFnArrayMetadata {
        child_dtypes,
        options,
    }
    .encode_to_vec())
}

/// Rebuild a scalar-fn array from metadata written by [`encode_children_and_options`], re-checking
/// the restored children against the function's own dtype rules.
///
/// Serves as the entire [`ScalarFnArrayVTable::deserialize`](
/// crate::arrays::scalar_fn::plugin::ScalarFnArrayVTable::deserialize) for any strict function whose
/// child dtypes its output cannot recover.
pub fn decode_scalar_fn_array<V: StrictScalarFnVTable>(
    vtable: &V,
    arity: usize,
    len: usize,
    metadata: &[u8],
    children: &dyn ArrayChildren,
    session: &VortexSession,
) -> VortexResult<ScalarFnArrayParts<V>> {
    let (children, options) = decode_children_and_options(metadata, arity, len, children, session)?;

    let child_dtypes = children
        .iter()
        .map(|child| child.dtype().clone())
        .collect::<Vec<_>>();
    vtable.return_element_dtype(&options, &child_dtypes)?;

    Ok(ScalarFnArrayParts { options, children })
}

/// Decodes metadata written by [`encode_children_and_options`], requiring exactly `arity` child
/// dtypes, and fetches each child from `children` at its persisted dtype.
pub fn decode_children_and_options<O: PersistableOptions>(
    metadata: &[u8],
    arity: usize,
    len: usize,
    children: &dyn ArrayChildren,
    session: &VortexSession,
) -> VortexResult<(Vec<ArrayRef>, O)> {
    let metadata = ScalarFnArrayMetadata::decode(metadata)
        .map_err(|e| vortex_err!("Failed to decode scalar function array metadata: {e}"))?;
    vortex_ensure_eq!(
        metadata.child_dtypes.len(),
        arity,
        "expected {arity} serialized child dtypes",
    );

    let children = metadata
        .child_dtypes
        .iter()
        .map(|dtype| DType::from_proto(dtype, session))
        .enumerate()
        .map(|(i, dtype)| children.get(i, &dtype?, len))
        .collect::<VortexResult<Vec<_>>>()?;
    let options = O::deserialize(metadata.options.as_deref().unwrap_or_default(), session)?;

    Ok((children, options))
}
