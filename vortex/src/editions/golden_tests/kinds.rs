// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Golden serialization fixtures for layouts, aggregate functions, expressions, and
//! extension dtypes.
//!
//! - **Layouts** are pinned as whole (tiny) Vortex files: a file's footer is the durable
//!   serialized form of its layout tree, and the read-forever check is a plain open + scan.
//!   Each layout id owns a golden file whose tree contains that layout.
//! - **Aggregate functions** are pinned as their serialized options bytes, the exact
//!   payload zone maps store next to the function id.
//! - **Expressions** are pinned as their protobuf encoding.
//! - **Extension dtypes** are pinned as the flatbuffer encoding of a `DType` using them,
//!   the form embedded in every file schema.

use std::num::NonZeroUsize;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::all_nan::AllNan;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::Time;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::extension::uuid::Uuid;
use vortex_array::extension::uuid::UuidMetadata;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_edition::EditionSessionExt;
use vortex_edition::ObjectKind;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::WriteFlatBufferExt;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::list::writer::ListLayoutStrategy;
use vortex_session::VortexSession;

use super::assert_fixture_completeness;
use super::check_golden;
use crate::VortexSessionDefault;
use crate::editions::DEFAULT_CORE_EDITION;

fn golden_session() -> VortexSession {
    VortexSession::default()
}

/// The session used to *write* layout golden files: everything the default session
/// registers, but with only the core edition enabled. The footer serializes the enabled
/// encoding table, so enabling the unstable edition (a compile-feature decision) would
/// change the golden bytes.
fn layout_write_session() -> VortexResult<VortexSession> {
    use vortex_array::array_session;
    use vortex_edition::EditionSession;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::session::LayoutSession;

    let session = array_session()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    crate::editions::register_default_editions(&session);
    session
        .enable_edition(DEFAULT_CORE_EDITION)
        .map_err(|error| vortex_err!("{error}"))?;
    Ok(session)
}

// ---------------------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------------------

/// A struct-of-columns fixture whose default-strategy layout tree contains the struct,
/// chunked, zoned, flat, and dict layouts.
fn layout_fixture_array() -> VortexResult<ArrayRef> {
    let ints: PrimitiveArray = (0..64i64).map(|i| i * 5 - 32).collect();
    let categories =
        VarBinViewArray::from_iter_str((0..64).map(|i| ["red", "green", "blue"][i % 3]));
    let flags = BoolArray::from_iter((0..64).map(|i| i % 2 == 0));
    Ok(StructArray::from_fields(&[
        ("ints", ints.into_array()),
        ("categories", categories.into_array()),
        ("flags", flags.into_array()),
    ])?
    .into_array())
}

/// A list-typed fixture, written as the file root so the list layout strategy applies to
/// it directly rather than through its non-list fallback.
fn list_fixture_array() -> VortexResult<ArrayRef> {
    use vortex_array::arrays::ListArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    let elements: PrimitiveArray = (0..12i32).collect();
    let offsets = PrimitiveArray::new(buffer![0i64, 3, 5, 6, 12], Validity::NonNullable);
    Ok(ListArray::try_new(
        elements.into_array(),
        offsets.into_array(),
        Validity::NonNullable,
    )?
    .into_array())
}

async fn write_file(
    session: &VortexSession,
    array: ArrayRef,
    strategy: Option<Arc<dyn LayoutStrategy>>,
) -> VortexResult<Vec<u8>> {
    let mut buffer = ByteBufferMut::empty();
    let options = match strategy {
        Some(strategy) => session.write_options().with_strategy(strategy),
        None => session.write_options(),
    };
    options.write(&mut buffer, array.to_array_stream()).await?;
    Ok(buffer.freeze().to_vec())
}

/// Open golden file bytes, assert the expected layout id is (still) in the footer tree for
/// the newest golden, and assert a full scan reproduces the fixture's logical value.
async fn decode_file(
    bytes: &[u8],
    id: Option<&str>,
    expected: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<()> {
    let file = session
        .open_options()
        .open_buffer(ByteBuffer::from(bytes.to_vec()))?;
    if let Some(id) = id {
        let root = file.footer().layout();
        let mut ids = vec![root.encoding_id().to_string()];
        let mut stack = root.children()?;
        while let Some(layout) = stack.pop() {
            ids.push(layout.encoding_id().to_string());
            stack.extend(layout.children()?);
        }
        if !ids.iter().any(|found| found == id) {
            return Err(vortex_err!(
                "layout {id} not present in the written file's tree: {ids:?}"
            ));
        }
    }
    let scanned = file.scan()?.into_array_stream()?.read_all().await?;
    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(scanned, expected.clone(), &mut ctx);
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[tokio::test(flavor = "multi_thread")]
async fn layout_goldens() -> VortexResult<()> {
    let session = golden_session();

    let write_session = layout_write_session()?;

    let default_array = layout_fixture_array()?;
    let default_file = write_file(&write_session, default_array.clone(), None).await?;

    let flat_array = PrimitiveArray::from_iter(0..64i32).into_array();
    let flat_file = write_file(
        &write_session,
        flat_array.clone(),
        Some(Arc::new(FlatLayoutStrategy::default())),
    )
    .await?;

    let list_array = list_fixture_array()?;
    let list_file = write_file(
        &write_session,
        list_array.clone(),
        Some(Arc::new(ListLayoutStrategy::default())),
    )
    .await?;

    // A single small input never repartitions into multiple chunks, so the chunked layout
    // gets an explicit chunked-over-flat strategy fed by a multi-chunk stream.
    let chunked_array = {
        use vortex_array::arrays::ChunkedArray;
        let chunk_a = PrimitiveArray::from_iter(0..32i64).into_array();
        let chunk_b = PrimitiveArray::from_iter(32..64i64).into_array();
        let dtype = chunk_a.dtype().clone();
        ChunkedArray::try_new(vec![chunk_a, chunk_b], dtype)?.into_array()
    };
    let chunked_file = write_file(
        &write_session,
        chunked_array.clone(),
        Some(Arc::new(ChunkedLayoutStrategy::new(
            FlatLayoutStrategy::default(),
        ))),
    )
    .await?;

    // Each layout id pins a file whose tree contains it. `vortex.stats` is exempted: it is
    // a read-only legacy layout no current writer can produce.
    let fixtures: Vec<(&'static str, &Vec<u8>, &ArrayRef)> = vec![
        ("vortex.chunked", &chunked_file, &chunked_array),
        ("vortex.dict", &default_file, &default_array),
        ("vortex.flat", &flat_file, &flat_array),
        ("vortex.list", &list_file, &list_array),
        ("vortex.struct", &default_file, &default_array),
        ("vortex.zoned", &default_file, &default_array),
    ];

    let ids: Vec<&str> = fixtures.iter().map(|(id, ..)| *id).collect();
    assert_fixture_completeness(ObjectKind::Layout, &ids);

    for (id, file, expected) in fixtures {
        decode_file(file, Some(id), expected, &session).await?;
        // `check_golden` takes a synchronous decoder; historical layout goldens are read
        // back on the ambient runtime via a nested handle.
        let handle = tokio::runtime::Handle::current();
        check_golden(ObjectKind::Layout, id, file, |bytes| {
            let newest = bytes == file.as_slice();
            tokio::task::block_in_place(|| {
                handle.block_on(decode_file(bytes, newest.then_some(id), expected, &session))
            })
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Aggregate functions
// ---------------------------------------------------------------------------------------

fn aggregation_fixtures() -> VortexResult<Vec<(&'static str, AggregateFnRef)>> {
    let bounded_bytes = NonZeroUsize::new(64).ok_or_else(|| vortex_err!("64 is non-zero"))?;
    Ok(vec![
        ("vortex.all_nan", AllNan.bind(EmptyOptions)),
        ("vortex.all_non_nan", AllNonNan.bind(EmptyOptions)),
        ("vortex.all_non_null", AllNonNull.bind(EmptyOptions)),
        ("vortex.all_null", AllNull.bind(EmptyOptions)),
        (
            "vortex.bounded_max",
            BoundedMax.bind(BoundedMaxOptions {
                max_bytes: bounded_bytes,
            }),
        ),
        (
            "vortex.bounded_min",
            BoundedMin.bind(BoundedMinOptions {
                max_bytes: bounded_bytes,
            }),
        ),
        ("vortex.max", Max.bind(NumericalAggregateOpts::skip_nans())),
        ("vortex.min", Min.bind(NumericalAggregateOpts::skip_nans())),
        ("vortex.nan_count", NanCount.bind(EmptyOptions)),
        ("vortex.null_count", NullCount.bind(EmptyOptions)),
        ("vortex.sum", Sum.bind(NumericalAggregateOpts::skip_nans())),
        (
            "vortex.uncompressed_size_in_bytes",
            UncompressedSizeInBytes.bind(EmptyOptions),
        ),
    ])
}

#[cfg_attr(miri, ignore)]
#[test]
fn aggregation_goldens() -> VortexResult<()> {
    let session = golden_session();
    let fixtures = aggregation_fixtures()?;

    let ids: Vec<&str> = fixtures.iter().map(|(id, _)| *id).collect();
    assert_fixture_completeness(ObjectKind::Aggregation, &ids);

    for (id, aggregate_fn) in &fixtures {
        assert_eq!(aggregate_fn.id().as_str(), *id);
        let current = aggregate_fn
            .options()
            .serialize()?
            .ok_or_else(|| vortex_err!("aggregate function {id} is not serializable"))?;
        check_golden(ObjectKind::Aggregation, id, &current, |bytes| {
            let plugin = session
                .aggregate_fns()
                .find_plugin(&aggregate_fn.id())
                .ok_or_else(|| vortex_err!("aggregate function {id} is not registered"))?;
            let decoded = plugin.deserialize(bytes, &session)?;
            if &decoded != aggregate_fn {
                return Err(vortex_err!(
                    "golden deserialized to {decoded} instead of {aggregate_fn}"
                ));
            }
            Ok(())
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------------------

#[cfg(feature = "unstable_encodings")]
#[cfg_attr(miri, ignore)]
#[test]
fn expression_goldens() -> VortexResult<()> {
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
    use vortex_tensor::scalar_fns::inner_product::InnerProduct;
    use vortex_tensor::scalar_fns::l2_norm::L2Norm;
    use vortex_tensor::vector::Vector;

    use super::arrays::decode_ipc;
    use super::arrays::ipc_bytes;

    // Expressions persist as scalar-fn arrays: the array encoding id is the scalar
    // function id, and the serialized metadata is the function's options plus the expression
    // structure. Registering the plugins is normally gated behind
    // `vortex_tensor::SCALAR_FN_ARRAY_TENSOR_PLUGIN_ENV`.
    let session = golden_session();
    session
        .arrays()
        .register(ScalarFnArrayPlugin::new(CosineSimilarity));
    session
        .arrays()
        .register(ScalarFnArrayPlugin::new(InnerProduct));
    session.arrays().register(ScalarFnArrayPlugin::new(L2Norm));

    let vectors = |seed: f32| -> VortexResult<ArrayRef> {
        let elements: PrimitiveArray = (0..12).map(|i| (i as f32) * 0.5 + seed).collect();
        let storage =
            FixedSizeListArray::try_new(elements.into_array(), 4, Validity::NonNullable, 3)?;
        Vector::try_new_vector_array(storage.into_array())
    };

    let fixtures: Vec<(&'static str, ArrayRef)> = vec![
        (
            "vortex.tensor.cosine_similarity",
            CosineSimilarity::try_new_array(vectors(1.0)?, vectors(2.0)?)?.into_array(),
        ),
        (
            "vortex.tensor.inner_product",
            InnerProduct::try_new_array(vectors(1.0)?, vectors(2.0)?)?.into_array(),
        ),
        (
            "vortex.tensor.l2_norm",
            L2Norm::try_new_array(vectors(1.0)?)?.into_array(),
        ),
    ];

    let ids: Vec<&str> = fixtures.iter().map(|(id, _)| *id).collect();
    assert_fixture_completeness(ObjectKind::Expression, &ids);

    for (id, array) in &fixtures {
        assert_eq!(array.encoding_id().as_str(), *id);
        let current = ipc_bytes(array, &session)?;
        check_golden(ObjectKind::Expression, id, &current, |bytes| {
            decode_ipc(bytes, array, &session)
        });
    }
    Ok(())
}

#[cfg(not(feature = "unstable_encodings"))]
#[test]
fn expression_goldens_completeness() {
    // Without the unstable feature no expression members are required; keep the
    // completeness check running so a core expression member cannot land uncovered.
    assert_fixture_completeness(ObjectKind::Expression, &[]);
}

// ---------------------------------------------------------------------------------------
// Extension dtypes
// ---------------------------------------------------------------------------------------

fn extension_dtype_fixtures() -> VortexResult<Vec<(&'static str, DType)>> {
    #[cfg_attr(
        not(feature = "unstable_encodings"),
        expect(unused_mut, reason = "extended only by the unstable tensor dtypes")
    )]
    let mut fixtures: Vec<(&'static str, DType)> = vec![
        (
            "vortex.date",
            DType::Extension(Date::new(TimeUnit::Days, Nullability::NonNullable).erased()),
        ),
        (
            "vortex.time",
            DType::Extension(Time::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
        ),
        (
            "vortex.timestamp",
            DType::Extension(
                Timestamp::new(TimeUnit::Microseconds, Nullability::Nullable).erased(),
            ),
        ),
        (
            "vortex.uuid",
            DType::Extension(
                ExtDType::try_with_vtable(
                    Uuid,
                    UuidMetadata::default(),
                    DType::FixedSizeList(
                        Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
                        16,
                        Nullability::NonNullable,
                    ),
                )?
                .erased(),
            ),
        ),
    ];

    #[cfg(feature = "unstable_encodings")]
    fixtures.extend(tensor_dtype_fixtures()?);

    Ok(fixtures)
}

#[cfg(feature = "unstable_encodings")]
fn tensor_dtype_fixtures() -> VortexResult<Vec<(&'static str, DType)>> {
    use vortex_array::EmptyMetadata;
    use vortex_tensor::fixed_shape_tensor::FixedShapeTensor;
    use vortex_tensor::fixed_shape_tensor::FixedShapeTensorMetadata;
    use vortex_tensor::vector::Vector;

    let f32_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    Ok(vec![
        (
            "vortex.tensor.fixed_shape_tensor",
            DType::Extension(
                ExtDType::try_with_vtable(
                    FixedShapeTensor,
                    FixedShapeTensorMetadata::new(vec![2, 3]),
                    DType::FixedSizeList(Arc::new(f32_dtype.clone()), 6, Nullability::NonNullable),
                )?
                .erased(),
            ),
        ),
        (
            "vortex.tensor.vector",
            DType::Extension(
                ExtDType::try_with_vtable(
                    Vector,
                    EmptyMetadata,
                    DType::FixedSizeList(Arc::new(f32_dtype), 4, Nullability::NonNullable),
                )?
                .erased(),
            ),
        ),
    ])
}

#[cfg_attr(miri, ignore)]
#[test]
fn extension_dtype_goldens() -> VortexResult<()> {
    let session = golden_session();
    let fixtures = extension_dtype_fixtures()?;

    let ids: Vec<&str> = fixtures.iter().map(|(id, _)| *id).collect();
    assert_fixture_completeness(ObjectKind::ExtensionDType, &ids);

    for (id, dtype) in &fixtures {
        let DType::Extension(ext) = dtype else {
            return Err(vortex_err!("fixture for {id} is not an extension dtype"));
        };
        assert_eq!(ext.id().as_str(), *id);
        // The session must be able to resolve the id, or files with this schema would be
        // unreadable.
        assert!(
            session.dtypes().registry().get(&ext.id()).is_some(),
            "{id} is not registered in the default session"
        );

        let current = dtype.write_flatbuffer_bytes()?.to_vec();
        check_golden(ObjectKind::ExtensionDType, id, &current, |bytes| {
            let decoded = DType::from_flatbuffer(FlatBuffer::copy_from(bytes), &session)?;
            if &decoded != dtype {
                return Err(vortex_err!(
                    "golden deserialized to {decoded} instead of {dtype}"
                ));
            }
            Ok(())
        });
    }
    Ok(())
}
