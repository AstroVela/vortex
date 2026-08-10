// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Golden serialization fixtures for every array encoding in an edition.
//!
//! The serialized surface is the IPC stream of a single-chunk array iterator: it is
//! self-describing (dtype plus per-message encoding-id table), it is exactly what crosses
//! process boundaries, and its body is the same serialized array form the file format
//! embeds in segments.
//!
//! Fixtures are frozen: once a golden exists for an encoding, its fixture must never
//! change, or the old goldens would no longer describe the fixture's logical value.

#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture sizes are small compile-time constants"
)]

use std::io::Cursor;

use vortex_alp::RDEncoder;
use vortex_alp::RDEncoderExt;
use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::VariantArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_bytebool::ByteBool;
use vortex_datetime_parts::DateTimeParts;
use vortex_datetime_parts::split_temporal;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_edition::ObjectKind;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::FoR;
use vortex_fastlanes::RLE;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_ipc::iterator::ArrayIteratorIPC;
use vortex_ipc::iterator::SyncIPCReader;
use vortex_pco::Pco;
use vortex_runend::RunEnd;
use vortex_runend::compress::runend_encode;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;
use vortex_zigzag::zigzag_encode;

use super::assert_fixture_completeness;
use super::check_golden;
use crate::VortexSessionDefault;

const N: usize = 32;

/// The session used to serialize and deserialize array fixtures: the default session plus
/// the edition members that are not registered by default.
fn golden_session() -> VortexSession {
    let session = VortexSession::default();
    // `vortex.patched` is an unstable edition member whose default registration is gated
    // behind an experiment flag.
    session.arrays().register(Patched);
    #[cfg(feature = "unstable_encodings")]
    vortex_parquet_variant::initialize(&session);
    session
}

pub(super) fn ipc_bytes(array: &ArrayRef, session: &VortexSession) -> VortexResult<Vec<u8>> {
    let iter = ArrayIteratorAdapter::new(array.dtype().clone(), std::iter::once(Ok(array.clone())));
    iter.write_ipc(Vec::new(), session)
}

pub(super) fn decode_ipc(
    bytes: &[u8],
    expected: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<()> {
    let reader = SyncIPCReader::try_new(Cursor::new(bytes), session)?;
    if reader.dtype() != expected.dtype() {
        return Err(vortex_err!(
            "golden dtype {} != fixture dtype {}",
            reader.dtype(),
            expected.dtype()
        ));
    }
    let chunks: Vec<ArrayRef> = reader.collect::<VortexResult<_>>()?;
    if chunks.len() != 1 {
        return Err(vortex_err!("expected one chunk, got {}", chunks.len()));
    }
    let decoded = chunks.into_iter().next().vortex_expect("one chunk");
    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(decoded, expected.clone(), &mut ctx);
    Ok(())
}

fn prim_i32() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..N as i32).map(|i| i * 3 - 7))
}

#[expect(
    clippy::too_many_lines,
    reason = "one fixture per edition array encoding"
)]
fn fixtures(ctx: &mut ExecutionCtx) -> VortexResult<Vec<(&'static str, ArrayRef)>> {
    let mut fixtures: Vec<(&'static str, ArrayRef)> = Vec::new();

    let small_ints: PrimitiveArray = (0..N as u32).map(|i| i % 16).collect();
    fixtures.push((
        "fastlanes.bitpacked",
        bitpack_encode(&small_ints, 4, None, ctx)?.into_array(),
    ));

    let offset_ints: PrimitiveArray = (0..N as i32).map(|i| 1_000_000 + (i % 50)).collect();
    fixtures.push(("fastlanes.for", FoR::encode(offset_ints, ctx)?.into_array()));

    let runs: PrimitiveArray = (0..N as i32).map(|i| i / 8).collect();
    fixtures.push((
        "fastlanes.rle",
        RLE::encode(runs.as_view(), ctx)?.into_array(),
    ));

    let prices: PrimitiveArray = (0..N).map(|i| 100.0 + (i as f64) * 0.25).collect();
    fixtures.push((
        "vortex.alp",
        alp_encode(prices.as_view(), None, ctx)?.into_array(),
    ));

    let reals: PrimitiveArray = (0..N)
        .map(|i| 98.6 + ((i * 7 + 13) % 100) as f64 / 1000.0)
        .collect();
    let rd_encoder = RDEncoder::new::<f64>(reals.as_slice::<f64>());
    fixtures.push((
        "vortex.alprd",
        rd_encoder.encode(reals.as_view()).into_array(),
    ));

    fixtures.push((
        "vortex.bool",
        BoolArray::from_iter((0..N).map(|i| (i % 7 != 0).then_some(i % 3 == 0))).into_array(),
    ));

    fixtures.push((
        "vortex.bytebool",
        ByteBool::from_vec((0..N).map(|i| i % 2 == 0).collect(), Validity::NonNullable)
            .into_array(),
    ));

    let chunks = vec![prim_i32().into_array(), prim_i32().into_array()];
    let chunk_dtype = chunks[0].dtype().clone();
    fixtures.push((
        "vortex.chunked",
        ChunkedArray::try_new(chunks, chunk_dtype)?.into_array(),
    ));

    fixtures.push(("vortex.constant", ConstantArray::new(42i32, N).into_array()));

    let base_us: i64 = 1_704_067_200_000_000;
    let ts_us: PrimitiveArray = (0..N as i64).map(|i| base_us + i * 3_600_000_000).collect();
    let temporal = TemporalArray::new_timestamp(ts_us.into_array(), TimeUnit::Microseconds, None);
    let temporal_dtype = temporal.dtype().clone();
    let parts = split_temporal(temporal, ctx)?;
    fixtures.push((
        "vortex.datetimeparts",
        DateTimeParts::try_new(temporal_dtype, parts.days, parts.seconds, parts.subseconds)?
            .into_array(),
    ));

    fixtures.push((
        "vortex.decimal",
        DecimalArray::new(
            buffer![10025i128, -3550, 0, 99999],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        )
        .into_array(),
    ));

    let dbp_values: PrimitiveArray = (0..N as i64).map(|i| i * 100 + (i % 100)).collect();
    fixtures.push((
        "vortex.decimal_byte_parts",
        DecimalByteParts::try_new(dbp_values.into_array(), DecimalDType::new(10, 2))?.into_array(),
    ));

    let codes: PrimitiveArray = (0..N as u32).map(|i| i % 4).collect();
    let dict_values = VarBinViewArray::from_iter_str(["red", "green", "blue", "yellow"]);
    fixtures.push((
        "vortex.dict",
        DictArray::try_new(codes.into_array(), dict_values.into_array())?.into_array(),
    ));

    let base_ms: i64 = 1_704_067_200_000;
    let ts_ms: PrimitiveArray = (0..N as i64).map(|i| base_ms + i * 1000).collect();
    fixtures.push((
        "vortex.ext",
        TemporalArray::new_timestamp(ts_ms.into_array(), TimeUnit::Milliseconds, None).into_array(),
    ));

    let fsl_elements: PrimitiveArray = (0..12i32).collect();
    fixtures.push((
        "vortex.fixed_size_list",
        FixedSizeListArray::try_new(fsl_elements.into_array(), 3, Validity::NonNullable, 4)?
            .into_array(),
    ));

    let urls: Vec<String> = (0..N)
        .map(|i| format!("https://example.com/api/v1/users/{i}"))
        .collect();
    let urls: ArrayRef =
        VarBinArray::from_strs(urls.iter().map(|s| s.as_str()).collect::<Vec<_>>()).into_array();
    let fsst_compressor = fsst_train_compressor(&urls, ctx)?;
    fixtures.push((
        "vortex.fsst",
        fsst_compress(&urls, &fsst_compressor, ctx)?.into_array(),
    ));

    let list_elements: PrimitiveArray = (0..10i32).collect();
    let list_offsets = PrimitiveArray::new(buffer![0i64, 3, 5, 6, 10], Validity::NonNullable);
    fixtures.push((
        "vortex.list",
        ListArray::try_new(
            list_elements.into_array(),
            list_offsets.into_array(),
            Validity::NonNullable,
        )?
        .into_array(),
    ));

    let lv_elements: PrimitiveArray = (0..10i32).collect();
    let lv_offsets = PrimitiveArray::new(buffer![0u32, 3, 5, 6], Validity::NonNullable);
    let lv_sizes = PrimitiveArray::new(buffer![3u32, 2, 1, 4], Validity::NonNullable);
    fixtures.push((
        "vortex.listview",
        ListViewArray::try_new(
            lv_elements.into_array(),
            lv_offsets.into_array(),
            lv_sizes.into_array(),
            Validity::NonNullable,
        )?
        .into_array(),
    ));

    let map_keys = VarBinViewArray::from_iter_str(["a", "b", "c", "d"]);
    let map_values: PrimitiveArray = (0..4i32).collect();
    let entries_struct = StructArray::from_fields(&[
        ("key", map_keys.into_array()),
        ("value", map_values.into_array()),
    ])?;
    let entry_offsets = PrimitiveArray::new(buffer![0u32, 2, 3], Validity::NonNullable);
    let entry_sizes = PrimitiveArray::new(buffer![2u32, 1, 1], Validity::NonNullable);
    let entries = ListViewArray::try_new(
        entries_struct.into_array(),
        entry_offsets.into_array(),
        entry_sizes.into_array(),
        Validity::NonNullable,
    )?;
    let map_dtype = MapDType::try_new(
        DType::Utf8(Nullability::NonNullable),
        DType::Primitive(PType::I32, Nullability::NonNullable),
        false,
    )?;
    fixtures.push((
        "vortex.map",
        MapArray::try_new(map_dtype, entries)?.into_array(),
    ));

    let mask_validity = Validity::from_iter((0..N).map(|i| i % 5 != 0));
    fixtures.push((
        "vortex.masked",
        MaskedArray::try_new(prim_i32().into_array(), mask_validity)?.into_array(),
    ));

    fixtures.push(("vortex.null", NullArray::new(N).into_array()));

    let pco_ints: PrimitiveArray = (0..N as i64).map(|i| i * i + (i % 17) * 1000).collect();
    fixtures.push((
        "vortex.pco",
        Pco::from_primitive(pco_ints.as_view(), 8, 0, ctx)?.into_array(),
    ));

    fixtures.push(("vortex.primitive", prim_i32().into_array()));

    let re_runs: PrimitiveArray = (0..N as i64).map(|i| i / 8).collect();
    let (re_ends, re_values) = runend_encode(re_runs.as_view(), ctx);
    fixtures.push((
        "vortex.runend",
        RunEnd::try_new(re_ends.into_array(), re_values, ctx)?.into_array(),
    ));

    fixtures.push((
        "vortex.sequence",
        Sequence::try_new_typed::<i32>(7, 3, Nullability::NonNullable, N)?.into_array(),
    ));

    let sparse_indices: PrimitiveArray = [2u64, 9, 20].into_iter().collect();
    let sparse_values: PrimitiveArray = [100i32, 200, 300].into_iter().collect();
    fixtures.push((
        "vortex.sparse",
        Sparse::try_new(
            sparse_indices.into_array(),
            sparse_values.into_array(),
            N,
            Scalar::primitive(0i32, Nullability::NonNullable),
        )?
        .into_array(),
    ));

    fixtures.push((
        "vortex.struct",
        StructArray::from_fields(&[
            ("ints", prim_i32().into_array()),
            (
                "flags",
                BoolArray::from_iter((0..N).map(|i| i % 2 == 0)).into_array(),
            ),
        ])?
        .into_array(),
    ));

    fixtures.push((
        "vortex.varbin",
        VarBinArray::from_nullable_strs(vec![Some("hello"), None, Some("world"), Some("")])
            .into_array(),
    ));

    fixtures.push((
        "vortex.varbinview",
        VarBinViewArray::from_iter_str(["", "hello", "こんにちは", "a-longer-string-fixture"])
            .into_array(),
    ));

    let variant_storage = ConstantArray::new(
        Scalar::variant(Scalar::primitive(1i32, Nullability::NonNullable)),
        8,
    )
    .into_array();
    fixtures.push((
        "vortex.variant",
        VariantArray::try_new(variant_storage, None)?.into_array(),
    ));

    let signed: PrimitiveArray = (0..N as i32)
        .map(|i| if i % 2 == 0 { i } else { -i })
        .collect();
    fixtures.push((
        "vortex.zigzag",
        zigzag_encode(signed.as_view())?.into_array(),
    ));

    #[cfg(feature = "zstd")]
    {
        let zstd_ints: PrimitiveArray = (0..N as i32).map(|i| i / 8).collect();
        fixtures.push((
            "vortex.zstd",
            vortex_zstd::Zstd::from_primitive(&zstd_ints, 3, 128, ctx)?.into_array(),
        ));
    }

    #[cfg(feature = "unstable_encodings")]
    fixtures.extend(unstable_fixtures(ctx)?);

    Ok(fixtures)
}

#[cfg(feature = "unstable_encodings")]
fn unstable_fixtures(ctx: &mut ExecutionCtx) -> VortexResult<Vec<(&'static str, ArrayRef)>> {
    use vortex_array::patches::Patches;
    use vortex_fastlanes::Delta;
    use vortex_onpair::onpair_compress;
    use vortex_parquet_variant::ParquetVariant;
    use vortex_tensor::encodings::normalized::normalize;
    use vortex_tensor::vector::Vector;

    let mut fixtures: Vec<(&'static str, ArrayRef)> = Vec::new();

    let monotonic: PrimitiveArray = (0..N as u64).map(|i| i * 3 + 1000).collect();
    fixtures.push((
        "fastlanes.delta",
        Delta::try_from_primitive_array(&monotonic, ctx)?.into_array(),
    ));

    let repeated: Vec<String> = (0..N)
        .map(|i| format!("common-prefix-value-{}", i % 4))
        .collect();
    let repeated = VarBinViewArray::from_iter_str(repeated.iter().map(|s| s.as_str())).into_array();
    fixtures.push((
        "vortex.onpair",
        onpair_compress(&repeated, Default::default(), ctx)?,
    ));

    let metadata =
        VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00", b"\x01\x00"]).into_array();
    let typed_value = PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array();
    fixtures.push((
        "vortex.parquet.variant",
        ParquetVariant::try_new(Validity::NonNullable, metadata, None, Some(typed_value))?
            .into_array(),
    ));

    let base = PrimitiveArray::from_option_iter((0..N as u64).map(Some)).into_array();
    let patches = Patches::new(
        N,
        0,
        PrimitiveArray::from_iter([1u32, 5, 6]).into_array(),
        PrimitiveArray::from_option_iter([Some(100u64), Some(200), Some(300)]).into_array(),
        None,
    )?;
    fixtures.push((
        "vortex.patched",
        Patched::from_array_and_patches(base, &patches, ctx)?.into_array(),
    ));

    let vec_elements: PrimitiveArray = (0..12).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let vec_storage =
        FixedSizeListArray::try_new(vec_elements.into_array(), 4, Validity::NonNullable, 3)?;
    let vectors = Vector::try_new_vector_array(vec_storage.into_array())?;
    fixtures.push((
        "vortex.tensor.normalized",
        normalize(vectors, ctx)?.into_array(),
    ));

    #[cfg(feature = "zstd")]
    {
        let strings = VarBinViewArray::from_iter_str((0..N).map(|i| {
            if i % 2 == 0 {
                "zstd-buffers-fixture"
            } else {
                "another-repeated-value"
            }
        }))
        .into_array();
        fixtures.push((
            "vortex.zstd_buffers",
            vortex_zstd::ZstdBuffers::compress(&strings, 3, &golden_session())?.into_array(),
        ));
    }

    Ok(fixtures)
}

#[cfg_attr(miri, ignore)]
#[test]
fn array_goldens() -> VortexResult<()> {
    let session = golden_session();
    let mut ctx = session.create_execution_ctx();
    let fixtures = fixtures(&mut ctx)?;

    let ids: Vec<&str> = fixtures.iter().map(|(id, _)| *id).collect();
    assert_fixture_completeness(ObjectKind::Array, &ids);

    for (id, array) in &fixtures {
        assert_eq!(
            array.encoding_id().as_str(),
            *id,
            "fixture for {id} produced root encoding {}",
            array.encoding_id()
        );
        let current = ipc_bytes(array, &session)?;
        check_golden(ObjectKind::Array, id, &current, |bytes| {
            decode_ipc(bytes, array, &session)
        });
    }
    Ok(())
}
