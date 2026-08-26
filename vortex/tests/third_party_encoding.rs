// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::tests_outside_test_module)]

//! Can an application write an array encoding it registered itself?
//!
//! `AppIdentity` stands in for a third-party encoding: it is registered on the session exactly
//! the way an application plugin would be, and it is a member of no Vortex edition.

use std::fmt::Display;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex::VortexSessionDefault;
use vortex::array::Array;
use vortex::array::ArrayEq;
use vortex::array::ArrayHash;
use vortex::array::ArrayId;
use vortex::array::ArrayParts;
use vortex::array::ArrayRef;
use vortex::array::ArrayView;
use vortex::array::Canonical;
use vortex::array::EqMode;
use vortex::array::ExecutionCtx;
use vortex::array::ExecutionResult;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::array_slots;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::dtype::DType;
use vortex::array::serde::ArrayChildren;
use vortex::array::session::ArraySessionExt;
use vortex::array::smallvec::smallvec;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::vtable::NotSupported;
use vortex::array::vtable::VTable;
use vortex::array::vtable::ValidityChild;
use vortex::array::vtable::ValidityVTableFromChild;
use vortex::array::vtable::with_empty_buffers;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_panic;
use vortex::editions::CORE_2026_08_1;
use vortex::editions::ComponentKind;
use vortex::editions::Edition;
use vortex::editions::EditionDeclaration;
use vortex::editions::EditionId;
use vortex::editions::EditionInclusion;
use vortex::editions::EditionMember;
use vortex::editions::EditionSessionExt;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexWriteOptions;
use vortex::file::WriteOptionsSessionExt;
use vortex::layout::LayoutStrategy;
use vortex::layout::LayoutStrategyEncodingValidator;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::session::VortexSession;
use vortex::session::registry::CachedId;
use vortex::utils::aliases::hash_set::HashSet;

// -- the third-party encoding ------------------------------------------------------------------

/// Counts `AppIdentity::deserialize` calls, so a test can tell whether the encoding actually
/// reached the file or was normalized away before serialization.
static DESERIALIZE_CALLS: AtomicUsize = AtomicUsize::new(0);

/// A minimal third-party encoding: one child array, executed through unchanged.
#[derive(Clone, Debug)]
struct AppIdentity;

#[array_slots(AppIdentity)]
struct AppIdentitySlots {
    /// The wrapped values.
    #[slot(0)]
    #[expect(dead_code, reason = "read through the generated slots view")]
    values: ArrayRef,
}

#[derive(Clone, Debug)]
struct AppIdentityData;

impl Display for AppIdentityData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArrayHash for AppIdentityData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl ArrayEq for AppIdentityData {
    fn array_eq(&self, _other: &Self, _accuracy: EqMode) -> bool {
        true
    }
}

impl VTable for AppIdentity {
    type TypedArrayData = AppIdentityData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("app.identity");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let values = AppIdentitySlotsView::from_slots(slots).values;
        vortex_ensure!(values.dtype() == dtype, "AppIdentity dtype mismatch");
        vortex_ensure!(values.len() == len, "AppIdentity len mismatch");
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("AppIdentity buffer index {idx} out of bounds")
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

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        AppIdentitySlots::NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        if !metadata.is_empty() {
            vortex_bail!("AppIdentity expects empty metadata");
        }
        DESERIALIZE_CALLS.fetch_add(1, Ordering::Relaxed);
        let values = children.get(0, dtype, len)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, AppIdentityData)
            .with_slots(smallvec![Some(values)]))
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            array.values().clone().execute::<Canonical>(ctx)?,
        ))
    }
}

impl ValidityChild<AppIdentity> for AppIdentity {
    fn validity_child(array: ArrayView<'_, AppIdentity>) -> ArrayRef {
        array.values().clone()
    }
}

impl AppIdentity {
    fn wrap(values: ArrayRef) -> VortexResult<ArrayRef> {
        let dtype = values.dtype().clone();
        let len = values.len();
        Array::try_from_parts(
            ArrayParts::new(AppIdentity, dtype, len, AppIdentityData)
                .with_slots(smallvec![Some(values)]),
        )
        .map(IntoArray::into_array)
    }
}

// -- the tests ---------------------------------------------------------------------------------

const APP_EDITION: EditionId = EditionId::new("app", 2026, 8, 0);

static APP_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: APP_EDITION,
        min_vortex_version: None,
    },
    added: &[EditionMember::array(&"app.identity")],
};

/// A default session with the third-party encoding registered, exactly as an application would.
fn app_session() -> VortexSession {
    let session = VortexSession::default();
    session.arrays().register(AppIdentity);
    session
}

fn app_array() -> VortexResult<ArrayRef> {
    AppIdentity::wrap(PrimitiveArray::from_iter(0..1024i32).into_array())
}

/// A leaf-only strategy: no repartitioning and no compression, so an already-encoded custom
/// array reaches serialization instead of being normalized away first.
fn preserving_strategy(allow: Option<HashSet<ArrayId>>) -> Arc<dyn LayoutStrategy> {
    let flat = FlatLayoutStrategy::default();
    match allow {
        // The same wrapper `WriteStrategyBuilder::with_allow_encodings` installs.
        Some(allow) => Arc::new(LayoutStrategyEncodingValidator::new(flat, allow)),
        None => Arc::new(flat),
    }
}

fn all_registered(session: &VortexSession) -> HashSet<ArrayId> {
    session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect())
}

async fn write_with(options: VortexWriteOptions, array: ArrayRef) -> VortexResult<ByteBufferMut> {
    let mut bytes = ByteBufferMut::empty();
    options.write(&mut bytes, array.to_array_stream()).await?;
    Ok(bytes)
}

/// Read the file back and return how many `AppIdentity` nodes it deserialized.
async fn read_back(session: &VortexSession, bytes: ByteBufferMut) -> VortexResult<usize> {
    DESERIALIZE_CALLS.store(0, Ordering::Relaxed);
    let file = session.open_options().open_buffer(bytes)?;
    let read = file.scan()?.into_array_stream()?.read_all().await?;
    let mut ctx = session.create_execution_ctx();
    let read = read.execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(read.as_slice::<i32>(), &(0..1024i32).collect::<Vec<_>>());
    Ok(DESERIALIZE_CALLS.load(Ordering::Relaxed))
}

/// The default strategy recompresses every chunk, so an already-encoded custom array is
/// canonicalized away before serialization. The write succeeds and the values survive, but the
/// application's encoding is silently gone: no error tells it the encoding was dropped.
#[tokio::test]
async fn default_strategy_silently_discards_an_already_encoded_custom_array() -> VortexResult<()> {
    let session = app_session();
    let bytes = write_with(session.write_options(), app_array()?).await?;
    assert_eq!(read_back(&session, bytes).await?, 0);
    Ok(())
}

/// Once the custom encoding actually reaches serialization, the write fails: the file's
/// `ArrayContext` is derived from the enabled editions and `app.identity` is in none of them.
#[tokio::test]
async fn custom_encoding_is_rejected_when_it_is_in_no_edition() -> VortexResult<()> {
    let session = app_session();
    let error = write_with(
        session
            .write_options()
            .with_strategy(preserving_strategy(None)),
        app_array()?,
    )
    .await
    .expect_err("app.identity is a member of no enabled edition");
    assert!(
        error
            .to_string()
            .contains("Array encoding app.identity not permitted by ctx"),
        "unexpected error: {error}"
    );
    Ok(())
}

/// A strategy-level encoding allowlist was documented as an explicit opt-out from editions.
/// Widening it to every registered encoding no longer readmits the third-party encoding, because
/// `VortexWriteOptions` derives the array context from the editions and nothing can override it.
#[tokio::test]
async fn with_allow_encodings_cannot_readmit_a_third_party_encoding() -> VortexResult<()> {
    let session = app_session();
    let allow = all_registered(&session);
    assert!(allow.iter().any(|id| id.as_str() == "app.identity"));

    let error = write_with(
        session
            .write_options()
            .with_strategy(preserving_strategy(Some(allow))),
        app_array()?,
    )
    .await
    .expect_err("with_allow_encodings cannot widen the edition policy");
    assert!(
        error
            .to_string()
            .contains("Array encoding app.identity not permitted by ctx"),
        "unexpected error: {error}"
    );
    Ok(())
}

/// The one path that works today: the application declares its own edition family, enables it,
/// and the encoding is written and read back.
#[tokio::test]
async fn third_party_encoding_writes_under_its_own_edition_family() -> VortexResult<()> {
    let session = app_session();
    session
        .register_edition(&APP_DECLARATION)
        .map_err(|error| vortex_err!("{error}"))?;
    session
        .enable_edition(APP_EDITION)
        .map_err(|error| vortex_err!("{error}"))?;

    let bytes = write_with(
        session
            .write_options()
            .with_strategy(preserving_strategy(None)),
        app_array()?,
    )
    .await?;
    assert_eq!(read_back(&session, bytes).await?, 1);
    Ok(())
}

/// Nothing stops an application from adding its encoding to a *frozen* first-party edition:
/// `declare_inclusion` has no freeze check. The write then succeeds while the file still claims
/// to be a `core2026.08.1` file, which no released reader can decode.
#[tokio::test]
async fn an_application_can_widen_a_frozen_core_edition() -> VortexResult<()> {
    let session = app_session();
    session
        .editions()
        .declare_inclusion(EditionInclusion::array(&"app.identity", CORE_2026_08_1))
        .map_err(|error| vortex_err!("{error}"))?;

    let bytes = write_with(
        session
            .write_options()
            .with_strategy(preserving_strategy(None)),
        app_array()?,
    )
    .await?;
    assert_eq!(read_back(&session, bytes).await?, 1);
    Ok(())
}

/// Two independently developed encodings coexist without coordinating, as long as each picks a
/// distinct family: `EnabledEditions` is keyed by family and the writer unions the families.
#[test]
fn independent_families_coexist() -> VortexResult<()> {
    const OTHER_EDITION: EditionId = EditionId::new("other", 2026, 8, 0);
    static OTHER_DECLARATION: EditionDeclaration = EditionDeclaration {
        edition: Edition {
            id: OTHER_EDITION,
            min_vortex_version: None,
        },
        added: &[EditionMember::array(&"other.encoding")],
    };

    let session = app_session();
    for (declaration, edition) in [
        (&APP_DECLARATION, APP_EDITION),
        (&OTHER_DECLARATION, OTHER_EDITION),
    ] {
        session
            .register_edition(declaration)
            .map_err(|error| vortex_err!("{error}"))?;
        session
            .enable_edition(edition)
            .map_err(|error| vortex_err!("{error}"))?;
    }

    let enabled = session.enabled_component_ids(ComponentKind::Array);
    assert!(enabled.iter().any(|id| id.as_str() == "app.identity"));
    assert!(enabled.iter().any(|id| id.as_str() == "other.encoding"));
    // ...and the first-party core encodings are still there.
    assert!(enabled.iter().any(|id| id.as_str() == "vortex.primitive"));
    Ok(())
}
