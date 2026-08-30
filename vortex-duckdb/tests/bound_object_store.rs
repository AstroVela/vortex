// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(vortex_vane_distributed)]
// This is an integration-test crate; its tests intentionally live at crate root.
#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;

use bound_object_store::BoundObjectStore;
use futures::executor::block_on;
use object_store::Error;
use object_store::GetOptions;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::memory::InMemory;
use object_store::path::Path;

#[path = "../src/bound_object_store.rs"]
mod bound_object_store;

fn assert_precondition(error: Error) {
    assert!(
        matches!(error, Error::Precondition { .. }),
        "expected a precondition failure, got {error}"
    );
}

#[test]
fn test_rejects_same_size_replacement_before_metadata_check() {
    block_on(async {
        let location = Path::from("same-size.vortex");
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        store.put(&location, "first".into()).await.unwrap();
        let original = store.head(&location).await.unwrap();
        let original_size = original.size;
        let bound_store =
            BoundObjectStore::new(Arc::clone(&store), original.e_tag, original.version);

        store.put(&location, "other".into()).await.unwrap();
        let replacement = store.head(&location).await.unwrap();
        assert_eq!(replacement.size, original_size);
        let error = bound_store.head(&location).await.unwrap_err();
        assert_precondition(error);

        let error = bound_store
            .get_opts(
                &location,
                GetOptions::new().with_if_match(replacement.e_tag),
            )
            .await
            .unwrap_err();
        assert_precondition(error);
    });
}

#[test]
fn test_rejects_same_size_replacement_between_metadata_and_range_read() {
    block_on(async {
        let location = Path::from("same-size.vortex");
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        store.put(&location, "first".into()).await.unwrap();
        let original = store.head(&location).await.unwrap();
        let original_size = original.size;
        let bound_store =
            BoundObjectStore::new(Arc::clone(&store), original.e_tag, original.version);

        bound_store.head(&location).await.unwrap();
        store.put(&location, "other".into()).await.unwrap();
        assert_eq!(store.head(&location).await.unwrap().size, original_size);
        let error = bound_store
            .get_opts(
                &location,
                GetOptions::new().with_range(Some(0..original_size)),
            )
            .await
            .unwrap_err();
        assert_precondition(error);

        let ranges = [0..2, 2..original_size];
        let error = bound_store
            .get_ranges(&location, &ranges)
            .await
            .unwrap_err();
        assert_precondition(error);
    });
}
