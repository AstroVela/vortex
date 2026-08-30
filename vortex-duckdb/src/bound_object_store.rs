// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Object-store adapter that preserves the exact object selected by a Vane bind.

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use object_store::CopyOptions;
use object_store::GetOptions;
use object_store::GetResult;
use object_store::ListResult;
use object_store::MultipartUpload;
use object_store::ObjectMeta;
use object_store::ObjectStore;
use object_store::PutMultipartOptions;
use object_store::PutOptions;
use object_store::PutPayload;
use object_store::PutResult;
use object_store::Result;
use object_store::path::Path;

/// Pins every HEAD and data request to the object identity selected at bind time.
#[derive(Debug)]
pub(crate) struct BoundObjectStore {
    inner: Arc<dyn ObjectStore>,
    e_tag: Option<String>,
    version: Option<String>,
}

impl BoundObjectStore {
    /// Wrap an object store with the version and/or ETag selected by the coordinator.
    pub(crate) fn new(
        inner: Arc<dyn ObjectStore>,
        e_tag: Option<String>,
        version: Option<String>,
    ) -> Self {
        Self {
            inner,
            e_tag,
            version,
        }
    }
}

impl Display for BoundObjectStore {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "BoundObjectStore<{}>", self.inner)
    }
}

#[async_trait]
// Keep ObjectStore's default range helpers: they route every request through
// this implementation's identity-pinned `get_opts` instead of bypassing it.
impl ObjectStore for BoundObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> Result<PutResult> {
        self.inner.put_opts(location, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        options: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, options).await
    }

    async fn get_opts(&self, location: &Path, mut options: GetOptions) -> Result<GetResult> {
        options.if_match.clone_from(&self.e_tag);
        options.version.clone_from(&self.version);
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}
