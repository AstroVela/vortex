// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::sync::Arc;

use compio::buf::BufResult;
use compio::buf::IntoInner;
use compio::buf::IoBuf;
use compio::io::AsyncReadAtExt;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;

/// Default number of concurrent reads issued for a local Compio file.
pub const DEFAULT_CONCURRENCY: usize = 32;

/// A completion-based positioned reader for a local file.
///
/// Reads are issued through the active Compio runtime directly into aligned Vortex buffers. On
/// Linux, Compio uses io_uring by default. Files are opened without `O_DIRECT`, so reads retain the
/// operating system page cache.
#[derive(Clone)]
pub struct CompioFileReadAt {
    uri: Arc<str>,
    file: Arc<compio::fs::File>,
    size: u64,
    handle: crate::CompioHandle,
}

impl CompioFileReadAt {
    /// Open a local file for completion-based positioned reads.
    ///
    /// The associated [`crate::CompioRuntime`] must be driven while this future and subsequent read
    /// futures are pending.
    pub async fn open(path: impl AsRef<Path>, handle: crate::CompioHandle) -> VortexResult<Self> {
        let path = path.as_ref().to_path_buf();
        let uri: Arc<str> = path.to_string_lossy().into();
        let (file, size) = handle
            .spawn_local(move || async move {
                let file = compio::fs::File::open(path).await?;
                let size = file.metadata().await?.len();
                std::io::Result::Ok((file, size))
            })
            .await?;
        Ok(Self {
            uri,
            file: Arc::new(file),
            size,
            handle,
        })
    }
}

impl VortexReadAt for CompioFileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn concurrency(&self) -> usize {
        DEFAULT_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.size;
        async move { Ok(size) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let file = Arc::clone(&self.file);
        self.handle
            .spawn_local(move || async move {
                let buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
                // `BufferMut` may over-allocate to achieve alignment. Restrict Compio's owned view to
                // the requested length so `read_exact_at` does not fill the spare capacity too.
                let buffer = buffer.slice(..length);
                let BufResult(result, buffer) = file.read_exact_at(buffer, offset).await;
                result?;

                let buffer = buffer.into_inner();
                Ok(BufferHandle::new_host(buffer.freeze()))
            })
            .boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use futures::future::join_all;
    use tempfile::NamedTempFile;
    use vortex_buffer::Alignment;
    use vortex_error::VortexError;
    use vortex_error::VortexResult;
    use vortex_io::VortexReadAt;
    use vortex_io::runtime::BlockingRuntime;

    use crate::CompioFileReadAt;
    use crate::CompioRuntime;

    const DATA: &[u8] = b"completion-based Vortex reads";

    #[test]
    fn reads_exact_aligned_ranges() -> VortexResult<()> {
        let mut temp = NamedTempFile::new()?;
        temp.write_all(DATA)?;
        temp.flush()?;

        let runtime = CompioRuntime::new()?;
        let handle = runtime.compio_handle();
        runtime.block_on(async move {
            let reader = CompioFileReadAt::open(temp.path(), handle).await?;
            assert_eq!(reader.size().await?, DATA.len() as u64);

            let alignment = Alignment::new(4096);
            let buffer = reader.read_at(17, 6, alignment).await?.unwrap_host();
            assert_eq!(buffer.as_slice(), b"Vortex");
            assert_eq!(buffer.len(), 6);
            assert!(alignment.is_ptr_aligned(buffer.as_ptr()));
            VortexResult::Ok(())
        })
    }

    #[test]
    fn supports_concurrent_positioned_reads() -> VortexResult<()> {
        let mut temp = NamedTempFile::new()?;
        temp.write_all(DATA)?;
        temp.flush()?;

        let runtime = CompioRuntime::new()?;
        let handle = runtime.compio_handle();
        runtime.block_on(async move {
            let reader = CompioFileReadAt::open(temp.path(), handle).await?;
            let reads = [
                reader.read_at(0, 10, Alignment::none()),
                reader.read_at(17, 6, Alignment::none()),
                reader.read_at(24, 5, Alignment::none()),
            ];
            let buffers = join_all(reads)
                .await
                .into_iter()
                .collect::<VortexResult<Vec<_>>>()?;

            assert_eq!(buffers[0].as_host().as_slice(), b"completion");
            assert_eq!(buffers[1].as_host().as_slice(), b"Vortex");
            assert_eq!(buffers[2].as_host().as_slice(), b"reads");
            VortexResult::Ok(())
        })
    }

    #[test]
    fn reports_unexpected_eof() -> VortexResult<()> {
        let mut temp = NamedTempFile::new()?;
        temp.write_all(DATA)?;
        temp.flush()?;

        let runtime = CompioRuntime::new()?;
        let handle = runtime.compio_handle();
        runtime.block_on(async move {
            let reader = CompioFileReadAt::open(temp.path(), handle).await?;
            let error = reader
                .read_at(DATA.len() as u64 - 2, 8, Alignment::none())
                .await
                .expect_err("short positioned read must fail");
            assert!(matches!(
                &error,
                VortexError::Io(error, _) if error.kind() == std::io::ErrorKind::UnexpectedEof
            ));
            VortexResult::Ok(())
        })
    }
}
