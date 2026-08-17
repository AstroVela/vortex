// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(target_os = "linux")]
mod cufile;
#[cfg(target_os = "linux")]
mod direct;

use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::error::VortexResult;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::Handle;
use vortex::io::std_file::read_exact_at;

#[cfg(target_os = "linux")]
use self::cufile::CuFileReadBackend;
#[cfg(target_os = "linux")]
use self::direct::DirectFileReadBackend;
use crate::pinned::PinnedByteBufferPool;
use crate::pinned::PooledPinnedBuffer;
use crate::stream::VortexCudaStream;

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_FILE_CONCURRENCY: usize = 32;

/// Options controlling how [`PooledFileReadAt`] opens and reads a local file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PooledFileReadAtOptions {
    direct_io: bool,
    cufile: bool,
}

impl PooledFileReadAtOptions {
    /// Bypass the operating system page cache for pooled file reads.
    ///
    /// This option is available only on Linux. Unaligned logical reads are widened to satisfy the
    /// filesystem's direct-I/O requirements and sliced back to the requested range after transfer
    /// to the device.
    #[cfg(target_os = "linux")]
    pub fn with_direct_io(mut self) -> Self {
        self.direct_io = true;
        self
    }

    /// Read file data into CUDA device memory with cuFile's POSIX compatibility path.
    ///
    /// This works on storage without native GPUDirect Storage support, but cuFile may bounce the
    /// data through internal host memory.
    #[cfg(target_os = "linux")]
    pub fn with_cufile(mut self) -> Self {
        self.cufile = true;
        self
    }
}

struct PooledHostRead {
    buffer: PooledPinnedBuffer,
    requested_range: Range<usize>,
}

trait HostFileReadBackend: Send + Sync {
    fn size(&self) -> VortexResult<u64>;

    fn read(
        &self,
        pool: &Arc<PinnedByteBufferPool>,
        offset: u64,
        length: usize,
    ) -> VortexResult<PooledHostRead>;
}

struct BufferedFileReadBackend {
    file: File,
}

impl BufferedFileReadBackend {
    fn open(path: &Path) -> VortexResult<Self> {
        Ok(Self {
            file: File::open(path)?,
        })
    }
}

impl HostFileReadBackend for BufferedFileReadBackend {
    fn size(&self) -> VortexResult<u64> {
        Ok(self.file.metadata()?.len())
    }

    fn read(
        &self,
        pool: &Arc<PinnedByteBufferPool>,
        offset: u64,
        length: usize,
    ) -> VortexResult<PooledHostRead> {
        let mut buffer = pool.get(length)?;
        read_exact_at(&self.file, buffer.as_mut_slice(), offset)?;
        Ok(PooledHostRead {
            buffer,
            requested_range: 0..length,
        })
    }
}

enum FileReadBackend {
    Host(Arc<dyn HostFileReadBackend>),
    #[cfg(target_os = "linux")]
    CuFile(Arc<CuFileReadBackend>),
}

impl FileReadBackend {
    fn size(&self) -> VortexResult<u64> {
        match self {
            Self::Host(backend) => backend.size(),
            #[cfg(target_os = "linux")]
            Self::CuFile(backend) => backend.size(),
        }
    }
}

#[cfg(target_os = "linux")]
fn open_backend(path: &Path, options: PooledFileReadAtOptions) -> VortexResult<FileReadBackend> {
    if options.cufile {
        return Ok(FileReadBackend::CuFile(Arc::new(CuFileReadBackend::open(
            path,
        )?)));
    }
    if options.direct_io {
        Ok(FileReadBackend::Host(Arc::new(
            DirectFileReadBackend::open(path)?,
        )))
    } else {
        Ok(FileReadBackend::Host(Arc::new(
            BufferedFileReadBackend::open(path)?,
        )))
    }
}

#[cfg(not(target_os = "linux"))]
fn open_backend(path: &Path, _options: PooledFileReadAtOptions) -> VortexResult<FileReadBackend> {
    Ok(FileReadBackend::Host(Arc::new(
        BufferedFileReadBackend::open(path)?,
    )))
}

/// File reader that returns local file data in CUDA device memory.
///
/// The default backend reads into a pooled pinned buffer and submits an H2D transfer. On Linux,
/// [`PooledFileReadAtOptions::with_cufile`] selects cuFile's POSIX compatibility path instead.
///
/// This is a data-plane reader. To open a complete local Vortex file, prefer
/// [`crate::CudaOpenOptionsExt::with_cuda`], which keeps the footer and zone maps on the host.
#[derive(Clone)]
pub struct PooledFileReadAt {
    uri: Arc<str>,
    backend: Arc<FileReadBackend>,
    handle: Handle,
    pool: Arc<PinnedByteBufferPool>,
    stream: VortexCudaStream,
}

impl PooledFileReadAt {
    /// Open a file for pooled reading with direct device transfer.
    pub fn open(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> VortexResult<Self> {
        Self::open_with_options(
            path,
            handle,
            pool,
            stream,
            PooledFileReadAtOptions::default(),
        )
    }

    /// Open a file for pooled reading with explicit options.
    pub fn open_with_options(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
        options: PooledFileReadAtOptions,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = Arc::from(path.to_string_lossy().to_string());
        let backend = Arc::new(open_backend(path, options)?);
        Ok(Self {
            uri,
            backend,
            handle,
            pool,
            stream,
        })
    }
}

impl VortexReadAt for PooledFileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn concurrency(&self) -> usize {
        DEFAULT_FILE_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let backend = Arc::clone(&self.backend);
        async move { backend.size() }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let backend = Arc::clone(&self.backend);
        let handle = self.handle.clone();
        let stream = self.stream.clone();
        let pool = Arc::clone(&self.pool);

        async move {
            match backend.as_ref() {
                FileReadBackend::Host(backend) => {
                    let backend = Arc::clone(backend);
                    let read = handle
                        .spawn_blocking(move || backend.read(&pool, offset, length))
                        .await?;
                    let cuda_buf = read.buffer.transfer_to_device(&stream)?;
                    Ok(BufferHandle::new_device(Arc::new(cuda_buf)).slice(read.requested_range))
                }
                #[cfg(target_os = "linux")]
                FileReadBackend::CuFile(backend) => {
                    let backend = Arc::clone(backend);
                    handle
                        .spawn_blocking(move || backend.read(&stream, offset, length))
                        .await
                }
            }
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooled_file_read_options_default_to_buffered_io() {
        assert!(!PooledFileReadAtOptions::default().direct_io);
        assert!(!PooledFileReadAtOptions::default().cufile);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pooled_file_read_options_enable_direct_io() {
        assert!(
            PooledFileReadAtOptions::default()
                .with_direct_io()
                .direct_io
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pooled_file_read_options_enable_cufile() {
        assert!(PooledFileReadAtOptions::default().with_cufile().cufile);
    }
}
