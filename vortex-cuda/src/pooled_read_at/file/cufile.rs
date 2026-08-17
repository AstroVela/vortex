// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! cuFile POSIX compatibility reads into CUDA device memory.

use std::ffi::c_int;
use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr;
use std::sync::OnceLock;

use cudarc::driver::DevicePtrMut;
use libloading::Library;
use vortex::array::buffer::BufferHandle;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::CudaDeviceBuffer;
use crate::VortexCudaStream;

const CU_FILE_SUCCESS: c_int = 0;
const CU_FILE_HANDLE_TYPE_OPAQUE_FD: c_int = 1;
const CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE: c_int = 1;
const CUFILE_PARAM_FORCE_COMPAT_MODE: c_int = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct CuFileError {
    err: c_int,
    cu_err: c_int,
}

#[repr(C)]
union CuFileDescrHandle {
    fd: c_int,
    handle: *mut c_void,
}

#[repr(C)]
struct CuFileDescr {
    type_: c_int,
    handle: CuFileDescrHandle,
    fs_ops: *const c_void,
}

type CuFileHandle = *mut c_void;
type DriverOpenFn = unsafe extern "C" fn() -> CuFileError;
type SetParameterBoolFn = unsafe extern "C" fn(c_int, bool) -> CuFileError;
type HandleRegisterFn = unsafe extern "C" fn(*mut CuFileHandle, *mut CuFileDescr) -> CuFileError;
type HandleDeregisterFn = unsafe extern "C" fn(CuFileHandle);
type ReadFn = unsafe extern "C" fn(CuFileHandle, *mut c_void, usize, i64, i64) -> isize;

struct CuFileLibrary {
    _library: Library,
    handle_register: HandleRegisterFn,
    handle_deregister: HandleDeregisterFn,
    read: ReadFn,
}

static CUFILE: OnceLock<Result<CuFileLibrary, String>> = OnceLock::new();

impl CuFileLibrary {
    fn get() -> VortexResult<&'static Self> {
        CUFILE
            .get_or_init(Self::load)
            .as_ref()
            .map_err(|error| vortex_err!("load cuFile compatibility backend: {error}"))
    }

    fn load() -> Result<Self, String> {
        // SAFETY: Symbols below are copied function pointers with signatures from cufile.h. The
        // Library remains owned by the returned value for at least as long as those pointers.
        unsafe {
            let library = Library::new("libcufile.so.0")
                .or_else(|_| Library::new("libcufile.so"))
                .map_err(|error| format!("open libcufile: {error}"))?;
            let set_parameter_bool =
                load_symbol::<SetParameterBoolFn>(&library, b"cuFileSetParameterBool\0")?;
            check_status(
                set_parameter_bool(CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE, true),
                "enable cuFile compatibility mode",
            )?;
            check_status(
                set_parameter_bool(CUFILE_PARAM_FORCE_COMPAT_MODE, true),
                "force cuFile compatibility mode",
            )?;
            let driver_open = load_symbol::<DriverOpenFn>(&library, b"cuFileDriverOpen\0")?;
            check_status(driver_open(), "open cuFile driver")?;

            Ok(Self {
                handle_register: load_symbol(&library, b"cuFileHandleRegister\0")?,
                handle_deregister: load_symbol(&library, b"cuFileHandleDeregister\0")?,
                read: load_symbol(&library, b"cuFileRead\0")?,
                _library: library,
            })
        }
    }
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, String> {
    // SAFETY: The caller supplies the cufile.h signature corresponding to `name`.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| {
            format!(
                "load {}: {error}",
                String::from_utf8_lossy(name).trim_end_matches('\0')
            )
        })
}

fn check_status(status: CuFileError, operation: &str) -> Result<(), String> {
    if status.err == CU_FILE_SUCCESS {
        Ok(())
    } else {
        Err(format!(
            "{operation} failed: {} ({}), CUDA error {}",
            status_name(status.err),
            status.err,
            status.cu_err,
        ))
    }
}

fn status_name(status: c_int) -> &'static str {
    match status {
        5001 => "CU_FILE_DRIVER_NOT_INITIALIZED",
        5004 => "CU_FILE_DRIVER_VERSION_MISMATCH",
        5007 => "CU_FILE_PLATFORM_NOT_SUPPORTED",
        5008 => "CU_FILE_IO_NOT_SUPPORTED",
        5009 => "CU_FILE_DEVICE_NOT_SUPPORTED",
        5010 => "CU_FILE_NVFS_DRIVER_ERROR",
        5011 => "CU_FILE_CUDA_DRIVER_ERROR",
        5012 => "CU_FILE_CUDA_POINTER_INVALID",
        5015 => "CU_FILE_CUDA_CONTEXT_MISMATCH",
        5019 => "CU_FILE_INVALID_FILE_OPEN_FLAG",
        5020 => "CU_FILE_DIO_NOT_SET",
        5022 => "CU_FILE_INVALID_VALUE",
        5025 => "CU_FILE_PERMISSION_DENIED",
        5048 => "CU_FILE_NVFS_INTERNAL_DRIVER_ERROR",
        _ => "unknown cuFile error",
    }
}

struct RegisteredHandle {
    library: &'static CuFileLibrary,
    handle: usize,
}

impl RegisteredHandle {
    fn new(file: &File) -> VortexResult<Self> {
        let library = CuFileLibrary::get()?;
        let mut handle = ptr::null_mut();
        let mut descriptor = CuFileDescr {
            type_: CU_FILE_HANDLE_TYPE_OPAQUE_FD,
            handle: CuFileDescrHandle {
                fd: file.as_raw_fd(),
            },
            fs_ops: ptr::null(),
        };
        // SAFETY: `file` remains owned by CuFileReadBackend until after this handle is deregistered,
        // and the descriptor layout and function signature match cufile.h.
        let status = unsafe { (library.handle_register)(&raw mut handle, &raw mut descriptor) };
        check_status(status, "register file with cuFile")
            .map_err(|error| vortex_err!("{error}"))?;
        Ok(Self {
            library,
            handle: handle as usize,
        })
    }

    fn as_raw(&self) -> CuFileHandle {
        self.handle as CuFileHandle
    }
}

impl Drop for RegisteredHandle {
    fn drop(&mut self) {
        // SAFETY: The handle was registered by this library and is deregistered exactly once.
        unsafe { (self.library.handle_deregister)(self.as_raw()) };
    }
}

pub(super) struct CuFileReadBackend {
    handle: RegisteredHandle,
    file: File,
}

impl CuFileReadBackend {
    pub(super) fn open(path: &Path) -> VortexResult<Self> {
        let file = File::open(path)?;
        let handle = RegisteredHandle::new(&file)?;
        Ok(Self { handle, file })
    }

    pub(super) fn size(&self) -> VortexResult<u64> {
        Ok(self.file.metadata()?.len())
    }

    pub(super) fn read(
        &self,
        stream: &VortexCudaStream,
        offset: u64,
        length: usize,
    ) -> VortexResult<BufferHandle> {
        if length == 0 {
            return Ok(BufferHandle::new_device(std::sync::Arc::new(
                CudaDeviceBuffer::new(stream.device_alloc::<u8>(0)?),
            )));
        }
        let file_offset = i64::try_from(offset)?;
        stream
            .context()
            .bind_to_thread()
            .map_err(|error| vortex_err!("bind CUDA context for cuFile read: {error}"))?;

        let mut cuda_slice = stream.device_alloc::<u8>(length)?;
        // Materialize the stream-ordered destination before copying into it outside the stream.
        stream
            .synchronize()
            .map_err(|error| vortex_err!("synchronize cuFile device allocation: {error}"))?;
        let (device_ptr, write_record) = cuda_slice.device_ptr_mut(stream);
        let read_result = self.read_unregistered(device_ptr as *mut c_void, length, file_offset);
        // Publish the completed external write to cudarc's event tracker before downstream kernels
        // can consume the destination on another managed stream.
        drop(write_record);
        read_result?;

        Ok(BufferHandle::new_device(std::sync::Arc::new(
            CudaDeviceBuffer::new(cuda_slice),
        )))
    }

    fn read_unregistered(
        &self,
        device_address: *mut c_void,
        length: usize,
        file_offset: i64,
    ) -> VortexResult<()> {
        let library = self.handle.library;
        // SAFETY: `device_address` names a live allocation of `length` bytes. Compatibility mode
        // supports unregistered device buffers and this synchronous call completes before return.
        let bytes_read =
            unsafe { (library.read)(self.handle.as_raw(), device_address, length, file_offset, 0) };

        if bytes_read == -1 {
            return Err(io::Error::last_os_error().into());
        }
        vortex_ensure!(
            bytes_read >= 0,
            "cuFileRead failed with cuFile error {}",
            -bytes_read
        );
        vortex_ensure!(
            usize::try_from(bytes_read)? >= length,
            "cuFileRead returned {bytes_read} bytes, but {length} bytes were required"
        );
        Ok(())
    }
}
