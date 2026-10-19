// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Loadable DuckDB 1.5 extension built entirely from this workspace.

use std::ffi::c_char;
use std::ffi::c_void;
use std::sync::Arc;

use vortex_error::VortexExpect;

unsafe extern "C-unwind" {
    fn vortex_fts_initialize(
        loader: *mut c_void,
        initialize: unsafe extern "C-unwind" fn(*mut c_void),
    );
}

unsafe extern "C-unwind" fn initialize_database(database: *mut c_void) {
    // The C++ bridge supplies a live C API DatabaseWrapper for this callback.
    unsafe { vortex_duckdb::initialize_extension_from_raw(database) };
}

/// Register Vortex IO and Tantivy full-text SQL through DuckDB's C++ loader ABI.
///
/// # Safety
/// DuckDB must pass a live `ExtensionLoader` from the same build as the headers
/// and library used to compile this extension. The pointer is borrowed only for
/// the duration of this call and must not be null.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vortex_duckdb_cpp_init(loader: *mut c_void) {
    vortex_duckdb::index::register_index_provider_factory(
        vortex_fts::TANTIVY_ID,
        |session, scratch| {
            Ok(Arc::new(vortex_fts::TantivyProvider::try_new(
                session,
                scratch.to_path_buf(),
            )?))
        },
    )
    .vortex_expect("Failed to register Tantivy provider");
    // The bridge keeps the C API wrapper alive until all registration completes.
    unsafe { vortex_fts_initialize(loader, initialize_database) };
}

/// Return the linked DuckDB library version for extension tooling.
#[unsafe(no_mangle)]
pub extern "C" fn vortex_version() -> *const c_char {
    vortex_duckdb::duckdb_library_version()
}
