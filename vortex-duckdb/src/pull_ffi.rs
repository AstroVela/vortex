// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FFI helpers for the C++ pull-based scan path (`cpp/pull_table_function.cpp`).
//!
//! The C++ side owns files, footers, shards, and reads (through the DuckDB `FileSystem`) and
//! drives `vx_pull_scan` coroutines from `vortex-ffi`. The helpers here cover the two pieces
//! that must stay in Rust: building pushdown expressions from DuckDB bind/init state, and
//! exporting decoded arrays into DuckDB `DataChunk`s.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute as _;
use vortex::error::VortexExpect;
use vortex::error::vortex_ensure;
use vortex::scan::DataSource as _;
use vortex::scan::selection::Selection;
use vortex_ffi::vx_array;
use vortex_ffi::vx_array_ref;
use vortex_ffi::vx_expression;
use vortex_ffi::vx_expression_new;
use vortex_ffi::vx_session;
use vortex_ffi::vx_session_new_with;

use crate::SESSION;
use crate::cpp;
use crate::duckdb::DataChunk;
use crate::duckdb::TableInitInput;
use crate::duckdb::try_or;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::projection::Filter;
use crate::projection::Projection;
use crate::table_function::TableFunctionBind;
use crate::table_function::convert_result;

/// The pushdown plan for a pull scan, or `supported = false` when the query needs a feature
/// the pull path does not cover (aggregates, virtual columns, row selections) and the caller
/// must use the legacy scan path.
#[repr(C)]
pub struct duckdb_vx_pull_plan {
    /// Projection expression handle; owned by the caller (free with vx_expression_free).
    pub projection: *mut vx_expression,
    /// Filter expression handle or NULL; owned by the caller.
    pub filter: *mut vx_expression,
    /// False when the caller must fall back to the legacy scan path.
    pub supported: bool,
}

/// Build the pull-scan pushdown plan from DuckDB init state.
///
/// Mirrors the projection/filter construction of the legacy `init_global`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_pull_plan(
    init_input: *const cpp::duckdb_vx_tfunc_init_input,
    plan_out: *mut duckdb_vx_pull_plan,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let init_input =
        TableInitInput::new(unsafe { init_input.as_ref() }.vortex_expect("null init_input"));
    try_or(error_out, || {
        let bind_data = init_input.bind_data();
        let out = unsafe { &mut *plan_out };
        out.projection = std::ptr::null_mut();
        out.filter = std::ptr::null_mut();
        out.supported = false;

        if !bind_data.aggregates.is_empty() {
            return Ok(false);
        }

        let column_ids = init_input.column_ids();
        let Projection {
            projection,
            file_index_column_pos,
            file_row_number_column_pos,
        } = Projection::new(
            init_input.projection_ids(),
            column_ids,
            &bind_data.column_fields,
        );
        if file_index_column_pos.is_some() || file_row_number_column_pos.is_some() {
            return Ok(false);
        }

        let Filter {
            filter,
            row_selection,
            row_range,
            file_selection,
            file_range,
            has_non_optional_filter,
        } = Filter::new(
            init_input.table_filter_set(),
            column_ids,
            &bind_data.column_fields,
            &bind_data.filter_exprs,
            bind_data.data_source.dtype(),
        )?;
        if !matches!(row_selection, Selection::All)
            || !matches!(file_selection, Selection::All)
            || row_range.is_some()
            || file_range.is_some()
        {
            return Ok(false);
        }
        if has_non_optional_filter {
            bind_data
                .has_non_optional_filter
                .store(true, Ordering::Relaxed);
        }

        out.projection = vx_expression_new(projection);
        out.filter = filter.map_or(std::ptr::null_mut(), vx_expression_new);
        out.supported = true;
        Ok(true)
    })
}

/// A clone of the extension's Vortex session for use with `vx_pull_*` functions.
///
/// The caller owns the handle and must free it with vx_session_free.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_vortex_session() -> *mut vx_session {
    vx_session_new_with(|_| SESSION.clone())
}

/// Per-file conversion cache shared by all exporters of one file's shards.
///
/// Free with [`duckdb_pull_cache_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_pull_cache_new(file_index: usize) -> *mut c_void {
    Box::into_raw(Box::new(Arc::new(ConversionCache {
        file_index,
        ..Default::default()
    })))
    .cast()
}

/// Free a conversion cache created by [`duckdb_pull_cache_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_pull_cache_free(cache: *mut c_void) {
    drop(unsafe { Box::from_raw(cache.cast::<Arc<ConversionCache>>()) });
}

/// Exports one decoded batch into DuckDB chunks, at most one chunk per call.
pub struct PullExporter {
    exporter: ArrayExporter,
}

/// Create an exporter for a decoded batch.
///
/// Borrows "array" (the caller still frees its handle) and the cache. On error returns NULL
/// and sets "error_out".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_pull_exporter_new(
    array: *const vx_array,
    cache: *const c_void,
    error_out: *mut cpp::duckdb_vx_error,
) -> *mut c_void {
    try_or(error_out, || {
        let array = unsafe { vx_array_ref(array) }?.clone();
        let cache = unsafe { cache.cast::<Arc<ConversionCache>>().as_ref() }
            .vortex_expect("null conversion cache");
        let mut ctx: ExecutionCtx = SESSION.create_execution_ctx();
        let array = convert_result(array, &mut ctx)?;
        let exporter = ArrayExporter::try_new(&array, cache, ctx)?;
        Ok(Box::into_raw(Box::new(PullExporter { exporter })).cast::<c_void>())
    })
}

/// Export the next chunk of the batch into "output".
///
/// Returns true while more data remains after this chunk; false once the batch is fully
/// exported (the chunk may still contain rows). On error sets "error_out".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_pull_exporter_next(
    exporter: *mut c_void,
    output: cpp::duckdb_data_chunk,
    error_out: *mut cpp::duckdb_vx_error,
) -> bool {
    let exporter =
        unsafe { exporter.cast::<PullExporter>().as_mut() }.vortex_expect("null exporter pointer");
    let chunk = unsafe { DataChunk::borrow_mut(output) };
    try_or(error_out, || {
        vortex_ensure!(chunk.is_empty(), "output chunk must be empty");
        exporter.exporter.export(chunk, None, None)
    })
}

/// Free an exporter created by [`duckdb_pull_exporter_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn duckdb_pull_exporter_free(exporter: *mut c_void) {
    drop(unsafe { Box::from_raw(exporter.cast::<PullExporter>()) });
}
