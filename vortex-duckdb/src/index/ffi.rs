// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::num::NonZeroUsize;
use std::path::Path;
use std::path::PathBuf;
use std::ptr;
use std::slice;

use bytes::Bytes;
use parking_lot::Mutex;
use vortex::array::IntoArray;
use vortex::array::RecursiveCanonical;
use vortex::array::VortexSessionExecute;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::io::runtime::BlockingRuntime;

use super::FACTORIES;
use super::MAX_K;
use super::MAX_OPTIONS_BYTES;
use super::Request;
use super::read_reference;
use super::root;
use crate::RUNTIME;
use crate::SESSION;
use crate::cpp;
use crate::duckdb::DataChunk;
use crate::duckdb::ExtractedValue;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;
use crate::duckdb::ValueRef;
use crate::duckdb::try_or;
use crate::duckdb::try_or_null;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;

fn string(value: &ValueRef) -> VortexResult<String> {
    // The C++ binder rejects NUL in the full VARCHAR before this C API extraction.
    match value.extract() {
        ExtractedValue::Varchar(value) => Ok(value.to_string()),
        _ => vortex_bail!("Index arguments require non-NULL UTF-8 strings"),
    }
}

fn options(value: &ValueRef) -> VortexResult<Bytes> {
    let options = string(value)?;
    if options.len() > MAX_OPTIONS_BYTES {
        vortex_bail!("Index backend options exceed the byte limit");
    }
    Ok(Bytes::from(options))
}

fn bind(build: bool, inputs: &[&ValueRef]) -> VortexResult<Request> {
    if build && inputs.len() == 5 {
        let ExtractedValue::List(files) = inputs[0].extract() else {
            vortex_bail!("Index build requires a non-NULL list of local filenames");
        };
        if files.is_empty() || files.len() > 4096 {
            vortex_bail!("Index build requires between 1 and 4096 files");
        }
        let files = files
            .iter()
            .map(|file| string(file))
            .collect::<VortexResult<Vec<_>>>()?;
        if files.iter().collect::<BTreeSet<_>>().len() != files.len() {
            vortex_bail!("Index build files must be unique");
        }
        let reference = PathBuf::from(string(inputs[1])?);
        root(&reference)?;
        let field = string(inputs[2])?;
        if field.trim().is_empty() {
            vortex_bail!("Index field must be nonempty");
        }
        let backend = string(inputs[3])?;
        if !FACTORIES.read().contains_key(&backend) {
            vortex_bail!("Unknown SQL index backend: {backend}");
        }
        Ok(Request::Build {
            files,
            reference,
            field,
            backend,
            options: options(inputs[4])?,
        })
    } else if !build && inputs.len() == 4 {
        let reference = PathBuf::from(string(inputs[0])?);
        let (descriptor, identity) = read_reference(&reference)?;
        let ExtractedValue::List(values) = inputs[1].extract() else {
            vortex_bail!("Index search requires a non-NULL Float32 query list");
        };
        if values.is_empty() || values.len() > 4096 {
            vortex_bail!("Index query dimension must be between 1 and 4096");
        }
        let query = values
            .iter()
            .map(|value| match value.extract() {
                ExtractedValue::Float(value) if value.is_finite() => Ok(value),
                _ => vortex_bail!("Index query requires finite, non-NULL Float32 elements"),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let ExtractedValue::BigInt(k) = inputs[2].extract() else {
            vortex_bail!("Index k requires a non-NULL integer");
        };
        if k <= 0 || k > i64::try_from(MAX_K)? {
            vortex_bail!("Index k must be between 1 and {MAX_K}");
        }
        let Some(k) = NonZeroUsize::new(usize::try_from(k)?) else {
            vortex_bail!("Index k must be positive");
        };
        Ok(Request::Search {
            reference,
            identity,
            descriptor,
            query,
            k,
            options: options(inputs[3])?,
        })
    } else {
        vortex_bail!("Invalid SQL index argument count");
    }
}

#[derive(Default)]
struct ReferencePins(Mutex<BTreeMap<PathBuf, String>>);

impl ReferencePins {
    fn record(mut groups: Vec<&Self>, references: &[(&Path, &str)]) -> VortexResult<()> {
        // A wrapper and its SQL owner can share pins. Deduplicate and lock in
        // address order, then validate every owner before remembering a new bind.
        groups.sort_unstable_by_key(|group| ptr::from_ref(*group));
        groups.dedup_by(|a, b| ptr::eq(*a, *b));
        let mut guards = groups
            .iter()
            .map(|group| group.0.lock())
            .collect::<Vec<_>>();
        for pins in &guards {
            for &(reference, identity) in references {
                if pins.get(reference).is_some_and(|pinned| pinned != identity) {
                    vortex_bail!("Index reference changed after bind; prepare a new query");
                }
            }
        }
        for pins in &mut guards {
            for &(reference, identity) in references {
                pins.entry(reference.to_owned())
                    .or_insert_with(|| identity.to_owned());
            }
        }
        Ok(())
    }

    fn inherit(&self, groups: Vec<&Self>) -> VortexResult<bool> {
        // Identities are immutable once pinned. Release the source lock before
        // locking destinations, which can include the source itself.
        let pins = self.0.lock().clone();
        if pins.is_empty() {
            return Ok(false);
        }
        let references = pins
            .iter()
            .map(|(reference, identity)| (reference.as_path(), identity.as_str()))
            .collect::<Vec<_>>();
        Self::record(groups, &references)?;
        Ok(true)
    }
}

#[unsafe(no_mangle)]
extern "C-unwind" fn vortex_index_pins_new() -> *mut c_void {
    Box::into_raw(Box::new(ReferencePins::default())).cast()
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn vortex_index_pins_free(pins: *mut c_void) {
    // The owning prepared plan frees its pin set after all active borrows end.
    unsafe { drop(Box::from_raw(pins.cast::<ReferencePins>())) };
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn vortex_index_pins_record(
    groups: *const *const c_void,
    count: usize,
    bind: *const c_void,
    error: *mut cpp::duckdb_vx_error,
) -> bool {
    // C++ lends a nonempty slice of live prepared owners and FunctionData for
    // this synchronous call; no owner or request is freed until it returns.
    let groups = unsafe { slice::from_raw_parts(groups, count) }
        .iter()
        .map(|pins| unsafe { &*pins.cast::<ReferencePins>() })
        .collect::<Vec<_>>();
    let request = unsafe { &*bind.cast::<Request>() };
    try_or(error, || {
        if let Request::Search {
            reference,
            identity,
            ..
        } = request
        {
            ReferencePins::record(groups, &[(reference, identity)])?;
        }
        Ok(true)
    })
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn vortex_index_pins_inherit(
    source: *const c_void,
    groups: *const *const c_void,
    count: usize,
    error: *mut cpp::duckdb_vx_error,
) -> bool {
    // C++ keeps the source and this nonempty slice of destinations alive for
    // the synchronous call. False without an error means the source is empty.
    let source = unsafe { &*source.cast::<ReferencePins>() };
    let groups = unsafe { slice::from_raw_parts(groups, count) }
        .iter()
        .map(|pins| unsafe { &*pins.cast::<ReferencePins>() })
        .collect::<Vec<_>>();
    try_or(error, || source.inherit(groups))
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn vortex_index_bind(
    build: bool,
    inputs: *const cpp::duckdb_value,
    count: usize,
    result_type: *mut cpp::duckdb_logical_type,
    error: *mut cpp::duckdb_vx_error,
) -> *mut c_void {
    // C++ supplies live Value pointers for this synchronous bind and owns the
    // returned logical type and request independently after this call.
    let inputs = unsafe { slice::from_raw_parts(inputs, count) }
        .iter()
        .map(|value| unsafe { Value::borrow(*value) })
        .collect::<Vec<_>>();
    try_or_null(error, || {
        let request = bind(build, &inputs)?;
        let logical_type = LogicalType::try_from(request.result_dtype()?)?;
        unsafe { result_type.write(logical_type.into_ptr()) };
        Ok(Box::into_raw(Box::new(request)).cast())
    })
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn vortex_index_bind_copy(bind: *const c_void) -> *mut c_void {
    // Only the owning C++ FunctionData passes these typed request pointers.
    Box::into_raw(Box::new(unsafe { &*bind.cast::<Request>() }.clone())).cast()
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn vortex_index_bind_free(bind: *mut c_void) {
    unsafe { drop(Box::from_raw(bind.cast::<Request>())) };
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn vortex_index_execute(
    bind: *const c_void,
    error: *mut cpp::duckdb_vx_error,
) -> *mut c_void {
    let request = unsafe { &*bind.cast::<Request>() };
    try_or_null(error, || {
        // Ranked take can retain nested dictionaries, chunks, or sequences.
        // Materialize only the bounded SQL result, not the source files.
        let mut ctx = SESSION.create_execution_ctx();
        let array = RUNTIME
            .block_on(request.execute())?
            .into_array()
            .execute::<RecursiveCanonical>(&mut ctx)?
            .0
            .into_struct();
        let exporter = ArrayExporter::try_new(&array, &ConversionCache::default(), ctx)?;
        Ok(Box::into_raw(Box::new(exporter)).cast())
    })
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn vortex_index_scan(
    state: *mut c_void,
    chunk: cpp::duckdb_data_chunk,
    error: *mut cpp::duckdb_vx_error,
) -> bool {
    // The single-threaded global state owns the exporter and DuckDB lends its
    // output DataChunk exclusively for the duration of this scan callback.
    let exporter = unsafe { &mut *state.cast::<ArrayExporter>() };
    let chunk = unsafe { DataChunk::borrow_mut(chunk) };
    try_or(error, || exporter.export(chunk, None, None))
}

#[unsafe(no_mangle)]
unsafe extern "C-unwind" fn vortex_index_state_free(state: *mut c_void) {
    unsafe { drop(Box::from_raw(state.cast::<ArrayExporter>())) };
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use vortex::error::VortexResult;
    use vortex::error::vortex_err;

    use super::ReferencePins;

    #[test]
    fn test_rejected_bind_does_not_change_any_owner() -> VortexResult<()> {
        let owner = ReferencePins::default();
        let empty = ReferencePins::default();
        let reference = Path::new("index.json");
        ReferencePins::record(vec![&owner], &[(reference, "original")])?;
        let error =
            ReferencePins::record(vec![&empty, &owner, &empty], &[(reference, "replacement")])
                .err()
                .ok_or_else(|| vortex_err!("Changed reference identity was accepted"))?;
        assert!(error.to_string().contains("reference changed"), "{error}");
        assert!(empty.0.lock().is_empty());
        ReferencePins::record(vec![&owner, &empty, &owner], &[(reference, "original")])?;
        for pins in [&owner, &empty] {
            assert_eq!(
                pins.0.lock().get(reference).map(String::as_str),
                Some("original")
            );
        }
        Ok(())
    }

    #[test]
    fn test_rejected_inheritance_does_not_change_any_owner() -> VortexResult<()> {
        let source = ReferencePins::default();
        let owner = ReferencePins::default();
        let empty = ReferencePins::default();
        let first = Path::new("a.json");
        let last = Path::new("z.json");
        ReferencePins::record(vec![&source], &[(first, "first"), (last, "replacement")])?;
        ReferencePins::record(vec![&owner], &[(last, "original")])?;
        let original = owner.0.lock().clone();
        let error = source
            .inherit(vec![&empty, &owner, &empty])
            .err()
            .ok_or_else(|| vortex_err!("Conflicting inherited identity was accepted"))?;
        assert!(error.to_string().contains("reference changed"), "{error}");
        assert!(empty.0.lock().is_empty());
        assert_eq!(*owner.0.lock(), original);
        Ok(())
    }

    #[test]
    fn test_inheritance_is_directional_and_deduplicates_owners() -> VortexResult<()> {
        let source = ReferencePins::default();
        let owner = ReferencePins::default();
        let inherited = Path::new("inherited.json");
        let unrelated = Path::new("unrelated.json");
        ReferencePins::record(vec![&owner], &[(unrelated, "outer")])?;
        assert!(!source.inherit(vec![&owner])?);
        ReferencePins::record(vec![&source], &[(inherited, "inner")])?;
        assert!(source.inherit(vec![&source, &owner, &owner])?);
        assert_eq!(source.0.lock().len(), 1);
        assert_eq!(owner.0.lock().len(), 2);
        assert_eq!(
            owner.0.lock().get(inherited).map(String::as_str),
            Some("inner")
        );
        Ok(())
    }
}
