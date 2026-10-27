// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use tantivy::schema::Field;
use tantivy::schema::STORED;
use tantivy::schema::Schema;
use tantivy::schema::TEXT;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

pub(crate) const ROW_FIELD: &str = "_vortex_row_id";

pub(crate) fn validate_names(columns: &[String]) -> VortexResult<()> {
    if columns.is_empty() {
        vortex_bail!("at least one UTF-8 column is required");
    }
    for (i, name) in columns.iter().enumerate() {
        // Restrict names so field-qualified queries have an unambiguous interpretation.
        if name == ROW_FIELD
            || name.is_empty()
            || !name
                .chars()
                .enumerate()
                .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        {
            vortex_bail!(
                "unsupported indexed column name {name:?}; use ASCII identifiers other than {ROW_FIELD}"
            );
        }
        if columns[..i].contains(name) {
            vortex_bail!("duplicate indexed column {name:?}");
        }
    }
    Ok(())
}

pub(crate) fn validate_columns(dtype: &DType, columns: &[String]) -> VortexResult<()> {
    validate_names(columns)?;
    let fields = dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("full-text indexing requires a struct/table"))?;
    for name in columns {
        if fields
            .names()
            .iter()
            .filter(|n| n.as_ref() == name.as_str())
            .count()
            != 1
        {
            vortex_bail!("indexed column {name:?} must exist exactly once");
        }
        if !matches!(fields.field(name), Some(DType::Utf8(_))) {
            vortex_bail!("indexed column {name:?} must be UTF-8");
        }
    }
    Ok(())
}

pub(crate) fn make_schema(columns: &[String]) -> (Schema, Field, Vec<Field>) {
    let mut builder = Schema::builder();
    let row_field = builder.add_u64_field(ROW_FIELD, STORED);
    let text_fields = columns
        .iter()
        .map(|name| builder.add_text_field(name, TEXT))
        .collect();
    (builder.build(), row_field, text_fields)
}
