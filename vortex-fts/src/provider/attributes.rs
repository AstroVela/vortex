// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::ops::Bound;

use serde::Deserialize;
use serde::Serialize;
use tantivy::query::AllQuery;
use tantivy::query::BooleanQuery;
use tantivy::query::EmptyQuery;
use tantivy::query::Occur;
use tantivy::query::Query;
use tantivy::query::RangeQuery;
use tantivy::query::TermQuery;
use tantivy::query::TermSetQuery;
use tantivy::schema::Field;
use tantivy::schema::INDEXED;
use tantivy::schema::IndexRecordOption;
use tantivy::schema::STRING;
use tantivy::schema::SchemaBuilder;
use tantivy::schema::Term;
use tantivy::tokenizer::MAX_TOKEN_LEN;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::bool::BoolArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex_index::IndexMetadata;
use vortex_index::attributes::AttributePredicate;
use vortex_index::attributes::AttributeValue;
use vortex_index::attributes::FloatValue;

mod blocks;

/// Scalar column type indexed by Tantivy, with no implicit query coercion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeType {
    /// Boolean.
    Bool,
    /// Signed 64-bit integer.
    Int,
    /// Unsigned 64-bit integer.
    UInt,
    /// Finite Float64; positive and negative zero compare equal.
    Float,
    /// Exact, case-sensitive UTF-8 value (no tokenization).
    String,
}

/// Build options for Tantivy backend format 2.
///
/// Serialize into `IndexBuildRequest::backend_options`. `metadata.fields` must
/// equal the union of these fields. A UTF-8 column can have both roles in the
/// same generation; text uses the existing analyzer, attributes use exact terms.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TantivyBuildOptions {
    /// Fields analyzed for full-text search, in query-parser default order.
    pub text_fields: Vec<String>,
    /// Fields indexed for typed attribute predicates.
    pub attribute_fields: BTreeMap<String, AttributeType>,
}

impl TantivyBuildOptions {
    pub(super) fn validate(&self, metadata: &IndexMetadata) -> VortexResult<()> {
        let mut fields: BTreeSet<_> = self.text_fields.iter().collect();
        if fields.len() != self.text_fields.len() {
            vortex_bail!("Duplicate Tantivy text fields");
        }
        for (field, kind) in &self.attribute_fields {
            if fields.contains(field) && *kind != AttributeType::String {
                vortex_bail!("A full-text field must have string attribute type");
            }
            fields.insert(field);
        }
        if fields != metadata.fields.iter().collect()
            || fields.iter().any(|name| name.starts_with("_vortex_"))
        {
            vortex_bail!("Tantivy field roles must match metadata and avoid reserved names");
        }
        Ok(())
    }
}

pub(super) struct AttributeFields {
    fields: BTreeMap<String, (Field, AttributeType)>,
    exists: Field,
}

impl AttributeFields {
    pub(super) fn add(schema: &mut SchemaBuilder, options: &TantivyBuildOptions) -> Self {
        let exists = schema.add_text_field("_vortex_attr_exists", STRING);
        let fields = options
            .attribute_fields
            .iter()
            .enumerate()
            .map(|(i, (name, kind))| {
                let internal = format!("_vortex_attr_{i}");
                let field = match kind {
                    AttributeType::Bool => schema.add_bool_field(&internal, INDEXED),
                    AttributeType::Int => schema.add_i64_field(&internal, INDEXED),
                    AttributeType::UInt => schema.add_u64_field(&internal, INDEXED),
                    AttributeType::Float => schema.add_f64_field(&internal, INDEXED),
                    AttributeType::String => schema.add_text_field(&internal, STRING),
                };
                (name.clone(), (field, *kind))
            })
            .collect();
        Self { fields, exists }
    }

    pub(super) fn decode(
        &self,
        data: &StructArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<Vec<Option<AttributeValue>>>> {
        let parent = data.validity()?.execute_mask(data.len(), ctx)?;
        self.fields
            .iter()
            .map(|(name, (_, kind))| {
                let column = data.unmasked_field_by_name(name)?.clone();
                let valid = column.validity()?.execute_mask(column.len(), ctx)?;
                let values: Vec<AttributeValue> = match (kind, column.dtype()) {
                    (AttributeType::String, DType::Utf8(_)) => {
                        let array = column.execute::<VarBinViewArray>(ctx)?;
                        (0..array.len())
                            .zip(valid.iter())
                            .zip(parent.iter())
                            .map(|((i, valid), parent)| {
                                if !valid || !parent {
                                    return Ok(AttributeValue::String(String::new()));
                                }
                                let bytes = array.bytes_at(i);
                                if bytes.len() > MAX_TOKEN_LEN {
                                    vortex_bail!(
                                        "Attribute {name} exceeds Tantivy's exact term length limit"
                                    );
                                }
                                Ok(AttributeValue::String(
                                    std::str::from_utf8(&bytes)
                                        .map_err(|err| vortex_err!(External: err))?
                                        .into(),
                                ))
                            })
                            .collect::<VortexResult<_>>()?
                    }
                    (AttributeType::Bool, DType::Bool(_)) => column
                        .execute::<BoolArray>(ctx)?
                        .to_bit_buffer()
                        .iter()
                        .map(AttributeValue::Bool)
                        .collect(),
                    (AttributeType::Int, DType::Primitive(PType::I64, _)) => column
                        .execute::<PrimitiveArray>(ctx)?
                        .as_slice::<i64>()
                        .iter()
                        .copied()
                        .map(AttributeValue::Int)
                        .collect(),
                    (AttributeType::UInt, DType::Primitive(PType::U64, _)) => column
                        .execute::<PrimitiveArray>(ctx)?
                        .as_slice::<u64>()
                        .iter()
                        .copied()
                        .map(AttributeValue::UInt)
                        .collect(),
                    (AttributeType::Float, DType::Primitive(PType::F64, _)) => column
                        .execute::<PrimitiveArray>(ctx)?
                        .as_slice::<f64>()
                        .iter()
                        .copied()
                        .zip(valid.iter())
                        .zip(parent.iter())
                        .map(|((value, valid), parent)| {
                            Ok(AttributeValue::Float(FloatValue::try_from(
                                if valid && parent { value } else { 0.0 },
                            )?))
                        })
                        .collect::<VortexResult<_>>()?,
                    _ => vortex_bail!("Attribute {name} does not match declared type {kind:?}"),
                };
                Ok(values
                    .into_iter()
                    .zip(valid.iter())
                    .zip(parent.iter())
                    .map(|((value, valid), parent)| (valid && parent).then_some(value))
                    .collect())
            })
            .collect()
    }

    pub(super) fn add_values(
        &self,
        doc: &mut tantivy::TantivyDocument,
        columns: &[Vec<Option<AttributeValue>>],
        offset: usize,
    ) {
        for ((name, (field, _)), column) in self.fields.iter().zip(columns) {
            if let Some(value) = &column[offset] {
                doc.add_text(self.exists, name);
                match value {
                    AttributeValue::Bool(value) => doc.add_bool(*field, *value),
                    AttributeValue::Int(value) => doc.add_i64(*field, *value),
                    AttributeValue::UInt(value) => doc.add_u64(*field, *value),
                    AttributeValue::Float(value) => doc.add_f64(*field, f64::from(*value)),
                    AttributeValue::String(value) => doc.add_text(*field, value),
                }
            }
        }
    }

    fn field(&self, name: &str) -> VortexResult<(Field, AttributeType)> {
        self.fields
            .get(name)
            .copied()
            .ok_or_else(|| vortex_err!("Unknown attribute field: {name}"))
    }

    fn term(&self, name: &str, value: &AttributeValue) -> VortexResult<Term> {
        let (field, kind) = self.field(name)?;
        Ok(match (kind, value) {
            (AttributeType::Bool, AttributeValue::Bool(value)) => {
                Term::from_field_bool(field, *value)
            }
            (AttributeType::Int, AttributeValue::Int(value)) => Term::from_field_i64(field, *value),
            (AttributeType::UInt, AttributeValue::UInt(value)) => {
                Term::from_field_u64(field, *value)
            }
            (AttributeType::Float, AttributeValue::Float(value)) => {
                Term::from_field_f64(field, f64::from(*value))
            }
            (AttributeType::String, AttributeValue::String(value)) => {
                if value.len() > MAX_TOKEN_LEN {
                    vortex_bail!("Attribute query exceeds Tantivy's exact term length limit");
                }
                Term::from_field_text(field, value)
            }
            _ => vortex_bail!("Attribute value type does not match {name}: {kind:?}"),
        })
    }

    pub(super) fn query(
        &self,
        predicate: &AttributePredicate,
        depth: usize,
        budget: &mut usize,
    ) -> VortexResult<Box<dyn Query>> {
        if depth > 64 || *budget == 0 {
            vortex_bail!("Attribute predicate exceeds depth/size limit");
        }
        *budget -= 1;
        Ok(match predicate {
            AttributePredicate::Eq(name, value) => Box::new(TermQuery::new(
                self.term(name, value)?,
                IndexRecordOption::Basic,
            )),
            AttributePredicate::In(name, values) => {
                self.field(name)?;
                if values.len() > *budget {
                    vortex_bail!("Attribute IN exceeds term limit");
                }
                *budget -= values.len();
                Box::new(TermSetQuery::new(
                    values
                        .iter()
                        .map(|value| self.term(name, value))
                        .collect::<VortexResult<Vec<_>>>()?,
                ))
            }
            AttributePredicate::Range(name, lower, upper) => {
                let lower_term = self.term(name, lower)?;
                let upper_term = self.term(name, upper)?;
                if lower > upper {
                    vortex_bail!("Attribute range bounds are reversed");
                }
                Box::new(RangeQuery::new(
                    Bound::Included(lower_term),
                    Bound::Included(upper_term),
                ))
            }
            AttributePredicate::Exists(name) => {
                self.field(name)?;
                Box::new(TermQuery::new(
                    Term::from_field_text(self.exists, name),
                    IndexRecordOption::Basic,
                ))
            }
            AttributePredicate::And(children) | AttributePredicate::Or(children) => {
                let and = matches!(predicate, AttributePredicate::And(_));
                if children.is_empty() {
                    if and {
                        Box::new(AllQuery)
                    } else {
                        Box::new(EmptyQuery)
                    }
                } else {
                    if children.len() > *budget {
                        vortex_bail!("Attribute predicate exceeds clause limit");
                    }
                    let occur = if and { Occur::Must } else { Occur::Should };
                    Box::new(BooleanQuery::new(
                        children
                            .iter()
                            .map(|child| Ok((occur, self.query(child, depth + 1, budget)?)))
                            .collect::<VortexResult<_>>()?,
                    ))
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::session::VortexSession;

    use super::*;

    #[test]
    fn test_null_parent_suppresses_attribute_and_nonfinite_child() -> VortexResult<()> {
        let data = StructArray::try_new(
            ["price"].into(),
            [buffer![1.0f64, f64::NAN].into_array()],
            2,
            Validity::from_iter([true, false]),
        )?;
        let config = TantivyBuildOptions {
            text_fields: vec![],
            attribute_fields: [("price".into(), AttributeType::Float)].into(),
        };
        let fields = AttributeFields::add(&mut tantivy::schema::Schema::builder(), &config);
        let session = VortexSession::default();
        let columns = fields.decode(&data, &mut session.create_execution_ctx())?;
        assert_eq!(
            columns,
            vec![vec![
                Some(AttributeValue::Float(FloatValue::try_from(1.0)?)),
                None
            ]]
        );
        Ok(())
    }
}
