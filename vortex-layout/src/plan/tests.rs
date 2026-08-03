// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::get_item;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use super::*;
use crate::LayoutRef;
use crate::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::dict::DictLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::foreign::new_foreign_layout;
use crate::layouts::list::ListLayout;
use crate::layouts::row_idx::row_idx;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentId;

fn primitive(ptype: PType, nullability: Nullability) -> DType {
    DType::Primitive(ptype, nullability)
}

fn flat(row_count: u64, dtype: DType, segment: u32) -> LayoutRef {
    FlatLayout::new(
        row_count,
        dtype,
        SegmentId::from(segment),
        ReadContext::new([]),
    )
    .into_layout()
}

fn unsupported(row_count: u64, dtype: DType) -> LayoutRef {
    static ID: CachedId = CachedId::new("vortex.test.unsupported");
    new_foreign_layout(*ID, dtype, row_count, Vec::new(), Vec::new(), Vec::new())
}

fn make_plan(layout: LayoutRef) -> VortexResult<PlanRef> {
    new_plan(&layout)
}

#[test]
fn unsupported_layout_has_no_plan() -> VortexResult<()> {
    let layout = unsupported(3, DType::Null);

    let error = new_plan(&layout)
        .err()
        .ok_or_else(|| vortex_err!("unsupported layout unexpectedly produced a plan"))?;
    assert!(
        error
            .to_string()
            .contains("No physical plan implementation for layout 'vortex.test.unsupported'")
    );
    Ok(())
}

#[test]
fn flat_plan_has_no_children() -> VortexResult<()> {
    let plan = make_plan(flat(3, primitive(PType::I32, Nullability::NonNullable), 0))?;

    assert!(plan.as_any().is::<FlatPlan>());
    assert_eq!(plan.child_count(), 0);
    assert!(plan.child(0).is_err());
    Ok(())
}

#[test]
fn chunked_plan_exposes_chunks() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)]),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    assert!(plan.as_any().is::<ChunkedPlan>());
    assert_eq!(plan.child_count(), 2);
    assert_eq!(
        plan.child(0)?
            .ok_or_else(|| vortex_err!("missing first chunk"))?
            .row_count(),
        2
    );
    assert_eq!(
        plan.child(1)?
            .ok_or_else(|| vortex_err!("missing second chunk"))?
            .row_count(),
        1
    );
    Ok(())
}

#[test]
fn chunked_plan_defers_unrequested_chunks_through_optimization() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        2,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![
            flat(1, dtype.clone(), 0),
            unsupported(1, dtype),
        ]),
    )
    .into_layout();
    let plan = make_plan(layout)?.optimize()?;

    let first = plan
        .child(0)?
        .ok_or_else(|| vortex_err!("missing first chunk"))?;
    assert!(first.as_any().is::<FlatPlan>());
    let cached = plan
        .child(0)?
        .ok_or_else(|| vortex_err!("missing cached first chunk"))?;
    assert!(Arc::ptr_eq(&first, &cached));

    let error = plan
        .child(1)
        .err()
        .ok_or_else(|| vortex_err!("unsupported chunk unexpectedly produced a plan"))?;
    assert!(
        error
            .to_string()
            .contains("No physical plan implementation for layout 'vortex.test.unsupported'")
    );
    Ok(())
}

#[test]
fn dict_plan_orders_codes_before_values() -> VortexResult<()> {
    let values_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::NonNullable);
    let layout = DictLayout::new(
        flat(2, values_dtype.clone(), 0),
        flat(3, codes_dtype.clone(), 1),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    assert!(plan.as_any().is::<DictPlan>());
    assert_eq!(plan.child_count(), 2);
    assert_eq!(
        plan.child(0)?
            .ok_or_else(|| vortex_err!("missing codes"))?
            .dtype(),
        &codes_dtype
    );
    assert_eq!(
        plan.child(1)?
            .ok_or_else(|| vortex_err!("missing values"))?
            .dtype(),
        &values_dtype
    );
    Ok(())
}

#[test]
fn list_plan_has_stable_optional_validity_slot() -> VortexResult<()> {
    let element_dtype = primitive(PType::I32, Nullability::NonNullable);
    let offsets_dtype = primitive(PType::U32, Nullability::NonNullable);
    let non_nullable = ListLayout::new(
        DType::List(Arc::new(element_dtype.clone()), Nullability::NonNullable),
        flat(3, element_dtype.clone(), 0),
        flat(3, offsets_dtype.clone(), 1),
        None,
    )
    .into_layout();
    let plan = make_plan(non_nullable)?;

    assert!(plan.as_any().is::<ListPlan>());
    assert_eq!(plan.child_count(), 3);
    assert_eq!(
        plan.child(0)?
            .ok_or_else(|| vortex_err!("missing elements"))?
            .dtype(),
        &element_dtype
    );
    assert_eq!(
        plan.child(1)?
            .ok_or_else(|| vortex_err!("missing offsets"))?
            .dtype(),
        &offsets_dtype
    );
    assert!(plan.child(2)?.is_none());

    let nullable = ListLayout::new(
        DType::List(Arc::new(element_dtype.clone()), Nullability::Nullable),
        flat(3, element_dtype, 2),
        flat(3, offsets_dtype, 3),
        Some(flat(2, DType::Bool(Nullability::NonNullable), 4)),
    )
    .into_layout();
    let nullable_plan = make_plan(nullable)?;
    assert_eq!(
        nullable_plan
            .child(2)?
            .ok_or_else(|| vortex_err!("missing validity"))?
            .dtype(),
        &DType::Bool(Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn struct_plan_orders_fields_before_optional_validity() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let fields = StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype.clone())]);
    let non_nullable = StructLayout::new(
        3,
        DType::Struct(fields.clone(), Nullability::NonNullable),
        vec![
            flat(3, field_dtype.clone(), 0),
            flat(3, field_dtype.clone(), 1),
        ],
    )
    .into_layout();
    let plan = make_plan(non_nullable)?;

    assert!(plan.as_any().is::<StructPlan>());
    assert_eq!(plan.child_count(), 3);
    assert_eq!(
        plan.child(0)?
            .ok_or_else(|| vortex_err!("missing field a"))?
            .dtype(),
        &field_dtype
    );
    assert_eq!(
        plan.child(1)?
            .ok_or_else(|| vortex_err!("missing field b"))?
            .dtype(),
        &field_dtype
    );
    assert!(plan.child(2)?.is_none());

    let nullable = StructLayout::new(
        3,
        DType::Struct(fields, Nullability::Nullable),
        vec![
            flat(3, DType::Bool(Nullability::NonNullable), 2),
            flat(3, field_dtype.clone(), 3),
            flat(3, field_dtype, 4),
        ],
    )
    .into_layout();
    let nullable_plan = make_plan(nullable)?;
    assert_eq!(
        nullable_plan
            .child(2)?
            .ok_or_else(|| vortex_err!("missing validity"))?
            .dtype(),
        &DType::Bool(Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn struct_plan_defers_unrequested_fields_through_optimization() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = StructLayout::new(
        1,
        DType::Struct(
            StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![flat(1, field_dtype.clone(), 0), unsupported(1, field_dtype)],
    )
    .into_layout();
    let plan = make_plan(layout)?.optimize()?;

    assert!(
        plan.child(0)?
            .ok_or_else(|| vortex_err!("missing field a"))?
            .as_any()
            .is::<FlatPlan>()
    );
    let error = plan
        .child(1)
        .err()
        .ok_or_else(|| vortex_err!("unsupported field unexpectedly produced a plan"))?;
    assert!(
        error
            .to_string()
            .contains("No physical plan implementation for layout 'vortex.test.unsupported'")
    );
    Ok(())
}

#[test]
fn plan_display_matches_array_tree_display_shape() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![flat(3, field_dtype.clone(), 0), flat(3, field_dtype, 1)],
    )
    .into_layout();
    let plan: PlanRef = Arc::new(ExpressionPlan::try_new(
        get_item("a", root()),
        make_plan(layout)?,
    )?);

    assert_eq!(plan.to_string(), "ExpressionPlan(i32, rows=3)");
    insta::assert_snapshot!(plan.tree_display(), @r"
    root: ExpressionPlan(i32, rows=3) expr=$.a
      child: StructPlan({a=i32, b=i32}, rows=3)
        a: FlatPlan(i32, rows=3)
        b: FlatPlan(i32, rows=3)
    ");

    struct DepthExtractor;

    impl PlanTreeExtractor for DepthExtractor {
        fn write_header(
            &self,
            _plan: &dyn Plan,
            context: &PlanTreeContext,
            formatter: &mut fmt::Formatter<'_>,
        ) -> fmt::Result {
            write!(formatter, " depth={}", context.depth())
        }
    }

    insta::assert_snapshot!(plan.tree_display_builder().with(DepthExtractor), @r"
    root: depth=0
      child: depth=1
        a: depth=2
        b: depth=2
    ");

    let nullable_fields = StructFields::from_iter([
        ("a", primitive(PType::I32, Nullability::NonNullable)),
        ("b", primitive(PType::I32, Nullability::NonNullable)),
    ]);
    let nullable_layout = StructLayout::new(
        3,
        DType::Struct(nullable_fields, Nullability::Nullable),
        vec![
            flat(3, DType::Bool(Nullability::NonNullable), 2),
            flat(3, primitive(PType::I32, Nullability::NonNullable), 3),
            flat(3, primitive(PType::I32, Nullability::NonNullable), 4),
        ],
    )
    .into_layout();
    let nullable = make_plan(nullable_layout)?;
    insta::assert_snapshot!(nullable.tree_display_builder(), @r"
    root:
      a:
      b:
      validity:
    ");
    Ok(())
}

#[test]
fn chunked_plan_display_names_chunks() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)]),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    insta::assert_snapshot!(plan.display_tree(), @r"
    root: ChunkedPlan(i32, rows=3)
      chunks[0]: FlatPlan(i32, rows=2)
      chunks[1]: FlatPlan(i32, rows=1)
    ");
    Ok(())
}

#[test]
fn dict_plan_display_names_logical_children() -> VortexResult<()> {
    let layout = DictLayout::new(
        flat(2, primitive(PType::I32, Nullability::NonNullable), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    insta::assert_snapshot!(plan.tree_display(), @r"
    root: DictPlan(i32, rows=3)
      codes: FlatPlan(u8, rows=3)
      values: FlatPlan(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn list_plan_display_handles_optional_validity() -> VortexResult<()> {
    let element_dtype = primitive(PType::I32, Nullability::NonNullable);
    let offsets_dtype = primitive(PType::U32, Nullability::NonNullable);
    let non_nullable_layout = ListLayout::new(
        DType::List(Arc::new(element_dtype.clone()), Nullability::NonNullable),
        flat(4, element_dtype.clone(), 0),
        flat(3, offsets_dtype.clone(), 1),
        None,
    )
    .into_layout();
    let non_nullable = make_plan(non_nullable_layout)?;

    insta::assert_snapshot!(non_nullable.tree_display(), @r"
    root: ListPlan(list(i32), rows=2)
      elements: FlatPlan(i32, rows=4)
      offsets: FlatPlan(u32, rows=3)
    ");

    let nullable_layout = ListLayout::new(
        DType::List(Arc::new(element_dtype.clone()), Nullability::Nullable),
        flat(4, element_dtype, 2),
        flat(3, offsets_dtype, 3),
        Some(flat(2, DType::Bool(Nullability::NonNullable), 4)),
    )
    .into_layout();
    let nullable = make_plan(nullable_layout)?;

    insta::assert_snapshot!(nullable.tree_display(), @r"
    root: ListPlan(list(i32)?, rows=2)
      elements: FlatPlan(i32, rows=4)
      offsets: FlatPlan(u32, rows=3)
      validity: FlatPlan(bool, rows=2)
    ");
    Ok(())
}

#[test]
fn row_idx_plan_preserves_row_index_expressions() -> VortexResult<()> {
    let layout = flat(3, primitive(PType::I32, Nullability::NonNullable), 0);
    let plan = RowIdxPlan::new_ref(10, make_plan(layout)?);
    let plan = ExpressionPlan::try_new(row_idx(), plan)?.optimize()?;
    let expression = plan
        .as_any()
        .downcast_ref::<ExpressionPlan>()
        .ok_or_else(|| vortex_err!("optimized plan is not an expression plan"))?;

    assert_eq!(expression.expression(), &row_idx());
    assert!(expression.child_plan().as_any().is::<RowIdxPlan>());
    assert_eq!(expression.row_count(), 3);
    Ok(())
}
