// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::and;
use vortex_array::expr::checked_add;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
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
fn struct_plan_optimization_visits_all_fields() -> VortexResult<()> {
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
    let error = make_plan(layout)?
        .optimize()
        .err()
        .ok_or_else(|| vortex_err!("unsupported field was not visited during optimization"))?;
    assert!(
        error
            .to_string()
            .contains("No physical plan implementation for layout 'vortex.test.unsupported'")
    );
    Ok(())
}

#[test]
fn chunked_plan_optimization_visits_all_chunks() -> VortexResult<()> {
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

    let error = make_plan(layout)?
        .optimize()
        .err()
        .ok_or_else(|| vortex_err!("unsupported chunk was not visited during optimization"))?;
    assert!(
        error
            .to_string()
            .contains("No physical plan implementation for layout 'vortex.test.unsupported'")
    );
    Ok(())
}

#[test]
fn row_idx_only_expression_uses_generated_values_plan() -> VortexResult<()> {
    let layout = flat(3, primitive(PType::I32, Nullability::NonNullable), 0);
    let plan: PlanRef = Arc::new(ExpressionPlan::try_new(
        row_idx(),
        RowIdxPlan::new_ref(10, make_plan(layout)?),
    )?);

    insta::assert_snapshot!(plan.tree_display(), @r"
    root: ExpressionPlan(u64, rows=3) expr=#row_idx
      child: RowIdxPlan(i32, rows=3)
        child: FlatPlan(i32, rows=3)
    ");

    let optimized = plan.optimize()?;
    insta::assert_snapshot!(
        optimized.tree_display(),
        @"root: RowIdxValuesPlan(u64, rows=3)"
    );
    let values = optimized
        .downcast_ref::<RowIdxValuesPlan>()
        .ok_or_else(|| vortex_err!("optimized plan does not generate row-index values"))?;
    assert_eq!(values.row_offset(), 10);
    Ok(())
}

#[test]
fn expression_partitions_across_row_idx_and_struct() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![dictionary, flat(3, value_dtype, 2)],
    )
    .into_layout();
    let expression = and(
        gt(row_idx(), lit(11_u64)),
        and(
            gt(get_item("a", root()), lit(5_i32)),
            gt(get_item("b", root()), lit(7_i32)),
        ),
    );
    let plan: PlanRef = Arc::new(ExpressionPlan::try_new(
        expression,
        RowIdxPlan::new_ref(10, make_plan(layout)?),
    )?);

    insta::assert_snapshot!(plan.tree_display(), @r"
    root: ExpressionPlan(bool, rows=3) expr=((#row_idx > 11u64) and (($.a > 5i32) and ($.b > 7i32)))
      child: RowIdxPlan({a=i32, b=i32}, rows=3)
        child: StructPlan({a=i32, b=i32}, rows=3)
          a: DictPlan(i32, rows=3)
            codes: FlatPlan(u8, rows=3)
            values: FlatPlan(i32, rows=2)
          b: FlatPlan(i32, rows=3)
    ");

    let optimized = plan.optimize()?;
    insta::assert_snapshot!(optimized.tree_display(), @r"
    root: ExpressionPlan(bool, rows=3) expr=(($.row_idx and $.child.child_0) and $.child.child_1)
      child: RowIdxPartitionPlan({row_idx=bool, child={child_0=bool, child_1=bool}}, rows=3)
        row_idx: ExpressionPlan(bool, rows=3) expr=($ > 11u64)
          child: RowIdxValuesPlan(u64, rows=3)
        child: ExpressionPlan({child_0=bool, child_1=bool}, rows=3) expr=pack(child_0: $.a, child_1: $.b)
          child: StructPlan({a=bool, b=bool}, rows=3)
            a: DictPlan(bool, rows=3)
              codes: FlatPlan(u8, rows=3)
              values: ExpressionPlan(bool, rows=2) expr=($ > 5i32)
                child: FlatPlan(i32, rows=2)
            b: ExpressionPlan(bool, rows=3) expr=($ > 7i32)
              child: FlatPlan(i32, rows=3)
    ");
    let residual = optimized
        .downcast_ref::<ExpressionPlan>()
        .ok_or_else(|| vortex_err!("optimized plan has no residual expression"))?;
    let partitions = residual
        .child_plan()
        .downcast_ref::<RowIdxPartitionPlan>()
        .ok_or_else(|| vortex_err!("optimized plan has no row-index partitions"))?;
    let row_idx_expression = partitions
        .row_idx_plan()
        .downcast_ref::<ExpressionPlan>()
        .ok_or_else(|| vortex_err!("row-index partition has no expression"))?;
    let values = row_idx_expression
        .child_plan()
        .downcast_ref::<RowIdxValuesPlan>()
        .ok_or_else(|| vortex_err!("row-index partition has no generated values"))?;
    assert_eq!(values.row_offset(), 10);
    Ok(())
}

#[test]
fn chunked_plan_preserves_global_row_index_expressions() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(1, dtype.clone(), 0), flat(2, dtype, 1)]),
    )
    .into_layout();
    let plan = ExpressionPlan::try_new(row_idx(), make_plan(layout)?)?.optimize()?;
    let expression = plan
        .downcast_ref::<ExpressionPlan>()
        .ok_or_else(|| vortex_err!("Row-index expression unexpectedly pushed into chunks"))?;

    assert_eq!(expression.expression(), &row_idx());
    assert!(expression.child_plan().is::<ChunkedPlan>());
    Ok(())
}

#[test]
fn expression_pushes_through_struct_field_and_dictionary_values() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::NonNullable);
    let dictionary =
        DictLayout::new(flat(2, value_dtype.clone(), 0), flat(3, codes_dtype, 1)).into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![dictionary, flat(3, value_dtype, 2)],
    )
    .into_layout();
    let plan: PlanRef = Arc::new(ExpressionPlan::try_new(
        gt(get_item("a", root()), lit(5_i32)),
        make_plan(layout)?,
    )?);

    insta::assert_snapshot!(plan.tree_display(), @r"
    root: ExpressionPlan(bool, rows=3) expr=($.a > 5i32)
      child: StructPlan({a=i32, b=i32}, rows=3)
        a: DictPlan(i32, rows=3)
          codes: FlatPlan(u8, rows=3)
          values: FlatPlan(i32, rows=2)
        b: FlatPlan(i32, rows=3)
    ");

    let optimized = plan.optimize()?;

    insta::assert_snapshot!(optimized.tree_display(), @r"
    root: DictPlan(bool, rows=3)
      codes: FlatPlan(u8, rows=3)
      values: ExpressionPlan(bool, rows=2) expr=($ > 5i32)
        child: FlatPlan(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn expression_pushes_through_struct_field_with_heterogeneous_chunks() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let chunks = ChunkedLayout::new(
        5,
        value_dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![dictionary, flat(2, value_dtype.clone(), 2)]),
    )
    .into_layout();
    let layout = StructLayout::new(
        5,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![chunks, flat(5, value_dtype, 3)],
    )
    .into_layout();
    let plan: PlanRef = Arc::new(ExpressionPlan::try_new(
        gt(get_item("a", root()), lit(5_i32)),
        make_plan(layout)?,
    )?);

    insta::assert_snapshot!(plan.tree_display(), @r"
    root: ExpressionPlan(bool, rows=5) expr=($.a > 5i32)
      child: StructPlan({a=i32, b=i32}, rows=5)
        a: ChunkedPlan(i32, rows=5)
          chunks[0]: DictPlan(i32, rows=3)
            codes: FlatPlan(u8, rows=3)
            values: FlatPlan(i32, rows=2)
          chunks[1]: FlatPlan(i32, rows=2)
        b: FlatPlan(i32, rows=5)
    ");

    let optimized = plan.optimize()?;

    insta::assert_snapshot!(optimized.tree_display(), @r"
    root: ChunkedPlan(bool, rows=5)
      chunks[0]: DictPlan(bool, rows=3)
        codes: FlatPlan(u8, rows=3)
        values: ExpressionPlan(bool, rows=2) expr=($ > 5i32)
          child: FlatPlan(i32, rows=2)
      chunks[1]: ExpressionPlan(bool, rows=2) expr=($ > 5i32)
        child: FlatPlan(i32, rows=2)
    ");
    Ok(())
}

#[test]
fn multi_field_struct_expression_pushes_into_each_field() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([
                ("a", value_dtype.clone()),
                ("b", value_dtype.clone()),
                ("c", value_dtype.clone()),
            ]),
            Nullability::NonNullable,
        ),
        vec![
            dictionary,
            flat(3, value_dtype.clone(), 2),
            flat(3, value_dtype, 3),
        ],
    )
    .into_layout();
    let expression = and(
        gt(get_item("a", root()), lit(5_i32)),
        gt(get_item("b", root()), lit(7_i32)),
    );
    let plan: PlanRef = Arc::new(ExpressionPlan::try_new(expression, make_plan(layout)?)?);

    insta::assert_snapshot!(plan.tree_display(), @r"
    root: ExpressionPlan(bool, rows=3) expr=(($.a > 5i32) and ($.b > 7i32))
      child: StructPlan({a=i32, b=i32, c=i32}, rows=3)
        a: DictPlan(i32, rows=3)
          codes: FlatPlan(u8, rows=3)
          values: FlatPlan(i32, rows=2)
        b: FlatPlan(i32, rows=3)
        c: FlatPlan(i32, rows=3)
    ");

    let optimized = plan.optimize()?;

    insta::assert_snapshot!(optimized.tree_display(), @r"
    root: ExpressionPlan(bool, rows=3) expr=($.a and $.b)
      child: StructPlan({a=bool, b=bool, c=i32}, rows=3)
        a: DictPlan(bool, rows=3)
          codes: FlatPlan(u8, rows=3)
          values: ExpressionPlan(bool, rows=2) expr=($ > 5i32)
            child: FlatPlan(i32, rows=2)
        b: ExpressionPlan(bool, rows=3) expr=($ > 7i32)
          child: FlatPlan(i32, rows=3)
        c: FlatPlan(i32, rows=3)
    ");
    assert_eq!(
        optimized.tree_display().to_string(),
        optimized.optimize()?.tree_display().to_string()
    );
    Ok(())
}

#[test]
fn multi_field_struct_expression_keeps_cross_field_refinement() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype.clone(), 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", value_dtype.clone()), ("b", value_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![dictionary, flat(3, value_dtype, 2)],
    )
    .into_layout();
    let expression = gt(
        checked_add(get_item("a", root()), get_item("b", root())),
        lit(10_i32),
    );
    let plan = ExpressionPlan::try_new(expression, make_plan(layout)?)?;

    let optimized = plan.optimize()?;

    insta::assert_snapshot!(optimized.tree_display(), @r"
    root: ExpressionPlan(bool, rows=3) expr=(($.a + $.b) > 10i32)
      child: StructPlan({a=i32, b=i32}, rows=3)
        a: DictPlan(i32, rows=3)
          codes: FlatPlan(u8, rows=3)
          values: FlatPlan(i32, rows=2)
        b: FlatPlan(i32, rows=3)
    ");
    Ok(())
}

#[test]
fn dictionary_pushdown_rejects_unsafe_expressions() -> VortexResult<()> {
    let value_dtype = primitive(PType::I32, Nullability::NonNullable);
    let dictionary = DictLayout::new(
        flat(2, value_dtype, 0),
        flat(3, primitive(PType::U8, Nullability::NonNullable), 1),
    )
    .into_layout();

    for expression in [
        lit(false),
        is_null(root()),
        gt(checked_add(root(), lit(1_i32)), lit(5_i32)),
    ] {
        let plan =
            ExpressionPlan::try_new(expression.clone(), make_plan(Arc::clone(&dictionary))?)?;
        let optimized = plan.optimize()?;
        let expression_plan = optimized.downcast_ref::<ExpressionPlan>().ok_or_else(|| {
            vortex_err!("Expression unexpectedly pushed into dictionary: {expression}")
        })?;
        assert!(expression_plan.child_plan().is::<DictPlan>());
    }
    Ok(())
}

#[test]
fn nullable_struct_keeps_expression_above_parent_validity() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = StructLayout::new(
        3,
        DType::Struct(
            StructFields::from_iter([("a", field_dtype.clone())]),
            Nullability::Nullable,
        ),
        vec![
            flat(3, DType::Bool(Nullability::NonNullable), 0),
            flat(3, field_dtype, 1),
        ],
    )
    .into_layout();
    let plan = ExpressionPlan::try_new(gt(get_item("a", root()), lit(5_i32)), make_plan(layout)?)?;

    let optimized = plan.optimize()?;
    let expression_plan = optimized
        .downcast_ref::<ExpressionPlan>()
        .ok_or_else(|| vortex_err!("Nullable struct expression unexpectedly pushed down"))?;
    assert!(expression_plan.child_plan().is::<StructPlan>());
    Ok(())
}
