// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for decimal arithmetic in filters and projections.
//!
//! Decimal Add/Sub below the physical type's precision ceiling is pushed down into the Vortex
//! scan with operand-aligning casts (see `convert::decimal`). Decimal Mul/Div and arithmetic at
//! the precision ceiling stay in DataFusion, so those queries must still work — just without
//! pushdown.

use std::sync::Arc;

use datafusion::arrow::array::Decimal128Array;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion_common::assert_batches_eq;
use rstest::rstest;

use crate::common_tests::TestSessionContext;

/// Schema: {a: Decimal128(9, 2), b: Decimal128(9, 2), c: Decimal128(9, 4), big1/big2:
/// Decimal128(38, 0)}
fn make_decimal_batch() -> RecordBatch {
    // a = [10.50, 20.00, 5.25], b = [19.50, 10.00, 4.75]
    let col_a = Decimal128Array::from(vec![1050i128, 2000, 525])
        .with_precision_and_scale(9, 2)
        .unwrap();
    let col_b = Decimal128Array::from(vec![1950i128, 1000, 475])
        .with_precision_and_scale(9, 2)
        .unwrap();
    // c = [0.0001, 2.5000, 1.0000]
    let col_c = Decimal128Array::from(vec![1i128, 25000, 10000])
        .with_precision_and_scale(9, 4)
        .unwrap();
    let big1 = Decimal128Array::from(vec![1i128, 2, 3])
        .with_precision_and_scale(38, 0)
        .unwrap();
    let big2 = Decimal128Array::from(vec![10i128, 20, 30])
        .with_precision_and_scale(38, 0)
        .unwrap();

    RecordBatch::try_from_iter(vec![
        ("a", Arc::new(col_a) as _),
        ("b", Arc::new(col_b) as _),
        ("c", Arc::new(col_c) as _),
        ("big1", Arc::new(big1) as _),
        ("big2", Arc::new(big2) as _),
    ])
    .unwrap()
}

async fn decimal_table(ctx: &TestSessionContext) -> anyhow::Result<()> {
    let batch = make_decimal_batch();
    ctx.write_arrow_batch("files/decimals.vortex", &batch)
        .await?;
    let provider = ctx
        .table_provider("tbl", "/files/", batch.schema().as_ref().clone())
        .await?;
    ctx.session.register_table("tbl", provider)?;
    Ok(())
}

/// Collect the physical plan of `sql` as a string.
async fn explain(ctx: &TestSessionContext, sql: &str) -> anyhow::Result<String> {
    let batches = ctx
        .session
        .sql(&format!("EXPLAIN {sql}"))
        .await?
        .collect()
        .await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

#[tokio::test]
async fn test_decimal_add_filter_is_pushed_down() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    decimal_table(&ctx).await?;

    let sql = "SELECT a FROM tbl WHERE a + b = 30.00 ORDER BY a";
    let result = ctx.session.sql(sql).await?.collect().await?;

    assert_batches_eq!(
        [
            "+-------+",
            "| a     |",
            "+-------+",
            "| 10.50 |",
            "| 20.00 |",
            "+-------+",
        ],
        &result
    );

    // The filter runs inside the Vortex scan, so no FilterExec remains in the plan.
    let plan = explain(&ctx, sql).await?;
    assert!(!plan.contains("FilterExec"), "expected pushdown:\n{plan}");
    Ok(())
}

#[tokio::test]
async fn test_decimal_sub_filter_is_pushed_down() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    decimal_table(&ctx).await?;

    let sql = "SELECT a FROM tbl WHERE a - b > 0 ORDER BY a";
    let result = ctx.session.sql(sql).await?.collect().await?;

    assert_batches_eq!(
        [
            "+-------+",
            "| a     |",
            "+-------+",
            "| 5.25  |",
            "| 20.00 |",
            "+-------+",
        ],
        &result
    );

    let plan = explain(&ctx, sql).await?;
    assert!(!plan.contains("FilterExec"), "expected pushdown:\n{plan}");
    Ok(())
}

/// Operands with different scales are rescaled inside the pushed-down expression, matching
/// DataFusion's arrow kernel semantics exactly.
#[tokio::test]
async fn test_decimal_mixed_scale_filter_is_pushed_down() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    decimal_table(&ctx).await?;

    let sql = "SELECT a FROM tbl WHERE a + c > 22.0";
    let result = ctx.session.sql(sql).await?.collect().await?;

    assert_batches_eq!(
        [
            "+-------+",
            "| a     |",
            "+-------+",
            "| 20.00 |",
            "+-------+",
        ],
        &result
    );

    let plan = explain(&ctx, sql).await?;
    assert!(!plan.contains("FilterExec"), "expected pushdown:\n{plan}");
    Ok(())
}

/// At the Decimal128 precision ceiling arrow saturates the result precision while Vortex would
/// widen it, so the filter must stay in DataFusion — and still produce correct results.
#[tokio::test]
async fn test_decimal_add_at_precision_ceiling_falls_back() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    decimal_table(&ctx).await?;

    let sql = "SELECT big1 FROM tbl WHERE big1 + big2 = 33";
    let result = ctx.session.sql(sql).await?.collect().await?;

    assert_batches_eq!(
        ["+------+", "| big1 |", "+------+", "| 3    |", "+------+"],
        &result
    );

    let plan = explain(&ctx, sql).await?;
    assert!(
        plan.contains("FilterExec"),
        "expected DataFusion fallback:\n{plan}"
    );
    Ok(())
}

/// Vortex has no decimal Mul/Div kernels; the filter stays in DataFusion instead of failing the
/// query at scan time.
#[tokio::test]
async fn test_decimal_mul_filter_falls_back() -> anyhow::Result<()> {
    let ctx = TestSessionContext::default();
    decimal_table(&ctx).await?;

    let sql = "SELECT a FROM tbl WHERE a * b > 100.0 ORDER BY a";
    let result = ctx.session.sql(sql).await?.collect().await?;

    assert_batches_eq!(
        [
            "+-------+",
            "| a     |",
            "+-------+",
            "| 10.50 |",
            "| 20.00 |",
            "+-------+",
        ],
        &result
    );

    let plan = explain(&ctx, sql).await?;
    assert!(
        plan.contains("FilterExec"),
        "expected DataFusion fallback:\n{plan}"
    );
    Ok(())
}

/// Decimal Add/Sub in projections, with and without projection pushdown.
#[rstest]
#[tokio::test]
async fn test_decimal_add_projection(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);
    decimal_table(&ctx).await?;

    let result = ctx
        .session
        .sql("SELECT a + b AS total, a - c AS diff FROM tbl ORDER BY total")
        .await?
        .collect()
        .await?;

    assert_batches_eq!(
        [
            "+-------+---------+",
            "| total | diff    |",
            "+-------+---------+",
            "| 10.00 | 4.2500  |",
            "| 30.00 | 10.4999 |",
            "| 30.00 | 17.5000 |",
            "+-------+---------+",
        ],
        &result
    );
    Ok(())
}

/// Decimal Mul in a projection must fall back to DataFusion evaluation.
#[rstest]
#[tokio::test]
async fn test_decimal_mul_projection_falls_back(
    #[values(false, true)] projection_pushdown: bool,
) -> anyhow::Result<()> {
    let ctx = TestSessionContext::new(projection_pushdown);
    decimal_table(&ctx).await?;

    let result = ctx
        .session
        .sql("SELECT a * b AS product FROM tbl ORDER BY product")
        .await?
        .collect()
        .await?;

    assert_batches_eq!(
        [
            "+----------+",
            "| product  |",
            "+----------+",
            "| 24.9375  |",
            "| 200.0000 |",
            "| 204.7500 |",
            "+----------+",
        ],
        &result
    );
    Ok(())
}
