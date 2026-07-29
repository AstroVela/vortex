// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::analysis::make_free_field_annotator;
use vortex_array::expr::get_item;
use vortex_array::expr::root;
use vortex_array::expr::transform::partition;
use vortex_array::expr::transform::replace;
use vortex_array::expr::transform::replace_root_fields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_layout::ArrayFuture;
use vortex_layout::scan::plan_v2::ScanPlan;
use vortex_layout::scan::plan_v2::ScanPlanRef;
use vortex_layout::scan::plan_v2::StructScanPlan;
use vortex_mask::Mask;

struct DTypeScanPlan {
    name: Arc<str>,
    dtype: DType,
}

impl ScanPlan for DTypeScanPlan {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn apply_expr(self: Arc<Self>, _expr: Expression) -> VortexResult<ScanPlanRef> {
        vortex_bail!("not needed by planning benchmark")
    }

    fn optimize(self: Arc<Self>) -> VortexResult<ScanPlanRef> {
        vortex_bail!("not needed by planning benchmark")
    }

    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        1
    }

    fn pruning_evaluation(&self, _row_range: &Range<u64>, _mask: Mask) -> VortexResult<MaskFuture> {
        vortex_bail!("not needed by planning benchmark")
    }

    fn filter_evaluation(
        &self,
        _row_range: &Range<u64>,
        _mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        vortex_bail!("not needed by planning benchmark")
    }

    fn projection_evaluation(
        &self,
        _row_range: &Range<u64>,
        _mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        vortex_bail!("not needed by planning benchmark")
    }
}

struct PlanningFixture {
    dtype: DType,
    expanded_root: Expression,
    get_item: Expression,
    struct_plan: ScanPlanRef,
}

impl PlanningFixture {
    fn new(width: usize) -> Self {
        let fields = StructFields::from_iter((0..width).map(|idx| {
            (
                format!("field_{idx}"),
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )
        }));
        let dtype = DType::Struct(fields.clone(), Nullability::NonNullable);
        let get_item = get_item(format!("field_{}", width - 1), root());
        let expanded_root = replace_root_fields(root(), &fields);
        let source: ScanPlanRef = Arc::new(DTypeScanPlan {
            name: Arc::from("benchmark"),
            dtype: dtype.clone(),
        });
        let struct_plan: ScanPlanRef = Arc::new(
            StructScanPlan::try_new(source)
                .vortex_expect("benchmark dtype must be a non-nullable struct"),
        );

        Self {
            dtype,
            expanded_root,
            get_item,
            struct_plan,
        }
    }

    fn partition_expr(&self) {
        let expr = replace(self.get_item.clone(), &root(), self.expanded_root.clone())
            .optimize_recursive(&self.dtype)
            .vortex_expect("benchmark expression must optimize");
        let partitioned = partition(
            expr,
            &self.dtype,
            make_free_field_annotator(self.dtype.as_struct_fields()),
        )
        .vortex_expect("benchmark expression must partition");
        black_box(partitioned);
    }

    fn reduce_scan_plan(&self) {
        let plan = Arc::clone(&self.struct_plan)
            .apply_expr(self.get_item.clone())
            .vortex_expect("benchmark expression must reduce");
        black_box(plan);
    }
}

fn bench_get_item_planning(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_item_planning");
    for width in [10, 100, 1_000] {
        let fixture = PlanningFixture::new(width);
        group.bench_with_input(BenchmarkId::new("expr_partition", width), &width, |b, _| {
            b.iter(|| fixture.partition_expr());
        });
        group.bench_with_input(
            BenchmarkId::new("struct_scan_plan_reduce", width),
            &width,
            |b, _| {
                b.iter(|| fixture.reduce_scan_plan());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_get_item_planning);
criterion_main!(benches);
