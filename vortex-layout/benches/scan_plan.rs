// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use futures::FutureExt;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::root;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_layout::LayoutChildren;
use vortex_layout::LayoutParts;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutRef;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::struct_::Struct;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::scan::plan_v2::LayoutReaderScanPlanV2;
use vortex_layout::scan::plan_v2::ScanPlanRef;
use vortex_layout::scan::plan_v2::StructScanPlan;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

#[derive(Clone)]
struct CountingLayoutChildren {
    children: Arc<[LayoutRef]>,
    materializations: Arc<AtomicUsize>,
}

impl LayoutChildren for CountingLayoutChildren {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        Arc::new(self.clone())
    }

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef> {
        let child = self
            .children
            .get(idx)
            .vortex_expect("benchmark child index must be valid");
        vortex_ensure!(
            child.dtype() == dtype,
            "benchmark child dtype mismatch: {} != {dtype}",
            child.dtype()
        );
        self.materializations.fetch_add(1, Ordering::Relaxed);
        Ok(Arc::clone(child))
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        self.children
            .get(idx)
            .vortex_expect("benchmark child index must be valid")
            .row_count()
    }

    fn nchildren(&self) -> usize {
        self.children.len()
    }
}

struct NoSegments;

impl SegmentSource for NoSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        async move { vortex_bail!("benchmark must not poll segment {id}") }.boxed()
    }
}

struct ColdGetItemFixture {
    layout: StructLayout,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    reader_context: LayoutReaderContext,
    get_item: Expression,
    child_materializations: Arc<AtomicUsize>,
}

impl ColdGetItemFixture {
    fn new(width: usize) -> Self {
        let child_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let fields = StructFields::from_iter(
            (0..width).map(|idx| (format!("field_{idx}"), child_dtype.clone())),
        );
        let dtype = DType::Struct(fields, Nullability::NonNullable);
        let children = (0..width)
            .map(|idx| {
                FlatLayout::new(
                    1,
                    child_dtype.clone(),
                    SegmentId::try_from(idx).vortex_expect("benchmark width must fit in SegmentId"),
                    ReadContext::new([]),
                )
                .into_layout()
            })
            .collect::<Vec<_>>();
        let child_materializations = Arc::new(AtomicUsize::new(0));
        let counting_children = CountingLayoutChildren {
            children: children.into(),
            materializations: Arc::clone(&child_materializations),
        };
        let layout = LayoutParts::new(
            Struct,
            dtype,
            1,
            Vec::new(),
            Arc::new(counting_children),
            (),
        )
        .into_typed();

        Self {
            layout,
            segment_source: Arc::new(NoSegments),
            session: VortexSession::empty(),
            reader_context: LayoutReaderContext::default(),
            get_item: get_item(format!("field_{}", width - 1), root()),
            child_materializations,
        }
    }

    fn fresh_reader(&self) -> LayoutReaderRef {
        self.layout
            .new_reader(
                Arc::from("benchmark"),
                Arc::clone(&self.segment_source),
                &self.session,
                &self.reader_context,
            )
            .vortex_expect("benchmark struct reader must be constructed")
    }

    fn child_materializations(&self) -> usize {
        self.child_materializations.load(Ordering::Relaxed)
    }

    // A fresh reader gives every timed iteration a new LazyReaderChildren cache. The counter
    // deltas below prove that construction is lazy and first access materializes one child.
    fn layout_reader(&self) {
        let before = self.child_materializations();
        let reader = self.fresh_reader();
        assert_eq!(
            self.child_materializations(),
            before,
            "constructing a fresh StructReader must not materialize children"
        );

        let future = reader
            .projection_evaluation(
                &(0..1),
                &self.get_item,
                MaskFuture::ready(Mask::new_true(1)),
            )
            .vortex_expect("benchmark projection must be planned");
        assert_eq!(
            self.child_materializations(),
            before + 1,
            "a cold GetItem must materialize exactly one child reader"
        );
        drop(black_box(future));
    }

    fn struct_scan_plan(&self) {
        let before = self.child_materializations();
        let reader = self.fresh_reader();
        let source: ScanPlanRef = Arc::new(LayoutReaderScanPlanV2::new(reader));
        let struct_plan: ScanPlanRef = Arc::new(
            StructScanPlan::try_new(source)
                .vortex_expect("benchmark source must construct a struct scan plan"),
        );
        let field = Arc::clone(&struct_plan)
            .apply_expr(self.get_item.clone())
            .vortex_expect("benchmark GetItem must reduce")
            .optimize()
            .vortex_expect("benchmark field plan must optimize");
        assert_eq!(
            self.child_materializations(),
            before,
            "constructing and reducing a StructScanPlan must not materialize children"
        );

        let future = field
            .projection_evaluation(&(0..1), MaskFuture::ready(Mask::new_true(1)))
            .vortex_expect("benchmark projection must be planned");
        assert_eq!(
            self.child_materializations(),
            before + 1,
            "a cold reduced GetItem must materialize exactly one child reader"
        );
        drop(black_box(future));
    }
}

fn bench_cold_get_item(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_get_item");
    for width in [10, 100, 1_000] {
        let fixture = ColdGetItemFixture::new(width);
        group.bench_with_input(BenchmarkId::new("layout_reader", width), &width, |b, _| {
            b.iter(|| fixture.layout_reader());
        });
        group.bench_with_input(
            BenchmarkId::new("struct_scan_plan", width),
            &width,
            |b, _| {
                b.iter(|| fixture.struct_scan_plan());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_cold_get_item);
criterion_main!(benches);
