// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::expect_used)]

//! Compare list-aware layout reads against flat whole-array reads.
//!
//! This benchmark writes the same list array two ways:
//!
//! - `ListLayoutStrategy`, with elements/offsets/validity stored as independently compressed
//!   children.
//! - `FlatLayoutStrategy` wrapped in `CompressingStrategy`, which serializes the whole list array
//!   as one compressed blob.
//!
//! The measured loops reuse an already-opened `LayoutReader`, so they isolate reader behavior:
//! full and partial projections, sparse projections, list-length projection pushdown, and filters.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p vortex-layout --features _test-harness --bench list_vs_flat
//! ```

use std::fmt;
use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::sync::Arc;
use std::sync::LazyLock;

use async_trait::async_trait;
use divan::Bencher;
use divan::black_box;
use futures::FutureExt;
use parking_lot::Mutex;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::runtime::Builder;
use tokio::runtime::Runtime;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::expr::Expression;
use vortex_array::expr::gt;
use vortex_array::expr::is_not_null;
use vortex_array::expr::list_length;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar_fn::session::ScalarFnSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::list::writer::ListLayoutStrategy;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSink;
use vortex_layout::segments::SegmentSource;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;

const N_LISTS: usize = 20_000;
const AVG_LIST_LEN: usize = 64;
const LENGTH_FILTER_THRESHOLD: u64 = AVG_LIST_LEN as u64;

fn main() {
    print_byte_sizes();
    divan::main();
}

#[derive(Copy, Clone, Debug)]
enum DataSet {
    RandomI32,
    LowCardinalityI32,
    NullableI32,
    LowCardinalityUtf8,
}

impl DataSet {
    fn name(self) -> &'static str {
        match self {
            Self::RandomI32 => "random_i32",
            Self::LowCardinalityI32 => "low_cardinality_i32",
            Self::NullableI32 => "nullable_i32",
            Self::LowCardinalityUtf8 => "low_cardinality_utf8",
        }
    }
}

impl fmt::Display for DataSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

const DATA_SETS: [DataSet; 4] = [
    DataSet::RandomI32,
    DataSet::LowCardinalityI32,
    DataSet::NullableI32,
    DataSet::LowCardinalityUtf8,
];

#[derive(Copy, Clone, Debug)]
enum LayoutKind {
    List,
    Flat,
}

fn make_list_array(data_set: DataSet) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let max_len = u32::try_from(AVG_LIST_LEN.saturating_mul(2)).expect("max list len fits in u32");

    let mut offsets = Vec::with_capacity(N_LISTS + 1);
    offsets.push(0u32);

    let mut i32_elements = Vec::with_capacity(N_LISTS * AVG_LIST_LEN);
    let mut utf8_elements = Vec::with_capacity(N_LISTS * AVG_LIST_LEN);
    let words = [
        "AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG",
        "TT",
    ];

    for row in 0..N_LISTS {
        let len = rng.random_range(0..max_len);
        for _ in 0..len {
            match data_set {
                DataSet::RandomI32 => i32_elements.push(rng.random::<i32>()),
                DataSet::LowCardinalityI32 | DataSet::NullableI32 => {
                    i32_elements.push(rng.random_range(0i32..100))
                }
                DataSet::LowCardinalityUtf8 => {
                    let idx = (row + rng.random_range(0..words.len())) % words.len();
                    utf8_elements.push(words[idx]);
                }
            }
        }
        let n_elements = match data_set {
            DataSet::LowCardinalityUtf8 => utf8_elements.len(),
            _ => i32_elements.len(),
        };
        offsets.push(u32::try_from(n_elements).expect("element count fits in u32"));
    }

    let elements = match data_set {
        DataSet::LowCardinalityUtf8 => VarBinArray::from(utf8_elements).into_array(),
        _ => Buffer::from(i32_elements).into_array(),
    };

    let validity = match data_set {
        DataSet::NullableI32 => {
            Validity::Array(BoolArray::from_iter((0..N_LISTS).map(|i| i % 7 != 0)).into_array())
        }
        _ => Validity::NonNullable,
    };

    ListArray::try_new(elements, Buffer::from(offsets).into_array(), validity)
        .expect("valid list")
        .into_array()
}

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
});

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>();
    vortex_alp::initialize(&session);
    vortex_fastlanes::initialize(&session);
    vortex_fsst::initialize(&session);
    vortex_runend::initialize(&session);
    vortex_sequence::initialize(&session);
    session
});

fn compressed_flat_strategy() -> Arc<dyn LayoutStrategy> {
    Arc::new(
        CompressingStrategy::new(
            FlatLayoutStrategy::default(),
            BtrBlocksCompressor::default(),
        )
        .with_concurrency(1),
    )
}

fn list_compressed_strategy() -> Arc<dyn LayoutStrategy> {
    let compressed_flat = compressed_flat_strategy();
    Arc::new(
        ListLayoutStrategy::default()
            .with_elements(Arc::clone(&compressed_flat))
            .with_offsets(Arc::clone(&compressed_flat))
            .with_validity(Arc::clone(&compressed_flat))
            .with_fallback(compressed_flat),
    )
}

fn strategy(kind: LayoutKind) -> Arc<dyn LayoutStrategy> {
    match kind {
        LayoutKind::List => list_compressed_strategy(),
        LayoutKind::Flat => compressed_flat_strategy(),
    }
}

struct DiskSegments {
    file: Arc<Mutex<File>>,
    specs: Arc<Mutex<Vec<(u64, u32)>>>,
    next_offset: Arc<Mutex<u64>>,
}

impl DiskSegments {
    fn create() -> std::io::Result<Self> {
        let file = tempfile::tempfile()?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            specs: Arc::new(Mutex::new(Vec::new())),
            next_offset: Arc::new(Mutex::new(0)),
        })
    }

    fn total_bytes(&self) -> u64 {
        *self.next_offset.lock()
    }
}

#[async_trait]
impl SegmentSink for DiskSegments {
    async fn write(
        &self,
        _sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        let mut file = self.file.lock();
        let mut specs = self.specs.lock();
        let mut next_offset = self.next_offset.lock();

        let segment_id = SegmentId::from(u32::try_from(specs.len())?);
        let offset = *next_offset;
        file.seek(SeekFrom::Start(offset))?;

        let mut len = 0u32;
        for buf in buffers {
            file.write_all(buf.as_ref())?;
            len = len
                .checked_add(u32::try_from(buf.len())?)
                .ok_or_else(|| vortex_err!("segment length exceeds u32"))?;
        }

        specs.push((offset, len));
        *next_offset = offset + u64::from(len);
        Ok(segment_id)
    }
}

impl SegmentSource for DiskSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let file = Arc::clone(&self.file);
        let specs = Arc::clone(&self.specs);
        async move {
            let (offset, len) = specs
                .lock()
                .get(*id as usize)
                .copied()
                .ok_or_else(|| vortex_err!("Segment {} not found", *id))?;

            let mut bytes = vec![0u8; len as usize];
            let mut file = file.lock();
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut bytes)?;
            Ok(BufferHandle::new_host(ByteBuffer::from(bytes)))
        }
        .boxed()
    }
}

struct Fixture {
    reader: LayoutReaderRef,
    row_count: u64,
    bytes: u64,
}

async fn write_layout(kind: LayoutKind, array: ArrayRef) -> (LayoutRef, Arc<DiskSegments>) {
    let segments = Arc::new(DiskSegments::create().expect("temp file"));
    let (ptr, eof) = SequenceId::root().split();
    let stream = array.to_array_stream().sequenced(ptr);
    let segment_sink: Arc<dyn SegmentSink> = Arc::<DiskSegments>::clone(&segments);
    let layout = strategy(kind)
        .write_stream(ArrayContext::empty(), segment_sink, stream, eof, &SESSION)
        .await
        .expect("write layout");
    (layout, segments)
}

fn make_fixture(kind: LayoutKind, data_set: DataSet) -> Fixture {
    RUNTIME.block_on(async {
        let (layout, segments) = write_layout(kind, make_list_array(data_set)).await;
        let row_count = layout.row_count();
        let bytes = segments.total_bytes();
        let segment_source: Arc<dyn SegmentSource> = segments;
        let reader = layout
            .new_reader(
                "bench".into(),
                segment_source,
                &SESSION,
                &LayoutReaderContext::new(),
            )
            .expect("reader");
        Fixture {
            reader,
            row_count,
            bytes,
        }
    })
}

fn whole_range(row_count: u64) -> std::ops::Range<u64> {
    0..row_count
}

fn partial_range(row_count: u64) -> std::ops::Range<u64> {
    let len = row_count / 100;
    let start = row_count / 2;
    start..start.saturating_add(len).min(row_count)
}

fn sparse_mask(row_count: u64) -> Mask {
    Mask::from_indices(
        usize::try_from(row_count).expect("row count fits"),
        (0..usize::try_from(row_count).expect("row count fits")).step_by(17),
    )
}

fn force_canonical_len(array: ArrayRef) -> VortexResult<usize> {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = array.execute::<Canonical>(&mut ctx)?;
    Ok(canonical.len())
}

async fn project_len_only(fixture: &Fixture, range: std::ops::Range<u64>, mask: MaskFuture) -> u64 {
    let array = fixture
        .reader
        .projection_evaluation(&range, &root(), mask)
        .expect("projection")
        .await
        .expect("projection result");
    u64::try_from(array.len()).expect("array len fits")
}

async fn project_materialized(
    fixture: &Fixture,
    range: std::ops::Range<u64>,
    expr: &Expression,
    mask: MaskFuture,
) -> usize {
    let array = fixture
        .reader
        .projection_evaluation(&range, expr, mask)
        .expect("projection")
        .await
        .expect("projection result");
    force_canonical_len(array).expect("canonicalize projection")
}

async fn filter_true_count(fixture: &Fixture, expr: &Expression) -> usize {
    let len = usize::try_from(fixture.row_count).expect("row count fits");
    fixture
        .reader
        .filter_evaluation(
            &whole_range(fixture.row_count),
            expr,
            MaskFuture::new_true(len),
        )
        .expect("filter")
        .await
        .expect("filter result")
        .true_count()
}

fn print_byte_sizes() {
    eprintln!("---- list_vs_flat byte sizes ({N_LISTS} lists x ~{AVG_LIST_LEN} elements) ----");
    eprintln!("  data_set                  list_bytes    flat_bytes   list/flat");
    for data_set in DATA_SETS {
        let list = make_fixture(LayoutKind::List, data_set);
        let flat = make_fixture(LayoutKind::Flat, data_set);
        let ratio = list.bytes as f64 / flat.bytes as f64;
        eprintln!(
            "  {:<24} {:>10}    {:>10}    {:>5.3}",
            data_set.name(),
            list.bytes,
            flat.bytes,
            ratio,
        );
    }
    eprintln!();
}

fn run_project_root_full(fixture: &Fixture) -> u64 {
    RUNTIME.block_on(project_len_only(
        fixture,
        whole_range(fixture.row_count),
        MaskFuture::new_true(usize::try_from(fixture.row_count).expect("row count fits")),
    ))
}

fn run_project_root_partial(fixture: &Fixture) -> u64 {
    let range = partial_range(fixture.row_count);
    let len = usize::try_from(range.end - range.start).expect("range len fits");
    RUNTIME.block_on(project_len_only(fixture, range, MaskFuture::new_true(len)))
}

fn run_project_root_sparse(fixture: &Fixture) -> u64 {
    let mask = sparse_mask(fixture.row_count);
    RUNTIME.block_on(project_len_only(
        fixture,
        whole_range(fixture.row_count),
        MaskFuture::ready(mask),
    ))
}

fn run_project_list_length(fixture: &Fixture) -> usize {
    let expr = list_length(root());
    RUNTIME.block_on(project_materialized(
        fixture,
        whole_range(fixture.row_count),
        &expr,
        MaskFuture::new_true(usize::try_from(fixture.row_count).expect("row count fits")),
    ))
}

fn run_filter_list_length(fixture: &Fixture) -> usize {
    let expr = gt(list_length(root()), lit(LENGTH_FILTER_THRESHOLD));
    RUNTIME.block_on(filter_true_count(fixture, &expr))
}

fn run_filter_is_not_null(fixture: &Fixture) -> usize {
    let expr = is_not_null(root());
    RUNTIME.block_on(filter_true_count(fixture, &expr))
}

macro_rules! reader_bench {
    ($fn_name:ident, $layout_kind:expr, $runner:ident) => {
        #[divan::bench(args = DATA_SETS)]
        fn $fn_name(bencher: Bencher, data_set: DataSet) {
            let fixture = make_fixture($layout_kind, data_set);
            bencher.bench_local(|| black_box($runner(&fixture)));
        }
    };
}

reader_bench!(
    project_root_full_list,
    LayoutKind::List,
    run_project_root_full
);
reader_bench!(
    project_root_full_flat,
    LayoutKind::Flat,
    run_project_root_full
);
reader_bench!(
    project_root_partial_list,
    LayoutKind::List,
    run_project_root_partial
);
reader_bench!(
    project_root_partial_flat,
    LayoutKind::Flat,
    run_project_root_partial
);
reader_bench!(
    project_root_sparse_list,
    LayoutKind::List,
    run_project_root_sparse
);
reader_bench!(
    project_root_sparse_flat,
    LayoutKind::Flat,
    run_project_root_sparse
);

reader_bench!(
    project_list_length_list,
    LayoutKind::List,
    run_project_list_length
);

reader_bench!(
    project_list_length_flat,
    LayoutKind::Flat,
    run_project_list_length
);

reader_bench!(
    filter_list_length_list,
    LayoutKind::List,
    run_filter_list_length
);

reader_bench!(
    filter_list_length_flat,
    LayoutKind::Flat,
    run_filter_list_length
);

reader_bench!(
    filter_is_not_null_list,
    LayoutKind::List,
    run_filter_is_not_null
);

reader_bench!(
    filter_is_not_null_flat,
    LayoutKind::Flat,
    run_filter_is_not_null
);
