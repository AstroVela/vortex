// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures the framework overhead of storing array data as a buffer on the array node
//! (e.g. `PrimitiveArray`) versus storing it as an uncompressed child array (an extra
//! `ArrayNode` in the serialized tree).
//!
//! The "child" form is modeled by a minimal identity `Wrapper` encoding with exactly one
//! child, no buffers, and empty metadata, so the measured delta is purely the cost the
//! Vortex framework charges for one extra node in the encoding tree.
//!
//! Running `cargo bench --bench serde_buffer_vs_child` first prints a size study (both
//! forms serialized to real files with the same code path the file writer uses for leaf
//! segments) and then runs timing benchmarks for serialize, decode, and decode+access.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::serde::ArrayChildren;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_array::vtable::with_empty_buffers;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

fn main() {
    LazyLock::force(&SESSION);
    print_size_study().unwrap();
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session();
    session.arrays().register(Wrapper);
    session
});

/// An identity encoding that stores its data as a single uncompressed child array, with no
/// buffers and no metadata: the minimal possible "store as child" encoding.
#[derive(Clone, Debug)]
struct Wrapper;

#[derive(Clone, Debug)]
struct WrapperData;

impl Display for WrapperData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("wrapper")
    }
}

impl ArrayHash for WrapperData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _eq_mode: EqMode) {}
}

impl ArrayEq for WrapperData {
    fn array_eq(&self, _other: &Self, _eq_mode: EqMode) -> bool {
        true
    }
}

impl ValidityChild<Wrapper> for Wrapper {
    fn validity_child(array: ArrayView<'_, Wrapper>) -> ArrayRef {
        child(&array)
    }
}

fn child(array: &ArrayView<'_, Wrapper>) -> ArrayRef {
    array.slots()[0]
        .clone()
        .unwrap_or_else(|| vortex_panic!("Wrapper child slot is missing"))
}

impl Wrapper {
    fn wrap(child: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(Array::try_from_parts(
            ArrayParts::new(Wrapper, child.dtype().clone(), child.len(), WrapperData)
                .with_slots(smallvec![Some(child)]),
        )?
        .into_array())
    }

    fn wrap_depth(mut array: ArrayRef, depth: usize) -> VortexResult<ArrayRef> {
        for _ in 0..depth {
            array = Self::wrap(array)?;
        }
        Ok(array)
    }
}

impl VTable for Wrapper {
    type TypedArrayData = WrapperData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.bench.wrapper");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(slots.len() == 1, "Wrapper must have one child slot");
        let Some(child) = &slots[0] else {
            vortex_error::vortex_bail!("Wrapper child slot is missing");
        };
        vortex_ensure!(child.dtype() == dtype, "Wrapper child dtype mismatch");
        vortex_ensure!(child.len() == len, "Wrapper child length mismatch");
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("Wrapper buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(children.len() == 1, "Wrapper expects one child");
        let child = children.get(0, dtype, len)?;
        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, WrapperData)
                .with_slots(smallvec![Some(child)]),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, _idx: usize) -> String {
        "child".to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(child(&array.as_view())))
    }
}

fn make_primitive(n: usize) -> ArrayRef {
    (0..n as u64).collect::<PrimitiveArray>().into_array()
}

/// Serialize an array exactly as the file writer serializes a leaf segment, returning the
/// concatenated bytes.
fn serialize_to_bytes(array: &ArrayRef, with_stats: bool) -> VortexResult<ByteBuffer> {
    if with_stats {
        let mut ctx = SESSION.create_execution_ctx();
        // Populate the pruning-style statistics the file writer typically records.
        array.statistics().compute_min::<u64>(&mut ctx);
        array.statistics().compute_max::<u64>(&mut ctx);
        array.statistics().compute_null_count(&mut ctx);
    }
    let buffers = array.serialize(
        &ArrayContext::empty(),
        &SESSION,
        &SerializeOptions {
            offset: 0,
            include_padding: true,
        },
    )?;
    let mut bytes = ByteBufferMut::empty();
    for buffer in buffers {
        bytes.extend_from_slice(buffer.as_ref());
    }
    Ok(bytes.freeze())
}

fn print_size_study() -> VortexResult<()> {
    let dir = std::env::temp_dir().join("vortex-buffer-vs-child");
    std::fs::create_dir_all(&dir)?;

    println!(
        "=== Serialized size: buffer-on-node (depth 0) vs uncompressed child (depth 1..8) ==="
    );
    println!("u64 primitive data, non-nullable; raw data bytes = 8 * n");
    println!();
    println!(
        "{:>10} {:>6} {:>6} {:>12} {:>10} {:>10} {:>12}",
        "n", "stats", "depth", "file bytes", "overhead", "fb bytes", "delta/node"
    );

    for with_stats in [false, true] {
        for n in [1usize, 1 << 10, 1 << 16, 1 << 20] {
            let raw = 8 * n;
            let mut prev = None;
            for depth in [0usize, 1, 2, 4, 8] {
                let array = Wrapper::wrap_depth(make_primitive(n), depth)?;
                let bytes = serialize_to_bytes(&array, with_stats)?;

                let path = dir.join(format!("n{n}_stats{with_stats}_depth{depth}.vxarr"));
                std::fs::write(&path, bytes.as_slice())?;
                let file_bytes = std::fs::metadata(&path)?.len() as usize;

                let tail = &bytes.as_slice()[bytes.len() - 4..];
                let fb_len = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]) as usize;
                let overhead = file_bytes - raw;
                // Report the marginal cost of the nodes added since the previous row.
                let delta_per_node = prev
                    .map(|p: usize| (file_bytes - p) as f64 / (depth_step(depth) as f64))
                    .map(|d| format!("{d:+.1}"))
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "{n:>10} {with_stats:>6} {depth:>6} {file_bytes:>12} {overhead:>10} {fb_len:>10} {delta_per_node:>12}"
                );
                prev = Some(file_bytes);
            }
            println!();
        }
    }
    Ok(())
}

/// Number of wrapper nodes added between consecutive rows of the depth sweep.
fn depth_step(depth: usize) -> usize {
    match depth {
        1 => 1,
        2 => 1,
        4 => 2,
        8 => 4,
        _ => 1,
    }
}

const N: usize = 1 << 16;
const DEPTHS: &[usize] = &[0, 1, 2, 4, 8];

fn serialized(depth: usize) -> (ByteBuffer, DType, ReadContext) {
    let array = Wrapper::wrap_depth(make_primitive(N), depth).unwrap();
    let dtype = array.dtype().clone();
    let ctx = ArrayContext::empty();
    let buffers = array
        .serialize(
            &ctx,
            &SESSION,
            &SerializeOptions {
                offset: 0,
                include_padding: true,
            },
        )
        .unwrap();
    let mut bytes = ByteBufferMut::empty();
    for buffer in buffers {
        bytes.extend_from_slice(buffer.as_ref());
    }
    (bytes.freeze(), dtype, ReadContext::new(ctx.to_ids()))
}

/// Time to serialize the array (the write-side framework cost).
#[divan::bench(args = DEPTHS)]
fn serialize(bencher: Bencher, depth: usize) {
    let array = Wrapper::wrap_depth(make_primitive(N), depth).unwrap();
    let ctx = ArrayContext::empty();
    bencher.bench(|| {
        array
            .serialize(
                &ctx,
                &SESSION,
                &SerializeOptions {
                    offset: 0,
                    include_padding: true,
                },
            )
            .unwrap()
    });
}

/// Time to parse + decode the serialized bytes back into an `ArrayRef`.
#[divan::bench(args = DEPTHS)]
fn decode(bencher: Bencher, depth: usize) {
    let (bytes, dtype, read_ctx) = serialized(depth);
    bencher.bench(|| {
        SerializedArray::try_from(bytes.clone())
            .unwrap()
            .decode(&dtype, N, &read_ctx, &SESSION)
            .unwrap()
    });
}

/// Time to decode and then reach the actual data: execute to a canonical
/// `PrimitiveArray` and touch an element.
#[divan::bench(args = DEPTHS)]
fn decode_and_access(bencher: Bencher, depth: usize) {
    let (bytes, dtype, read_ctx) = serialized(depth);
    bencher.bench(|| {
        let array = SerializedArray::try_from(bytes.clone())
            .unwrap()
            .decode(&dtype, N, &read_ctx, &SESSION)
            .unwrap();
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = array.execute::<PrimitiveArray>(&mut ctx).unwrap();
        divan::black_box(primitive.as_slice::<u64>()[N / 2])
    });
}

/// Access cost alone: the array is already decoded, measure getting to the data.
#[divan::bench(args = DEPTHS)]
fn access_decoded(bencher: Bencher, depth: usize) {
    let array = Wrapper::wrap_depth(make_primitive(N), depth).unwrap();
    bencher.bench(|| {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        divan::black_box(primitive.as_slice::<u64>()[N / 2])
    });
}
