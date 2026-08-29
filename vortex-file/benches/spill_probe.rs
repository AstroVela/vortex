// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex counterpart to the Feldera `spill_probe` benchmark.
//!
//! Generates the same records that the DBSP probe spills to a layer file — a `u64` key, ten
//! `u64` value fields and an `i64` weight — writes them to a Vortex file, and reports on-disk
//! bytes and sequential scan throughput. The two probes share a seed and generation order so
//! their outputs are directly comparable.

#![expect(clippy::cast_possible_truncation, clippy::print_stdout)]

use std::sync::LazyLock;
use std::time::Instant;

use futures::StreamExt;
use futures::pin_mut;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_edition::ComponentKind;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::session::LayoutSession;
use vortex_layout::session::LayoutSessionExt;
use vortex_session::VortexSession;

const NUM_RECORDS: usize = 2_000_000;
const NUM_BATCHES: usize = 8;
const KEY_RANGES: &[u64] = &[100, 100_000_000];

/// Shape of the ten value fields.
///
/// `Random` matches the DBSP `list_merger` benchmark exactly: ten independent random `u64`s,
/// which no columnar format can compress. `Realistic` approximates a Nexmark-style bid row,
/// where most fields are low-cardinality, clustered, or monotonic, which is what production
/// spilled batches actually look like.
#[derive(Clone, Copy, Debug)]
enum Profile {
    Random,
    Realistic,
}
const VALUE_FIELDS: usize = 10;

/// Raw logical bytes per record: 1 key `u64` + 10 value `u64` + 1 `i64` weight.
const RAW_BYTES_PER_RECORD: usize = 8 * 12;

static RUNTIME: LazyLock<tokio::runtime::Runtime> =
    LazyLock::new(|| tokio::runtime::Runtime::new().unwrap());

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let _guard = RUNTIME.enter();
    let session = array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_tokio();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);
    session
});

const BENCH_EDITION: EditionId = EditionId::new("spill_probe", 2026, 8, 0);

/// Enables every registered component so the write path may use the full encoding and
/// statistics surface, matching what a production writer would have available.
fn enable_all_registered_array_encodings(session: &VortexSession) {
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: BENCH_EDITION,
            min_vortex_version: None,
        })
        .unwrap();
    let component_ids = [
        (
            ComponentKind::Array,
            session
                .arrays()
                .registry()
                .read(|map| map.keys().copied().collect::<Vec<_>>()),
        ),
        (
            ComponentKind::Layout,
            session
                .layouts()
                .registry()
                .read(|map| map.keys().copied().collect::<Vec<_>>()),
        ),
        (
            ComponentKind::DType,
            session
                .dtypes()
                .registry()
                .read(|map| map.keys().copied().collect::<Vec<_>>()),
        ),
    ];
    for (kind, ids) in component_ids {
        for id in ids {
            editions
                .declare_inclusion(EditionInclusion::new(kind, &id, BENCH_EDITION))
                .unwrap();
        }
    }
    for id in [
        "vortex.bounded_max",
        "vortex.bounded_min",
        "vortex.max",
        "vortex.min",
        "vortex.nan_count",
        "vortex.null_count",
    ] {
        editions
            .declare_inclusion(EditionInclusion::new(
                ComponentKind::Aggregate,
                id,
                BENCH_EDITION,
            ))
            .unwrap();
    }
    session.enable_edition(BENCH_EDITION).unwrap();
}

/// Xoshiro256**, matching the generator the DBSP probe uses so both see identical records.
struct Xoshiro {
    s: [u64; 4],
}

impl Xoshiro {
    fn from_seed(seed: [u8; 32]) -> Self {
        let mut s = [0u64; 4];
        for (i, chunk) in seed.chunks_exact(8).enumerate() {
            s[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `0..range`, using the same widening-multiply reduction as `rand`'s
    /// `gen_range` fast path.
    fn gen_range(&mut self, range: u64) -> u64 {
        ((u128::from(self.next_u64()) * u128::from(range)) >> 64) as u64
    }
}

const SEED: [u8; 32] = [
    0x7f, 0xc3, 0x59, 0x18, 0x45, 0x19, 0xc0, 0xaa, 0xd2, 0xec, 0x31, 0x26, 0xbb, 0x74, 0x2f, 0x8b,
    0x11, 0x7d, 0x0c, 0xe4, 0x64, 0xbf, 0x72, 0x17, 0x46, 0x28, 0x46, 0x42, 0xb2, 0x4b, 0x72, 0x18,
];

fn records_in_batch(batch_index: usize) -> usize {
    let base = NUM_RECORDS / NUM_BATCHES;
    let remainder = NUM_RECORDS % NUM_BATCHES;
    base + usize::from(batch_index < remainder)
}

/// Builds one chunk per batch, sorted by key, mirroring how DBSP seals a spilled batch.
/// Fills one row's value fields according to `profile`.
fn fill_value(rng: &mut Xoshiro, profile: Profile, row: usize, value: &mut [u64; VALUE_FIELDS]) {
    match profile {
        Profile::Random => {
            for slot in value.iter_mut() {
                *slot = rng.next_u64();
            }
        }
        Profile::Realistic => {
            value[0] = rng.gen_range(8); // small enum, e.g. channel
            value[1] = rng.gen_range(64); // small enum, e.g. category
            value[2] = 1_700_000_000_000 + row as u64 * 7; // monotonic timestamp
            value[3] = 1_700_000_000_000 + row as u64 * 7 + rng.gen_range(1_000); // near-copy
            value[4] = rng.gen_range(10_000); // medium-cardinality id
            value[5] = rng.gen_range(1_000_000); // high-cardinality id
            value[6] = rng.gen_range(100); // price bucket
            value[7] = u64::from(rng.gen_range(2) == 0); // boolean-ish flag
            value[8] = rng.gen_range(256); // byte-ranged field
            value[9] = rng.next_u64(); // one genuinely random field
        }
    }
}

fn generate_chunks(key_range: u64, profile: Profile) -> Vec<ArrayRef> {
    let mut rng = Xoshiro::from_seed(SEED);

    (0..NUM_BATCHES)
        .map(|batch_index| {
            let n = records_in_batch(batch_index);
            let mut records: Vec<(u64, [u64; VALUE_FIELDS])> = (0..n)
                .map(|row| {
                    let key = rng.gen_range(key_range);
                    let mut value = [0u64; VALUE_FIELDS];
                    fill_value(&mut rng, profile, row, &mut value);
                    (key, value)
                })
                .collect();
            // A spilled batch is always sorted by key.
            records.sort_unstable_by_key(|(key, _)| *key);

            let mut keys = BufferMut::with_capacity(n);
            let mut values: Vec<BufferMut<u64>> = (0..VALUE_FIELDS)
                .map(|_| BufferMut::with_capacity(n))
                .collect();
            let mut weights = BufferMut::with_capacity(n);
            for (key, value) in &records {
                keys.push(*key);
                for (field, slot) in values.iter_mut().zip(value.iter()) {
                    field.push(*slot);
                }
                weights.push(1i64);
            }

            let mut fields: Vec<(String, ArrayRef)> = Vec::with_capacity(VALUE_FIELDS + 2);
            fields.push(("key".to_string(), keys.freeze().into_array()));
            for (i, field) in values.into_iter().enumerate() {
                fields.push((format!("v{i}"), field.freeze().into_array()));
            }
            fields.push(("weight".to_string(), weights.freeze().into_array()));

            let refs: Vec<(&str, ArrayRef)> = fields
                .iter()
                .map(|(name, array)| (name.as_str(), array.clone()))
                .collect();
            StructArray::from_fields(&refs).unwrap().into_array()
        })
        .collect()
}

async fn run(key_range: u64, profile: Profile) {
    let gen_start = Instant::now();
    let chunks = generate_chunks(key_range, profile);
    let gen_elapsed = gen_start.elapsed();

    let chunked = ChunkedArray::from_iter(chunks).into_array();

    let write_start = Instant::now();
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, chunked.to_array_stream())
        .await
        .unwrap();
    let write_elapsed = write_start.elapsed();
    let bytes = buf.len();

    let scan_start = Instant::now();
    let stream = SESSION
        .open_options()
        .open_buffer(buf.freeze())
        .unwrap()
        .scan()
        .unwrap()
        .into_array_stream()
        .unwrap();
    pin_mut!(stream);
    let mut rows = 0;
    while let Some(array) = stream.next().await {
        rows += array.unwrap().len();
    }
    let scan_elapsed = scan_start.elapsed();

    let raw = (NUM_RECORDS * RAW_BYTES_PER_RECORD) as f64;
    println!(
        "{profile:>9?}  key_range={key_range:>9}  gen={:>7.2}s  write={:>7.2}s  write_bytes={bytes:>10}  \
         ratio_vs_raw={:>5.2}x  scan={:>7.2}s ({:>5.2} M rec/s)  rows={rows}",
        gen_elapsed.as_secs_f64(),
        write_elapsed.as_secs_f64(),
        raw / bytes.max(1) as f64,
        scan_elapsed.as_secs_f64(),
        NUM_RECORDS as f64 / scan_elapsed.as_secs_f64() / 1e6,
    );
}

fn main() {
    println!(
        "vortex spill_probe: {NUM_RECORDS} records across {NUM_BATCHES} chunks, \
         raw logical size = {:.1} MiB\n",
        (NUM_RECORDS * RAW_BYTES_PER_RECORD) as f64 / (1024.0 * 1024.0)
    );
    for profile in [Profile::Random, Profile::Realistic] {
        for &key_range in KEY_RANGES {
            RUNTIME.block_on(run(key_range, profile));
        }
    }
}
