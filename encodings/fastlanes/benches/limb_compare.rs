// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! How much of the limb-split compare advantage survives when the limbs are *bit-packed*, which is
//! the state a wide decimal column is actually in after compression.
//!
//! Every scenario evaluates the same predicate — `value < constant` over a 128-bit column — four
//! ways, so rows within a scenario are directly comparable:
//!
//! * `contiguous_raw` — the predicate over an in-memory `Buffer<i128>`, no decode. This is today's
//!   path for a wide decimal, which gets no compression at all
//!   (`vortex-btrblocks/src/schemes/decimal.rs:79` bails for I128/I256), so the buffer is read
//!   as-is. Best case for the contiguous layout.
//! * `contiguous_rebuild` — decode both limbs, interleave them back into a contiguous `i128`
//!   buffer, then compare. This is what canonicalizing a limb-split array costs, i.e. the price of
//!   keeping contiguous canonical while storing limbs compressed.
//! * `limbs_unpacked` — decode both limbs to plain slices, then compare lexicographically without
//!   interleaving. Isolates the layout change from the compressed-domain change.
//! * `limbs_bitpacked` — [`limbs_lt_constant`] straight onto the compressed limbs. Never
//!   materializes them, folds a constant limb into a scalar comparison, and skips trailing limbs
//!   once no row is still tied.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench limb_compare`. `layout_report` is not a timing
//! bench; it prints which encoding each limb compressed to and how many limbs the kernel actually
//! decoded, which is what the timings should be read against.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::cast_sign_loss)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_buffer::pack_bools_into_words;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::limbs_lt_constant;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

/// Vortex splits scans at 100k rows (`vortex-layout/src/scan/mod.rs`), so measure at that order
/// plus one L2-resident batch below it.
const LENS: &[usize] = &[8 * 1024, 100 * 1024];

/// Deterministic xorshift, so the data depends on neither a PRNG crate nor run order.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// The regimes that decide whether a limb split can do anything at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scenario {
    /// `DECIMAL(38,2)` holding money: every value fits `i64`, so the high limb is all-zero and
    /// compresses to a constant. The low limb decides every row.
    Money,
    /// Money plus one value above `i64::MAX`. That single row defeats the min/max narrowing in
    /// `narrowed_decimal`, so today the whole chunk is stored raw at 16 B/value — while the limb
    /// split still folds the high limb to a constant and only stores the low limb.
    Outlier,
    /// Values near 2^100, with a predicate constant whose high limb matches a value present in the
    /// column. A single tie is enough to force the low limb to be decoded, so this is the
    /// pessimistic case for the early exit.
    WideTie,
    /// The same values, with a constant whose high limb appears nowhere. No row survives the high
    /// limb, so the low limb is never decoded — the optimistic case.
    WideNoTie,
}

impl Scenario {
    const ALL: &'static [Scenario] = &[
        Scenario::Money,
        Scenario::Outlier,
        Scenario::WideTie,
        Scenario::WideNoTie,
    ];

    fn name(self) -> &'static str {
        match self {
            Scenario::Money => "money",
            Scenario::Outlier => "outlier",
            Scenario::WideTie => "wide_tie",
            Scenario::WideNoTie => "wide_no_tie",
        }
    }

    fn values(self, len: usize) -> Vec<i128> {
        let mut rng = Rng(0x5DEECE66D);
        match self {
            // Unscaled cents up to ~10^12, the range a real money column occupies.
            Scenario::Money => (0..len)
                .map(|_| i128::from(rng.next() % 1_000_000_000_000))
                .collect(),
            Scenario::Outlier => {
                let mut v: Vec<i128> = (0..len)
                    .map(|_| i128::from(rng.next() % 1_000_000_000_000))
                    .collect();
                v[len / 2] = (1i128 << 63) + 12345;
                v
            }
            // ~2^100. High limbs are forced even so a no-tie constant can be built simply by making
            // the constant's high limb odd, which keeps selectivity at ~50% either way.
            Scenario::WideTie | Scenario::WideNoTie => (0..len)
                .map(|_| {
                    let hi = i128::from(2 * (rng.next() % (1 << 35)));
                    (hi << 64) | i128::from(rng.next())
                })
                .collect(),
        }
    }

    /// A constant at roughly median selectivity. For [`Scenario::WideNoTie`] the high limb is
    /// bumped to an odd value, which no row can equal, so no row survives the leading limb.
    fn constant(self, values: &[i128]) -> i128 {
        let mut sorted: Vec<i128> = values.to_vec();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2];
        match self {
            Scenario::WideNoTie => {
                let (hi, _) = split(median);
                join(hi + 1, u64::MAX)
            }
            _ => median,
        }
    }
}

fn split(v: i128) -> (i64, u64) {
    ((v >> 64) as i64, v as u64)
}

fn join(hi: i64, lo: u64) -> i128 {
    (i128::from(hi) << 64) | i128::from(lo)
}

/// Page-align a packed buffer so cache-line placement, and therefore the measurement, does not
/// drift with allocator state. Same reasoning as `bitpack_compare.rs`.
fn page_aligned(array: BitPackedArray) -> BitPackedArray {
    let ptype = array.dtype().as_ptype();
    let parts = BitPacked::into_parts(array);
    BitPacked::try_new(
        parts.packed.ensure_aligned(Alignment::new(4096)).unwrap(),
        ptype,
        parts.validity,
        parts.patches,
        parts.bit_width,
        parts.len,
        parts.offset,
    )
    .unwrap()
}

/// Compress one limb the way the cascading compressor would: a constant when uniform, otherwise
/// bit-packed at the narrowest width that fits, otherwise raw — `bitpack_encode` needs a width
/// below 64 and rejects negatives, so a full-entropy or mixed-sign limb stays uncompressed here.
fn compress_limb<T>(values: &[T], ctx: &mut ExecutionCtx) -> ArrayRef
where
    T: NativePType + Ord + Into<i128>,
    Scalar: From<T>,
{
    let prim = PrimitiveArray::new(
        values.iter().copied().collect::<BufferMut<T>>().freeze(),
        Validity::NonNullable,
    );
    let min: i128 = (*values.iter().min().unwrap()).into();
    let max: i128 = (*values.iter().max().unwrap()).into();
    if min == max {
        return ConstantArray::new(Scalar::from(values[0]), values.len()).into_array();
    }
    if min < 0 {
        return prim.into_array();
    }
    let bits = 128 - u128::try_from(max).unwrap_or(0).leading_zeros();
    if bits >= 64 {
        return prim.into_array();
    }
    match bitpack_encode(&prim, bits.max(1) as u8, None, ctx) {
        Ok(packed) => page_aligned(packed).into_array(),
        Err(_) => prim.into_array(),
    }
}

/// One scenario materialized in both physical layouts, with its predicate constant.
struct Case {
    /// `[signed high, unsigned low]`, each compressed independently.
    limbs: Vec<ArrayRef>,
    /// The contiguous canonical form: the `Buffer<i128>` a `DecimalArray` holds today. It cannot be
    /// a `PrimitiveArray`, because `PType` has no `i128` — which is exactly why wide decimals are
    /// locked out of the integer compression schemes.
    contiguous: Vec<i128>,
    constant: i128,
}

impl Case {
    fn new(scenario: Scenario, len: usize) -> Self {
        let mut ctx = SESSION.create_execution_ctx();
        let values = scenario.values(len);
        let constant = scenario.constant(&values);
        let his: Vec<i64> = values.iter().map(|v| split(*v).0).collect();
        let los: Vec<u64> = values.iter().map(|v| split(*v).1).collect();
        Self {
            limbs: vec![compress_limb(&his, &mut ctx), compress_limb(&los, &mut ctx)],
            contiguous: values,
            constant,
        }
    }

    fn len(&self) -> usize {
        self.contiguous.len()
    }

    fn constant_limbs(&self) -> [u64; 2] {
        let (hi, lo) = split(self.constant);
        [hi as u64, lo]
    }
}

/// Trailing component of an encoding id, e.g. `bitpacked`.
fn short_id(array: &ArrayRef) -> String {
    let id = array.encoding_id().to_string();
    id.rsplit('.').next().unwrap_or(&id).to_string()
}

/// Decode a limb to a plain slice — what every non-pushdown variant must do first.
fn decode_limb(limb: &ArrayRef, ctx: &mut ExecutionCtx) -> PrimitiveArray {
    limb.clone().execute::<PrimitiveArray>(ctx).unwrap()
}

fn contiguous_lt(values: &[i128], c: i128, words: &mut [u64]) {
    pack_bools_into_words(words, 0, values.len(), |i| values[i] < c);
}

fn limbs_lt(his: &[i64], los: &[u64], c: i128, words: &mut [u64]) {
    let (c_hi, c_lo) = split(c);
    pack_bools_into_words(words, 0, his.len(), |i| {
        his[i] < c_hi || (his[i] == c_hi && los[i] < c_lo)
    });
}

/// Generate the four measurement variants for each scenario, so a scenario's rows sit together in
/// the output.
macro_rules! scenario_benches {
    ($($module:ident => $scenario:expr),* $(,)?) => {
        $(
            mod $module {
                use super::*;

                /// Today's wide-decimal path: an uncompressed contiguous `i128` buffer, no decode.
                #[divan::bench(args = LENS)]
                fn contiguous_raw(bencher: Bencher, len: usize) {
                    let case = Case::new($scenario, len);
                    let values = case.contiguous.clone();
                    let c = case.constant;
                    let mut words = vec![0u64; len.div_ceil(64)];
                    bencher
                        .counter(ItemsCount::new(len))
                        .bench_local(|| contiguous_lt(&values, c, &mut words));
                }

                /// Canonicalize the limbs back to a contiguous `i128` buffer, then compare.
                #[divan::bench(args = LENS)]
                fn contiguous_rebuild(bencher: Bencher, len: usize) {
                    let case = Case::new($scenario, len);
                    let mut ctx = SESSION.create_execution_ctx();
                    let c = case.constant;
                    let mut words = vec![0u64; len.div_ceil(64)];
                    bencher.counter(ItemsCount::new(len)).bench_local(|| {
                        let his = decode_limb(&case.limbs[0], &mut ctx);
                        let los = decode_limb(&case.limbs[1], &mut ctx);
                        let his = his.as_slice::<i64>();
                        let los = los.as_slice::<u64>();
                        let values: Vec<i128> = (0..len).map(|i| join(his[i], los[i])).collect();
                        contiguous_lt(&values, c, &mut words);
                    });
                }

                /// Decode the limbs, then compare limb-major without interleaving.
                #[divan::bench(args = LENS)]
                fn limbs_unpacked(bencher: Bencher, len: usize) {
                    let case = Case::new($scenario, len);
                    let mut ctx = SESSION.create_execution_ctx();
                    let c = case.constant;
                    let mut words = vec![0u64; len.div_ceil(64)];
                    bencher.counter(ItemsCount::new(len)).bench_local(|| {
                        let his = decode_limb(&case.limbs[0], &mut ctx);
                        let los = decode_limb(&case.limbs[1], &mut ctx);
                        limbs_lt(his.as_slice::<i64>(), los.as_slice::<u64>(), c, &mut words);
                    });
                }

                /// Compare directly on the compressed limbs.
                #[divan::bench(args = LENS)]
                fn limbs_bitpacked(bencher: Bencher, len: usize) {
                    let case = Case::new($scenario, len);
                    let mut ctx = SESSION.create_execution_ctx();
                    let cl = case.constant_limbs();
                    bencher.counter(ItemsCount::new(len)).bench_local(|| {
                        limbs_lt_constant(&case.limbs, &cl, case.len(), &mut ctx).unwrap()
                    });
                }
            }
        )*
    };
}

scenario_benches! {
    money => Scenario::Money,
    outlier => Scenario::Outlier,
    wide_tie => Scenario::WideTie,
    wide_no_tie => Scenario::WideNoTie,
}

/// Not a timing bench: prints the encoding each limb compressed to, how many limbs the kernel had
/// to decode, the resulting selectivity, and the compressed size of the limb pair against the
/// 16 B/value a contiguous buffer costs.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn layout_report(bencher: Bencher) {
    bencher.bench_local(|| {
        let mut ctx = SESSION.create_execution_ctx();
        eprintln!();
        for &len in LENS {
            for &scenario in Scenario::ALL {
                let case = Case::new(scenario, len);
                let out =
                    limbs_lt_constant(&case.limbs, &case.constant_limbs(), case.len(), &mut ctx)
                        .unwrap();
                let selectivity =
                    (0..case.len()).filter(|&i| out.bits.value(i)).count() * 100 / case.len();
                let limb_bytes: u64 = case.limbs.iter().map(|l| l.nbytes()).sum();
                eprintln!(
                    "[layout] len={len:>6} {:<12} limbs={:<24} decoded={}/2 \
                     sel={selectivity:>2}% limb_bytes/val={:>5.2} vs contiguous 16.00",
                    scenario.name(),
                    format!("{}+{}", short_id(&case.limbs[0]), short_id(&case.limbs[1])),
                    out.limbs_decoded,
                    limb_bytes as f64 / case.len() as f64,
                );
            }
        }
    });
}
