// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized run-end decoding for boolean arrays.
//!
//! Two kernels, picked per array by [`prefer_splat`]:
//!
//! * *splat* accumulates runs into a 64-bit word and stores each output word exactly once. Head
//!   and tail masking is paid per output word rather than per run, and the run's value only
//!   feeds an `AND` mask, so the value-dependent branch disappears. This is the kernel for short
//!   runs, where it is up to 4x the prefill.
//! * *prefill* memsets the whole output with the majority value and then patches only the runs
//!   that differ. One big memset beats a per-run one, so this stays ahead once runs are long
//!   enough that the splat pays a whole-word fill per run and the patched runs are rare.
//!
//! See `benches/run_end_decode_bool_ablation.rs` for the A/B both the crossover and the splat's
//! output-buffer strategy come from.

use std::mem::MaybeUninit;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

/// Bits per accumulator word.
const WORD_BITS: usize = u64::BITS as usize;

/// Decodes run-end encoded boolean values into a flat `BoolArray`.
pub fn runend_decode_bools(
    ends: PrimitiveArray,
    values: BoolArray,
    offset: usize,
    length: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity = values
        .as_ref()
        .validity()?
        .execute_mask(values.as_ref().len(), ctx)?;
    let values_buf = values.to_bit_buffer();
    let nullability = values.dtype().nullability();

    Ok(match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        runend_decode_typed_bool(
            ends.as_slice::<E>(),
            offset,
            &values_buf,
            validity,
            nullability,
            length,
        )
    }))
}

/// Decodes run-end encoded boolean values into a flat `BoolArray`.
///
/// Run ends are taken as a slice and trimmed by `offset` inline. Consuming them through an
/// iterator chain instead costs a significant fraction of decode time: the per-run work is a
/// handful of instructions, so `trimmed_ends_iter`'s `map`s plus a `zip_eq` plus a bit-at-a-time
/// [`BitBuffer`] iterator are not amortized by anything.
pub fn runend_decode_typed_bool<E: IntegerPType>(
    run_ends: &[E],
    offset: usize,
    values: &BitBuffer,
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> ArrayRef {
    match values_validity {
        Mask::AllTrue(_) => {
            decode_bool_non_nullable(run_ends, offset, values, values_nullability, length)
                .into_array()
        }
        Mask::AllFalse(_) => {
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), length)
                .into_array()
        }
        Mask::Values(mask) => {
            decode_bool_nullable(run_ends, offset, values, mask.bit_buffer(), length).into_array()
        }
    }
}

/// A bit sink that overwrites a pre-sized word buffer, one 64-bit word at a time.
///
/// Bits are accumulated in a register and each word is written to the buffer exactly once, so a
/// run never reads back what an earlier run stored. Compare a per-run `fill_range`, which
/// read-modify-writes a partial byte at each end of every run.
///
/// The buffer is allocated once at the exact final word count and written by index. Appending it
/// with `BufferMut::push`/`push_n` instead keeps the accumulator in memory rather than in
/// registers and puts the spanning-run path behind an out-of-line call; measured, the index
/// writes are worth up to 1.9x. See `push_splat` in `benches/run_end_decode_bool_ablation.rs`.
struct BitSplat<'a> {
    /// Exactly the output words, written in ascending order and each at most once.
    words: &'a mut [MaybeUninit<u64>],
    /// Number of words already written, so `words[..idx]` is initialized.
    idx: usize,
    /// The word under construction. Bits at and above `bits` are always zero.
    word: u64,
    /// Number of bits already accumulated into `word`, always `< WORD_BITS`.
    bits: usize,
}

impl<'a> BitSplat<'a> {
    /// Creates a sink over the output words, which it initializes in full.
    fn new(words: &'a mut [MaybeUninit<u64>]) -> Self {
        Self {
            words,
            idx: 0,
            word: 0,
            bits: 0,
        }
    }

    /// Appends `n` copies of `value`.
    #[inline(always)]
    fn append_run(&mut self, value: bool, n: usize) {
        // 0 or all-ones; the value never branches, it only masks.
        let splat = u64::from(value).wrapping_neg();
        if self.bits + n < WORD_BITS {
            // `n < WORD_BITS`, so the mask shift is in range.
            self.word |= splat & (((1u64 << n) - 1) << self.bits);
            self.bits += n;
            return;
        }
        self.append_spanning(splat, n);
    }

    /// Appends a run that reaches at least to the end of the current word.
    ///
    /// Whole words are filled in one pass rather than splatted one at a time.
    ///
    /// Inlined on purpose: leaving it out of line spills the accumulator to the stack for the
    /// whole loop, since `&mut self` escapes into the call. That costs a read-modify-write per
    /// short run, which is the common case, and `#[inline]` alone is not enough — the crate
    /// build leaves it out of line even where a smaller bench-local copy of the same code is
    /// inlined.
    #[inline(always)]
    fn append_spanning(&mut self, splat: u64, n: usize) {
        // Complete the current word. `self.bits < WORD_BITS`, so the shift is in range, and the
        // run covers every bit from `self.bits` up.
        self.words[self.idx] = MaybeUninit::new(self.word | (splat << self.bits));
        self.idx += 1;
        let rest = n - (WORD_BITS - self.bits);

        let whole = rest / WORD_BITS;
        if whole > 0 {
            self.words[self.idx..self.idx + whole].fill(MaybeUninit::new(splat));
            self.idx += whole;
        }

        let tail = rest % WORD_BITS;
        self.word = splat & ((1u64 << tail) - 1);
        self.bits = tail;
    }

    /// Flushes the accumulator and zero-fills any words the runs did not reach, leaving every
    /// word of the output initialized.
    fn finish(self) {
        let mut idx = self.idx;
        if self.bits > 0 {
            self.words[idx] = MaybeUninit::new(self.word);
            idx += 1;
        }
        self.words[idx..].fill(MaybeUninit::new(0));
    }
}

/// Splats `length` bits into a fresh buffer, driving `fill` over the sink.
fn splat_bits(length: usize, fill: impl FnOnce(&mut BitSplat<'_>)) -> BitBufferMut {
    let n_words = length.div_ceil(WORD_BITS);
    let mut buf = BufferMut::<u64>::with_capacity(n_words);
    {
        let words = &mut buf.spare_capacity_mut()[..n_words];
        let mut splat = BitSplat::new(words);
        fill(&mut splat);
        splat.finish();
    }
    // SAFETY: the sink covered exactly `n_words` words and was finished, so every word in
    // `0..n_words` is initialized, and `n_words` is the capacity requested above.
    unsafe { buf.set_len(n_words) };
    into_bits(buf, length)
}

/// Two bit sinks advanced in lockstep: the decoded bits and the validity bits of the same runs.
///
/// Every run appends the same number of bits to both, so the run length, the word-boundary test,
/// the head mask and the bit counter are computed once for the pair rather than twice. Driving
/// two independent [`BitSplat`]s instead costs a second mask chain and a second boundary branch
/// per run, and the doubled live state spills; measured, fusing them is worth up to 1.4x.
struct BitSplatPair<'a> {
    /// Exactly the decoded output words, written in ascending order and each at most once.
    words: &'a mut [MaybeUninit<u64>],
    /// Exactly the validity output words, written in step with `words`.
    validity: &'a mut [MaybeUninit<u64>],
    /// Number of words already written to each, so `..idx` of both is initialized.
    idx: usize,
    /// The decoded word under construction. Bits at and above `bits` are always zero.
    word: u64,
    /// The validity word under construction, at the same bit position.
    validity_word: u64,
    /// Number of bits already accumulated into both words, always `< WORD_BITS`.
    bits: usize,
}

impl<'a> BitSplatPair<'a> {
    /// Creates a pair of sinks over the two output buffers, which it initializes in full.
    fn new(words: &'a mut [MaybeUninit<u64>], validity: &'a mut [MaybeUninit<u64>]) -> Self {
        Self {
            words,
            validity,
            idx: 0,
            word: 0,
            validity_word: 0,
            bits: 0,
        }
    }

    /// Appends `n` copies of `value` to the decoded bits and of `is_valid` to the validity bits.
    #[inline(always)]
    fn append_run(&mut self, value: bool, is_valid: bool, n: usize) {
        let splat = u64::from(value).wrapping_neg();
        let validity_splat = u64::from(is_valid).wrapping_neg();
        if self.bits + n < WORD_BITS {
            // `n < WORD_BITS`, so the mask shift is in range. One mask serves both words.
            let mask = ((1u64 << n) - 1) << self.bits;
            self.word |= splat & mask;
            self.validity_word |= validity_splat & mask;
            self.bits += n;
            return;
        }
        self.append_spanning(splat, validity_splat, n);
    }

    /// Appends a run that reaches at least to the end of the current word. Inlined for the same
    /// reason as [`BitSplat::append_spanning`].
    #[inline(always)]
    fn append_spanning(&mut self, splat: u64, validity_splat: u64, n: usize) {
        self.words[self.idx] = MaybeUninit::new(self.word | (splat << self.bits));
        self.validity[self.idx] =
            MaybeUninit::new(self.validity_word | (validity_splat << self.bits));
        self.idx += 1;
        let rest = n - (WORD_BITS - self.bits);

        let whole = rest / WORD_BITS;
        if whole > 0 {
            self.words[self.idx..self.idx + whole].fill(MaybeUninit::new(splat));
            self.validity[self.idx..self.idx + whole].fill(MaybeUninit::new(validity_splat));
            self.idx += whole;
        }

        let tail = rest % WORD_BITS;
        let mask = (1u64 << tail) - 1;
        self.word = splat & mask;
        self.validity_word = validity_splat & mask;
        self.bits = tail;
    }

    /// Flushes both accumulators and zero-fills the words the runs did not reach.
    fn finish(self) {
        let mut idx = self.idx;
        if self.bits > 0 {
            self.words[idx] = MaybeUninit::new(self.word);
            self.validity[idx] = MaybeUninit::new(self.validity_word);
            idx += 1;
        }
        self.words[idx..].fill(MaybeUninit::new(0));
        self.validity[idx..].fill(MaybeUninit::new(0));
    }
}

/// Splats `length` decoded bits and `length` validity bits into two fresh buffers, driving `fill`
/// over the fused sink.
fn splat_bits_pair(
    length: usize,
    fill: impl FnOnce(&mut BitSplatPair<'_>),
) -> (BitBufferMut, BitBufferMut) {
    let n_words = length.div_ceil(WORD_BITS);
    let mut buf = BufferMut::<u64>::with_capacity(n_words);
    let mut validity_buf = BufferMut::<u64>::with_capacity(n_words);
    {
        let words = &mut buf.spare_capacity_mut()[..n_words];
        let validity_words = &mut validity_buf.spare_capacity_mut()[..n_words];
        let mut splat = BitSplatPair::new(words, validity_words);
        fill(&mut splat);
        splat.finish();
    }
    // SAFETY: the sink covered exactly `n_words` words of each buffer and was finished, so every
    // word in `0..n_words` of both is initialized, and `n_words` is the capacity of each.
    unsafe {
        buf.set_len(n_words);
        validity_buf.set_len(n_words);
    }
    (into_bits(buf, length), into_bits(validity_buf, length))
}

/// Reinterprets initialized words as a `length`-bit buffer. Little-endian only, like the rest of
/// the crate's bit packing.
fn into_bits(words: BufferMut<u64>, length: usize) -> BitBufferMut {
    let mut bytes = words.into_byte_buffer();
    bytes.truncate(length.div_ceil(8));
    BitBufferMut::from_buffer(bytes, 0, length)
}

/// Convert `offset`/`length` into `E` once, outside the per-run loop.
#[inline(always)]
fn trim_bounds<E: IntegerPType>(offset: usize, length: usize) -> (E, E) {
    let offset_e = E::from_usize(offset).unwrap_or_else(|| {
        vortex_panic!(
            "offset {} cannot be converted to {}",
            offset,
            std::any::type_name::<E>()
        )
    });
    let length_e = E::from_usize(length).unwrap_or_else(|| {
        vortex_panic!(
            "length {} cannot be converted to {}",
            length,
            std::any::type_name::<E>()
        )
    });
    (offset_e, length_e)
}

/// Trim a raw run end down by `offset` and clamp it to `length`.
#[inline(always)]
fn trim_end<E: IntegerPType>(end: E, offset: E, length: E) -> usize {
    if end < offset {
        vortex_panic!("run end {end} must be >= offset {offset}");
    }
    std::cmp::min(end - offset, length).as_()
}

/// Length of the run ending at `end`, given the position the previous run reached.
///
/// Saturating, so that run ends which are not strictly increasing yield empty runs rather than
/// running the output position backwards. The splat sinks are sized to exactly `length` bits, so
/// a position that could move backwards would let the total appended exceed the buffer and turn
/// a malformed array into a panic. Run ends are only checked for sortedness under
/// `debug_assertions`, so a release build does see them.
#[inline(always)]
fn run_length<E: IntegerPType>(end: E, offset: E, length: E, pos: usize) -> usize {
    trim_end(end, offset, length).saturating_sub(pos)
}

/// Number of runs whose value differs from the majority: exactly the runs the prefill kernel
/// has to patch after its memset.
fn minority_runs(values: &BitBuffer) -> usize {
    let true_count = values.true_count();
    std::cmp::min(true_count, values.len() - true_count)
}

/// Chooses between the two kernels from how many runs the prefill would have to patch.
///
/// The splat pays per output word; the prefill pays one memset over the whole output plus one
/// range fill per patched run, and a range fill is far dearer than a splat step — a partial byte
/// at each end, a `memset` call, and a data-dependent branch to decide whether to do it at all.
/// Measured, one patched run costs about what the splat spends on three output words. So the
/// prefill wins exactly when patches are rarer than that.
///
/// A second output buffer does not double the splat's side of that: [`BitSplatPair`] shares the
/// run length, the word-boundary test and the head mask, so the second buffer adds about half the
/// cost of the first. Hence `buffers + 1` against a doubled constant rather than `buffers`.
///
/// Re-derive by returning a constant from here and running `run_end_decode_bool_ablation` with
/// each kernel pinned. On that grid this picks the faster kernel at 25 of 26 points; the miss is
/// non-nullable 1024-element runs with 50/50 values, where it gives up 6%.
fn prefer_splat(patched_runs: usize, buffers: usize, length: usize) -> bool {
    patched_runs * 6 * WORD_BITS >= length * (buffers + 1)
}

/// Decodes run-end encoded booleans when all values are valid (non-nullable).
fn decode_bool_non_nullable<E: IntegerPType>(
    run_ends: &[E],
    offset: usize,
    values: &BitBuffer,
    nullability: Nullability,
    length: usize,
) -> BoolArray {
    let (offset_e, length_e) = trim_bounds::<E>(offset, length);

    if !prefer_splat(minority_runs(values), 1, length) {
        return prefill_bool_non_nullable(
            run_ends,
            offset_e,
            length_e,
            values,
            nullability,
            length,
        );
    }

    let decoded = splat_bits(length, |decoded| {
        let mut pos = 0usize;
        for (i, &end) in run_ends.iter().enumerate() {
            let run = run_length(end, offset_e, length_e, pos);
            pos += run;
            decoded.append_run(values.value(i), run);
        }
    });

    BoolArray::new(decoded.freeze(), nullability.into())
}

/// Decodes run-end encoded booleans when values may be null (nullable).
fn decode_bool_nullable<E: IntegerPType>(
    run_ends: &[E],
    offset: usize,
    values: &BitBuffer,
    validity_mask: &BitBuffer,
    length: usize,
) -> BoolArray {
    let (offset_e, length_e) = trim_bounds::<E>(offset, length);

    // Both buffers are prefilled and patched independently, so both sets of minority runs count.
    let patched_runs = minority_runs(values) + minority_runs(validity_mask);
    if !prefer_splat(patched_runs, 2, length) {
        return prefill_bool_nullable(run_ends, offset_e, length_e, values, validity_mask, length);
    }

    let (decoded, decoded_validity) = splat_bits_pair(length, |decoded| {
        let mut pos = 0usize;
        for (i, &end) in run_ends.iter().enumerate() {
            let run = run_length(end, offset_e, length_e, pos);
            pos += run;

            // The decoded bit is the value where valid, and false where null.
            let is_valid = validity_mask.value(i);
            decoded.append_run(is_valid && values.value(i), is_valid, run);
        }
    });

    BoolArray::new(decoded.freeze(), Validity::from(decoded_validity.freeze()))
}

/// Memsets the output with the majority value, then patches the runs that differ.
fn prefill_bool_non_nullable<E: IntegerPType>(
    run_ends: &[E],
    offset: E,
    length_e: E,
    values: &BitBuffer,
    nullability: Nullability,
    length: usize,
) -> BoolArray {
    let prefill = 2 * values.true_count() > values.len();
    let mut decoded = BitBufferMut::full(prefill, length);

    let mut pos = 0usize;
    for (i, &end) in run_ends.iter().enumerate() {
        let end = trim_end(end, offset, length_e);
        if end > pos && values.value(i) != prefill {
            // SAFETY: pos < end <= length == decoded.len()
            unsafe { decoded.fill_range_unchecked(pos, end, !prefill) };
        }
        pos = pos.max(end);
    }

    BoolArray::new(decoded.freeze(), nullability.into())
}

/// Memsets both output buffers with their majority value, then patches the runs that differ.
fn prefill_bool_nullable<E: IntegerPType>(
    run_ends: &[E],
    offset: E,
    length_e: E,
    values: &BitBuffer,
    validity_mask: &BitBuffer,
    length: usize,
) -> BoolArray {
    let prefill_decoded = 2 * values.true_count() > values.len();
    let prefill_valid = 2 * validity_mask.true_count() > validity_mask.len();

    let mut decoded = BitBufferMut::full(prefill_decoded, length);
    let mut decoded_validity = BitBufferMut::full(prefill_valid, length);

    let mut pos = 0usize;
    for (i, &end) in run_ends.iter().enumerate() {
        let end = trim_end(end, offset, length_e);
        if end > pos {
            let is_valid = validity_mask.value(i);
            // SAFETY: pos < end <= length == decoded.len() == decoded_validity.len()
            if is_valid != prefill_valid {
                unsafe { decoded_validity.fill_range_unchecked(pos, end, is_valid) };
            }
            // The decoded bit is the value where valid, and false where null.
            let want_decoded = is_valid && values.value(i);
            if want_decoded != prefill_decoded {
                unsafe { decoded.fill_range_unchecked(pos, end, want_decoded) };
            }
        }
        pos = pos.max(end);
    }

    BoolArray::new(decoded.freeze(), Validity::from(decoded_validity.freeze()))
}

#[cfg(test)]
mod tests {
    // Test data is sized well within `u32`.
    #![expect(clippy::cast_possible_truncation)]

    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::runend_decode_bools;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn decode_bools_alternating() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Alternating true/false: [T, T, F, F, F, T, T, T, T, T]
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false, true]));
        let decoded = runend_decode_bools(ends, values, 0, 10, &mut ctx)?;

        let expected = BoolArray::from(BitBuffer::from(vec![
            true, true, false, false, false, true, true, true, true, true,
        ]));
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn decode_bools_mostly_true() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Mostly true: [T, T, T, T, T, F, T, T, T, T]
        let ends = PrimitiveArray::from_iter([5u32, 6, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false, true]));
        let decoded = runend_decode_bools(ends, values, 0, 10, &mut ctx)?;

        let expected = BoolArray::from(BitBuffer::from(vec![
            true, true, true, true, true, false, true, true, true, true,
        ]));
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn decode_bools_mostly_false() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Mostly false: [F, F, F, F, F, T, F, F, F, F]
        let ends = PrimitiveArray::from_iter([5u32, 6, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![false, true, false]));
        let decoded = runend_decode_bools(ends, values, 0, 10, &mut ctx)?;

        let expected = BoolArray::from(BitBuffer::from(vec![
            false, false, false, false, false, true, false, false, false, false,
        ]));
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn decode_bools_single_run(#[case] value: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ends = PrimitiveArray::from_iter([10u32]);
        let values = BoolArray::from(BitBuffer::from(vec![value]));
        let decoded = runend_decode_bools(ends, values, 0, 10, &mut ctx)?;

        let expected = BoolArray::from(BitBuffer::from(vec![value; 10]));
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    /// Runs long enough to span whole accumulator words, at every alignment of the word grid.
    #[rstest]
    fn decode_bools_word_spanning(
        #[values(1, 7, 63, 64, 65, 127, 128, 129, 1000)] first_run: usize,
        #[values(1, 63, 64, 65, 200)] second_run: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let length = first_run + second_run;
        let ends = PrimitiveArray::from_iter([first_run as u32, length as u32]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false]));
        let decoded = runend_decode_bools(ends, values, 0, length, &mut ctx)?;

        let mut expected = vec![true; first_run];
        expected.extend(std::iter::repeat_n(false, second_run));
        let expected = BoolArray::from(BitBuffer::from(expected));
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    /// Run ends are only checked for sortedness under `debug_assertions`, so a release build can
    /// be handed ends that go backwards. They must decode to something of the right length rather
    /// than running the splat sink past the end of its buffer.
    #[test]
    fn decode_bools_non_monotonic_ends() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ends = PrimitiveArray::from_iter([100u32, 50, 100]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false, true]));
        let decoded =
            runend_decode_bools(ends, values, 0, 100, &mut ctx)?.execute::<BoolArray>(&mut ctx)?;

        assert_eq!(decoded.len(), 100);
        assert_eq!(decoded.into_bit_buffer().true_count(), 100);
        Ok(())
    }

    #[test]
    fn decode_bools_with_offset() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Test with offset: [T, T, F, F, F, T, T, T, T, T] -> slice [2..8] = [F, F, F, T, T, T]
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false, true]));
        let decoded = runend_decode_bools(ends, values, 2, 6, &mut ctx)?;

        let expected =
            BoolArray::from(BitBuffer::from(vec![false, false, false, true, true, true]));
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn decode_bools_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 3 runs: T (valid), F (null), T (valid) -> [T, T, null, null, null, T, T, T, T, T]
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = BoolArray::new(
            BitBuffer::from(vec![true, false, true]),
            Validity::from(BitBuffer::from(vec![true, false, true])),
        );
        let decoded = runend_decode_bools(ends, values, 0, 10, &mut ctx)?;

        // Expected: values=[T, T, F, F, F, T, T, T, T, T], validity=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
        let expected = BoolArray::new(
            BitBuffer::from(vec![
                true, true, false, false, false, true, true, true, true, true,
            ]),
            Validity::from(BitBuffer::from(vec![
                true, true, false, false, false, true, true, true, true, true,
            ])),
        );
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn decode_bools_nullable_long_runs() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 5 runs of length 2000 each, alternating validity.
        let ends = PrimitiveArray::from_iter([2000u32, 4000, 6000, 8000, 10000]);
        let values = BoolArray::new(
            BitBuffer::from(vec![true, false, true, false, true]),
            Validity::from(BitBuffer::from(vec![true, false, true, false, true])),
        );
        let decoded = runend_decode_bools(ends, values, 0, 10000, &mut ctx)?;

        let mut expected_values = Vec::with_capacity(10000);
        let mut expected_validity = Vec::with_capacity(10000);
        for run in 0..5 {
            let is_valid = run % 2 == 0;
            expected_values.extend(std::iter::repeat_n(is_valid, 2000));
            expected_validity.extend(std::iter::repeat_n(is_valid, 2000));
        }
        let expected = BoolArray::new(
            BitBuffer::from(expected_values),
            Validity::from(BitBuffer::from(expected_validity)),
        );
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    /// Differential test against a naive element-at-a-time decode over randomised inputs.
    #[rstest]
    fn decode_bools_matches_naive(
        #[values(1, 2, 5, 37, 64, 300)] avg_run_length: usize,
        #[values(false, true)] nullable: bool,
        #[values(0, 1, 63, 64, 100)] offset: usize,
    ) -> VortexResult<()> {
        use rand::RngExt;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut ctx = SESSION.create_execution_ctx();
        let mut rng = StdRng::seed_from_u64(0x5eed ^ (avg_run_length as u64) << 8 ^ offset as u64);
        let total = 4096usize;

        let mut ends = Vec::new();
        let mut values = Vec::new();
        let mut validity = Vec::new();
        let mut expected_values = Vec::new();
        let mut expected_validity = Vec::new();

        let mut pos = 0usize;
        while pos < total {
            let run = rng
                .random_range(1..=(2 * avg_run_length).max(1))
                .min(total - pos);
            pos += run;
            ends.push(pos as u32);
            let value = rng.random_bool(0.5);
            let is_valid = !nullable || rng.random_bool(0.7);
            values.push(value);
            validity.push(is_valid);
            expected_values.extend(std::iter::repeat_n(is_valid && value, run));
            expected_validity.extend(std::iter::repeat_n(is_valid, run));
        }

        // Drop the runs that end before `offset`, mirroring what slicing produces.
        let first_run = ends.iter().position(|&e| e as usize > offset).unwrap_or(0);
        let length = total - offset;
        let ends = PrimitiveArray::from_iter(ends[first_run..].iter().copied());
        let values_array = if nullable {
            BoolArray::new(
                BitBuffer::from(values[first_run..].to_vec()),
                Validity::from(BitBuffer::from(validity[first_run..].to_vec())),
            )
        } else {
            BoolArray::from(BitBuffer::from(values[first_run..].to_vec()))
        };

        let decoded = runend_decode_bools(ends, values_array, offset, length, &mut ctx)?;

        let expected = if nullable {
            BoolArray::new(
                BitBuffer::from(expected_values[offset..].to_vec()),
                Validity::from(BitBuffer::from(expected_validity[offset..].to_vec())),
            )
        } else {
            BoolArray::from(BitBuffer::from(expected_values[offset..].to_vec()))
        };
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }
}
