// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lexicographic compare of a wide integer column stored as 64-bit limbs, evaluated directly on
//! the limbs' compressed form.
//!
//! This is the compare kernel a limb-split wide-decimal encoding needs, written against
//! [`BitPackedArray`] limbs so it can be measured against the alternative of canonicalizing to a
//! contiguous `i128`/`i256` buffer first. See `benches/limb_compare.rs`.
//!
//! # Representation
//!
//! A wide signed value is `limbs[0]` (signed, most significant) followed by the remaining limbs in
//! descending significance (unsigned). Under that layout the value order is *exactly* the
//! lexicographic order of the limbs, which is what makes a limb-by-limb evaluation possible:
//!
//! ```text
//! v < c  <=>  v[0] < c[0]  or  (v[0] == c[0] and (v[1..] < c[1..] lexicographically))
//! ```
//!
//! # Why this can beat canonicalizing
//!
//! Two properties, both of which come from the limbs being *separate* arrays:
//!
//! 1. **Limb elision.** A limb that compressed to a [`ConstantArray`] is answered by one scalar
//!    comparison; its bytes are never read. After compression the high limb of a wide decimal
//!    column is usually exactly that.
//! 2. **Tie-driven early exit.** Once no row is still tied on the limbs examined so far, the
//!    remaining limbs cannot change any row's answer, so they are never decoded. For a predicate
//!    against a constant with high-entropy leading limbs this is the common case.
//!
//! Both are unavailable to a contiguous layout, where every limb of an element shares its cache
//! lines and is therefore read unconditionally.
//!
//! [`BitPackedArray`]: crate::BitPackedArray
//! [`ConstantArray`]: vortex_array::arrays::ConstantArray

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_buffer::pack_bools_into_words;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use super::stream_predicate::splice_patches;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::unpack_iter::BitPacked as BitPackedIter;

/// A per-row bit mask, kept symbolic while it is uniform so that a limb which compressed to a
/// constant costs no word traffic at all.
#[derive(Debug)]
enum LimbMask {
    AllTrue,
    AllFalse,
    Words(BufferMut<u64>),
}

impl LimbMask {
    fn from_bool(value: bool) -> Self {
        if value { Self::AllTrue } else { Self::AllFalse }
    }
}

/// The outcome of [`limbs_lt_constant`], including how much of the column it had to decode.
#[derive(Debug)]
pub struct LimbCompare {
    /// One bit per row: the value at that row is less than the constant.
    pub bits: BitBuffer,
    /// How many limbs were decoded. Limbs answered from a constant, and limbs skipped because no
    /// row was still tied, are not counted.
    pub limbs_decoded: usize,
}

/// Evaluate `value < constant` over a wide integer column held as `limbs`, most significant first.
///
/// `limbs[0]` must be a signed 64-bit column and the rest unsigned 64-bit columns, all of length
/// `len`. `constant` gives the constant's limbs in the same order as raw bit patterns; the leading
/// one is reinterpreted as `i64` to match `limbs[0]`.
///
/// Each limb may be in any encoding. [`BitPacked`] limbs are streamed a FastLanes block at a time
/// and never fully materialized; a constant limb is folded into a scalar comparison; anything else
/// is canonicalized to a [`PrimitiveArray`] first.
///
/// # Errors
///
/// Returns an error if the limb list is empty, if a limb's length disagrees with `len`, if the
/// limb and constant counts differ, or if a limb is not a 64-bit integer column of the expected
/// signedness.
pub fn limbs_lt_constant(
    limbs: &[ArrayRef],
    constant: &[u64],
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<LimbCompare> {
    vortex_ensure!(!limbs.is_empty(), "wide value must have at least one limb");
    vortex_ensure!(
        limbs.len() == constant.len(),
        "got {} limbs but {} constant limbs",
        limbs.len(),
        constant.len(),
    );

    let words = len.div_ceil(u64::BITS as usize);

    // `lt` accumulates rows already decided less-than. `eq` tracks rows still tied on every limb
    // examined so far, so it starts all-true: before looking at any limb, every row is tied.
    let mut lt = LimbMask::AllFalse;
    let mut eq = LimbMask::AllTrue;
    let mut limbs_decoded = 0;

    for (idx, (limb, &c)) in limbs.iter().zip(constant).enumerate() {
        vortex_ensure!(
            limb.len() == len,
            "limb {idx} has length {} but the column length is {len}",
            limb.len(),
        );
        let is_last = idx + 1 == limbs.len();
        // The trailing limb only narrows `lt`; nothing consumes a tie after it.
        let need_eq = !is_last;

        let (limb_lt, limb_eq, decoded) = limb_masks(limb, c, idx == 0, words, need_eq, ctx)?;
        limbs_decoded += usize::from(decoded);

        // lt |= eq & limb_lt
        let contribution = and(&eq, &limb_lt, words);
        lt = or_assign(lt, &contribution, words);

        if is_last {
            break;
        }

        // eq &= limb_eq
        eq = and(&eq, &limb_eq.unwrap_or(LimbMask::AllTrue), words);

        // No row is still tied, so no later limb can change any answer: stop before decoding them.
        if is_all_false(&eq, len) {
            break;
        }
    }

    Ok(LimbCompare {
        bits: into_bits(lt, len, words),
        limbs_decoded,
    })
}

/// Produce the `< c` and (optionally) `== c` masks for one limb, reporting whether the limb's
/// values had to be decoded.
fn limb_masks(
    limb: &ArrayRef,
    c: u64,
    is_msp: bool,
    words: usize,
    need_eq: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(LimbMask, Option<LimbMask>, bool)> {
    let expected = if is_msp { PType::I64 } else { PType::U64 };
    vortex_ensure!(
        limb.dtype().is_int(),
        "limb must be an integer column, got {}",
        limb.dtype(),
    );
    let ptype = limb.dtype().as_ptype();
    vortex_ensure!(
        ptype == expected,
        "limb has ptype {ptype} but {expected} was expected ({} limb)",
        if is_msp {
            "leading signed"
        } else {
            "trailing unsigned"
        },
    );

    // A constant limb is answered without touching any values.
    if let Some(scalar) = limb.as_constant() {
        let Some(prim) = scalar.as_primitive_opt() else {
            vortex_bail!("constant limb must hold a primitive scalar");
        };
        let (is_lt, is_eq) = if is_msp {
            let v = prim
                .typed_value::<i64>()
                .ok_or_else(|| vortex_error::vortex_err!("limb constant must be non-null"))?;
            (v.is_lt(c as i64), v.is_eq(c as i64))
        } else {
            let v = prim
                .typed_value::<u64>()
                .ok_or_else(|| vortex_error::vortex_err!("limb constant must be non-null"))?;
            (v.is_lt(c), v.is_eq(c))
        };
        return Ok((
            LimbMask::from_bool(is_lt),
            need_eq.then(|| LimbMask::from_bool(is_eq)),
            false,
        ));
    }

    // A bit-packed limb streams through the FastLanes block unpacker, so the limb is never
    // materialized in full and both masks come out of a single decode.
    if limb.as_opt::<BitPacked>().is_some() {
        let (lt, eq) = if is_msp {
            packed_masks::<i64>(limb, c as i64, words, need_eq, ctx)?
        } else {
            packed_masks::<u64>(limb, c, words, need_eq, ctx)?
        };
        return Ok((lt, eq, true));
    }

    // Any other encoding: canonicalize, then compare the slice.
    let prim = limb.clone().execute::<PrimitiveArray>(ctx)?;
    let (lt, eq) = if is_msp {
        slice_masks::<i64>(prim.as_slice::<i64>(), c as i64, words, need_eq)
    } else {
        slice_masks::<u64>(prim.as_slice::<u64>(), c, words, need_eq)
    };
    Ok((lt, eq, true))
}

/// Stream a [`BitPacked`] limb one FastLanes block at a time, folding both masks out of the same
/// unpacked block.
fn packed_masks<T>(
    limb: &ArrayRef,
    c: T,
    words: usize,
    need_eq: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(LimbMask, Option<LimbMask>)>
where
    T: NativePType + BitPackedIter + Copy,
{
    let array = limb.as_::<BitPacked>();
    let mut lt: BufferMut<u64> = BufferMut::zeroed(words);
    let mut eq: BufferMut<u64> = BufferMut::zeroed(if need_eq { words } else { 0 });

    let mut chunks = array.unpacked_chunks::<T>()?;

    // Patches hold the values that did not fit the bit width, so they must be spliced into each
    // block before the masks are folded, exactly as `stream_predicate` does.
    let patches = match array.patches() {
        Some(p) => Some((
            p.indices().clone().execute::<PrimitiveArray>(ctx)?,
            p.values().clone().execute::<PrimitiveArray>(ctx)?,
            p.offset(),
        )),
        None => None,
    };

    {
        let lt = lt.as_mut_slice();
        let eq = eq.as_mut_slice();
        let mut fold = |block: &mut [T], start: usize| {
            pack_bools_into_words(lt, start, block.len(), |i| block[i].is_lt(c));
            if need_eq {
                pack_bools_into_words(eq, start, block.len(), |i| block[i].is_eq(c));
            }
        };

        match &patches {
            Some((indices, values, offset)) => {
                let values = values.as_slice::<T>();
                let mut cursor = 0usize;
                vortex_array::match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
                    let indices = indices.as_slice::<I>();
                    chunks.for_each_unpacked_chunk(|block, range| {
                        cursor = splice_patches::<T, I>(
                            block,
                            range.start,
                            cursor,
                            indices,
                            values,
                            *offset,
                        );
                        fold(block, range.start);
                    });
                });
            }
            None => chunks.for_each_unpacked_chunk(|block, range| fold(block, range.start)),
        }
    }

    Ok((LimbMask::Words(lt), need_eq.then_some(LimbMask::Words(eq))))
}

/// Fold both masks over an already-materialized limb.
fn slice_masks<T: NativePType>(
    values: &[T],
    c: T,
    words: usize,
    need_eq: bool,
) -> (LimbMask, Option<LimbMask>) {
    let mut lt: BufferMut<u64> = BufferMut::zeroed(words);
    pack_bools_into_words(lt.as_mut_slice(), 0, values.len(), |i| values[i].is_lt(c));
    let eq = need_eq.then(|| {
        let mut eq: BufferMut<u64> = BufferMut::zeroed(words);
        pack_bools_into_words(eq.as_mut_slice(), 0, values.len(), |i| values[i].is_eq(c));
        LimbMask::Words(eq)
    });
    (LimbMask::Words(lt), eq)
}

fn and(lhs: &LimbMask, rhs: &LimbMask, words: usize) -> LimbMask {
    match (lhs, rhs) {
        (LimbMask::AllFalse, _) | (_, LimbMask::AllFalse) => LimbMask::AllFalse,
        (LimbMask::AllTrue, LimbMask::AllTrue) => LimbMask::AllTrue,
        (LimbMask::AllTrue, LimbMask::Words(w)) | (LimbMask::Words(w), LimbMask::AllTrue) => {
            LimbMask::Words(w.clone())
        }
        (LimbMask::Words(a), LimbMask::Words(b)) => {
            let mut out: BufferMut<u64> = BufferMut::zeroed(words);
            let (a, b, out_slice) = (a.as_slice(), b.as_slice(), out.as_mut_slice());
            for i in 0..words {
                out_slice[i] = a[i] & b[i];
            }
            LimbMask::Words(out)
        }
    }
}

fn or_assign(lhs: LimbMask, rhs: &LimbMask, words: usize) -> LimbMask {
    match (lhs, rhs) {
        (LimbMask::AllTrue, _) | (_, LimbMask::AllTrue) => LimbMask::AllTrue,
        (lhs, LimbMask::AllFalse) => lhs,
        (LimbMask::AllFalse, LimbMask::Words(w)) => LimbMask::Words(w.clone()),
        (LimbMask::Words(mut a), LimbMask::Words(b)) => {
            let b = b.as_slice();
            let a_slice = a.as_mut_slice();
            for i in 0..words {
                a_slice[i] |= b[i];
            }
            LimbMask::Words(a)
        }
    }
}

/// True when no row's bit is set. Bits at or beyond `len` are padding and must not count, which is
/// what makes the all-true seed for `eq` safe to test.
fn is_all_false(mask: &LimbMask, len: usize) -> bool {
    match mask {
        LimbMask::AllFalse => true,
        LimbMask::AllTrue => len == 0,
        LimbMask::Words(w) => {
            let bits = u64::BITS as usize;
            let full = len / bits;
            let w = w.as_slice();
            if w[..full].iter().any(|&word| word != 0) {
                return false;
            }
            let rem = len % bits;
            rem == 0 || w[full] & ((1u64 << rem) - 1) == 0
        }
    }
}

fn into_bits(mask: LimbMask, len: usize, words: usize) -> BitBuffer {
    match mask {
        LimbMask::AllTrue => BitBuffer::full(true, len),
        LimbMask::AllFalse => BitBuffer::full(false, len),
        LimbMask::Words(w) => {
            debug_assert_eq!(w.len(), words);
            BitBufferMut::from_buffer(w.into_byte_buffer(), 0, len).freeze()
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;

    use super::*;
    use crate::bitpack_compress::bitpack_encode;

    /// Split a two's-complement `i128` into the `[signed high, unsigned low]` limb pair this
    /// kernel expects.
    fn split(v: i128) -> (i64, u64) {
        ((v >> 64) as i64, v as u64)
    }

    fn packed<T: NativePType>(values: &[T], bit_width: u8) -> VortexResult<ArrayRef> {
        let mut ctx = array_session().create_execution_ctx();
        let prim = PrimitiveArray::new(
            values.iter().copied().collect::<BufferMut<T>>().freeze(),
            Validity::NonNullable,
        );
        Ok(bitpack_encode(&prim, bit_width, None, &mut ctx)?.into_array())
    }

    /// An uncompressed limb. This is what the compressor leaves behind for a limb with full
    /// 64-bit entropy, since `bitpack_encode` requires a bit width below 64.
    fn raw<T: NativePType>(values: &[T]) -> ArrayRef {
        PrimitiveArray::new(
            values.iter().copied().collect::<BufferMut<T>>().freeze(),
            Validity::NonNullable,
        )
        .into_array()
    }

    /// Reference answer, computed on the reconstructed `i128` values.
    fn reference(values: &[i128], c: i128) -> Vec<bool> {
        values.iter().map(|v| *v < c).collect()
    }

    fn run(limbs: &[ArrayRef], c: i128, len: usize) -> VortexResult<(Vec<bool>, usize)> {
        let (c_hi, c_lo) = split(c);
        let mut ctx = array_session().create_execution_ctx();
        let out = limbs_lt_constant(limbs, &[c_hi as u64, c_lo], len, &mut ctx)?;
        Ok((
            (0..len).map(|i| out.bits.value(i)).collect(),
            out.limbs_decoded,
        ))
    }

    /// Build the `[high, low]` limb pair for a set of `i128` values.
    fn limb_pair(values: &[i128], hi_width: u8, lo_width: u8) -> VortexResult<Vec<ArrayRef>> {
        let his: Vec<i64> = values.iter().map(|v| split(*v).0).collect();
        let los: Vec<u64> = values.iter().map(|v| split(*v).1).collect();
        Ok(vec![packed(&his, hi_width)?, packed(&los, lo_width)?])
    }

    #[rstest]
    // Every high limb is 0 and the constant's high limb is 0 too, so the tie survives and the low
    // limb must decide: 2 limbs decoded.
    #[case(&[1i128, 2, 3, 1000], 500, 2)]
    // The constant's high limb is 1, above every row's high limb of 0, so every row is decided by
    // the high limb alone and the low limb is never touched.
    #[case(&[1i128, 2, 3, 1000], 1i128 << 64, 1)]
    fn elides_untied_limbs(
        #[case] values: &[i128],
        #[case] c: i128,
        #[case] expect_decoded: usize,
    ) -> VortexResult<()> {
        let limbs = limb_pair(values, 8, 16)?;
        let (got, decoded) = run(&limbs, c, values.len())?;
        assert_eq!(got, reference(values, c));
        assert_eq!(
            decoded, expect_decoded,
            "unexpected number of limbs decoded"
        );
        Ok(())
    }

    #[test]
    fn constant_high_limb_is_free() -> VortexResult<()> {
        // The shape compression actually produces for a money column: the high limb folds to a
        // constant, so only the low limb is ever decoded.
        let values: Vec<i128> = (0..2048).map(|i| i as i128 * 7).collect();
        let los: Vec<u64> = values.iter().map(|v| split(*v).1).collect();
        let limbs = vec![
            ConstantArray::new(Scalar::from(0i64), values.len()).into_array(),
            packed(&los, 16)?,
        ];
        let c = 7000i128;
        let (got, decoded) = run(&limbs, c, values.len())?;
        assert_eq!(got, reference(&values, c));
        assert_eq!(decoded, 1, "a constant limb must not be decoded");
        Ok(())
    }

    #[test]
    fn negative_high_limb_orders_correctly() -> VortexResult<()> {
        // The leading limb is signed, so negative values must order below every non-negative one.
        //
        // `bitpack_encode` rejects negative integers, so a mixed-sign leading limb is never a bare
        // `BitPackedArray` in practice - the compressor puts FoR or ZigZag in front of it. That
        // routes through the generic canonicalize-then-compare arm, which is what this exercises.
        let values: Vec<i128> = vec![-1, -(1i128 << 70), 0, 5, 1i128 << 70, -3];
        let his: Vec<i64> = values.iter().map(|v| split(*v).0).collect();
        let los: Vec<u64> = values.iter().map(|v| split(*v).1).collect();
        let limbs = vec![raw(&his), raw(&los)];
        for c in [-5i128, 0, 1, 1i128 << 70, -(1i128 << 70)] {
            let (got, _) = run(&limbs, c, values.len())?;
            assert_eq!(got, reference(&values, c), "wrong answer for constant {c}");
        }
        Ok(())
    }

    #[test]
    fn patched_limb_uses_patch_values() -> VortexResult<()> {
        // One value overflows the low limb's bit width and lands in patches. The mask must reflect
        // the real value, not the placeholder left in the packed lane.
        let mut values: Vec<i128> = (0..1200).map(i128::from).collect();
        values[700] = 1i128 << 40;
        let limbs = limb_pair(&values, 8, 11)?;
        for c in [500i128, 1i128 << 40, (1i128 << 40) + 1] {
            let (got, _) = run(&limbs, c, values.len())?;
            assert_eq!(got, reference(&values, c), "wrong answer for constant {c}");
        }
        Ok(())
    }

    #[test]
    fn four_limbs_stop_at_the_first_deciding_limb() -> VortexResult<()> {
        // A 256-bit value as four limbs. The constant's leading limb is above every row's, so all
        // three trailing limbs are skipped.
        let len = 512;
        let msp: Vec<i64> = (0..len as i64).map(|i| i % 64).collect();
        let mid: Vec<u64> = (0..len as u64).collect();
        let limbs = vec![
            packed(&msp, 8)?,
            packed(&mid, 16)?,
            packed(&mid, 16)?,
            packed(&mid, 16)?,
        ];
        let mut ctx = array_session().create_execution_ctx();
        let out = limbs_lt_constant(&limbs, &[100u64, 0, 0, 0], len, &mut ctx)?;
        assert!((0..len).all(|i| out.bits.value(i)));
        assert_eq!(out.limbs_decoded, 1);
        Ok(())
    }

    #[test]
    fn rejects_mismatched_limb_types() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // The leading limb must be signed; an unsigned one is rejected.
        let limbs = vec![packed(&[1u64, 2], 8)?, packed(&[1u64, 2], 8)?];
        assert!(limbs_lt_constant(&limbs, &[0, 0], 2, &mut ctx).is_err());
        Ok(())
    }
}
