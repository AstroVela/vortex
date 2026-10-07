// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::Path;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_index::RowAddress;

use crate::HnswBundle;
use crate::bundle::ROW_MAGIC;

fn word<const N: usize>(input: &mut impl Read) -> VortexResult<[u8; N]> {
    let mut bytes = [0u8; N];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn u64_value(input: &mut impl Read) -> VortexResult<u64> {
    Ok(u64::from_le_bytes(word(input)?))
}

fn u32_value(input: &mut impl Read) -> VortexResult<u32> {
    Ok(u32::from_le_bytes(word(input)?))
}

fn fixed<const N: usize>(bytes: &[u8]) -> VortexResult<[u8; N]> {
    bytes
        .try_into()
        .map_err(|err| vortex_err!("Invalid hnswlib word: {err}"))
}

pub(crate) fn mapping(path: &Path, rows: u32) -> VortexResult<Vec<RowAddress>> {
    let file = File::open(path)?;
    if file.metadata()?.len() != 16 + u64::from(rows) * 16 {
        vortex_bail!("hnswlib row mapping length mismatch");
    }
    let mut input = BufReader::new(file);
    if &word::<8>(&mut input)? != ROW_MAGIC || u64_value(&mut input)? != u64::from(rows) {
        vortex_bail!("hnswlib row mapping header mismatch");
    }
    (0..rows)
        .map(|_| {
            Ok(RowAddress {
                file_id: u64_value(&mut input)?,
                row_offset: u64_value(&mut input)?,
            })
        })
        .collect()
}

fn links(bytes: &[u8], capacity: u32, rows: u32) -> VortexResult<Vec<u32>> {
    let header = u32::from_le_bytes(fixed(&bytes[..4])?);
    // Upper bits are deletion/reserved flags, never permitted in static bundles.
    if header > capacity {
        vortex_bail!("hnswlib invalid link count or deletion flags");
    }
    bytes[4..4 + header as usize * 4]
        .chunks_exact(4)
        .map(|bytes| {
            let id = u32::from_le_bytes(fixed(bytes)?);
            if id >= rows {
                vortex_bail!("hnswlib graph neighbor is outside the index");
            }
            Ok(id)
        })
        .collect()
}

/// Check the pinned little-endian 64-bit layout before hnswlib allocates or follows links.
pub(crate) fn native_file(path: &Path, bundle: HnswBundle) -> VortexResult<()> {
    bundle.validate()?;
    let file = File::open(path)?;
    let length = file.metadata()?.len();
    let mut input = BufReader::with_capacity(64 * 1024, file);
    let level_zero = u64_value(&mut input)?;
    let capacity = u64_value(&mut input)?;
    let count = u64_value(&mut input)?;
    let stride = u64_value(&mut input)?;
    let label_offset = u64_value(&mut input)?;
    let data_offset = u64_value(&mut input)?;
    let max_level = u32_value(&mut input)?;
    let entry = u32_value(&mut input)?;
    let max_m = u64_value(&mut input)?;
    let max_m_zero = u64_value(&mut input)?;
    let m = u64_value(&mut input)?;
    let multiplier = f64::from_le_bytes(word(&mut input)?);
    let construction = u64_value(&mut input)?;
    let expected_data = u64::from(bundle.m) * 8 + 4;
    let expected_label = expected_data + u64::from(bundle.dimension) * 4;
    if level_zero != 0
        || capacity != u64::from(bundle.rows)
        || count != capacity
        || stride != expected_label + 8
        || data_offset != expected_data
        || label_offset != expected_label
        || m != u64::from(bundle.m)
        || max_m != m
        || max_m_zero != 2 * m
        || construction != u64::from(bundle.ef_construction)
        || max_level > 64
        || entry >= bundle.rows
        || !multiplier.is_finite()
        || (multiplier - 1.0 / f64::from(bundle.m).ln()).abs() > 1e-12
        || length < 96 + count * (stride + 4)
    {
        vortex_bail!("hnswlib native header does not match the bounded static descriptor");
    }
    let mut labels = vec![false; bundle.rows as usize];
    let mut record = vec![0u8; usize::try_from(stride)?];
    let data_offset = usize::try_from(data_offset)?;
    let label_offset = usize::try_from(label_offset)?;
    for _ in 0..bundle.rows {
        input.read_exact(&mut record)?;
        links(&record[..data_offset], bundle.m * 2, bundle.rows)?;
        for bytes in record[data_offset..label_offset].chunks_exact(4) {
            if !f32::from_le_bytes(fixed(bytes)?).is_finite() {
                vortex_bail!("hnswlib native vector contains non-finite components");
            }
        }
        let label = u64::from_le_bytes(fixed(&record[label_offset..])?);
        let position = usize::try_from(label)?;
        if label >= count || labels[position] {
            vortex_bail!("hnswlib native labels are not a dense unique permutation");
        }
        labels[position] = true;
    }
    let upper_start = input.stream_position()?;
    let upper_stride = bundle.m as usize * 4 + 4;
    let mut level_counts = vec![0u8; bundle.rows as usize];
    let mut upper = vec![0u8; upper_stride];
    let mut consumed = upper_start;
    for levels in &mut level_counts {
        let size = u32_value(&mut input)? as usize;
        consumed += 4 + size as u64;
        if !size.is_multiple_of(upper_stride)
            || size / upper_stride > max_level as usize
            || consumed > length
        {
            vortex_bail!("hnswlib upper-level size is invalid or truncated");
        }
        *levels = u8::try_from(size / upper_stride)?;
        for _ in 0..*levels {
            input.read_exact(&mut upper)?;
            links(&upper, bundle.m, bundle.rows)?;
        }
    }
    let max_level = u8::try_from(max_level)?;
    if consumed != length
        || level_counts[entry as usize] != max_level
        || level_counts.iter().copied().max() != Some(max_level)
    {
        vortex_bail!("hnswlib upper-level entry point or trailing bytes are invalid");
    }
    // Follow only edges whose target actually owns that level; native traversal
    // otherwise dereferences an absent upper-level allocation.
    input.seek(SeekFrom::Start(upper_start))?;
    for _ in 0..bundle.rows {
        let levels = u32_value(&mut input)? as usize / upper_stride;
        for level in 1..=levels {
            input.read_exact(&mut upper)?;
            for neighbor in links(&upper, bundle.m, bundle.rows)? {
                if usize::from(level_counts[neighbor as usize]) < level {
                    vortex_bail!("hnswlib upper-level edge targets a missing level");
                }
            }
        }
    }
    Ok(())
}
