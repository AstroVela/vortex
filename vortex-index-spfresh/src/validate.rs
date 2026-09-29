// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming checks of the pinned little-endian native layout before C++ allocation.
//! These checks catch shape/format errors; upstream native files remain trusted input.

use std::collections::BTreeSet;
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

use crate::SpFreshBundle;
use crate::bundle::ROW_MAGIC;

fn bytes<const N: usize>(reader: &mut impl Read) -> VortexResult<[u8; N]> {
    let mut bytes = [0u8; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn int(reader: &mut impl Read) -> VortexResult<i32> {
    Ok(i32::from_le_bytes(bytes(reader)?))
}
fn uint(reader: &mut impl Read) -> VortexResult<u64> {
    Ok(u64::from_le_bytes(bytes(reader)?))
}

fn reader(path: &Path) -> VortexResult<(BufReader<File>, u64)> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    Ok((BufReader::new(file), size))
}

pub(crate) fn mapping(path: &Path, rows: u32) -> VortexResult<Vec<RowAddress>> {
    let (mut input, size) = reader(path)?;
    if size != 16 + 16 * u64::from(rows)
        || &bytes::<8>(&mut input)? != ROW_MAGIC
        || uint(&mut input)? != u64::from(rows)
    {
        vortex_bail!("Invalid SPFresh row mapping header or length");
    }
    (0..rows)
        .map(|_| {
            Ok(RowAddress {
                file_id: uint(&mut input)?,
                row_offset: uint(&mut input)?,
            })
        })
        .collect()
}

pub(crate) fn native_files(root: &Path, bundle: SpFreshBundle) -> VortexResult<()> {
    let (mut vectors, size) = reader(&root.join("vectors.bin"))?;
    let heads = int(&mut vectors)?;
    if heads <= 0
        || heads as u32 > bundle.rows
        || int(&mut vectors)? != bundle.dimension as i32
        || size != 8 + heads as u64 * u64::from(bundle.dimension) * 4
    {
        vortex_bail!("Invalid SPFresh head vector shape");
    }
    for _ in 0..heads as u64 * u64::from(bundle.dimension) {
        if !f32::from_le_bytes(bytes(&mut vectors)?).is_finite() {
            vortex_bail!("Non-finite SPFresh vector");
        }
    }
    let (mut ids, size) = reader(&root.join("head_ids.bin"))?;
    if size != heads as u64 * 8 {
        vortex_bail!("Invalid SPFresh head ID length");
    }
    let mut seen = vec![false; bundle.rows as usize];
    for _ in 0..heads {
        let id = uint(&mut ids)?;
        let offset =
            usize::try_from(id).map_err(|err| vortex_err!("Invalid SPFresh head ID: {err}"))?;
        if id >= u64::from(bundle.rows) || seen[offset] {
            vortex_bail!("Invalid or duplicate SPFresh head ID");
        }
        seen[offset] = true;
    }
    let (mut deletes, size) = reader(&root.join("deletes.bin"))?;
    if size != 12 + heads as u64
        || int(&mut deletes)? != 0
        || int(&mut deletes)? != heads
        || int(&mut deletes)? != 1
    {
        vortex_bail!("SPFresh deleted heads or invalid delete bitmap");
    }
    for _ in 0..heads {
        if !matches!(bytes::<1>(&mut deletes)?, [0] | [255]) {
            vortex_bail!("SPFresh deleted heads are unsupported");
        }
    }
    let duplicate_nodes = validate_tree(root, heads)?;
    let (mut graph, size) = reader(&root.join("graph.bin"))?;
    let graph_rows = int(&mut graph)?;
    let neighbors = int(&mut graph)?;
    if graph_rows != heads
        || !(2..=4096).contains(&neighbors)
        || size != 8 + heads as u64 * neighbors as u64 * 4
    {
        vortex_bail!("Invalid SPFresh graph shape");
    }
    for _ in 0..heads as u64 * neighbors as u64 {
        let id = int(&mut graph)?;
        if id >= heads || (id < -1 && !duplicate_nodes.contains(&(-2 - id))) {
            vortex_bail!("Invalid SPFresh graph ID");
        }
    }
    validate_postings(root, bundle, heads, &mut seen)?;
    if seen.iter().any(|present| !present) {
        vortex_bail!("SPFresh native files do not cover all mapped IDs");
    }
    Ok(())
}

fn validate_tree(root: &Path, heads: i32) -> VortexResult<BTreeSet<i32>> {
    let (mut tree, size) = reader(&root.join("tree.bin"))?;
    let trees = int(&mut tree)?;
    if !(1..=64).contains(&trees) {
        vortex_bail!("Invalid SPFresh tree count");
    }
    let roots = (0..trees)
        .map(|_| int(&mut tree))
        .collect::<VortexResult<Vec<_>>>()?;
    let nodes = int(&mut tree)?;
    if nodes <= 0
        || size != 8 + trees as u64 * 4 + nodes as u64 * 12
        || roots.iter().any(|root| *root < 0 || *root >= nodes)
        || roots.windows(2).any(|pair| pair[0] >= pair[1])
    {
        vortex_bail!("Invalid SPFresh tree length or roots");
    }
    let mut sentinels = BTreeSet::new();
    let mut children = Vec::new();
    let mut duplicate_nodes = BTreeSet::new();
    for node in 0..nodes {
        let center = int(&mut tree)?;
        let start = int(&mut tree)?;
        let end = int(&mut tree)?;
        // Upstream uses the head count as the non-leaf root's sentinel center.
        if center < -1
            || center > heads
            || (center == heads && (!roots.contains(&node) || start < 0))
            || start == i32::MIN
            || start.abs() > nodes
            || end >= nodes
        {
            vortex_bail!("Invalid SPFresh tree node {node}: {center}, {start}, {end}");
        }
        if center == -1 {
            if roots.contains(&node) || start != -1 || end != -1 {
                vortex_bail!("Invalid SPFresh tree sentinel");
            }
            sentinels.insert(node);
        }
        if start != -1 || end != -1 {
            let begin = start.abs();
            if begin <= node || end < begin {
                vortex_bail!("Invalid SPFresh tree child range or cycle");
            }
            children.push((begin, end));
            if start < 0 {
                duplicate_nodes.insert(node);
            }
        }
    }
    for (begin, end) in children {
        if sentinels.range(begin..end).next().is_some()
            || roots.iter().any(|root| (begin..end).contains(root))
        {
            vortex_bail!("SPFresh tree children reference a sentinel or root");
        }
    }
    Ok(duplicate_nodes)
}

fn validate_postings(
    root: &Path,
    bundle: SpFreshBundle,
    heads: i32,
    seen: &mut [bool],
) -> VortexResult<()> {
    let path = root.join("postings.bin");
    let (mut header, size) = reader(&path)?;
    if int(&mut header)? != heads
        || int(&mut header)? != bundle.rows as i32
        || int(&mut header)? != bundle.dimension as i32
    {
        vortex_bail!("SPFresh posting shape mismatch");
    }
    let header_pages = int(&mut header)?;
    if header_pages <= 0
        || (header_pages as u64 * 4096) < 16 + heads as u64 * 12
        || header_pages as u64 * 4096 > size
    {
        vortex_bail!("Invalid SPFresh posting header pages");
    }
    let (mut data, _) = reader(&path)?;
    let stride = 4 + u64::from(bundle.dimension) * 4;
    for _ in 0..heads {
        let page = int(&mut header)?;
        let offset = u16::from_le_bytes(bytes(&mut header)?);
        let count = int(&mut header)?;
        let pages = u16::from_le_bytes(bytes(&mut header)?);
        if page < 0
            || header_pages.checked_add(page).is_none()
            || count < 0
            || offset >= 4096
            || u32::from(pages) > bundle.posting_page_limit
            || u64::from(offset) + count as u64 * stride > u64::from(pages) * 4096
        {
            vortex_bail!("Invalid SPFresh posting layout");
        }
        let start = (header_pages as u64 + page as u64) * 4096;
        if start + u64::from(pages) * 4096 > size {
            vortex_bail!("Truncated SPFresh postings");
        }
        data.seek(SeekFrom::Start(start + u64::from(offset)))?;
        for _ in 0..count {
            let id = int(&mut data)?;
            if id < 0 || id as u32 >= bundle.rows {
                vortex_bail!("Invalid SPFresh posting ID");
            }
            seen[id as usize] = true;
            for _ in 0..bundle.dimension {
                if !f32::from_le_bytes(bytes(&mut data)?).is_finite() {
                    vortex_bail!("Non-finite SPFresh posting vector");
                }
            }
        }
    }
    Ok(())
}
