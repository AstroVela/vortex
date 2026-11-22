// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Versioned compact sidecar. Original f32 components stay in the native files.

use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::Path;

use hashbrown::HashMap;
use hashbrown::hash_map::Entry;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::DistanceEstimate;
use super::RaBitQ;
use super::RaBitQQuery;
use super::distance::ExactQuery;
use crate::SpFreshBundle;

#[cfg(feature = "native")]
pub(crate) const SIDECAR: &str = "spfresh/rabitq.bin";
const MAGIC: &[u8; 8] = b"VXSRBQ01";
const HEAD: u64 = 1 << 63;
const READ_CHUNK_BYTES: usize = 16 * 1024;
const MAX_READ_CACHE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct QuantizedIndex {
    pub(super) quantizer: RaBitQ,
    pub(super) codes: QuantizedCodes,
    pub(super) head_ids: Vec<u32>,
    pub(super) postings: Vec<Vec<u32>>,
    locations: Vec<u64>,
    vectors: File,
    original_postings: File,
}

#[derive(Debug)]
pub(super) struct QuantizedCodes {
    data: Vec<u8>,
    stride: usize,
    positions: Vec<u32>,
}

impl QuantizedCodes {
    pub(super) fn len(&self) -> usize {
        self.positions.len()
    }

    #[inline]
    fn corrections(record: &[u8]) -> (f64, f64, f64) {
        let mut word = [0; 8];
        let mut read = |offset| {
            word.copy_from_slice(&record[offset..offset + 8]);
            f64::from_le_bytes(word)
        };
        (read(0), read(8), read(16))
    }

    #[inline]
    pub(super) fn estimate(
        &self,
        query: &RaBitQQuery,
        index: usize,
    ) -> VortexResult<DistanceEstimate> {
        let position = self.positions[index] as usize;
        let record = &self.data[position * self.stride..(position + 1) * self.stride];
        let (norm_squared, scale, residual) = Self::corrections(record);
        query.estimate_parts(&record[24..], norm_squared, scale, residual)
    }

    #[inline]
    pub(super) fn estimate_four(
        &self,
        query: &RaBitQQuery,
        ids: [u32; 4],
    ) -> VortexResult<[DistanceEstimate; 4]> {
        let records = ids.map(|id| {
            let position = self.positions[id as usize] as usize;
            &self.data[position * self.stride..(position + 1) * self.stride]
        });
        query.estimate_four(
            records.map(|record| &record[24..]),
            records.map(Self::corrections),
        )
    }
}

fn bytes<const N: usize>(input: &mut impl Read) -> VortexResult<[u8; N]> {
    let mut value = [0; N];
    input.read_exact(&mut value)?;
    Ok(value)
}
fn u32_at(input: &mut impl Read) -> VortexResult<u32> {
    Ok(u32::from_le_bytes(bytes(input)?))
}
fn u64_at(input: &mut impl Read) -> VortexResult<u64> {
    Ok(u64::from_le_bytes(bytes(input)?))
}
fn f64_at(input: &mut impl Read) -> VortexResult<f64> {
    let value = f64::from_le_bytes(bytes(input)?);
    if !value.is_finite() {
        vortex_bail!("Non-finite RaBitQ metadata");
    }
    Ok(value)
}

fn original(
    vectors: &File,
    postings: &File,
    location: u64,
    dimension: usize,
) -> VortexResult<Vec<f32>> {
    let file = if location & HEAD == 0 {
        postings
    } else {
        vectors
    };
    let mut buffer = vec![0; dimension * 4];
    file.read_exact_at(&mut buffer, location & !HEAD)?;
    let mut values = Vec::with_capacity(dimension);
    for value in buffer.chunks_exact(4) {
        let value = f32::from_le_bytes([value[0], value[1], value[2], value[3]]);
        if !value.is_finite() {
            vortex_bail!("Non-finite RaBitQ original vector");
        }
        values.push(value);
    }
    Ok(values)
}

pub(crate) struct Routing {
    pub(crate) head_ids: Vec<u32>,
    pub(crate) postings: Vec<Vec<u32>>,
    pub(crate) locations: Vec<u64>,
}

// Parse only bounded routing records. Called after full native validation in production.
pub(crate) fn routing(root: &Path, bundle: SpFreshBundle) -> VortexResult<Routing> {
    bundle.validate()?;
    let mut vectors = BufReader::new(File::open(root.join("vectors.bin"))?);
    let heads = u32_at(&mut vectors)?;
    let dimension = u64::from(bundle.dimension);
    if heads == 0
        || heads > bundle.rows
        || u32_at(&mut vectors)? != bundle.dimension
        || vectors.get_ref().metadata()?.len() != 8 + u64::from(heads) * dimension * 4
    {
        vortex_bail!("Invalid RaBitQ native head shape");
    }
    let mut ids = BufReader::new(File::open(root.join("head_ids.bin"))?);
    if ids.get_ref().metadata()?.len() != u64::from(heads) * 8 {
        vortex_bail!("Invalid RaBitQ native head mapping length");
    }
    let mut locations = vec![u64::MAX; bundle.rows as usize];
    let mut head_ids = Vec::with_capacity(heads as usize);
    for head in 0..heads {
        let id = u32::try_from(u64_at(&mut ids)?)?;
        if id >= bundle.rows || locations[id as usize] != u64::MAX {
            vortex_bail!("Invalid RaBitQ native head ID");
        }
        head_ids.push(id);
        locations[id as usize] = HEAD | (8 + u64::from(head) * dimension * 4);
    }
    let mut header = BufReader::new(File::open(root.join("postings.bin"))?);
    let size = header.get_ref().metadata()?.len();
    if u32_at(&mut header)? != heads
        || u32_at(&mut header)? != bundle.rows
        || u32_at(&mut header)? != bundle.dimension
    {
        vortex_bail!("Invalid RaBitQ native posting shape");
    }
    let header_pages = u64::from(u32_at(&mut header)?);
    if header_pages * 4096 < 16 + u64::from(heads) * 12 || header_pages * 4096 > size {
        vortex_bail!("Invalid RaBitQ native posting header");
    }
    let mut input = BufReader::new(File::open(root.join("postings.bin"))?);
    let mut postings = Vec::with_capacity(heads as usize);
    let mut total_count = 0u64;
    for _ in 0..heads {
        let page = u64::from(u32_at(&mut header)?);
        let offset = u64::from(u16::from_le_bytes(bytes(&mut header)?));
        let count = u64::from(u32_at(&mut header)?);
        let pages = u64::from(u16::from_le_bytes(bytes(&mut header)?));
        let start = (header_pages + page) * 4096;
        let stride = 4 + dimension * 4;
        if offset >= 4096
            || pages > u64::from(bundle.posting_page_limit)
            || offset + count * stride > pages * 4096
            || start + pages * 4096 > size
        {
            vortex_bail!("Invalid RaBitQ native posting extent");
        }
        total_count = total_count
            .checked_add(count)
            .ok_or_else(|| vortex_err!("RaBitQ posting count overflow"))?;
        if total_count > (size - header_pages * 4096) / stride {
            vortex_bail!("RaBitQ posting counts exceed the original file capacity");
        }
        let mut list = Vec::with_capacity(usize::try_from(count)?);
        for index in 0..count {
            let position = start + offset + index * stride;
            input.seek(SeekFrom::Start(position))?;
            let id = u32_at(&mut input)?;
            if id >= bundle.rows {
                vortex_bail!("Invalid RaBitQ native posting ID");
            }
            list.push(id);
            if locations[id as usize] == u64::MAX {
                locations[id as usize] = position + 4;
            }
        }
        postings.push(list);
    }
    if locations.contains(&u64::MAX) {
        vortex_bail!("Incomplete RaBitQ row coverage");
    }
    Ok(Routing {
        head_ids,
        postings,
        locations,
    })
}

pub(crate) fn build_sidecar(root: &Path, bundle: SpFreshBundle, seed: u64) -> VortexResult<()> {
    let routing = routing(root, bundle)?;
    let vectors = File::open(root.join("vectors.bin"))?;
    let original_postings = File::open(root.join("postings.bin"))?;
    let dimension = bundle.dimension as usize;
    let mut center = vec![0.0; dimension];
    for location in &routing.locations {
        for (mean, value) in center.iter_mut().zip(original(
            &vectors,
            &original_postings,
            *location,
            dimension,
        )?) {
            *mean += f64::from(value);
        }
    }
    for mean in &mut center {
        *mean = (*mean / f64::from(bundle.rows)).clamp(-f64::from(f32::MAX), f64::from(f32::MAX));
    }
    let quantizer = RaBitQ::new(center, seed)?;
    let mut output = BufWriter::new(File::create_new(root.join("rabitq.bin"))?);
    output.write_all(MAGIC)?;
    for value in [
        bundle.dimension,
        bundle.rows,
        u32::try_from(routing.head_ids.len())?,
        0,
    ] {
        output.write_all(&value.to_le_bytes())?;
    }
    let total = routing
        .postings
        .iter()
        .map(|list| list.len() as u64)
        .sum::<u64>();
    output.write_all(&total.to_le_bytes())?;
    for value in quantizer
        .center
        .iter()
        .chain(&quantizer.reflectors)
        .chain(&quantizer.signs)
    {
        output.write_all(&value.to_le_bytes())?;
    }
    for value in &routing.locations {
        output.write_all(&value.to_le_bytes())?;
    }
    for value in &routing.head_ids {
        output.write_all(&value.to_le_bytes())?;
    }
    let mut offset = 0u64;
    output.write_all(&offset.to_le_bytes())?;
    for list in &routing.postings {
        offset += list.len() as u64;
        output.write_all(&offset.to_le_bytes())?;
    }
    for id in routing.postings.iter().flatten() {
        output.write_all(&id.to_le_bytes())?;
    }
    for location in &routing.locations {
        let code = quantizer.encode(&original(
            &vectors,
            &original_postings,
            *location,
            dimension,
        )?)?;
        for value in [code.norm_squared, code.scale, code.residual] {
            output.write_all(&value.to_le_bytes())?;
        }
        output.write_all(&code.bits)?;
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(())
}

impl QuantizedIndex {
    pub(crate) fn open(root: &Path, sidecar: &Path, bundle: SpFreshBundle) -> VortexResult<Self> {
        bundle.validate()?;
        let mut input = BufReader::new(File::open(sidecar)?);
        let size = input.get_ref().metadata()?.len();
        if &bytes::<8>(&mut input)? != MAGIC
            || u32_at(&mut input)? != bundle.dimension
            || u32_at(&mut input)? != bundle.rows
        {
            vortex_bail!("Invalid RaBitQ sidecar header");
        }
        let heads = u32_at(&mut input)?;
        if heads == 0 || heads > bundle.rows || u32_at(&mut input)? != 0 {
            vortex_bail!("Invalid RaBitQ sidecar head count or version");
        }
        let total = u64_at(&mut input)?;
        let dim = u64::from(bundle.dimension);
        let rotation = dim * (dim + 1) / 2;
        let fixed = 32
            + 8 * (rotation + 2 * dim)
            + 8 * u64::from(bundle.rows)
            + 4 * u64::from(heads)
            + 8 * (u64::from(heads) + 1)
            + u64::from(bundle.rows) * (24 + dim.div_ceil(8));
        if total.checked_mul(4).and_then(|n| n.checked_add(fixed)) != Some(size) {
            vortex_bail!("Invalid RaBitQ sidecar length");
        }
        let mut floats = |count: u64| {
            (0..count)
                .map(|_| f64_at(&mut input))
                .collect::<VortexResult<Vec<_>>>()
        };
        let center = floats(dim)?;
        if center.iter().any(|v| v.abs() > f64::from(f32::MAX)) {
            vortex_bail!("Invalid RaBitQ center");
        }
        let reflectors = floats(rotation)?;
        let signs = floats(dim)?;
        let mut offset = 0;
        for len in (1..=bundle.dimension as usize).rev() {
            let norm = reflectors[offset..offset + len]
                .iter()
                .map(|v| v * v)
                .sum::<f64>();
            if (norm - 1.0).abs() > 1e-12 {
                vortex_bail!("Non-unit RaBitQ reflector");
            }
            offset += len;
        }
        if signs.iter().any(|v| *v != -1.0 && *v != 1.0) {
            vortex_bail!("Invalid RaBitQ rotation sign");
        }
        let locations = (0..bundle.rows)
            .map(|_| u64_at(&mut input))
            .collect::<VortexResult<Vec<_>>>()?;
        let head_ids = (0..heads)
            .map(|_| u32_at(&mut input))
            .collect::<VortexResult<Vec<_>>>()?;
        let offsets = (0..=heads)
            .map(|_| u64_at(&mut input))
            .collect::<VortexResult<Vec<_>>>()?;
        if offsets[0] != 0
            || offsets[heads as usize] != total
            || offsets.windows(2).any(|p| p[0] > p[1])
        {
            vortex_bail!("Invalid RaBitQ posting offsets");
        }
        let mut postings = Vec::with_capacity(heads as usize);
        for pair in offsets.windows(2) {
            postings.push(
                (pair[0]..pair[1])
                    .map(|_| u32_at(&mut input))
                    .collect::<VortexResult<Vec<_>>>()?,
            );
        }
        let native = routing(root, bundle)?;
        if locations != native.locations
            || head_ids != native.head_ids
            || postings != native.postings
        {
            vortex_bail!("RaBitQ routing does not match original native files");
        }
        let stride = 24 + usize::try_from(dim.div_ceil(8))?;
        let code_bytes = (bundle.rows as usize)
            .checked_mul(stride)
            .ok_or_else(|| vortex_err!("RaBitQ code allocation overflow"))?;
        let mut code_data = vec![0; code_bytes];
        input.read_exact(&mut code_data)?;
        for record in code_data.chunks_exact(stride) {
            let (norm_squared, scale, residual) = QuantizedCodes::corrections(record);
            if !norm_squared.is_finite()
                || !scale.is_finite()
                || !residual.is_finite()
                || norm_squared < 0.0
                || scale < 0.0
                || residual < 0.0
                || norm_squared > 4.0 * dim as f64 * f64::from(f32::MAX).powi(2)
                || scale * scale > (norm_squared * dim as f64) * (1.0 + 1e-10)
                || scale * scale < norm_squared * (1.0 - 1e-10)
                || (residual * residual - (scale * scale - norm_squared).max(0.0)).abs()
                    > 1e-10 * (1.0 + scale * scale)
            {
                vortex_bail!("Invalid RaBitQ correction factors");
            }
            let bits = &record[24..];
            if dim % 8 != 0 && bits[bits.len() - 1] >> (dim % 8) != 0 {
                vortex_bail!("Invalid RaBitQ code padding");
            }
        }
        // Keep each posting's codes together; native IDs retain their original mapping.
        let mut positions = vec![u32::MAX; bundle.rows as usize];
        let mut packed_codes = Vec::with_capacity(code_bytes);
        for id in head_ids.iter().chain(postings.iter().flatten()) {
            let index = *id as usize;
            if positions[index] == u32::MAX {
                positions[index] = u32::try_from(packed_codes.len() / stride)?;
                packed_codes.extend_from_slice(&code_data[index * stride..(index + 1) * stride]);
            }
        }
        if packed_codes.len() != code_bytes {
            vortex_bail!("Incomplete RaBitQ packed code coverage");
        }
        Ok(Self {
            quantizer: RaBitQ {
                center,
                reflectors,
                signs,
            },
            codes: QuantizedCodes {
                data: packed_codes,
                stride,
                positions,
            },
            head_ids,
            postings,
            locations,
            vectors: File::open(root.join("vectors.bin"))?,
            original_postings: File::open(root.join("postings.bin"))?,
        })
    }

    pub(super) fn original_reader(
        &self,
        max_cache_bytes: usize,
    ) -> VortexResult<OriginalReader<'_>> {
        let lengths = [
            self.original_postings.metadata()?.len(),
            self.vectors.metadata()?.len(),
        ];
        if lengths.iter().any(|length| !length.is_multiple_of(4)) {
            vortex_bail!("Unaligned RaBitQ original file length");
        }
        Ok(OriginalReader {
            index: self,
            lengths,
            max_chunks: max_cache_bytes.min(MAX_READ_CACHE_BYTES) / READ_CHUNK_BYTES,
            chunks: HashMap::new(),
            raw: vec![0; READ_CHUNK_BYTES],
            fallback: vec![0.0; self.quantizer.center.len()],
            read_calls: 0,
            read_bytes: 0,
            cached_bytes: 0,
        })
    }
}

pub(super) struct OriginalReader<'a> {
    index: &'a QuantizedIndex,
    lengths: [u64; 2],
    max_chunks: usize,
    chunks: HashMap<u64, Vec<f32>>,
    raw: Vec<u8>,
    fallback: Vec<f32>,
    pub read_calls: usize,
    pub read_bytes: usize,
    pub cached_bytes: usize,
}

impl OriginalReader<'_> {
    fn read_vector(&mut self, id: u32) -> VortexResult<&[f32]> {
        let location = *self
            .index
            .locations
            .get(id as usize)
            .ok_or_else(|| vortex_err!("RaBitQ ID outside row map"))?;
        let kind = usize::from(location & HEAD != 0);
        let file = if kind == 0 {
            &self.index.original_postings
        } else {
            &self.index.vectors
        };
        let offset = location & !HEAD;
        let bytes = self.fallback.len() * 4;
        let end = offset
            .checked_add(u64::try_from(bytes)?)
            .ok_or_else(|| vortex_err!("RaBitQ original extent overflow"))?;
        if !offset.is_multiple_of(4) || end > self.lengths[kind] {
            vortex_bail!("Truncated or unaligned RaBitQ original vector");
        }
        let chunk_start = offset & !(u64::try_from(READ_CHUNK_BYTES)? - 1);
        let within = usize::try_from(offset - chunk_start)?;
        let key = (location & HEAD) | chunk_start;
        if within + bytes > READ_CHUNK_BYTES
            || (self.chunks.len() >= self.max_chunks && !self.chunks.contains_key(&key))
        {
            file.read_exact_at(&mut self.raw[..bytes], offset)?;
            self.read_calls += 1;
            self.read_bytes += bytes;
            for (value, raw) in self
                .fallback
                .iter_mut()
                .zip(self.raw[..bytes].chunks_exact(4))
            {
                *value = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            }
            return Ok(&self.fallback);
        }
        let chunk = match self.chunks.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let size = usize::try_from(
                    (self.lengths[kind] - chunk_start).min(u64::try_from(READ_CHUNK_BYTES)?),
                )?;
                file.read_exact_at(&mut self.raw[..size], chunk_start)?;
                self.read_calls += 1;
                self.read_bytes += size;
                let values = self.raw[..size]
                    .chunks_exact(4)
                    .map(|raw| f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
                    .collect::<Vec<_>>();
                self.cached_bytes += values.len() * 4;
                entry.insert(values)
            }
        };
        chunk
            .get(within / 4..(within + bytes) / 4)
            .ok_or_else(|| vortex_err!("RaBitQ original outside cached extent"))
    }

    pub(super) fn exact_score(
        &mut self,
        id: u32,
        query: &ExactQuery<'_>,
    ) -> VortexResult<(f32, bool)> {
        query.score(self.read_vector(id)?)
    }

    #[cfg(test)]
    pub(super) fn exact_distance(&mut self, id: u32, query: &[f32]) -> VortexResult<f64> {
        if query.len() != self.fallback.len() {
            vortex_bail!("Invalid RaBitQ original distance shape");
        }
        let vector = self.read_vector(id)?;
        if vector.iter().any(|value| !value.is_finite()) {
            vortex_bail!("Non-finite RaBitQ original vector");
        }
        Ok(vector
            .iter()
            .zip(query)
            .map(|(left, right)| {
                let delta = f64::from(*left) - f64::from(*right);
                delta * delta
            })
            .sum())
    }
}
