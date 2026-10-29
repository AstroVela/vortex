// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

const MAGIC: &[u8; 8] = b"VXHNPL02";

/// Immutable supplementary level-zero adjacency, addressed by native internal IDs.
/// The bundle descriptor binds this graph to the exact base graph and row mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswPayload {
    pub(crate) offsets: Vec<u64>,
    pub(crate) links: Vec<u32>,
}

impl HnswPayload {
    /// Construct an explicitly empty overlay for a build with payload_m=0.
    pub fn empty(rows: u32) -> Self {
        Self {
            offsets: vec![0; rows as usize + 1],
            links: Vec::new(),
        }
    }

    /// Validate and own a CSR graph. Neighbor IDs must be in range, strictly
    /// increasing within each row, and different from the source row.
    pub fn try_new(offsets: Vec<u64>, links: Vec<u32>) -> VortexResult<Self> {
        let graph = Self { offsets, links };
        let rows = graph
            .offsets
            .len()
            .checked_sub(1)
            .ok_or_else(|| vortex_err!("Missing payload offsets"))?;
        graph.validate(u32::try_from(rows)?)?;
        Ok(graph)
    }

    /// Number of directed supplementary edges after deduplication.
    pub fn edge_count(&self) -> usize {
        self.links.len()
    }

    pub(crate) fn validate(&self, rows: u32) -> VortexResult<()> {
        if self.offsets.len() != rows as usize + 1
            || self.offsets.first() != Some(&0)
            || self.offsets.last() != Some(&(self.links.len() as u64))
        {
            vortex_bail!("Invalid payload CSR shape");
        }
        for (row, bounds) in self.offsets.windows(2).enumerate() {
            if bounds[0] > bounds[1] || bounds[1] > self.links.len() as u64 {
                vortex_bail!("Invalid payload CSR offsets");
            }
            let links = &self.links[usize::try_from(bounds[0])?..usize::try_from(bounds[1])?];
            if links.iter().any(|id| *id >= rows || *id as usize == row)
                || links.windows(2).any(|pair| pair[0] >= pair[1])
            {
                vortex_bail!("Invalid payload neighbor ID or ordering");
            }
        }
        Ok(())
    }

    pub(crate) fn serialized_size(&self) -> VortexResult<usize> {
        self.offsets
            .len()
            .checked_mul(8)
            .and_then(|size| {
                self.links
                    .len()
                    .checked_mul(4)
                    .and_then(|links| size.checked_add(links))
            })
            .and_then(|size| size.checked_add(24))
            .ok_or_else(|| vortex_err!("Payload byte size overflow"))
    }

    pub(crate) fn encode(&self, rows: u32) -> VortexResult<Vec<u8>> {
        self.validate(rows)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.serialized_size()?)
            .map_err(|err| vortex_err!("Payload serialization allocation: {err}"))?;
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&u64::from(rows).to_le_bytes());
        bytes.extend_from_slice(&(self.links.len() as u64).to_le_bytes());
        for offset in &self.offsets {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        for link in &self.links {
            bytes.extend_from_slice(&link.to_le_bytes());
        }
        Ok(bytes)
    }

    #[cfg(any(feature = "native", test))]
    pub(crate) fn decode(
        bytes: &[u8],
        rows: u32,
        max_edges: u64,
        max_degree: u32,
    ) -> VortexResult<Self> {
        if bytes.len() < 24 || &bytes[..8] != MAGIC {
            vortex_bail!("Invalid payload header");
        }
        let count = u64::from_le_bytes(
            bytes[8..16]
                .try_into()
                .map_err(|err: std::array::TryFromSliceError| vortex_err!(External: err))?,
        );
        let edges = u64::from_le_bytes(
            bytes[16..24]
                .try_into()
                .map_err(|err: std::array::TryFromSliceError| vortex_err!(External: err))?,
        );
        let expected = (u64::from(rows) + 1)
            .checked_mul(8)
            .and_then(|size| {
                edges
                    .checked_mul(4)
                    .and_then(|edges| size.checked_add(edges))
            })
            .and_then(|size| size.checked_add(24));
        if count != u64::from(rows) || edges > max_edges || expected != Some(bytes.len() as u64) {
            vortex_bail!("Invalid payload length, row count or edge budget");
        }
        let split = 24 + (rows as usize + 1) * 8;
        let offsets = bytes[24..split]
            .chunks_exact(8)
            .map(|word| {
                Ok(u64::from_le_bytes(word.try_into().map_err(
                    |err: std::array::TryFromSliceError| vortex_err!(External: err),
                )?))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let links = bytes[split..]
            .chunks_exact(4)
            .map(|word| {
                Ok(u32::from_le_bytes(word.try_into().map_err(
                    |err: std::array::TryFromSliceError| vortex_err!(External: err),
                )?))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let graph = Self::try_new(offsets, links)?;
        if graph
            .offsets
            .windows(2)
            .any(|pair| pair[1] - pair[0] > u64::from(max_degree))
        {
            vortex_bail!("Payload degree exceeds provider limit");
        }
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_roundtrip_and_corruption() -> VortexResult<()> {
        let graph = HnswPayload::try_new(vec![0, 1, 2, 2], vec![1, 2])?;
        let bytes = graph.encode(3)?;
        assert_eq!(HnswPayload::decode(&bytes, 3, 10, 2)?, graph);
        for offset in [0, 8, 16, 24, 32, 48, 56] {
            let mut damaged = bytes.clone();
            damaged[offset] = 255;
            assert!(HnswPayload::decode(&damaged, 3, 10, 2).is_err());
        }
        assert!(HnswPayload::decode(&bytes[..bytes.len() - 1], 3, 10, 2).is_err());
        assert!(HnswPayload::decode(&bytes, 3, 1, 2).is_err());
        assert!(HnswPayload::decode(&bytes, 3, 10, 0).is_err());
        assert!(HnswPayload::try_new(vec![0, 1, 1], vec![0]).is_err());
        assert!(HnswPayload::try_new(vec![0, 2, 2], vec![1, 1]).is_err());
        Ok(())
    }
}
