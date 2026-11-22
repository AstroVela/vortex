// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::StandardNormal;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::RaBitQErrorBound;

/// Shared centering and orthogonal transform. Construction uses O(d²) space;
/// encoding and query preparation use O(d²) time. Scoring one code uses O(d).
#[derive(Debug, Clone)]
pub struct RaBitQ {
    pub(super) center: Vec<f64>,
    // Packed normalized Gaussian Householder vectors of lengths d, d-1, ..., 1.
    pub(super) reflectors: Vec<f64>,
    pub(super) signs: Vec<f64>,
}

/// Packed sign code and correction factors for a single centered data vector.
#[derive(Debug, Clone)]
pub struct EncodedVector {
    pub(super) bits: Vec<u8>,
    pub(super) norm_squared: f64,
    pub(super) scale: f64,
    pub(super) residual: f64,
}

impl EncodedVector {
    /// One bit per dimension, least significant bit first, with zero padding.
    pub fn code(&self) -> &[u8] {
        &self.bits
    }
}

/// Squared-L2 estimate and its absolute error radius before clipping to zero.
#[derive(Debug, Clone, Copy)]
pub struct DistanceEstimate {
    /// Unclipped estimator; it can be negative because quantization is lossy.
    pub estimate: f64,
    /// Error radius under the selected deterministic or probabilistic policy.
    pub error: f64,
}

impl DistanceEstimate {
    /// Lower endpoint, clipped to the squared-L2 distance domain.
    pub fn lower(self) -> f64 {
        (self.estimate - self.error).max(0.0)
    }

    /// Upper endpoint, clipped to the squared-L2 distance domain.
    pub fn upper(self) -> f64 {
        (self.estimate + self.error).max(0.0)
    }
}

/// Prepared full precision query in the quantizer's rotated coordinate system.
#[derive(Debug)]
pub struct RaBitQQuery {
    tables: Vec<[f64; 16]>,
    dimension: usize,
    norm_squared: f64,
    bound: RaBitQErrorBound,
}

impl RaBitQ {
    /// Construct a Haar-distributed rotation for a finite center of 1..=4096
    /// components. Unlike a signed Hadamard approximation, this construction
    /// preserves the distribution assumed by the RaBitQ concentration analysis.
    pub fn new(center: Vec<f64>, seed: u64) -> VortexResult<Self> {
        if !(1..=4096).contains(&center.len())
            || center
                .iter()
                .any(|x| !x.is_finite() || x.abs() > f64::from(f32::MAX))
        {
            vortex_bail!("RaBitQ requires a finite center of dimension 1..=4096");
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut reflectors = Vec::with_capacity(center.len() * (center.len() + 1) / 2);
        let mut signs = Vec::with_capacity(center.len());
        for len in (1..=center.len()).rev() {
            let mut values: Vec<f64> = (0..len).map(|_| StandardNormal.sample(&mut rng)).collect();
            let norm = values.iter().map(|x| x * x).sum::<f64>().sqrt();
            let sign = if values[0] < 0.0 { -1.0 } else { 1.0 };
            values[0] += sign * norm;
            let norm = values.iter().map(|x| x * x).sum::<f64>().sqrt();
            if !norm.is_finite() || norm == 0.0 {
                vortex_bail!("Degenerate RaBitQ random rotation");
            }
            reflectors.extend(values.into_iter().map(|x| x / norm));
            signs.push(-sign);
        }
        Ok(Self {
            center,
            reflectors,
            signs,
        })
    }

    fn rotate(&self, vector: &[f32]) -> VortexResult<(Vec<f64>, f64)> {
        if vector.len() != self.center.len() || vector.iter().any(|x| !x.is_finite()) {
            vortex_bail!("RaBitQ vector dimensions or components are invalid");
        }
        let mut values: Vec<f64> = vector
            .iter()
            .zip(&self.center)
            .map(|(value, center)| f64::from(*value) - center)
            .collect();
        let norm_squared = values.iter().map(|x| x * x).sum::<f64>();
        let mut offset = 0;
        for start in 0..values.len() {
            let len = values.len() - start;
            let reflector = &self.reflectors[offset..offset + len];
            let dot = values[start..]
                .iter()
                .zip(reflector)
                .map(|(x, y)| x * y)
                .sum::<f64>();
            for (value, direction) in values[start..].iter_mut().zip(reflector) {
                *value -= 2.0 * dot * direction;
            }
            offset += len;
        }
        for (value, sign) in values.iter_mut().zip(&self.signs) {
            *value *= sign;
        }
        if !norm_squared.is_finite() || values.iter().any(|x| !x.is_finite()) {
            vortex_bail!("RaBitQ centered vector exceeds finite f64 range");
        }
        Ok((values, norm_squared))
    }

    /// Encode a finite vector, retaining norm and projection correction. A vector
    /// equal to the center has zero norm and is represented without division.
    pub fn encode(&self, vector: &[f32]) -> VortexResult<EncodedVector> {
        let (rotated, norm_squared) = self.rotate(vector)?;
        let mut bits = vec![0u8; vector.len().div_ceil(8)];
        for (i, value) in rotated.iter().enumerate() {
            if *value >= 0.0 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        let projection =
            rotated.iter().map(|x| x.abs()).sum::<f64>() / (vector.len() as f64).sqrt();
        let scale = if norm_squared == 0.0 {
            0.0
        } else {
            norm_squared / projection
        };
        // Avoid cancellation in sqrt(scale² - norm²) near a codebook vertex:
        // an underestimated residual would make deterministic pruning unsound.
        let component = scale / (vector.len() as f64).sqrt();
        let residual = rotated
            .iter()
            .map(|value| {
                let reconstructed = if *value >= 0.0 { component } else { -component };
                (value - reconstructed).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        Ok(EncodedVector {
            bits,
            norm_squared,
            scale,
            residual,
        })
    }

    /// Prepare a query once per search. Data and query must use this same quantizer.
    pub fn query(&self, query: &[f32], bound: RaBitQErrorBound) -> VortexResult<RaBitQQuery> {
        bound.validate()?;
        let (rotated, norm_squared) = self.rotate(query)?;
        let dimension = rotated.len();
        let normalization = (dimension as f64).sqrt();
        let tables = (0..dimension.div_ceil(8) * 2)
            .map(|nibble| {
                let values: [f64; 4] = std::array::from_fn(|bit| {
                    rotated.get(nibble * 4 + bit).copied().unwrap_or(0.0) / normalization
                });
                let mut table = [0.0; 16];
                table[0] = -values.iter().sum::<f64>();
                for mask in 1usize..16 {
                    let bit = mask.trailing_zeros() as usize;
                    table[mask] = table[mask & (mask - 1)] + 2.0 * values[bit];
                }
                table
            })
            .collect();
        Ok(RaBitQQuery {
            tables,
            dimension,
            norm_squared,
            bound,
        })
    }
}

impl RaBitQQuery {
    /// Estimate distance without reading the original vector. Codes must have
    /// been produced by the same quantizer; mixing generations is unsupported.
    pub fn estimate(&self, code: &EncodedVector) -> VortexResult<DistanceEstimate> {
        if code.bits.len() != self.dimension.div_ceil(8) {
            vortex_bail!("RaBitQ code dimension mismatch");
        }
        let dot = code
            .bits
            .iter()
            .zip(self.tables.chunks_exact(2))
            .map(|(byte, tables)| {
                tables[0][usize::from(byte & 15)] + tables[1][usize::from(byte >> 4)]
            })
            .sum::<f64>();
        let estimate = code.norm_squared + self.norm_squared - 2.0 * code.scale * dot;
        // The error vector x - ||x|| z/alpha has norm ||x||sqrt(1-alpha²)/alpha.
        // Its direction is uniform in x's orthogonal complement under Haar rotation.
        let error =
            2.0 * code.residual * self.norm_squared.sqrt() * self.bound.factor(self.dimension);
        let slack = 128.0
            * (self.dimension as f64).powi(2)
            * f64::EPSILON
            * (code.norm_squared
                + self.norm_squared
                + 2.0 * code.scale * self.norm_squared.sqrt()
                + 1.0);
        Ok(DistanceEstimate {
            estimate,
            error: error + slack,
        })
    }
}
