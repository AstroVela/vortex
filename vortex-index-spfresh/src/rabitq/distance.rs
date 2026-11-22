// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parallel exact-distance accumulation with the original f32 rounding contract.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::search::score;

pub(super) struct ExactQuery<'a> {
    original: &'a [f32],
    values: Vec<f64>,
    rounding_factor: f64,
}

impl<'a> ExactQuery<'a> {
    pub(super) fn new(query: &'a [f32]) -> VortexResult<Self> {
        if !(1..=4096).contains(&query.len()) || query.iter().any(|value| !value.is_finite()) {
            vortex_bail!("Invalid RaBitQ exact query");
        }
        Ok(Self {
            original: query,
            values: query.iter().copied().map(f64::from).collect(),
            rounding_factor: 4.0 * query.len() as f64 * f64::EPSILON,
        })
    }

    pub(super) fn score(&self, vector: &[f32]) -> VortexResult<(f32, bool)> {
        if vector.len() != self.values.len() {
            vortex_bail!("Invalid RaBitQ original distance shape");
        }
        let mut sums = [0.0; 4];
        let mut finite = true;
        let mut values = vector.chunks_exact(4);
        let mut query = self.values.chunks_exact(4);
        for (values, query) in values.by_ref().zip(query.by_ref()) {
            for lane in 0..4 {
                finite &= values[lane].is_finite();
                let delta = f64::from(values[lane]) - query[lane];
                sums[lane] += delta * delta;
            }
        }
        let mut total = sums.into_iter().sum::<f64>();
        for (value, query) in values.remainder().iter().zip(query.remainder()) {
            finite &= value.is_finite();
            let delta = f64::from(*value) - query;
            total += delta * delta;
        }
        if !finite {
            vortex_bail!("Non-finite RaBitQ original vector");
        }
        // The squared terms are nonnegative and cannot underflow/overflow f64 for
        // finite f32 inputs. Each reduction's error is at most gamma_n times their
        // exact sum. For n <= 4096, 4*n*EPSILON conservatively bounds the difference
        // between both reductions, including rounding these interval endpoints.
        let slack = self.rounding_factor * total;
        let lower = score(total - slack);
        let upper = score(total + slack);
        if lower.to_bits() == upper.to_bits() {
            return Ok((lower, false));
        }
        Ok((
            score(
                vector
                    .iter()
                    .zip(self.original)
                    .map(|(left, right)| {
                        let delta = f64::from(*left) - f64::from(*right);
                        delta * delta
                    })
                    .sum(),
            ),
            true,
        ))
    }
}
