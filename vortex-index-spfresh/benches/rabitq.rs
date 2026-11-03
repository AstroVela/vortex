// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use vortex_error::VortexExpect;
use vortex_index_spfresh::rabitq::RaBitQ;
use vortex_index_spfresh::rabitq::RaBitQErrorBound;

fn main() {
    divan::main();
}

#[divan::bench(args = [128, 768, 1536])]
fn quantized_distance(bencher: Bencher, dimension: usize) {
    let quantizer = RaBitQ::new(vec![0.0; dimension], 42).vortex_expect("valid quantizer");
    let vector: Vec<f32> = (0..dimension).map(|i| (i % 17) as f32 - 8.0).collect();
    let code = quantizer.encode(&vector).vortex_expect("finite vector");
    let query = quantizer
        .query(
            &vec![1.0; dimension],
            RaBitQErrorBound::Probabilistic { epsilon: 4.0 },
        )
        .vortex_expect("finite query");
    bencher.bench_local(|| {
        query
            .estimate(divan::black_box(&code))
            .vortex_expect("matching code")
    });
}

#[divan::bench(args = [128, 768, 1536])]
fn exact_distance(bencher: Bencher, dimension: usize) {
    let vector: Vec<f32> = (0..dimension).map(|i| (i % 17) as f32 - 8.0).collect();
    let query = vec![1.0f32; dimension];
    bencher.bench_local(|| {
        divan::black_box(&vector)
            .iter()
            .zip(divan::black_box(&query))
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum::<f64>()
    });
}
