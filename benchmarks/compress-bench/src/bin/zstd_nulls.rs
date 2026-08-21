//! How the zstd encodings handle null-heavy string columns.
//!
//! Both store only the valid values, so nulls cost validity bits rather than value bytes. This
//! measures what that comes to across null fractions, and which scheme the cascade picks.
use std::time::Instant;

use anyhow::Result;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::VarBinViewArray;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex_bench::SESSION;
use vortex_zstd::Zstd;
use vortex_zstd::ZstdOptions;
use vortex_zstd_v2::ZstdV2;

const N: usize = 200_000;

/// A deterministic null pattern at the requested density, with values that share structure the
/// way real columns do.
fn values(null_fraction: f64, value_len: usize) -> VarBinViewArray {
    let nulls_per_thousand = (null_fraction * 1000.0).round() as usize;
    VarBinViewArray::from_iter_nullable_str((0..N).map(|i| {
        (i % 1000 >= nulls_per_thousand).then(|| {
            // Distinct values, so dictionary encoding cannot swallow the column and the zstd
            // schemes are what actually run.
            let body = format!("{i}-{}", "payload".repeat(value_len / 7 + 1));
            body[..body.len().min(value_len)].to_string()
        })
    }))
}

fn main() -> Result<()> {
    println!(
        "{:>6} {:>8} {:>11} {:>11} {:>11} {:>7} {:>11} {:>11} {:>7} {:>9} {:>9}  {}",
        "nulls",
        "str len",
        "raw",
        "v1 1frame",
        "v2 1frame",
        "v2/v1",
        "v1 8k",
        "v2 8k",
        "v2/v1",
        "v1 decode",
        "v2 decode",
        "cascade picks"
    );
    for value_len in [32usize, 256] {
        for null_fraction in [0.0f64, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let mut ctx = SESSION.create_execution_ctx();
            let array = values(null_fraction, value_len);
            let raw = array.as_ref().nbytes();

            let one_frame = ZstdOptions::new(6).with_dictionary(vortex_zstd::DictionaryMode::Never);
            let v1_one = Zstd::from_var_bin_view_with_options(&array, one_frame, &mut ctx)?
                .into_array();
            let v1_8k = Zstd::from_var_bin_view_with_options(
                &array,
                one_frame.with_values_per_frame(8192),
                &mut ctx,
            )?
            .into_array();
            let v2_one = ZstdV2::from_var_bin_view(&array, 6, 0, &mut ctx)?.into_array();
            let v2_8k = ZstdV2::from_var_bin_view(&array, 6, 8192, &mut ctx)?.into_array();

            let start = Instant::now();
            let decoded = v1_8k.clone().execute::<Canonical>(&mut ctx)?;
            let v1_decode = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(decoded.into_array().len(), N);

            let start = Instant::now();
            let decoded = v2_8k.clone().execute::<Canonical>(&mut ctx)?;
            let v2_decode = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(decoded.into_array().len(), N);

            // What the cascade would choose left to itself.
            let compressor = BtrBlocksCompressorBuilder::default().with_compact().build();
            let chosen = compressor.compress(&array.clone().into_array(), &mut ctx)?;

            let ratio = |a: &vortex::array::ArrayRef, b: &vortex::array::ArrayRef| {
                (a.nbytes() as f64 / b.nbytes() as f64 - 1.0) * 100.0
            };
            println!(
                "{:>5.0}% {:>8} {:>11} {:>11} {:>11} {:>6.1}% {:>11} {:>11} {:>6.1}% {:>7.1}ms {:>7.1}ms  {}",
                null_fraction * 100.0,
                value_len,
                raw,
                v1_one.nbytes(),
                v2_one.nbytes(),
                ratio(&v2_one, &v1_one),
                v1_8k.nbytes(),
                v2_8k.nbytes(),
                ratio(&v2_8k, &v1_8k),
                v1_decode,
                v2_decode,
                chosen.encoding_id(),
            );
        }
    }
    Ok(())
}
