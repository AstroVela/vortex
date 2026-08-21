//! Measures filtering a zstd array against decompressing it and filtering the result, across
//! framings and selectivities. Frames are compressed independently, so a mask that touches few of
//! them only pays for those.
use std::time::Instant;

use anyhow::Result;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builtins::ArrayBuiltins as _;
use vortex::mask::Mask;
use vortex_bench::SESSION;
use vortex_zstd::Zstd;

const N: usize = 1_000_000;

fn main() -> Result<()> {
    let values = VarBinViewArray::from_iter_str(
        (0..N).map(|i| format!("https://example.com/page/{}/section/{}", i % 9973, i % 71)),
    );

    println!(
        "{:>8} {:>7} {:>12} {:>12} {:>12} {:>9}",
        "vpf", "frames", "selectivity", "filter ms", "decode-all ms", "speedup"
    );
    for values_per_frame in [0usize, 65536, 8192, 1024] {
        let mut ctx = SESSION.create_execution_ctx();
        let array = Zstd::from_var_bin_view(&values, 3, values_per_frame, &mut ctx)?.into_array();
        let n_frames = array.nbuffers();

        for selectivity in [0.0001f64, 0.001, 0.01, 0.5] {
            let step = (1.0 / selectivity) as usize;
            let mask = Mask::from_indices(N, (0..N).step_by(step));
            let selected = mask.true_count();

            // Warm both paths, so neither timing carries first-touch allocation cost.
            let warmup = array.clone().filter(mask.clone())?.execute::<Canonical>(&mut ctx)?;
            drop(warmup);
            let warmup = array.clone().execute::<Canonical>(&mut ctx)?.into_array();
            let warmup = warmup.filter(mask.clone())?.execute::<Canonical>(&mut ctx)?;
            drop(warmup);

            let start = Instant::now();
            let filtered = array
                .clone()
                .filter(mask.clone())?
                .execute::<Canonical>(&mut ctx)?;
            let with_kernel = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(filtered.into_array().len(), selected);

            let start = Instant::now();
            let canonical = array.clone().execute::<Canonical>(&mut ctx)?.into_array();
            let filtered = canonical.filter(mask)?.execute::<Canonical>(&mut ctx)?;
            let decode_all = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(filtered.into_array().len(), selected);

            println!(
                "{values_per_frame:>8} {n_frames:>7} {selectivity:>12} {with_kernel:>12.2} \
                 {decode_all:>12.2} {:>8.1}x",
                decode_all / with_kernel
            );
        }
    }
    Ok(())
}
