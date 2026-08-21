//! Sweeps zstd scheme settings over parquet datasets, writing each as a Vortex file.
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use futures::StreamExt;
use vortex::array::IntoArray;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex_bench::SESSION;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::tpch::tpchgen::TpchGenOptions;
use vortex_bench::tpch::tpchgen::generate_tpch_tables;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt as _;
use vortex_btrblocks::schemes::string;
use vortex_file::WriteStrategyBuilder;
use vortex_zstd::DictionaryMode;
use vortex_zstd::ZstdOptions;

struct Config {
    name: &'static str,
    options: ZstdOptions,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut paths: Vec<PathBuf> = Vec::new();
    for arg in std::env::args().skip(1) {
        if let Some(scale_factor) = arg.strip_prefix("tpch:") {
            let output_dir = PathBuf::from("/tmp/zstd-sweep-tpch").join(scale_factor);
            let lineitem = output_dir.join("lineitem.parquet");
            if !lineitem.exists() {
                generate_tpch_tables(TpchGenOptions {
                    scale_factor: scale_factor.to_string(),
                    output_dir: output_dir.clone(),
                    format: vortex_bench::Format::Parquet,
                    batch_size: 65536,
                })
                .await?;
            }
            paths.push(lineitem);
        } else {
            paths.push(PathBuf::from(arg));
        }
    }

    let configs = vec![
        Config { name: "L3 vpf=8192 nodict (today)", options: ZstdOptions::new(3).with_values_per_frame(8192).with_dictionary(DictionaryMode::Never) },
        Config { name: "L3 vpf=8192 auto-dict", options: ZstdOptions::new(3).with_values_per_frame(8192) },
        Config { name: "L3 one frame nodict", options: ZstdOptions::new(3).with_dictionary(DictionaryMode::Never) },
        Config { name: "L6 one frame nodict", options: ZstdOptions::new(6).with_dictionary(DictionaryMode::Never) },
        Config { name: "L6 one frame auto-dict", options: ZstdOptions::new(6) },
        Config { name: "L9 one frame nodict", options: ZstdOptions::new(9).with_dictionary(DictionaryMode::Never) },
        Config { name: "L9 one frame auto-dict", options: ZstdOptions::new(9) },
    ];

    for path in &paths {
        let chunks = parquet_to_vortex_chunks(path.clone()).await?;
        let parquet_size = std::fs::metadata(path)?.len();
        println!("\n### {} ({:.1} MiB parquet)", path.display(), parquet_size as f64 / 1048576.0);
        println!("{:<30} {:>14} {:>8} {:>10} {:>10}", "config", "vortex bytes", "vs today", "write s", "scan s");
        let mut baseline = 0u64;
        for config in &configs {
            let scheme: &'static string::ZstdScheme =
                Box::leak(Box::new(string::ZstdScheme::new(config.options)));
            let builder = BtrBlocksCompressorBuilder::default()
                .with_compact()
                .exclude_schemes([string::ZstdScheme::DEFAULT.id()])
                .with_new_scheme(scheme);
            let strategy = WriteStrategyBuilder::default()
                .with_btrblocks_builder(builder)
                .build();

            let mut buf = Vec::new();
            let start = Instant::now();
            SESSION
                .write_options()
                .with_strategy(strategy)
                .write(&mut std::io::Cursor::new(&mut buf), chunks.clone().into_array().to_array_stream())
                .await?;
            let write_s = start.elapsed().as_secs_f64();
            let size = buf.len() as u64;
            if baseline == 0 {
                baseline = size;
            }

            let data = bytes::Bytes::from(buf);
            let start = Instant::now();
            let stream = SESSION.open_options().open_buffer(data)?.scan()?.into_array_stream()?;
            futures::pin_mut!(stream);
            let mut rows = 0usize;
            while let Some(chunk) = stream.next().await {
                rows += chunk?.len();
            }
            let scan_s = start.elapsed().as_secs_f64();
            assert!(rows > 0);
            println!("{:<30} {:>14} {:>7.1}% {:>10.2} {:>10.2}",
                config.name, size,
                (size as f64 / baseline as f64 - 1.0) * 100.0,
                write_s, scan_s);
        }
    }
    Ok(())
}
