// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Arc;

    use futures::future::try_join;
    use tempfile::NamedTempFile;
    use vortex_buffer::Alignment;
    use vortex_compio::CompioFileReadAt;
    use vortex_compio::CompioRuntime;
    use vortex_error::VortexResult;
    use vortex_file::SegmentSpec;
    use vortex_file::segments::FileSegmentSource;
    use vortex_file::segments::RequestMetrics;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_layout::segments::SegmentId;
    use vortex_layout::segments::SegmentSource;
    use vortex_metrics::DefaultMetricsRegistry;

    const DATA: &[u8] = b"completion-based Vortex reads";

    #[test]
    fn coalescing_driver_reads_with_compio() -> VortexResult<()> {
        let mut temp = NamedTempFile::new()?;
        temp.write_all(DATA)?;
        temp.flush()?;

        let runtime = CompioRuntime::new()?;
        let compio_handle = runtime.compio_handle();
        let handle = runtime.handle();
        runtime.block_on(async move {
            let reader = CompioFileReadAt::open(temp.path(), compio_handle).await?;
            let metrics = DefaultMetricsRegistry::default();
            let source = FileSegmentSource::open(
                Arc::from([
                    SegmentSpec {
                        offset: 0,
                        length: 10,
                        alignment: Alignment::none(),
                    },
                    SegmentSpec {
                        offset: 17,
                        length: 6,
                        alignment: Alignment::new(4096),
                    },
                ]),
                reader,
                handle,
                RequestMetrics::new(&metrics, vec![]),
            );

            let (first, second) = try_join(
                source.request(SegmentId::from(0)),
                source.request(SegmentId::from(1)),
            )
            .await?;
            assert_eq!(first.as_host().as_slice(), b"completion");
            assert_eq!(second.as_host().as_slice(), b"Vortex");
            assert!(Alignment::new(4096).is_ptr_aligned(second.as_host().as_ptr()));
            VortexResult::Ok(())
        })
    }
}
