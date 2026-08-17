// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::Barrier;
use std::time::Duration;
use std::time::Instant;

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_cuda::CudaSession;
use vortex_cuda::cuda_session;

const TOTAL_LAUNCHES: usize = 4096;
const REPEATS: usize = 5;

#[derive(Clone, Copy)]
struct Measurement {
    enqueue: Duration,
    drain: Duration,
    stream: Duration,
}

fn main() -> VortexResult<()> {
    let mut ctx = CudaSession::create_execution_ctx(&cuda_session())?;
    let function = Arc::new(ctx.load_function("constant_numeric", &[PType::U32])?);

    println!("elements,host_work_us,threads,launches,enqueue_us,stream_us,drain_us,launches_per_s");
    for elements in [1usize, 4096, 1 << 20] {
        for threads in [1usize, 2, 4, 8] {
            report(
                &mut ctx,
                Arc::clone(&function),
                elements,
                Duration::ZERO,
                threads,
                TOTAL_LAUNCHES,
            )?;
        }
    }
    for host_work_us in [10u64, 50] {
        for threads in [1usize, 2, 4, 8] {
            report(
                &mut ctx,
                Arc::clone(&function),
                1,
                Duration::from_micros(host_work_us),
                threads,
                1024,
            )?;
        }
    }
    Ok(())
}

fn report(
    ctx: &mut vortex_cuda::executor::CudaExecutionCtx,
    function: Arc<cudarc::driver::CudaFunction>,
    elements: usize,
    host_work: Duration,
    threads: usize,
    total_launches: usize,
) -> VortexResult<()> {
    let mut measurements = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        measurements.push(measure(
            ctx,
            Arc::clone(&function),
            elements,
            host_work,
            threads,
            total_launches,
        )?);
    }
    measurements.sort_by_key(|measurement| measurement.drain);
    let median = measurements[REPEATS / 2];
    let launches_per_second = total_launches as f64 / median.enqueue.as_secs_f64();
    println!(
        "{elements},{},{threads},{total_launches},{},{},{},{launches_per_second:.0}",
        host_work.as_micros(),
        median.enqueue.as_micros(),
        median.stream.as_micros(),
        median.drain.as_micros(),
    );
    Ok(())
}

fn measure(
    ctx: &mut vortex_cuda::executor::CudaExecutionCtx,
    function: Arc<cudarc::driver::CudaFunction>,
    elements: usize,
    host_work: Duration,
    threads: usize,
    total_launches: usize,
) -> VortexResult<Measurement> {
    let stream = ctx.stream().clone();
    let outputs = (0..threads)
        .map(|_| ctx.device_alloc::<u32>(elements))
        .collect::<VortexResult<Vec<_>>>()?;
    ctx.synchronize_stream()?;

    let cuda_context = stream.context();
    let start_event = cuda_context
        .new_event(Some(CU_EVENT_BLOCKING_SYNC))
        .map_err(|error| vortex_err!("create start event: {error}"))?;
    let end_event = cuda_context
        .new_event(Some(CU_EVENT_BLOCKING_SYNC))
        .map_err(|error| vortex_err!("create end event: {error}"))?;
    start_event
        .record(&stream)
        .map_err(|error| vortex_err!("record start event: {error}"))?;

    let start_barrier = Arc::new(Barrier::new(threads + 1));
    let done_barrier = Arc::new(Barrier::new(threads + 1));
    let exit_barrier = Arc::new(Barrier::new(threads + 1));
    let launches_per_thread = total_launches / threads;
    let array_len = elements as u64;
    let config = LaunchConfig {
        grid_dim: (u32::try_from(elements.div_ceil(256))?, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch_error = None;
    std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(threads);
        for (worker, mut output) in outputs.into_iter().enumerate() {
            let stream = stream.clone();
            let function = Arc::clone(&function);
            let start_barrier = Arc::clone(&start_barrier);
            let done_barrier = Arc::clone(&done_barrier);
            let exit_barrier = Arc::clone(&exit_barrier);
            workers.push(scope.spawn(move || -> VortexResult<()> {
                start_barrier.wait();
                for launch_index in 0..launches_per_thread {
                    let host_work_start = Instant::now();
                    while host_work_start.elapsed() < host_work {
                        std::hint::spin_loop();
                    }
                    let value = u32::try_from(worker * launches_per_thread + launch_index)?;
                    let mut launch = stream.launch_builder(&function);
                    launch.arg(&mut output).arg(&value).arg(&array_len);
                    unsafe {
                        launch
                            .launch(config)
                            .map_err(|error| vortex_err!("launch constant kernel: {error}"))?;
                    }
                }
                done_barrier.wait();
                exit_barrier.wait();
                Ok(())
            }));
        }

        let wall_start = Instant::now();
        start_barrier.wait();
        done_barrier.wait();
        let enqueue = wall_start.elapsed();
        if let Err(error) = end_event.record(&stream) {
            launch_error = Some(vortex_err!("record end event: {error}"));
        }
        if launch_error.is_none()
            && let Err(error) = end_event.synchronize()
        {
            launch_error = Some(vortex_err!("synchronize end event: {error}"));
        }
        let drain = wall_start.elapsed();
        exit_barrier.wait();
        for worker in workers {
            match worker.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => launch_error = Some(error),
                Err(_) => launch_error = Some(vortex_err!("dispatch worker panicked")),
            }
        }

        let stream = start_event
            .elapsed_ms(&end_event)
            .map(|milliseconds| Duration::from_secs_f32(milliseconds / 1000.0))
            .map_err(|error| vortex_err!("measure stream events: {error}"));
        match (launch_error.take(), stream) {
            (Some(error), _) => Err(error),
            (None, Err(error)) => Err(error),
            (None, Ok(stream)) => Ok(Measurement {
                enqueue,
                drain,
                stream,
            }),
        }
    })
}
