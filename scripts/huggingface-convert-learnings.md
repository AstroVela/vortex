# Hugging Face conversion pipeline learnings

This document records evidence from large `huggingface-convert.py` runs and turns it into a
post-run improvement plan. Keep the running converter unchanged: finish or stop it cleanly, retain
its plan, checkpoint, log, and reports, then make and benchmark changes against the same workload.

## Reference run

The first reference run converts `HuggingFaceFW/fineweb` at `sample/350BT` into both `vortex` and
`vortex-compact` in `vortex-data/fineweb`.

- Source revision: `9bb295ddab0e05d785b879661af7260fed5140fc`
- Destination baseline: `78d65b878d9f326612dc355e2c96c7de35c516de`
- Input: 510 Parquet files and 1,061,360,917,731 bytes
- Work: 1,020 output actions in 22 planned commits
- Limits: 100 source files or 100 GB downloaded, and 100 output files or 100 GB queued for upload
- Initial concurrency: 2 downloads, 20 single-core transcodes, and 2 uploads
- Durability: checkpointed apply, a restarting user service, user lingering, and a reboot resume unit

At 24 minutes, the checkpoint reported 45 downloads (96.6 GB), 42 completed shards, and 42
outputs in each format. Aggregate effective source conversion was about 62.5 MB/s. Mean per-task
throughput was 23.2 MB/s for `vortex` and 28.5 MB/s for `vortex-compact`. Live aggregate upload
ranged from 109 to 145 MB/s. The upload stage could drain output faster than conversion produced
it, so the sustained bottleneck was conversion admission/execution rather than network upload.

These numbers are an interim snapshot. Add the final `summary.json`, total elapsed time, bytes
downloaded, bytes committed, retries, failures, and final repository revision after the run.

## What worked

- The immutable plan makes the source, destination baseline, exact actions, target paths, and
  commit boundaries reviewable before any writes.
- The checkpoint distinguishes downloaded, converting, converted, queued, uploading, preuploaded,
  and committed work well enough to resume without reconverting durable local outputs.
- A global CPU-slot queue and `taskset` keep each `vx` process on one core.
- Download and upload byte/file bounds prevent unconstrained local growth.
- Xet preupload overlaps conversion with transfer, and planned batches keep commits bounded by
  target and represented source bytes.
- The production process survived independently of the terminal and had zero restarts during the
  observed interval.

## Observed limitations

### Wrapped Xet timeouts were incorrectly treated as permanent

The reference run converted every output but stopped with 96 paths absent from the repository. One
preupload raised `RuntimeError: Internal error: timed out reading request body`. The retry helper
recognized typed transport exceptions and HTTP status codes, but not a transient timeout wrapped in
a generic Xet `RuntimeError`, so it failed immediately and prevented the final planned Vortex batch
from flushing. A fresh destination diff found 46 missing Vortex files after the compact tail had
committed. Retry classification must inspect exception chains and known transient Xet messages,
use bounded jittered backoff, retain local output until remote commit, and verify the destination
after completion. A service restart remains a final recovery layer, not the primary retry policy.

### Queue state is inferred rather than reported

The CLI prints configured limits but not live queue occupancy, active counts, wait time, or byte
watermarks. Operators currently reconstruct queues from `checkpoint.json`, process listings, disk
usage, and network counters. Checkpoint status also describes durability, not always in-memory queue
membership, so a count such as "download complete" is not the same as "waiting to transcode."

### Plan-order waiting can cause head-of-line blocking

Downloads run concurrently, but the main loop consumes `download_futures` in selection order and
calls `.result()` for that specific shard. A slow earlier download can delay admission of already
downloaded later shards. A completion queue should admit whichever download finishes first while
preserving the planned ordinal on the work item for upload batching.

### Limits and concurrency are conflated

The download executor is sized from `download_buffer_files`, and the upload executor and adaptive
upload ceiling are also sized from `upload_buffer_files`. Buffer capacity, executor thread count,
initial concurrency, and maximum concurrency should be independent controls. A 100-file safety
buffer should not imply permission to create 100 transfer threads.

### Adaptive control uses noisy per-file feedback

Concurrency changes after individual file completion. File size, Xet chunk reuse, hashing,
finalization, disk contention, and server latency can make one file's apparent rate misleading.
The controller should use rolling aggregate bytes per wall-clock second, queue pressure, CPU idle,
and disk/network utilization. It should have minimum and maximum concurrency, hysteresis, and a
cooldown period so it does not oscillate.

### Conversion capacity is not kept consistently full

The run allowed 20 single-core transcodes but snapshots ranged from zero to ten active `vx`
processes. Some variation is expected at queue transitions, but sustained idle CPU while unplanned
work remains indicates admission or backpressure policy, not a lack of CPU capacity. Measure CPU
slot utilization and time spent waiting on download, upload capacity, and disk IO before changing
worker counts.

### Upload has two distinct completion states

Xet preupload transfers file data, but files do not appear in the repository tree until the planned
commit is created. The reference run had 78 preuploaded files and zero committed files at one
snapshot. Progress and status output must report `preuploaded` and `committed` separately, along
with the next batch's required and ready file counts and bytes.

### Live output is difficult to consume

Concurrent Hugging Face progress bars make the persistent log noisy. Reports are most useful at the
end, while long jobs need machine-readable live metrics. Logs should be line-oriented when stdout
is not a terminal, and a periodically replaced `status.json` should summarize each stage.

## Post-run evolution plan

Implement changes in small steps and replay a fixed 10-file local-sink test before another remote
run. Preserve plan schema compatibility or version it explicitly.

1. Add observability without changing scheduling.
   - Write `status.json` every 5–10 seconds.
   - Report waiting, active, succeeded, failed, and retried items and bytes for every stage.
   - Record aggregate and rolling download, conversion, and upload throughput.
   - Record CPU-slot utilization, queue wait time, buffer high-water marks, and estimated completion.
   - Disable interactive progress bars in detached mode.
2. Separate configuration dimensions.
   - Add explicit initial and maximum download/upload concurrency.
   - Keep file and byte buffer limits independent from executor sizes.
   - Validate that configured worst-case disk use fits available space before apply.
3. Replace plan-order download consumption with completion-driven admission.
   - Consume completed downloads with `as_completed` or a bounded completion queue.
   - Carry the immutable plan ordinal through conversion and upload.
   - Test a deliberately slow first download to prove later work reaches transcode workers.
4. Make backpressure resource-aware.
   - Reserve bytes when a task is admitted and release them at a documented lifecycle point.
   - Bound downloaded source, converted local output, queued upload, and preuploaded-uncommitted
     output independently.
   - Prefer conversion when CPU is idle and source is available; prefer upload when local output
     approaches its high-water mark.
5. Replace adaptive heuristics with aggregate control.
   - Use rolling 30–60 second throughput and resource utilization.
   - Increase concurrency only when throughput improves and the relevant resource is not saturated.
   - Back off on errors, latency spikes, or throughput regression, with hysteresis and cooldown.
6. Harden upload batching and recovery.
   - Expose batch readiness and commit state in the checkpoint and live status.
   - Test a crash after preupload but before commit, during commit, and after the remote commit but
     before the local checkpoint update.
   - Verify concurrent destination changes produce a clear stale-plan failure and safe re-plan path.
   - Treat wrapped Xet request-body timeouts and internal transport errors as transient, with a
     regression test using the exact production error text.
7. Add a benchmark matrix.
   - Compare 1, 2, 4, and 8 transfer workers with 8, 12, 16, and 20 transcode slots.
   - Run with a dummy sink, a throttled sink, and real Xet.
   - Select defaults from end-to-end elapsed time and resource headroom, not peak stage throughput.

## Acceptance criteria

- A detached run exposes exact queue items and bytes without inspecting processes or inferring from
  checkpoint internals.
- No CPU slot is idle for more than the reporting interval while convertible source is available,
  unless a configured disk or upload high-water mark is active.
- Download completion order cannot block ready conversion work.
- Transfer concurrency never exceeds its explicit maximum and does not derive from buffer length.
- Restart at every tested lifecycle boundary neither loses work nor creates duplicate repository
  paths or commits.
- A 10-file local replay and a small real-Xet replay produce identical planned paths and hashes to
  the current implementation.
- The final report accounts for all planned actions as skipped or committed and verifies them
  against the destination repository revision.

## Final-run record

- Start: 2026-08-27 18:56:24 UTC
- Remote repair verified: 2026-08-28, after the original tail was stopped and re-planned
- Final destination revision: `b04ab34c10c8a9e1a88767ea951d9a81e006277d`
- Source: 510 files and 1,061,360,917,731 bytes
- `vortex`: 756,043,634,688 bytes, ratio 0.712334
- `vortex-compact`: 655,931,845,640 bytes, ratio 0.618010
- Peak observed original working directory: about 215 GB; at least 4.5 TB remained free
- Original failure: one wrapped Xet request-body timeout; zero systemd restarts
- Repair: fresh destination diff with 46 Vortex creates and 974 skips; zero service restarts
- Final remote verification: 1,020/1,020 planned paths present, zero size mismatches, zero extras
- Retry-hardened local validation: 10 source files, 20 outputs, 28,550,034,024 sink bytes,
  372 seconds, zero failures, and all queues empty at the final status snapshot
