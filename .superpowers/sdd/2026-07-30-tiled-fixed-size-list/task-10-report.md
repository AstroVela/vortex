# Task 10: tiled fixed-size-list differential property testing

## Implementation

Completed the local differential property harness in `fuzz/`.

- Added the bounded `FuzzTiledFsl` generator and action sequence model.
  The generated canonical fixed-size lists cover all primitive physical types,
  independently nullable elements and rows, zero-width lists, empty arrays,
  and independently bounded tile geometry.
- Added the canonical-oracle runner.  It encodes a canonical fixed-size list,
  checks equality and scalar probes, verifies tile count calculations, and
  independently derives every tile's row-major physical positions before
  comparing values only for valid outer rows.
- Added composed `CheckTiles`, scalar, slice, take, and reconstruction actions.
  Degenerate empty results stop a sequence after their logical equality check.
- Added exactly three hand-built smoke fixtures: empty zero-width `i32`;
  nullable `u16` with duplicated nullable take; and independently nullable
  65-by-129 `f32` data with a boundary-crossing slice and reverse take.
- Added the native `tiled_fsl` target and README commands for a normal run and
  replaying a single artifact.

## Initial RED evidence

The prior agent's terminal output was not retained in this checkout, so the
requested original RED compilation failure cannot be recovered.  At takeover,
the smoke fixture and harness implementation were already present; the first
observed local smoke execution passed.  The recoverable initial failure from
this session was warning-denied Clippy: `expect_used` in the bounded geometry
and take conversion paths, plus `result_large_err` on the new public helper
and test.  Those harness-local findings were resolved without changing its
behavior.

## Verification

Passed:

- `cargo +nightly fmt --all`
- `git diff --check`
- `cargo nextest run -p vortex-fuzz deterministic_tiled_fsl_smoke`
  (three deterministic fixtures exercised by one passing test)
- `cargo +nightly clippy -p vortex-fuzz --all-targets --no-deps -- -D warnings`
- `cargo +nightly build -p vortex-fuzz --bin tiled_fsl --features native`

The full-workspace warning-denied Clippy command was also attempted.  It is
blocked by pre-existing warnings in `vortex-edition` and `vortex-array`, not by
this harness; the focused no-dependencies command above verifies the changed
crate under the same warning policy.

The requested `cargo +nightly fuzz build --dev --sanitizer=none tiled_fsl` and
10,000-input `cargo +nightly fuzz run` command could not run because this
checkout has no `cargo-fuzz` subcommand installed (`error: no such command:
fuzz`).  No install was attempted.  The directly built target was started with
the equivalent `-runs=10000 -max_len=4096` arguments, but the local execution
runner stopped that debug process after it reported progress through 2,048
inputs; therefore this report does not claim a completed 10,000-input campaign.
