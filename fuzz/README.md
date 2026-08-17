# Vortex Fuzz

This crate contains general fuzzing infrastructure and tooling for all public components of Vortex.

## Setup

Currently, the only thing required to run the fuzzing targets is [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz)

## Reproduce crash from CI

In the case of a crash in the nightly run, you can download the crash artifact and run `cargo-fuzz` with the exact same
input with the command `cargo fuzz run array_ops <path/to/artifact>` or `cargo fuzz run file_io <path/to/artifact>`

### ASAN

If there are any linking (on macOS) then run `cargo fuzz run --dev --sanitizer=none ...`. `--dev` runs the fuzzer in dev
profile.

### Replay without cargo-fuzz

To replay an `array_ops` input on stable Rust, without `cargo-fuzz` or a nightly toolchain:

```bash
cargo run -p vortex-fuzz --example replay --release -- <path/to/artifact>
```

The example decodes the input the same way `libfuzzer-sys` does and runs the action directly, so it
reproduces the panic under a normal debugger and works in environments where `cargo-fuzz` cannot
build. It replays a single input; use `cargo fuzz run` for actual fuzzing.
