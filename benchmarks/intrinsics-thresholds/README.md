# Intrinsics threshold matrix

This standalone binary measures scalar and explicitly selected intrinsic kernels used by Vortex. It
has no Vortex crate dependency, so a release binary can be copied to the target machine without
bringing dispatch code or benchmark data with it.

```bash
cargo run --release -p intrinsics-thresholds -- --min-time-ms 20
cargo run --release -p intrinsics-thresholds -- --format markdown > thresholds.md
```

`--format` accepts `matrix` and `markdown`. The default `matrix` format prints one block per case
for a terminal reader: an implementation per row, a measured length per column, and the
intrinsic/scalar ratio in each cell. Ratios below 1.00 are green on a terminal. The `markdown`
format prints the full pipe table with nanoseconds per call, for a pull request or an issue.

The matrix reports the intrinsic/scalar ratio for every input size. The crossover column is the
first size from which the intrinsic wins at every larger measured size.
Implementations whose required CPU feature is missing are omitted rather than replaced by a
fallback. Run on an otherwise idle machine and retain the complete matrix: crossovers can be
non-monotonic around vector-width and cache boundaries.

## Audited coverage

The inventory was produced by searching production Rust sources for `std::arch`, `core::arch`,
`#[target_feature]`, and runtime CPU-feature detection. Wrapper and dispatch-only files do not need
their own case; each distinct intrinsic loop does.

| benchmark case | production pattern | implementations |
| --- | --- | --- |
| `popcount` | `vortex-buffer/src/bit/count_ones.rs` | AVX2, AVX-512 VPOPCNTDQ, NEON |
| `select-chunk-scan` | `vortex-buffer/src/bit/select.rs` | AVX-512 VPOPCNTDQ, NEON via `popcount` |
| `pack-bools` | `vortex-buffer/src/bit/pack.rs` | SSE2, AVX2, AVX-512 BW, NEON |
| `select-word` | `vortex-buffer/src/bit/select.rs`, `vortex-mask/src/intersect_by_rank.rs` | BMI2 PDEP |
| `extract-words` | `vortex-array/src/arrays/bool/compute/filter.rs` | BMI2 PEXT |
| `deposit-words` | `vortex-mask/src/intersect_by_rank.rs` | BMI2 PDEP |
| `take-u32-random` | `vortex-array/src/arrays/fixed_width/take/avx2` | AVX2 gather |

The fixed-width take implementation has many type-specialized instruction sequences. The u32 case
measures the common gather loop and its scalar crossover; use the resulting threshold as a starting
point, not as proof that every value/index-width pair has the same crossover. Likewise, PEXT/PDEP
performance varies materially by microarchitecture and mask density. This tool answers the CPU
kernel question; end-to-end dispatch policy still needs validation with the crate benchmarks.
