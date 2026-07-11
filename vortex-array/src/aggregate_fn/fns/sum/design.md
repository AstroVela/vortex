# SQL semantics for the `Sum` aggregate

## Goal

`Sum` of zero valid values yields **null** (SQL `SUM`, ISO 9075-2 §10.9), everywhere:
scalar `sum()`, grouped aggregation, and `list_sum` — replacing today's contract of
"all-invalid sums to zero". Overflow keeps yielding null (unchanged). This matches
arrow-rs (`sum` returns `None` for empty/all-null), DuckDB (`SumState.isset` →
`ReturnNull`), and DataFusion (`SumAccumulator { sum: Option }`).

## Why this is not a one-line change

The current partial state is a single nullable sum value, and **null is already taken**:
a null partial means overflow and must *poison* merges (`combine(null, x) = null`), while
"empty" must be the *identity* (`combine(empty, x) = x`). One symbol cannot be both, and
the partial value crosses two persistence boundaries where merging happens:

1. **Array/layout statistics**: `Stat::Sum` is serialized as a dedicated flatbuffer field
   (`vortex-array/src/stats/flatbuffers.rs`) and merged as final scalars via
   `StatsSet::merge_sum` (`stats/stats_set.rs`, `checked_add`).
2. **Zoned layouts**: `vortex-layout/src/layouts/zoned/builder.rs` writes
   `accumulator.partial_scalar()` per zone; readers fold these back through
   `combine_partials` (see the "read legacy stat" path in
   `vortex-array/src/aggregate_fn/accumulator.rs`).

Additionally, `Accumulator::accumulate` short-circuits through cached `Stat::Sum` values
(`sum/mod.rs`, `try_accumulate`), so a batch's cached monoid sum can substitute for
actually reading the batch. Any design must keep these paths correct and mutually
consistent.

## Design options

### A. Struct partials (recommended)

Change `SumPartial`'s wire form to `Struct { sum: <widened, nullable>, seen: bool }`,
where `seen` records whether at least one valid value contributed (DuckDB's `isset`;
`sum = null` still means overflow). A valid-*count* would be redundant — `NullCount` is
already persisted alongside the sum in every stats pipeline, so any boundary that needs
a count derives it from existing stats — and the bool keeps the grouped kernel's partial
column bit-packed (1 bit per group instead of 64).

The algebra becomes unambiguous and total:

- identity: `{sum: 0, seen: false}`
- combine: `{sum: a.sum ⊕ b.sum, seen: a.seen ∨ b.seen}` where `⊕` is
  null-poisoning checked add
- finalize: `!seen → null`, else `sum` (null if overflow)
- checks: `combine(overflow, empty) = {null, true} → null`;
  `combine(empty, empty) = {0, false} → null`; `combine(v, empty) = {v, true} → v`

(If the field is named `all_null` instead, the combine flips to AND; `seen`/OR avoids
the double negative.)

Consequences:

- **Grouped machinery needs no special-casing at all.** The per-group fallback
  accumulator naturally produces `seen = false` for empty/all-null groups; `finalize`
  nulls them. The grouped sum kernel emits a struct column whose `seen` field is
  exactly the has-valid-element bitmap it already computes (via
  `BitBuffer::count_range` over the materialized element mask).
  `null_for_empty_groups`, `mask_empty_lists`, and every fix-up variant dissolve.
- **`list_sum_impl` becomes accumulate + finish**, no post-pass.
- **Mean** (`Combined<Sum, Count>`): its empty result changes from `NaN` to `null`
  (SQL-correct; matches DuckDB and Postgres). Its `finalize_scalar` comment and tests
  must be updated deliberately.
- **Persistence formats change**, which is the cost:
  - zoned stats: new files write struct partials; the existing legacy-stat shim in
    `accumulator.rs` (which already casts old stat dtypes) grows one rule: a legacy
    plain-primitive partial is read as `{sum: v, seen: true}` (preserves old behavior
    for old files, including treating legacy null as overflow-poison).
  - array stats flatbuffer: either keep the `sum` field as the *finalized* value and add
    a sibling field/valid-count consultation, or version the field. **This needs a
    format-owner decision** — the one open question to settle before implementing.
    Note `NullCount` is already persisted alongside, so "empty" is derivable at read
    time as `null_count == len` without any new field, which may make the flatbuffer
    change unnecessary: keep `Stat::Sum` as the monoid scalar on disk and apply
    `finalize` semantics at the read boundary using `NullCount`.
- Forward-compat: old readers encountering new zoned struct partials will fail the
  partial-dtype ensure and should degrade to "stat unavailable" rather than error —
  verify and test.

### B. In-memory `seen` bit only (fallback option, no format change)

`SumPartial` gains a non-serialized `seen: bool`; `finish`/`finalize_scalar` null when
unseen; persisted forms stay plain monoid sums; every stat-fed path (`try_accumulate`
short-circuit, zoned reads, `merge_sum`) marks `seen` conservatively and boundaries that
present SQL SUM consult `NullCount` (`null_count == row_count → null`).

Cheaper, but the grouped fallback still can't express empty-group nulls through
`to_scalar` (partials must stay monoid for the zoned writer), so the grouped machinery
still needs an explicit empty-group mechanism — i.e. this degenerates into the
`null_for_empty_groups` design we already have stashed, plus boundary guards. Choose B
only if the format-owner conversation for A stalls.

## Plan (assuming A)

1. **Core algebra** (`aggregate_fn/fns/sum/mod.rs`)
   - `SumPartial { sum: Option<SumState>, seen: bool }`; update `empty_partial`,
     `combine_partials`, `to_scalar`, `reset`, `is_saturated`, `finalize`,
     `finalize_scalar`; `partial_dtype` → struct; `return_dtype` unchanged.
   - `try_accumulate` stat short-circuit: derive `seen` from the batch's
     `NullCount`/len when consuming a cached `Stat::Sum`; if unavailable, fall through
     to real accumulation instead of guessing.
   - Rustdoc: sum of zero valid values is null; overflow is null; NaN handling
     unchanged (NaNs are valid values — an all-NaN sum under `skip_nans` is `0`, and
     `list_sum`'s `test_all_nan_list_sums_to_zero` pins this).
2. **Grouped paths** (`accumulator_grouped.rs`, `fns/sum/grouped.rs`, `fns/count/grouped.rs`)
   - Kernel emits `{sum, seen}` struct rows (`seen` via `BitBuffer::count_range > 0`
     over the materialized element mask — never `Mask::slice` per group).
   - Fallback: no changes beyond the struct builder working generically.
   - `finalize(states)`: struct → nullable sums in one pass (validity intersection).
3. **`list_sum`** — delete `mask_empty_lists` and the grouped-view plumbing; the
   existing 21 tests must pass unchanged (they pin the public semantics).
4. **Stats boundaries**
   - Legacy shim in `accumulator.rs` for old zoned partials.
   - `stats_set.rs::merge_sum`: decide monoid-on-disk vs struct (see A) and align.
   - `vortex-datafusion/src/convert/stats.rs`: export `sum_value` as null/absent when
     `null_count == row_count` — fixes the pre-existing wrong-results edge where
     DataFusion could answer `SELECT SUM(x)` as `0` for an all-null column.
   - `sum()`'s `Stat::Sum` cache read: apply the same `NullCount` guard.
5. **Mean** — accept null-for-empty (update `finalize_scalar` comment + tests).
6. **Tests**
   - Algebra: combine across empty/value/overflow in all orders; struct round-trip.
   - Grouped: empty/all-null groups null on kernel and fallback (bool elements) paths.
   - Legacy: old-format zoned partial read shim; forward-compat degradation.
   - End-to-end: file with an all-null zone → correct file-level sum; all-null file →
     null; DataFusion stats export edge.
   - Existing `list_sum` suite passes unmodified.
7. **Benches** — `aggregate_grouped` + `list_sum` A/B against develop (expect: nullable
   grouped cases improve from deleting the fix-up; small kernel cost for the count
   column; quantify). The stashed zero-fill kernel rewrite (stash@{0} on the list-sum
   worktree) composes with this and can follow as its own PR.

## Open questions (resolve before/while implementing)

1. Flatbuffer `Stat::Sum` field: keep monoid-on-disk + `NullCount` at boundaries, or
   version the field to carry count? (Recommend monoid-on-disk — no format change to
   array stats; only zoned partials change shape, where a legacy shim already exists.)
2. Forward-compat policy for old readers on new zoned partials: skip-stat vs error.
3. Mean: confirm null-for-empty is wanted (SQL says yes).
