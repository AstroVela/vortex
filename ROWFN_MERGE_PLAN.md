<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Plan: land the row framework as a PR series

Working note, branch-only, like `SCALAR_FN_HANDOFF.md`. This is the execution plan for splitting
this branch (`ct/row-fn` at `dfebd42115`, the prototype from #9128) into reviewable PRs against
`develop`. The handoff records the recommended landing order and asserts the seam supports it.
This note turns that order into per-PR file manifests, develop reconciliations, and gates, and
records the one extraction that was verified by building it rather than assumed.

## What was verified

The framework layer is additive. On `develop` at `a1057db58e`, copying these paths verbatim from
this branch compiles and passes all 68 row tests, with none of the numeric, tensor, or geo
migrations present:

- `vortex-array/src/scalar_fn/row/` (the whole module, tests included)
- `vortex-array/src/scalar_fn/mod.rs` (module wiring plus the "Choosing a trait" doc)
- `vortex-array/benches/row_fn_executor.rs` and `benches/strict_validity.rs`, with their
  `Cargo.toml` registrations

One reconciliation was required. Do not take this branch's `scalar_fn/vtable.rs`: #9228 changed
`fmt_sql` to take `&dyn ExprDisplay` on develop after this branch last merged, and this branch's
only real change to that file is the `Default` derive on `EmptyOptions`. Apply the derive to
develop's copy. The row blanket vtable does not implement `fmt_sql`, so nothing else collides.

Verified with `cargo check -p vortex-array --lib`, `cargo test -p vortex-array --lib
scalar_fn::row` (68 passed), and `cargo check` of both benches.

## Develop drift to reconcile

Develop has 21 commits this branch has not merged. Three touch files this branch changes:

- **#9228, bound expression formatting.** Resolved by the `vtable.rs` rule above.
- **#9210, vectorized checked multiplication.** Rewrote the hand-written checked kernels that the
  numeric port (PR 3) partly deletes. Resolve by taking develop's `checked.rs` where decimal still
  uses it and this branch's `row.rs` for primitive execution. The epic's mul benchmark numbers
  predate #9210 and flatter the port, so PR 3 must re-measure against a develop base that includes
  it.
- **#9224, ChunkedArray take routing.** Overlaps only in `vortex-array/Cargo.toml` bench
  registrations. Trivial.

`claude/collapse-checked-arith-macros`, the develop-side macro collapse the handoff mentions, was
never pushed and does not exist on origin. Decide during PR 3 whether the `CheckedArithmetic`
bodies that remain after the port still justify the collapse, and recreate it then if so.

## The PR series

Order: PR 1, then PR 2, then PRs 3, 4, and 5 in any order. PR 3 depends only on PR 1, since
primitive elements are dense-safe and the numeric port never reaches the valid-only strategies.
PR 5 requires PR 2, since costly geometry decode is the case the adaptive selection exists for.
Run PR 4 after PR 2 for the same reason. Each PR is individually revertible.

Author each PR from develop, not by rebasing this branch. Branch from develop, `git checkout
ct/row-fn -- <paths>` for the manifest, apply the named reconciliations, trim to the PR's scope,
and run its gates. This branch stays frozen as the reference implementation: it is publicly linked
from #9128, so do not rewrite or delete its history. Diff each extraction against this branch
before opening the PR, so that any divergence from the prototype is a deliberate edit rather than
a transcription error.

Every PR body links its tracking issue (#9129 and #9130 for PRs 1 and 2, "Progress towards #9128"
for the rest), and the epic's implementation history gains a checklist entry as each lands.

### PR 0: independent pieces (optional, any time)

Two changes have no dependency on the framework and pin existing develop behavior. Both document
why `like` and `list_length` are excluded from `RowFn`:

- `vortex-array/benches/like.rs`: the `like_per_row_distinct_patterns` benchmark that isolates
  pattern compilation from matching.
- `vortex-array/src/scalar_fn/fns/list_length.rs`: the test that a non-nullable fixed-size list
  length stays a `ConstantArray`.

About 55 lines. These can also ride along with PR 1 if a separate PR is not worth the churn.

### PR 1: the `RowFn` API and lifting, dense and filter only

Tracking issues: #9129 (API) and #9130 (execution).

Manifest:

- `vortex-array/src/scalar_fn/row/`, minus the branch-and-skip execution listed under PR 2
- `vortex-array/src/scalar_fn/mod.rs` wiring and module doc
- The `Default` derive on `EmptyOptions`, applied to develop's `vtable.rs`
- `vortex-array/benches/row_fn_executor.rs` plus its registration
- `row/tests/` except `null_strategies.rs` (1,816 lines of tests)

The full API surface lands here unchanged, including `InputElement::DENSE_SAFE`,
`decode_null_tolerant`, `FILTERED_DECODE_COST`, and `OutputSink::SUPPORTS_SKIPPED_ROWS`, so no
signature changes between steps. Only the executor is reduced: `ValidOnly` always
filter-and-scatters, and dense, dense-with-retry, and the deferred-error retry all land here.

This is the one extraction that is surgery rather than file copying. The `branch` closure threads
through `RowFnExecutor::execute`, `execute_filtered`, `execute_branched`, and
`branch_beats_filter` in `lift.rs`, the closure construction in `row/vtable.rs`, and the
`NullStrategy` and `execute_row_fn_with_strategy` exports in `row/mod.rs`. Removing it touches
roughly 300 to 400 lines. PR 2 must restore exactly the prototype's code, which the
diff-against-prototype step checks. If the surgery proves error-prone, the fallback is landing
PRs 1 and 2 as one PR: that combined extraction is the one verified above.

Gates: the row test suite, `cargo clippy --all-targets --all-features -p vortex-array`, nightly
fmt, `cargo test --doc -p vortex-array`. No production path moves, so CodSpeed only gains the new
`row_fn_executor` names.

Size: roughly 2,300 lines of implementation and docs, 1,816 of tests, 419 of benchmark.

### PR 2: branch-and-skip and adaptive selection

Tracking issue: #9130.

Manifest:

- The branch-and-skip execution carved out of PR 1: `execute_branched`, `branch_beats_filter` and
  its thresholds, the `branch` closure plumbing, and the null-tolerant decode path through
  `element/tuple.rs`
- `NullStrategy` and `execute_row_fn_with_strategy`, the forced-strategy test seam
- `row/tests/null_strategies.rs` (502 lines)
- `vortex-array/benches/strict_validity.rs` plus its registration

The PR body should carry the threshold evidence from the x86 sweep in #9128: cost 0 always
branches, cost 1 branches at 50% or more survivors, cost 2 or more branches at 85% or more. The
thresholds are executor policy, not API, and #9130's open question about a calibrated cost model
stays open.

Gates: as PR 1, plus the `strict_validity` benchmark on an x86 host with the forced-strategy
comparison, since the thresholds were calibrated there.

Size: roughly 300 to 400 lines of implementation, 502 of tests, 217 of benchmark.

### PR 3: `NumericBinary`

Progress towards #9128. First production user, and the PR that proves the API against real
kernels.

Manifest:

- `vortex-array/src/scalar_fn/fns/binary/numeric/row.rs` (new, 269 lines)
- `numeric/mod.rs`, `numeric/primitive.rs`, `numeric/checked.rs`: delegate primitive execution to
  `NumericBinary`, delete the replaced execution, keep `checked_lanes` for decimal
- `numeric/tests.rs` additions
- `compare/primitive.rs`: `PrimitiveOperand` moves here, its only remaining user
- `scalar/typed_view/primitive/numeric_operator.rs` (2 lines)
- `vortex-compute/src/lane_kernels/map_into.rs`: delete `map_checked_into`, no caller left

Reconcile with #9210 as described above. Decimal stays on `execute_numeric_decimal`. `Binary`
keeps its ID, options serialization, and strictness. `NumericBinary` is unregistered.

Gates: the full `binary/numeric` test suite unchanged, including
`test_decimal_overflow_on_null_lane_ignored`. CodSpeed on the stable `binary_ops` names from
#9136 is the performance gate. For any regression near the vectorizer's decision boundary,
compare emitted optimized IR before trusting single-host wall clock, per the handoff's measurement
notes.

Size: roughly +550/-400 lines.

### PR 4: tensor migration

Progress towards #9128.

Manifest: `vortex-tensor` in full, which is `scalar_fns/row.rs` (new), the rewritten `l2_norm`,
`inner_product`, and `cosine_similarity`, the new `scalar_fns/tests/` directory, the `l2_norm_row`
consolidation in `utils.rs`, `vector_search.rs`, `encodings/normalized/execute.rs` and `mod.rs`,
and the three bench registrations.

Gates: the 179 tensor tests, CodSpeed on the #9136 tensor bench names. The epic's cosine numbers
(1.40x to 30.13x faster than develop depending on shape) are the expectation to confirm, and the
`reduce_encoded` probes are pinned by `reduce_encoded_is_probed_before_and_after_filtering` and
`normalized_readthrough_survives_null_rows`.

Size: roughly +1,880/-1,570 lines.

### PR 5: geo migration

Progress towards #9128.

Manifest: `vortex-geo` in full, which is `scalar_fn/row.rs` (new), the rewritten `contains`,
`intersects`, and `distance`, the deleted `scalar_fn/execute.rs`, the bbox caching in
`extension/{mod,point,polygon}.rs`, `test_harness.rs`, `benches/null_strategies.rs`, plus the
workspace `geo = "=0.31.0"` pin in the root `Cargo.toml` with its `contains_route` justification
comment.

The x86 bbox widening from the #9128 follow-up comment is already on this branch, so the
extraction carries it.

Gates: the 230 geo tests, CodSpeed on the geo bench names, and the `null_strategies` forced-vs-auto
check on x86.

Size: roughly +1,210/-640 lines.

Open question: are the two remaining geo regressions acceptable to land, or does PR 5 wait for a
fix? `contains` constant x points is 8.6% to 14.2% slower by median and `intersects` points x
constant is 10.9% to 13.2% slower, both real and settled per the #9128 follow-up. The epic's
overlapping-polygon win (8.6x to 8.8x) is the other side of the same bbox mechanism.

## What never lands

- `SCALAR_FN_HANDOFF.md`, `STRICT_SCALAR_FN_RESEARCH.md`, `NUMERIC_ROWFN_PLAN.md`, and this file
  are branch-only working notes. Authoring PRs from develop keeps them out automatically.
- `docs/strictness-and-validity-pushdown.typ` is a typeset design artifact, and the docs tree has
  no other Typst source. Recommend attaching its rendered output to #9130 or folding its content
  into the issue, not landing the source.

## Issue edits that accompany the series

- Strike #9130's "avoid probing `reduce_encoded` twice when branch execution is unsupported"
  follow-up. The handoff records why it is unsound: the pre-filter probe is the only one that sees
  the arrays still encoded, and `L2Norm` over `Normalized` answers differently by design.
  `reduce_encoded_is_probed_before_and_after_filtering` pins the two probes.
- Record the PR series as a checklist in #9128's implementation history, and check entries off as
  they land.
- Nullable row outputs stay out of the initial API, as an additive follow-up. The
  `nullable_outputs.rs` tests pin the all-valid sink contract that makes the derivation sound.
  Resolve the epic's open question accordingly rather than leaving it dangling into PR 1 review.

## Superseded items from the handoff

The handoff's "Next action: rerun the benchmarks on x86" described a monolithic branch-vs-develop
comparison. The per-PR CodSpeed gates on the #9136 names supersede it: each migration is measured
against its own develop base at landing time, which is a cleaner comparison than the whole branch
against a drifting develop. The x86-specific items that survive are attached to their PRs above
(the forced-strategy check in PR 2 and PR 5, and the #9210 re-measurement in PR 3).
