<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Handoff: the row scalar-function framework

Written at the end of a long session, for whoever picks this up next. The companion document
`STRICT_SCALAR_FN_RESEARCH.md` is the full design record with derivations, measured tables, and the
alternatives that were rejected and why; this file is the orientation and the plan. Read this first,
then that. Both are working notes that live on this branch only: they are not meant to land in the
pull-request stack, and the tracking issue is their public form.

The design is proposed publicly in three issues that Connor wrote by hand. Their goals and scope
remain authoritative, but the API and executor sketches in 9129 and 9130 predate the final
sink-only refactor. This handoff records the current prototype until those issues are updated:

- **Epic 9128, Row-oriented scalar functions.** Goals, motivation (the `Hypot` walkthrough of
  everything an author has to get right today), non-goals, and the current benchmark/codegen
  record in its [follow-up comment](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802).
  Status is *Proposed*, and it links this branch and its diffshub comparison as the prototype, so
  the branch is now publicly referenced: do not rewrite or delete it.
- **9129, Define the `RowFn` API.** The author-facing surface, with trait sketches and worked
  `Hypot` and `CosineSimilarity` examples. Its current return-witness and three-visit design is
  obsolete.
- **9130, Execute `RowFn` over Vortex arrays.** The private lifting, the two null-handling
  contracts, and adaptive execution. Its current statement that sinks cannot branch-and-skip is
  obsolete.

Both tracking issues end with an "Implementation history" section reading "None yet". Add each pull
request there as it lands; that is the intended record of progress.

This branch is where the design was worked out and measured, and the plan below turns it into small
reviewable pull requests cut fresh from develop.

## Final prototype status

The prototype's design is settled at commit `b918a8fb1f`, on top of the loop work in `3e675c9aaa`.
The branch then added a decoded-length guard, merged `origin/develop` at `fae9da1ebb`, documented the
x86 runbook at `d293d3cdd59e`, and widened geo's constant-side bbox rejection after the measurement
confirmed the predicted #9076 regression. The API did not change in those follow-ups. Read
[the completed x86 record](#the-x86-avx-512-re-measurement-runbook) before citing performance.

The sink-only shape makes an output sink the only row-execution model. `RowVisitor` now has one
method, `visit_prepared_into`; ordinary scalar outputs use `ElementSink<T>`, and runtime-shaped
outputs use a custom sink. `RowFn` has no return witness because the sink owns the output dtype and
builder.

This is simpler than the three-mode API in issue 9129 and faster than the earlier returning loop.
The framework borrows the sink's row storage once, validates its length once, and passes one row
slot into an `Fn` closure. A captured mutable sink would require `FnMut` and measured 8 to 11%
slower because it inhibited vectorization. Keep the row slot explicit.

Fallible kernels now have two forms. `VortexResult<()>` still exits on the first real error and
therefore gets valid-row-only execution. `DeferredError` is for operations such as checked add that
can write a memory-safe provisional value for every row. The executor OR-reduces one error word
across the batch and asks the sink to report it from `finish`. A nullable dense batch that reports a
deferred error is retried over only its valid rows, which discards errors caused by garbage behind
nulls without putting a branch or `VortexResult` discriminant in the hot loop.

`L2Denorm` remains implemented through the row machinery on this prototype, but its normalized
child and authoritative stored norms are now classified publicly as an encoding. The baseline PR
therefore benchmarks the renamed `Normalized` encoding rather than presenting it as a scalar
function. This classification does not weaken the sink result: `TensorSink` remains a production
runtime-shaped sink, and checked add demonstrates the deferred-error form.

## Current benchmark and codegen record

The authoritative record is now the
[updated issue #9128 x86 AVX-512 comment](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802).
It measures candidate `d293d3cdd59e` plus the geo bbox widening against current develop
`876996fe7846` on a Ryzen 9 7950X. Runs were pinned to CPU 4 with TSC timing, performance governor
and EPP, two sequential repetitions, and ABBA ordering for cross-revision comparisons. Ratios below
are control/develop divided by candidate, so above 1x favors the row branch.

The results that constrain the implementation stack are:

- sink-only checked add is 1.018-1.022x faster by median for two columns, 1.218-1.226x for a
  column and constant, and 1.033-1.042x for nullable columns;
- `BytesLen` is 1.410-1.411x faster by median than resolving a byte slice for long strings and
  1.097x for short/inlined strings;
- non-null `l2_norm` is 1.038x faster by median at width 32 and within 0.4% at width 256; width 2
  has a repeatable fastest/median split, about 2% faster by fastest and 3.5% slower by median;
- cosine is 1.40-30.13x faster than develop across column, constant, and extension-constant shapes;
- prepared overlapping `contains` is 8.60-8.77x faster by median, disjoint polygons are at parity
  after the bbox widening, and overlapping/disjoint `intersects` are at parity or slightly faster;
- point/constant geo remains slower: `contains` constant x points is 8.6-14.2% higher by median,
  `intersects` points x constant is 10.9-13.2% higher, and column x column point `contains` is
  3.7-5.2% higher;
- direct bbox rejection is 33.4-33.5x faster by median on disjoint rows and adds 1.8-1.9% before
  exact evaluation on overlapping rows;
- the global 75% survivor threshold misses one-nullable/50%-null in favor of filter when branch is
  6-8% lower latency, and two-nullable/10%-null in favor of branch when filter is 2.5-2.8% lower.

The checked-add control is benchmark-local rather than a production function, and there is still no
production deferred-error user. Its native release IR contains `<8 x i64>` error accumulators and
`llvm.vector.reduce.or.v8i64`; assembly uses a four-way-unrolled AVX-512 `zmm` loop with no per-row
error branch. Current `l2_norm` remains a strict-order scalar reduction. The old four-`vmulps`,
64-`f32` loop belonged to `l2_denorm`, which is now `Normalized` encoding evidence rather than a
scalar-function claim.

Lazy/eager validity and distinct/repeated LIKE diagnostics were not rerun because the relevant code
did not move. Their earlier conclusions remain scoped to those in-binary controls: validity stayed
within 2%, while distinct LIKE patterns were 4.7x slower and justify keeping shared pattern state
outside `RowFn`.

## Where the work lives

Everything is on `claude/strict-scalar-fn-abstraction-ah88x3`. The branch is publicly linked from
epic 9128 and is a prototype record, not a merge candidate. It contains the sink-only work, the
decoded-length guard, the `origin/develop` merge at `fae9da1ebb`, the completed x86 measurement, and
the geo bbox correction. Do not rewrite the published history. Push final implementation and
documentation commits only when explicitly requested.

The merge is the only develop integration on this branch, and its five conflicts were all in
`vortex-geo`, all resolved to the row ports because #9076 changed bodies the row layer replaces.
`vortex-geo/src/scalar_fn/execute.rs` stays deleted, develop's new `is_fallible` overrides are gone
because the blanket impl derives that from the witness, and develop's two new behavior tests were
carried across (`constant_container_vs_row_rect_poking_outside` and
`distance_to_constant_polygon_is_exact`) since both constrain the verdict rather than the mechanism.
`git rerere` recorded the resolutions, so re-doing the merge will not re-ask.

The production benchmark baseline is separate: draft PR
[#9136](https://github.com/vortex-data/vortex/pull/9136), branch `ct/scalar-fn-baselines`, currently
at `bf814bbe02cb`. It includes public-path baselines for byte length, binary add, LIKE, tensor,
geo predicates, and geo distance; scales the expensive overlapping-contains arm to 1,024 rows;
and gives each allocating binary a vendored `mimalloc`. It is stacked on
`ct/scalar-fn-factory-ext` and currently reports `DIRTY`, so resolve the stack before treating it as
ready to merge.

## What exists, in dependency order

**`RowFn`** (`vortex-array/src/scalar_fn/row/`) is the only authoring trait above
`ScalarFnVTable`: an implementor names witness types, picks concrete element types per batch in
`dispatch`, and hands the framework a row closure. One blanket impl,
`impl<F: RowFn> ScalarFnVTable for F` in `row/row_fn.rs`, derives everything else. The prototype on
this branch still includes `l2_denorm`, but the intended adopters after the encoding rebase are
`byte_length`, geo `contains`/`intersects`/`distance`, and tensor `l2_norm`/`inner_product`/
`cosine_similarity`.

**The lifting** (`vortex-array/src/scalar_fn/row/lift.rs`, `pub(super)`) turns that row loop into a
full `execute`: null propagation, constant folding, nullability widening, output dtype
reconciliation, and per-batch null-strategy selection. `Batch` carries one batch's facts and its
`execute` takes the kernel as two closures, one running it over whichever arguments the lifting
hands over and one trying branch-and-skip over the conjoined mask. This used to be a public
`StrictScalarFnVTable`, which was deleted; see the last section of the research document for why and
for what replaced each of its members.

**`RowVisitor`** has one method: `visit_prepared_into`. It chooses an argument tuple, an output
sink, a batch preparation closure, and a row closure. Passing `|_| ()` is the unprepared case;
choosing `ElementSink<T>` is the ordinary one-value-per-row case. The API does not retain aliases
for those combinations because the executor never needs separate methods and every implementation
must already understand the complete primitive.

**`OutputSink`** owns the return contract. `sink_dtype` derives the non-nullable output dtype from
the input dtypes, `with_capacity` allocates once, `rows` exposes loop-local storage,
`row_count_matches` discharges the output bound once, `row` hands one slot to the closure, and
`finish` builds the array. `ElementSink<T>` adapts an ordinary `OutputElement`; `TensorSink<T>`
writes runtime-width tensor rows. `SUPPORTS_SKIPPED_ROWS` controls branch-and-skip, while
`ERRORS_ARE_DEFERRED` opts a sink into batch-wide error accumulation.

**`SinkResult`** is the only row return contract: `()` for infallible writes,
`VortexResult<()>` for early errors, and `DeferredError` for a failure bit reduced across the full
loop. The value itself is already in the sink, which is why no `RetWitness` or returning
`ApplyResult` remains.

**Adaptive null strategy** is implemented. `row_null_handling` (in
`row/execute.rs`) derives `Dense` (run over the garbage behind null rows, mask after) or `Filter`
(the kernel must never see a null row) from element dense-safety and fallibility. `Filter` names
that *contract*, and two mechanisms satisfy it: the original filter-and-scatter, and
branch-and-skip, which decodes full length null-tolerantly and visits only mask-set rows
word-at-a-time. A sink participates when `SUPPORTS_SKIPPED_ROWS` is true; otherwise the branch
attempt declines and filtering remains the fallback. The lifting chooses per batch from
`Mask::true_count` and
`InputElement::DECODE_SHRINKS_WHEN_FILTERED`, a defaulted const that is `true` only for elements
whose decode parses every row (geo geometries). Threshold
`BRANCH_MIN_SURVIVING_FRACTION = 0.75`, with the measured crossover tables cited in its doc
comment. Function authors write nothing to benefit. To compare or force a strategy from a test or
benchmark, `execute_row_fn_with_strategy` (test-harness gated) is the only seam.

## The essential invariants, so you do not break them by accident

- **Dispatch must be pure in `(options, args)`** and is value-blind: it sees dtypes, never data.
  Plan time and run time both go through it. This is why constant-compute hoisting had to live
  inside the visit (`prepare`) rather than in dispatch.
- **The argument witness and `RowFn::FALLIBLE` pin what the framework reads before dispatching**:
  arity, dense-safety, decode fallibility, early kernel fallibility, and
  `DECODE_SHRINKS_WHEN_FILTERED`. `assert_witness_agrees` makes a contradicting dispatch a compile
  error, even in an arm that never runs. There is no return witness; the selected sink supplies the
  dtype and its `SinkResult` supplies the row error shape.
- **`prepare` must never be load-bearing for validation.** An empty batch decodes as non-constant,
  so a check that only runs in `prepare` will not run at all. It is also infallible by design,
  because fallibility is read off the witnesses before dispatch.
- **`P` has no `Send`/`Sync` bounds, deliberately.** Geo's prepared geometry carries `Rc`/`RefCell`.
  Adding bounds later is a breaking change.
- **Every sink builds an all-valid column**, which is what licenses the derived
  `validity() = union_child_validities`. Anything that lets a kernel emit nulls must change that
  derivation in the same commit.
- **A `RowFn` cannot also implement `ScalarFnVTable`**, since the blanket impl claims the slot.
  `RowFn` therefore mirrors any `ScalarFnVTable` method an adopter needs to vary. Today that is
  `validity` and nothing else, and `reduce` was dropped when the strict trait went because no
  adopter used it. Mirror the next one when a real adopter needs it, not before.
- **An early-failing kernel must never run behind a null row**, which is why `VortexResult<()>`
  forces the `Filter` contract and branch-and-skip visits only set bits. A deferred-error kernel is
  different: it may run densely when its arguments are dense-safe, then retry over valid rows only
  if its reduced error bit is set. The hostile tests (views pointing at nonexistent buffers,
  poison zero divisors behind nulls) exist to catch a regression here; keep them.
- **`reduce_encoded` sees different row counts per strategy**: original count under `Dense` and
  branch-and-skip, filtered count under filter-and-scatter. Its doc says so; every current
  implementor is a dense-path tensor function, so nothing in tree depends on the difference yet.

## Decisions already made, with the reasoning, so they are not relitigated

**Array serde is per-function, not derived.** A blanket `ScalarFnArrayVTable` impl over `RowFn`
existed and was removed (`caa3df090`, Connor's commit): the four tensor functions now implement
serialize/deserialize themselves, sharing a `BinaryTensorOpMetadata` helper in
`vortex-tensor/src/utils.rs`, and `array_metadata.rs` is gone. The rationale: functions already
implement `ScalarFnVTable` concerns separately, so requiring array serde separately is consistent,
and a generic derivation can come later as its own PR if it ever earns one. Do not re-add it as
part of the row work.

**The separate strict-lifting PR was off the table, and the trait is now deleted.** It was measured
and reviewed and did not justify itself as a standalone change: +798 production lines against -210,
no `NullHandling::Filter` user among its three ports, mixed benchmark results, and no demonstrated
pre-existing bug fixed. A local review also made a naming objection that stood: `is_strict` is the
semantic property `valid(f(x)) ⊆ valid(x)` used for pushdown, while the trait actually demanded the
*operational* property that a kernel may run over garbage or over a filtered copy. Those are
independent (`Bytes` is strict and not dense-safe).

So the three columnar ports were reverted (`not`, `list_length` and `list_sum` are back on their
develop bodies, which are already known to be their optimum) and the trait was deleted rather than
merely made private: with the ports gone it had exactly one implementor, the blanket impl over
`RowFn`. The lifting *machinery* stayed, since `RowFn` depends on all of it, and now lives in
`row/lift.rs` behind `pub(super)`. Do not reintroduce a trait there. If a non-row columnar kernel
ever wants the lifting, extract one then, named for the lifting contract rather than for
strictness, with that kernel as its first user.

Two things fell out of the deletion. `reduce` was mirrored on the strict trait only because the
blanket impl occupied the `ScalarFnVTable` slot, and nothing used it, so it is gone; `validity`
stays mirrored on `RowFn`. And the runtime rejection of `Dense` paired with a fallible kernel is
gone, because `row_null_handling` derives the pairing from the same witnesses `is_fallible` reads,
which makes the combination unconstructible; the requirement is documented on
`NullHandling::Dense`.

**`NumericalAggregateOpts::serialize`/`deserialize` keep their original names, and nothing should
make them implement `PersistableOptions` again.** Porting `list_sum` had forced that impl, whose
trait methods collided with the inherent ones, and the resulting rename to `serialize_proto`/
`deserialize_proto` broke public source compatibility for downstream callers. A reviewer flagged it
and suggested forwarding aliases; reverting the port removed the cause instead, which is why the
original names are back with no aliases and no deprecation. Note the collision was never a compile
error (inherent methods win method resolution), so if some future change does need both,
fully-qualified calls are enough and a rename is not.

**`not` is back exactly as it is on develop.** Its `to_bit_buffer()` is a handle clone, and the
source array keeps the buffer shared, so in-place negation (a real 19% on uniquely owned buffers)
is unreachable without redesigning `ExecutionArgs` ownership. Encoded NOT flows through
`NotReduce` and per-encoding pushdown at 13-24x below canonical cost, so canonicalizing would be a
regression. This is the reference example for the exclusion taxonomy: an existing columnar body
that a port would make worse.

**The lifting's small-batch overhead is structural and accepted.** It is generic prelude
bookkeeping (collect inputs, compute the declared dtype, conjoin validity), not a single avoidable
allocation; ablations including SmallVec found nothing beneficial, and an earlier
"-10% at 100 rows" reading did not reproduce uniformly. The only structural answer is to
monomorphize the prelude over the compile-time arity (`[ArrayRef; N]` from the tuple witness),
which the row layer can do through its tuple witness.

**Sink-only execution is the final prototype shape.** Returning visits, sink visits, and prepared
visits were three spellings of the same executor. Keeping only `visit_prepared_into` removes the
return witness and makes preparation plus output construction one explicit contract. The simple
case is still short because `ElementSink<T>` and unit preparation supply the defaults. Do not add
the convenience methods back unless an independently simpler executor would consume them.

**Value errors should be deferred only when provisional computation is memory-safe.** Checked add
can write the wrapping sum and OR an overflow word, so a specialized sink avoids a per-row result
branch. Parsing, allocation, and errors that cannot produce a legal provisional row still use
`VortexResult<()>`. `DeferredError` is not a general replacement for ordinary error handling.

**Non-strict functions are out of scope, and the epic settles it.** Epic 9128 states the framework
will focus solely on strict functions, because the semantics around non-strict ones are complicated
enough that extending an already-somewhat-complicated API is probably not worth it. The research
backs that from the other direction: the survey found no in-tree customer for null-visible inputs,
since 13 of 15 non-strict functions are cheap columnar mask algebra, and a prototype row-function
Kleene AND measured 250-1030x slower than the in-tree fused word kernel. The two genuinely expensive
null-visible kernels (`RowEncode`/`RowSize`) are blocked by variadic heterogeneous arity, not by
nullability. Four functions have value-dependent output validity (`false AND null` is a valid
`false`), for which no validity expression over child validities exists even in principle. Keep the
survey's constraint list for reference; do not build the tier. Note this is a different question
from nullable *outputs* (step 5 below), which stay strict and are an explicit goal of the epic.

The epic's non-goals also match the exclusion taxonomy the research arrived at independently:
`RowFn` is not for columnar or zero-copy kernels (`not`, `list_length`), kernels with state shared
across rows (`like`), or heterogeneous variadic kernels (`RowEncode`, `pack`, `case_when`).

## What to do next, in order

The x86 AVX-512 re-measurement is complete, the #9128 comment is updated, and the predicted geo bbox
gap is fixed and tested on this branch. The remaining work starts with the durable baseline:

1. **Land PR #9136.** It now contains the durable public-path benchmark set needed before the
   implementation stack: byte length, signed and unsigned add including a constant operand, LIKE
   cache behavior, tensor functions, geo predicates, and geo distance. Resolve its stacked-base
   conflict, rerun its targeted checks and CodSpeed shard, then merge it before any benchmarked
   implementation change.
2. **Update issues 9129 and 9130 before cutting code.** Their API sketches still contain
   `RetWitness`, `visit`, `visit_prepared`, and `visit_into`; 9130 still says sinks cannot
   branch-and-skip. Replace those with the sink-only API, `SinkResult`, deferred-error retry, and
   `SUPPORTS_SKIPPED_ROWS`. The issues are the public contract and must not direct an implementing
   agent back to the discarded design.
3. **Cut the implementation stack fresh from current develop.** Do not merge this prototype
   branch. Keep each PR independently reviewable and link it from the appropriate tracking issue.
   A sensible split is row core plus `ElementSink` and `byte_length`; tensor elements and kernels;
   prepared execution with cosine and geo; adaptive null execution; then specialized sinks and
   deferred errors with a production checked arithmetic user. Preserve each migrated function's
   serialized metadata. Two amendments the measurement settled: the geo PR should carry the widened
   conservative bbox rejection and its NaN agreement tests, and should explicitly track the
   remaining point/constant regressions; the deferred-error PR has to bring its own production user,
   since `ERRORS_ARE_DEFERRED` currently has none outside a bench and a test.
4. **Use CodSpeed to decide whether handwritten add can be deleted.** The local executor harness
   establishes that the machinery can reach parity, but the gate is the stable production
   `binary_ops` names from #9136 on the actual implementation PR. Require non-null, nullable,
   constant, signed, and unsigned cases. Treat a sub-percent local difference as noise; investigate
   a repeatable CodSpeed regression from the generated IR before accepting it.
5. **Add nullable outputs separately.** `list_sum` and `variant_get` are strict but may return null
   from valid inputs. A sink that builds validity can express this, but the blanket
   `validity() = union_child_validities` derivation must become conditional in the same change. Do
   not smuggle this into the initial execution PR.

## Adaptive execution status

Adaptive execution is correct and tested, but its current selection rule is not fully calibrated.
The x86 run found two repeatable cases where auto chooses the slower forced strategy.

The rule today: for a mixed validity mask, use branch-and-skip unless some argument's element sets
`DECODE_SHRINKS_WHEN_FILTERED` and fewer than 75% of rows survive, in which case filter and scatter.
One boolean per element, one global threshold, no other inputs. The measured misses are:

- one nullable geometry at 50% nulls: auto/filter median 5.999-6.050 ms, branch
  5.560-5.642 ms, so branch is 6-8% lower latency;
- two nullable geometries at 10% nulls, about 81% conjoined survivors: auto/branch median
  10.40-10.60 ms, filter 10.20-10.34 ms, so filter is 2.5-2.8% lower latency;
- at two nullable geometries and 25% nulls, auto correctly filters and filter is 1.21-1.22x faster
  than branch; at 90% nulls it is about 11.6x faster.

The first two cases point in opposite directions, so conjoined survivor fraction alone is not enough.
The remaining questions are:

- **What should replace the global const?** The alternatives are per-element and arity-aware
  thresholds, or a small cost model comparing estimated full-row decode work against survivor-only
  decode plus filter and scatter. The one-versus-two-element reversal is now direct evidence that
  element/arity cost must be represented.
- **Should batch size be an input?** Filter's cost is dominated by allocation and copying, which
  does not amortize on small batches, while branch's cost is proportional to the full column. The
  crossover therefore probably moves with row count, and nothing measures that yet. The small-batch
  sweep in the measurement plan below feeds this directly.
- **How should multiple nullable per-row-decode arguments compose?** The flag is ORed across
  arguments, but decode cost grows with arity while the conjoined survivor fraction falls. The 10%
  two-input miss proves that the current OR plus one threshold does not compose those effects.
- **Does the crossover hold across architectures?** It trades memory-bandwidth-bound work against
  compute-bound work, so x86 and Apple Silicon need not agree. CodSpeed's x86/AVX2 simulation is the
  next stable data point, but it does not replace real wall-clock measurements on either machine.
- **Which sinks should branch-and-skip?** The question is now per sink, expressed by
  `SUPPORTS_SKIPPED_ROWS`. `ElementSink` supports it by pre-filling placeholders; a custom builder
  that cannot finish skipped rows declines and filters. No new execution model is required.
- **Is the strategy observable in production?** The only seam today is test-harness gated. A session
  option, or tracing the chosen strategy, may be wanted for debugging a slow query.
- Plus the recorded follow-up: avoid probing `reduce_encoded` twice when branch execution turns out
  unsupported.

## The x86 AVX-512 re-measurement runbook

The re-measurement was completed on 2026-08-04 and published in the
[#9128 follow-up comment](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802).
This section now records the completed protocol and remains the procedure for reproducing it.

### The one rule about machines

**Label every number with its host, and never mix hosts inside one table.** The authoritative issue
comment is x86 (7950X, AVX-512, CPU 4, TSC timer, performance governor and EPP). Anything measured on
Apple Silicon is separate evidence, not an update to that table.

- The checked-add audit is AVX-512 specific: four `zmm` `vpaddq` operations produce 32 `i64` rows
  per main-loop iteration and vector error words reduce after the loop. Do not translate that
  instruction claim to NEON.
- The old four-`vmulps`, 64-`f32` claim belongs to `l2_denorm`, now the `Normalized` encoding. The
  current `l2_norm` audit is scalar because strict reduction order prevents reassociation.
- `BRANCH_MIN_SURVIVING_FRACTION = 0.75` was tested on x86 and found imperfect in both directions.
  An architecture change may move the misses, but does not erase the x86 evidence.
- macOS has no equivalent core pinning and no TSC, so its per-sample variance is higher and its
  `fastest` is less meaningful.

### Revisions measured

| role | ref |
| --- | --- |
| candidate | `d293d3cdd59e` plus the geo bbox widening recorded in this update |
| baseline | `origin/develop` at `876996fe7846` |

The merge is what makes a single baseline valid for both suites. Before it, geo needed a pre-#9076
baseline and tensor a post-#9076 one. Confirm with `git merge-base --is-ancestor 2de0319312 HEAD`
before trusting any geo comparison.

### What each suite needs

Tensor and `byte_length` controls are in-binary, so one checkout measures both arms:

```bash
cargo bench -p vortex-tensor --bench l2_norm --bench cosine_similarity
cargo bench -p vortex-array --bench row_fn_executor --bench byte_length_element
```

Geo needs two checkouts, because its comparison is against develop's columnar path rather than an
in-binary control. Mind which bench exists where: develop registers only `envelope` and
`predicate_bbox`, so `binary_predicates` has to be carried into the develop tree. It is portable
because it touches nothing branch-only (`GeoContains` / `GeoIntersects` constructors and the
`geo_session` / `point_column` / `polygon_column` helpers all exist on develop unchanged):

```bash
git worktree add ../vortex-develop origin/develop
cp vortex-geo/benches/binary_predicates.rs ../vortex-develop/vortex-geo/benches/
# then add the matching [[bench]] stanza to ../vortex-develop/vortex-geo/Cargo.toml

# candidate tree
cargo bench -p vortex-geo --bench binary_predicates --bench predicate_bbox
cargo bench -p vortex-geo --bench null_strategies
# develop tree
cargo bench -p vortex-geo --bench binary_predicates --bench predicate_bbox
```

`predicate_bbox` arrived with #9076 and still builds against the row ports, since it goes through
the public path, which makes it the one geo bench already named identically on both sides and so the
cheapest signal on whether the ports kept develop's rejection win. `null_strategies` and
`null_strategy_bytes` force a strategy through the test-harness seam, so they cannot exist on
develop and can never be baselined; keep them as candidate-only diagnostics. `l2_denorm` measures
what is now an encoding, so run it only if the encoding claim is what is wanted.

Report `fastest` and median from at least two runs, sequentially, with the host recorded. Divan is
aliased to `codspeed-divan-compat` and behaves as ordinary divan locally.

### Outcome of the geo hypothesis

The hypothesis was correct. The pre-fix diagnostic showed the expected constant-point and
disjoint-`contains` losses. The final patch caches the constant bbox for `contains` and applies
conservative bbox rejection to every constant arrangement for both predicates. NaN agreement tests
cover both operand orders. The stabilized rerun restores disjoint polygons to parity, keeps
overlapping prepared `contains` 8.60-8.77x faster by median, and leaves the point/constant regressions
listed in the benchmark-record section. Carry the widened gate and tests into the geo implementation
PR; do not treat the remaining point cost as a bbox mystery.

## The measurement plan

The repo runs CodSpeed on every pull request, and the workspace aliases `divan` to
`codspeed-divan-compat`, so every registered bench is measured against a develop baseline
automatically and behaves as ordinary divan when run locally. That makes CI the most credible source
for the no-regression claim, since it is the team's own process rather than anyone's laptop.

**The constraint that shapes everything: CodSpeed compares a pull request against develop for the
same benchmark name.** A benchmark introduced in the same change it measures has no baseline and
produces no comparison. Draft PR #9136 exists to land those names first. Its current public-path
set covers `byte_length`; signed and unsigned add; repeated and distinct LIKE patterns; tensor
`l2_norm`, `normalized`, `inner_product`, and `cosine_similarity`; and geo `contains`, `intersects`,
and `distance`, including constant and nullable shapes. Two rules remain load-bearing:

- **Exercise the functions through the public expression and execution API only.** The versions on
  this branch reference framework internals (forced-strategy seams, hand-written control arms
  implementing `ScalarFnVTable`), which cannot exist on develop. A bench that goes through the
  public path is also the only kind that keeps measuring the same thing before and after a port.
- **Identical benchmark and arm names on both sides**, or CodSpeed cannot line them up.

Arms that compare framework strategies against each other (forced filter versus forced branch versus
auto) cannot be baselined, because they need machinery that does not exist on develop. Keep those as
implementation diagnostics. The durable CodSpeed suite should contain only public production paths
at representative null densities.

**Adaptive execution is less entangled with the other measurements than it looks.** Every adopter
except geo is dense-safe and infallible, so it takes the dense path and never reaches strategy
selection at all: `l2_norm`, `inner_product`, `cosine_similarity`, and `byte_length` (which ships at
`BytesLen`) are unaffected no matter what the rule becomes. The prototype's `l2_denorm` port is
dense-safe too, although that operation is now classified as the `Normalized` encoding. Only geo's
nullable batches are affected, which is why the geo baseline needs both nullable and non-nullable
arms: it keeps the port's parity claim separable from adaptive execution's win. The baseline can
land before the implementation stack without depending on the final threshold.

CodSpeed runs these CPU-bound microbenchmarks in simulation over compiled amd64/AVX2 machine code.
That is useful for the LLVM and vectorization questions that motivated the executor work, but it is
not real x86 wall time or a branch-predictor measurement. Continue to use local Divan runs for
wall-clock diagnosis, sequentially, with machine metadata, `fastest` and median from at least two
runs. Use CodSpeed as the merge gate once the stable baseline names are on develop.

## What the tracking issues ask that this branch has not answered

Four of their unresolved questions are genuinely open, and two of them this branch never considered:

- **Are `InputElement`, `OutputElement` and `OutputSink` public downstream extension points, or only
  cross-crate extension points within Vortex?** (9128, 9129.) This branch treats them as
  Vortex-internal: `vortex-tensor` and `vortex-geo` implement them, nothing outside does. It matters
  because a downstream implementor makes every associated constant a compatibility surface, and
  three already carry defaults chosen for internal convenience (`DECODE_SHRINKS_WHEN_FILTERED` is
  `false`, `decode_null_tolerant` forwards to `decode`). If they go public, audit which defaults are
  the *safe* answer rather than the common one.
- **Which sink contracts belong in the first public API?** (9129.) `ElementSink` now backs every
  ordinary row function and `TensorSink` proves runtime-shaped output, so the sink itself is no
  longer speculative. The open compatibility questions are whether downstream crates may implement
  it, whether `sink_dtype` needs options, and whether `SUPPORTS_SKIPPED_ROWS` and
  `ERRORS_ARE_DEFERRED` should be stable extension points.
- **What regression threshold gates replacing a hand-written implementation?** (9130.) PR #9136
  now supplies the benchmark set, but the policy still needs a number and a repeatability rule.
  Checked add shows the intended precedent: parity for ordinary and nullable columns, with a
  constant-side win. Make CodSpeed over those stable names the gate.
- **Are nullable outputs required for v1?** (9128, 9129.) The sink-only shape makes them possible,
  but the validity derivation remains the semantic change. Keep this separate from skipped-row sink
  support, which is already implemented.

9130's remaining non-blocking follow-ups are the double `reduce_encoded` probe when branch execution
is unsupported, replacing the global 75% threshold with an element/arity-aware decision, and the
branch loop's inability to stop `for_each_set_index` immediately after an early error. The x86
matrix proves the current threshold misses in opposite directions, but geo is still the only
per-row-decode element measured. Geo's null-tolerant decode still arrow-exports the full column;
slicing runs of valid rows could move or remove its filter crossover. The lifting's small-batch
prelude cost remains structural.

## Loose ends, and issues worth filing separately

The branch-only benchmark binaries are diagnostic, not the permanent CI suite. In particular,
`row_fn_executor`, `byte_length_element`, `strict_validity`, and the two forced null-strategy
matrices should travel only with the implementation work that needs them. PR #9136 contains the
durable public-path names. Commit `72e01f96c` (the first version of this file) is pushed unsigned,
so GitHub shows it unverified; fixing it would require rewriting published history and is not worth
doing.

Also worth an issue on its own, unrelated to this work: `Between::validity` declares the strict
three-way conjunction while its fallback execute path joins two comparisons with Kleene AND, so
with per-row nullable bounds the lazy validity and the executed result disagree (a valid `false`
reported as null). Verified on develop. Separately, `NotKernel` appears to have no implementations
and looks like dead code.

## How to work on this branch

Read `CLAUDE.md` and follow it. Invoke the `rust-style` skill before writing Rust: this branch is
written to Connor's personal style and reviewers will notice the difference. No em dashes anywhere,
in code, docs, or commit messages. Use `git commit -s` with the repository's configured identity;
this branch currently uses `Connor Tsui <connor.tsui20@gmail.com>`.

Verification that matters for changes here:

```bash
cargo nextest run -p vortex-array -p vortex-geo -p vortex-tensor
cargo clippy --all-targets --all-features -p vortex-array -p vortex-geo -p vortex-tensor
cargo +nightly fmt --all
git diff --check
cargo build -p vortex -p vortex-file -p vortex-datafusion
```

The final framework verification was: 65 focused RowFn tests, 164 tensor library tests, 72
`vortex-array` doctests with 10 ignored, targeted all-target/all-feature clippy for the three crates,
nightly fmt, and whitespace checks. After the geo bbox widening, 43 focused geo tests and all 230
`vortex-geo` tests passed, as did targeted all-target/all-feature `vortex-geo` clippy.

The prototype forked from development history at `bb4138d051` and has since merged `origin/develop`
at `fae9da1ebb`. That merge commit is an ancestor of the branch, but current `origin/develop` has
advanced and is not. Before the implementation stack is cut, fetch `origin/develop` and use `git
range-diff` rather than assuming either recorded hash or merge relationship is current. To recover
the pre-port body of `not`, `list_length`, or `list_sum` from the prototype history, use
`24d1933e1^`.

Benchmarks are divan (aliased to `codspeed-divan-compat`, so they also run in CI); report `fastest`
and `median` from at least two runs and **always state the machine**. Historical measurements used a
shared 4-vCPU VM, while the current issue comment and checked-add audit use the recorded 7950X setup.
See
[the x86 runbook](#the-x86-avx-512-re-measurement-runbook) for why mixing hosts in one table is not
recoverable after the fact. Any new bench must be
registered as a `[[bench]]` with `harness = false` in the crate's `Cargo.toml`. Per-adopter
benchmarks with constant and non-constant arms are the convention, and where a claim is "as fast as
the hand-written version", put both arms in one binary rather than comparing across builds. Disk is tight:
build narrowly and delete `target/debug` subdirectories on ENOSPC. If cargo fails with exactly
`sccache: error: Operation not permitted`, rerun that command with a `RUSTC_WRAPPER=` prefix.

Force-pushing and remote branch deletion are both blocked by the permission classifier, so plan
history edits before pushing rather than after. Do not open a pull request unless asked.
