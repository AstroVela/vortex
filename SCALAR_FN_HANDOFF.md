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
Two commits follow it and change no design: a decoded-length guard on the mixed constant path, and a
merge of `origin/develop` at `fae9da1ebb`. The merge matters for measurement rather than for the
API, and [the x86 re-measurement runbook](#the-x86-avx-512-re-measurement-runbook) is the section
to read before running anything.

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

**Every published ratio is stale. Do not cite the issue tables until they are re-run.** The
[issue #9128 benchmark and codegen follow-up](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802)
was the broad branch-versus-develop comparison for tensor and geo, and three independent things have
invalidated it since. They are separable, and only the second is confined to geo:

1. **The candidate moved.** The comment records candidate `0c0d1f80ee21`. `3e675c9aaa` then added
   the borrowed `Varying` view and hoisted the sink's row storage and bounds out of the loop, and
   `b918a8fb1f` made the sink the only execution model. Both rewrote the hot loop, so every tensor
   and geo row measures an executor that no longer exists.
2. **Develop moved, for geo only.** The comment's baseline is develop at `08c336f54a88`.
   [#9076](https://github.com/vortex-data/vortex/pull/9076) has since landed bounding-rect rejection
   for the geo predicates on develop, which is the same class of win the comment credits to the row
   ports ("`intersects` hoists the constant bounding rectangle and short-circuits disjoint cases").
   All nine geo rows therefore compare against a baseline that no longer exists, and the disjoint
   arms in particular should be assumed gone until re-measured. Develop's tensor code is unchanged
   over the same range, so the tensor baseline is still good and only reason 1 applies there.
3. **`l2_denorm` is no longer a scalar function.** Its normalized child and stored norms are
   classified as an encoding, so its row is not port-parity evidence for anything. Re-label it as
   the `Normalized` encoding or drop it.

The checked-add table below is the machinery measurement for the sink-only executor, and it carries
its own caveat: both arms live in `vortex-array/benches/row_fn_executor.rs`, so the "specialized"
control is bench-local rather than a production function, and there is no production deferred-error
user at all (`ERRORS_ARE_DEFERRED` is set only in that bench and in `row/tests/sink.rs`). State that
when the number goes public.

At 65,536 `i64` rows, 100 samples and a one-second minimum per arm, the sink-only checked-add
medians were:

| workload | specialized | `RowFn` | delta |
| --- | ---: | ---: | ---: |
| two columns | 13.54 us | 13.79 us | 1.8% slower |
| column and constant | 12.83 us | 11.33 us | 11.7% faster |
| nullable columns | 12.91 us | 12.95 us | 0.3% slower |

The generated LLVM IR contains vector error-word OR accumulators and
`llvm.vector.reduce.or.v2i64`; there is no per-row result discriminant or error branch. The
remaining ordinary two-column difference is framework setup and output construction around a hot
loop that has reached the intended shape, not a hidden `VortexResult` in that loop.

**The machine for that table is unrecorded, which by itself makes it unciteable.** Do not assume it
matches the 7950X in the issue comment. Re-run it under the runbook below and record the host.

The other final diagnostic runs support keeping the abstractions separate. Lazy and eager validity
were within 2% in every `strict_validity` arm. `BytesLen` was 28 to 29% faster than resolving a byte
slice for `byte_length`, which justifies the specialized input element but is not a permanent
production benchmark. Distinct per-row LIKE patterns were 4.7x slower than repeated patterns,
which confirms that state shared across rows belongs outside `RowFn`.

## Where the work lives

Everything is on `claude/strict-scalar-fn-abstraction-ah88x3`. The branch is publicly linked from
epic 9128 and is a prototype record, not a merge candidate. It is now several local commits ahead of
the remote branch: the sink-only work, the decoded-length guard, the `origin/develop` merge at
`fae9da1ebb`, and this handoff update. Do not rewrite the published history. Push the final
implementation and documentation commits only when explicitly requested.

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

0. **Re-measure on x86 AVX-512, before touching the issues.** Every published ratio is stale for the
   three reasons in the benchmark-record section, so the issue text cannot be corrected without
   numbers to correct it to. Follow
   [the x86 runbook](#the-x86-avx-512-re-measurement-runbook), which the `origin/develop` merge has
   already unblocked by giving both suites one valid baseline. Supersede the existing #9128 comment
   with a new one rather than editing it, so its AVX-512 codegen audit survives.
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
   serialized metadata. Two amendments the merge added: the geo PR also owns reconciling with #9076,
   including whether to widen the early-out gate, and the deferred-error PR has to bring its own
   production user, since `ERRORS_ARE_DEFERRED` currently has none outside a bench and a test.
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

Adaptive execution is no longer a design blocker for the prototype. The implementation is tested,
and the final measurements still show auto selecting the intended side of the crossover.

The rule today: for a mixed validity mask, use branch-and-skip unless some argument's element sets
`DECODE_SHRINKS_WHEN_FILTERED` and fewer than 75% of rows survive, in which case filter and scatter.
One boolean per element, one global threshold, no other inputs. The questions:

- **Is a global const the right shape at all?** The alternatives are a per-element threshold (each
  element knows its own decode-to-kernel cost ratio), or a small cost model comparing estimated
  decode work over all rows against decode over survivors plus filter and scatter. The measured
  crossover sat between 56% and 81% surviving rows for *one* element type, which is a single
  calibration point: a second per-row-decode element could easily want a different number, and that
  is the argument for moving the knob onto the element.
- **Should batch size be an input?** Filter's cost is dominated by allocation and copying, which
  does not amortize on small batches, while branch's cost is proportional to the full column. The
  crossover therefore probably moves with row count, and nothing measures that yet. The small-batch
  sweep in the measurement plan below feeds this directly.
- **Is the rule right with more than one nullable per-row-decode argument?** The flag is ORed across
  arguments, but with independent nulls the conjoined surviving fraction falls roughly quadratically,
  and the measurements did show the crossover moving down in that case (branch won only to about 10%
  null density with two nullable operands, versus 50% with one). One threshold against the conjoined
  fraction may already handle this correctly; confirm rather than assume.
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

This section is self-contained: it is what to run on an x86 AVX-512 host to replace the stale issue
tables, and it assumes nothing from the sections above except that they explain *why* the old
numbers cannot be cited.

### The one rule about machines

**Label every number with its host, and never mix hosts inside one table.** The issue comment's
numbers are x86 (7950X, AVX-512, pinned to one core, TSC timer). Anything measured on Apple Silicon
is separate evidence, not an update to that table, because three of its claims do not survive the
move at all:

- The folded IR and assembly are AVX-512 specific: `vbroadcastss`, four `vmulps`, zmm stores, "64
  `f32` coordinates per iteration". On NEON that is 128-bit vectors and a different unroll factor.
  Re-derive those claims on x86 or scope them explicitly to x86; do not restate them from an
  aarch64 run.
- `BRANCH_MIN_SURVIVING_FRACTION = 0.75` is calibrated on x86. It trades memory-bandwidth work
  against compute-bound work, so a macOS run disagreeing with it is not evidence the threshold is
  wrong.
- macOS has no equivalent core pinning and no TSC, so its per-sample variance is higher and its
  `fastest` is less meaningful.

If an interim aarch64 run happens first, post it as a clearly labeled interim comment and leave the
x86 re-verification open. Do not overwrite the existing comment: it holds the AVX-512 codegen audit,
which is the strongest part of the record and is not reproducible off x86.

### Revisions to measure

| role | ref |
| --- | --- |
| candidate | `claude/strict-scalar-fn-abstraction-ah88x3`, at or after the `origin/develop` merge |
| baseline | `origin/develop` at `fae9da1ebb` or later, which **must** include #9076 |

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

### The specific hypothesis this run should test

The merge left the candidate with a *narrower* bounding-rect early-out than develop, and this is the
most likely cause of a geo regression:

- `intersects_opens_with_bbox_check` admits the hoisted early-out only for pairings where geo itself
  opens with that same rect comparison. Develop's #9076 applies its rejection to every one-constant
  pairing. So for a pairing geo does not open with a rect test, `Point` against `Polygon` above all,
  develop gets an early-out the candidate does not.
- `contains` on the candidate routes through a prepared geometry and has **no** rect test at all,
  while develop now rejects on `!ra.contains(rb)`.

The gate was written to keep the hoist pure common-subexpression elimination, on the worry that a
rect test could change the verdict for NaN coordinates. That worry is settled and the answer is no:
geo 0.31's `Rect::intersects` returns false only after finding an ordered separation, so NaN bounds
make all four comparisons false, it falls through to true, and
`(!ra.intersects(rb)).then_some(false)` yields `None` and defers to the exact test. Develop's
unconditional form is therefore conservative and result-preserving.

So if `intersects points x constant` or either constant-side `contains` arm loses to develop,
widening the gate is the first thing to try, not a mystery. Both were already the weakest arms
against the *old* baseline (0.907-0.998x), and they are now measured against a faster one.

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
selection at all: `l2_norm`, `l2_denorm`, `inner_product`, `cosine_similarity`, and `byte_length` (which
ships at `BytesLen`) are unaffected no matter what the rule becomes. Only geo's nullable batches are
affected, which is why the geo baseline needs both nullable and non-nullable arms: it keeps the
port's parity claim separable from adaptive execution's win. The baseline can land before the
implementation stack without depending on the final threshold.

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
is unsupported, the global 75% threshold pending a second per-row-decode element, and the branch
loop's inability to stop `for_each_set_index` immediately after an early error. Geo's null-tolerant
decode still arrow-exports the full column; slicing runs of valid rows could move or remove its
filter crossover. The lifting's small-batch prelude cost remains structural.

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

The final implementation verification was: 65 focused RowFn tests, 225 geo library tests, 164 tensor
library tests, 72 `vortex-array` doctests with 10 ignored, targeted all-target/all-feature clippy for
the three crates, nightly fmt, and whitespace checks. The compile-fail witness doctest passed.

The prototype forked from development history at `bb4138d051` and has since merged `origin/develop`
at `fae9da1ebb`, so develop is an ancestor and there is no merge base to recompute for ordinary
diffing. Both refs will still move before the implementation stack is cut: fetch `origin/develop`
and use `git range-diff` rather than assuming either recorded hash is current. To recover the
pre-port body of `not`, `list_length`, or `list_sum` from the prototype history, use `24d1933e1^`.

Benchmarks are divan (aliased to `codspeed-divan-compat`, so they also run in CI); report `fastest`
and `median` from at least two runs and **always state the machine**, which is the one convention
that has actually been broken here: the historical measurements used a shared 4-vCPU VM, the issue
comment used a 7950X, and the checked-add table records no host at all. See
[the x86 runbook](#the-x86-avx-512-re-measurement-runbook) for why mixing hosts in one table is not
recoverable after the fact. Any new bench must be
registered as a `[[bench]]` with `harness = false` in the crate's `Cargo.toml`. Per-adopter
benchmarks with constant and non-constant arms are the convention, and where a claim is "as fast as
the hand-written version", put both arms in one binary rather than comparing across builds. Disk is tight:
build narrowly and delete `target/debug` subdirectories on ENOSPC. If cargo fails with exactly
`sccache: error: Operation not permitted`, rerun that command with a `RUSTC_WRAPPER=` prefix.

Force-pushing and remote branch deletion are both blocked by the permission classifier, so plan
history edits before pushing rather than after. Do not open a pull request unless asked.
