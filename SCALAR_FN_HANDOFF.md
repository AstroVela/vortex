# Handoff: the row scalar-function framework

Written at the end of a long session, for whoever picks this up next. The companion document
`STRICT_SCALAR_FN_RESEARCH.md` is the full design record with derivations, measured tables, and the
alternatives that were rejected and why; this file is the orientation and the plan. Read this first,
then that. Both are working notes that live on this branch only: they are not meant to land in the
pull-request stack, and the tracking issue is their public form.

The design is now proposed publicly, in three issues that Connor wrote by hand and that supersede
this file wherever they disagree:

- **Epic 9128, Row-oriented scalar functions.** Goals, motivation (the `Hypot` walkthrough of
  everything an author has to get right today), non-goals, and the current benchmark/codegen
  record in its [follow-up comment](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802).
  Status is *Proposed*, and it links this branch and its diffshub comparison as the prototype, so
  the branch is now publicly referenced: do not rewrite or delete it.
- **9129, Define the `RowFn` API.** The author-facing surface, with the trait sketches and worked
  `Hypot` and `CosineSimilarity` examples. Its Steps list is the API work plan.
- **9130, Execute `RowFn` over Vortex arrays.** The private lifting, the two null-handling
  contracts, and adaptive execution. Its Steps list is the machinery work plan. Marked WIP because
  the mechanics may optimize further.

Both tracking issues end with an "Implementation history" section reading "None yet". Add each pull
request there as it lands; that is the intended record of progress.

This branch is where the design was worked out and measured, and the plan below turns it into small
reviewable pull requests cut fresh from develop.

## Current benchmark and codegen record

The [issue #9128 benchmark and codegen follow-up](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802)
is the authoritative current comparison. It states the machine and harness, reports two-run
fastest and median results, documents the control-arm limitations, and includes representative
generated LLVM IR and assembly. Older shared-VM numbers in this handoff remain historical context,
not the current branch-versus-develop record.

## Where the work lives

Everything is on `claude/strict-scalar-fn-abstraction-ah88x3`, which is the designated development
branch and the only branch to push to. Two earlier split branches
(`claude/strict-scalar-fn-vtable-ah88x3`, `claude/row-fn-ah88x3`) are fully contained in this one
and should be deleted from the remote; `git push --delete` is blocked by the permission classifier,
so that is a manual step in the GitHub UI. There is no open pull request, by design: the plan is to
propose this as a clean stack of small PRs cut fresh from develop, not to merge this branch.

A scratch worktree at `/home/user/vortex-nullproto` (branch `proto/null-strategies`) holds an
unmerged prototype kept only for reference. It is not to be pushed.

Working notes from the session live in the session scratchpad, notably
`null_strategy_report.md` (prototype measurements), `branch_skip_impl_report.md` (implementation
report), `nonstrict_survey.md` (the non-strict function survey), and
`null_strategy_prototype.diff`. These are disposable; anything load-bearing was folded into
`STRICT_SCALAR_FN_RESEARCH.md`.

## What exists, in dependency order

**`RowFn`** (`vortex-array/src/scalar_fn/row/`) is the only authoring trait above
`ScalarFnVTable`: an implementor names witness types, picks concrete element types per batch in
`dispatch`, and hands the framework a row closure. One blanket impl,
`impl<F: RowFn> ScalarFnVTable for F` in `row/row_fn.rs`, derives everything else. Adopters:
`byte_length`, geo `contains`/`intersects`/`distance`, tensor `l2_norm`/`inner_product`/
`cosine_similarity`/`l2_denorm`.

**The lifting** (`vortex-array/src/scalar_fn/row/lift.rs`, `pub(super)`) turns that row loop into a
full `execute`: null propagation, constant folding, nullability widening, output dtype
reconciliation, and per-batch null-strategy selection. `Batch` carries one batch's facts and its
`execute` takes the kernel as two closures, one running it over whichever arguments the lifting
hands over and one trying branch-and-skip over the conjoined mask. This used to be a public
`StrictScalarFnVTable`, which was deleted; see the last section of the research document for why and
for what replaced each of its members.

**`RowVisitor`** has three methods. `visit_prepared` is the primitive for returning kernels and
runs a `prepare` closure once per batch over whichever operands are batch-constant, threading the
result to every row by `&P`. `visit` is a provided method deriving from it with unit state (a ZST,
so it monomorphizes to the identical loop; verified by benchmark parity). `visit_into` is the
primitive for kernels writing into an `OutputSink`, which exists for outputs whose width is runtime
data (a tensor row) or that append into one buffer for the batch.

**Adaptive null strategy** is the most recent framework work. `row_null_handling` (in
`row/execute.rs`) derives `Dense` (run over the garbage behind null rows, mask after) or `Filter`
(the kernel must never see a null row) from element dense-safety and fallibility. `Filter` names
that *contract*, and two mechanisms satisfy it: the original filter-and-scatter, and
branch-and-skip, which decodes full length null-tolerantly and visits only mask-set rows
word-at-a-time. The lifting chooses per batch from `Mask::true_count` and
`InputElement::DECODE_SHRINKS_WHEN_FILTERED`, a defaulted const that is `true` only for elements
whose decode parses every row (geo geometries). Threshold
`BRANCH_MIN_SURVIVING_FRACTION = 0.75`, with the measured crossover tables cited in its doc
comment. Function authors write nothing to benefit. To compare or force a strategy from a test or
benchmark, `execute_row_fn_with_strategy` (test-harness gated) is the only seam.

## The essential invariants, so you do not break them by accident

- **Dispatch must be pure in `(options, args)`** and is value-blind: it sees dtypes, never data.
  Plan time and run time both go through it. This is why constant-compute hoisting had to live
  inside the visit (`prepare`) rather than in dispatch.
- **The witnesses pin what the framework reads before dispatching**: arity, dense-safety,
  fallibility, and now `DECODE_SHRINKS_WHEN_FILTERED`. `assert_witness_agrees` makes a
  contradicting dispatch a compile error, even in an arm that never runs.
- **`prepare` must never be load-bearing for validation.** An empty batch decodes as non-constant,
  so a check that only runs in `prepare` will not run at all. It is also infallible by design,
  because fallibility is read off the witnesses before dispatch.
- **`P` has no `Send`/`Sync` bounds, deliberately.** Geo's prepared geometry carries `Rc`/`RefCell`.
  Adding bounds later is a breaking change.
- **Both output forms build an all-valid column**, which is what licenses the derived
  `validity() = union_child_validities`. Anything that lets a kernel emit nulls must change that
  derivation in the same commit (this is what step 3 of the plan below is about).
- **A `RowFn` cannot also implement `ScalarFnVTable`**, since the blanket impl claims the slot.
  `RowFn` therefore mirrors any `ScalarFnVTable` method an adopter needs to vary. Today that is
  `validity` and nothing else, and `reduce` was dropped when the strict trait went because no
  adopter used it. Mirror the next one when a real adopter needs it, not before.
- **A fallible kernel must never run behind a null row**, which is why fallibility forces the
  `Filter` contract, and why branch-and-skip visits only set bits. The hostile tests (views
  pointing at nonexistent buffers, poison zero divisors behind nulls) exist to catch a regression
  here; keep them.
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
from nullable *outputs* (step 3 below), which stay strict and are an explicit goal of the epic.

The epic's non-goals also match the exclusion taxonomy the research arrived at independently:
`RowFn` is not for columnar or zero-copy kernels (`not`, `list_length`), kernels with state shared
across rows (`like`), or heterogeneous variadic kernels (`RowEncode`, `pack`, `case_when`).

## What to do next, in order

0. ~~**Fold the lifting into the row layer and drop the public strict trait.**~~ Done, in three
   commits on this branch: revert the two remaining columnar ports, take the two benchmark control
   arms off the trait, then delete it. No PR in the stack has to argue for a public trait that the
   measurements did not support.
1. ~~**Write the tracking issue.**~~ Done, by hand: epic 9128 with sub-issues 9129 and 9130. Their
   Steps checklists are the authoritative work plan; read them before planning a change, and answer
   their unresolved questions with evidence from this branch where it exists.
2. **Settle adaptive execution.** This is the next design task, ahead of any porting, because it is
   the least settled thing in the proposal and design churn invalidates measurement. 9130 marks it
   WIP; the threshold is a global const calibrated on a shared VM whose noisiest arms are the ones
   that determine it; and the open question is not just the number but the *shape*. See "Settling
   adaptive execution" below for the specific questions to answer.
3. **Land the baseline benchmarks on develop, in parallel with step 2.** Independent of the design
   work, and required before any porting PR can show a comparison. See "The measurement plan"
   below.
4. **The PR stack**, cut fresh from develop, each linking 9129 or 9130 and recorded in that issue's
   Implementation history: row core plus `byte_length`; then `OutputSink` plus `l2_denorm`; then
   `l2_norm` and `inner_product` on the plain visit; then `visit_prepared` with both of its users,
   `cosine_similarity` and geo; then adaptive execution and its benches. Land each API surface with
   its first user, which is why `visit_prepared` travels with geo rather than with cosine (the
   prepared-geometry win carries the API, cosine's few percent does not) and why adaptive execution comes after geo
   (geo is its only production beneficiary, since every other adopter is dense-safe). Two of 9129's
   steps are already satisfied on this branch and only need carrying over: serialized metadata of
   migrated functions is preserved (the per-function array serde described above), and the "when to
   use `RowFn`" guidance is written in `vortex-array/src/scalar_fn/mod.rs`.
3. **Option outputs**, the one extension with demonstrated in-tree demand: `list_sum` (a valid
   empty list sums to null) and `variant_get` (missing path yields null) are strict but excluded
   from `RowFn` today only by the all-valid-output rule. Needs an `Option<T>`
   output form with a nullable element dtype, a nullability bit on the return witness, and derived
   `validity()` becoming `None` for such functions. `is_strict` stays true. Three tests were
   deleted with the strict trait for exactly this gap, since they used a synthetic kernel returning
   null for a valid row: `a_non_total_kernel_declines_precomputed_validity`,
   `a_non_total_kernel_is_still_strict` and `a_non_total_kernel_keeps_its_own_nulls`, recoverable
   from `git show a605e5779^:vortex-array/src/scalar_fn/strict/tests.rs`. Restore them as row
   functions when this lands.

## Settling adaptive execution

What exists works and is tested, and the numbers on this branch are real. What is unsettled is
whether the *selection rule* is the right shape, and that question should be answered before the
porting PRs, because every later measurement is taken through it.

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
  compute-bound work, so x86 and Apple Silicon need not agree. Partly blocked on hardware, and the
  answer decides whether a fixed number is defensible at all.
- **Do sinks need branch-and-skip?** They stay on dense and filter today because a sink has no
  skipped-row representation. Resolving this needs either a pre-filled sink row or a sparse writer,
  or a documented decision that sinks keep filtering.
- **Is the strategy observable in production?** The only seam today is test-harness gated. A session
  option, or tracing the chosen strategy, may be wanted for debugging a slow query.
- Plus the recorded follow-up: avoid probing `reduce_encoded` twice when branch execution turns out
  unsupported.

## The measurement plan

The repo runs CodSpeed on every pull request, and the workspace aliases `divan` to
`codspeed-divan-compat`, so every registered bench is measured against a develop baseline
automatically and behaves as ordinary divan when run locally. That makes CI the most credible source
for the no-regression claim, since it is the team's own process rather than anyone's laptop.

**The constraint that shapes everything: CodSpeed compares a pull request against develop for the
same benchmark name.** A benchmark introduced in the same change it measures has no baseline and
produces no comparison. Every benchmark on this branch is new, and develop currently has no
benchmark for any tensor or geo scalar function at all, so the baselines have to land first, as their
own pull request against develop.

That baseline pull request is purely additive test tooling, uncontroversial, valuable to the project
on its own, and a chance to get the team to agree on what counts as evidence before any design
debate. It should cover the current develop implementations of: tensor `l2_norm`, `l2_denorm`,
`inner_product` and `cosine_similarity`; geo `contains` and `intersects`, each with non-nullable and
nullable arms; and `byte_length`. Two rules make it work:

- **Exercise the functions through the public expression and execution API only.** The versions on
  this branch reference framework internals (forced-strategy seams, hand-written control arms
  implementing `ScalarFnVTable`), which cannot exist on develop. A bench that goes through the
  public path is also the only kind that keeps measuring the same thing before and after a port.
- **Identical benchmark and arm names on both sides**, or CodSpeed cannot line them up.

Arms that compare framework strategies against each other (forced filter versus forced branch versus
auto) cannot be baselined, because they need machinery that does not exist on develop. That is fine:
their claim is "auto tracks the faster forced arm", an internal comparison rather than a before and
after, so they land with adaptive execution.

**Adaptive execution is less entangled with the other measurements than it looks.** Every adopter
except geo is dense-safe and infallible, so it takes the dense path and never reaches strategy
selection at all: `l2_norm`, `l2_denorm`, `inner_product`, `cosine_similarity`, and `byte_length` (which
ships at `BytesLen`) are unaffected no matter what the rule becomes. Only geo's nullable batches are
affected, which is why the geo baseline needs both nullable and non-nullable arms: it keeps the
port's parity claim separable from adaptive execution's win. So steps 2 and 3 above genuinely can run
in parallel.

Priority for wall-clock runs on real hardware, highest first: parity for the dense-path adopters (all
four already have in-binary hand-written control arms, which is the credible form: same compiler,
same run, arms side by side); then the crossover sweep across 0 to 90% null density; then the
headline wins (geo `contains` prepared, `byte_length` adaptive); then a small-batch sweep from 100 to
1M rows, which is both the known weak spot and an input to the batch-size question above. Report
`fastest` and `median` from at least two runs with machine metadata, since `fastest` alone reads as a
cherry-picked minimum. Expect cosine's small prepare win to shrink or vanish on a wider machine: the
explanation for why it was only a few percent is that the hoisted arithmetic rode in spare slots of a
latency-bound FMA chain, and a machine with more slack removes even that. Better to drop the claim
ourselves than have a reviewer find it.

## What the tracking issues ask that this branch has not answered

Four of their unresolved questions are genuinely open, and two of them this branch never considered:

- **Are `InputElement`, `OutputElement` and `OutputSink` public downstream extension points, or only
  cross-crate extension points within Vortex?** (9128, 9129.) This branch treats them as
  Vortex-internal: `vortex-tensor` and `vortex-geo` implement them, nothing outside does. It matters
  because a downstream implementor makes every associated constant a compatibility surface, and
  three already carry defaults chosen for internal convenience (`DECODE_SHRINKS_WHEN_FILTERED` is
  `false`, `decode_null_tolerant` forwards to `decode`). If they go public, audit which defaults are
  the *safe* answer rather than the common one.
- **Does `OutputSink::sink_dtype` need access to function options?** (9129.) Today it takes only the
  input dtypes, which is what lets a tensor row's runtime width come from its arguments. No adopter
  wants options there yet, so the question is whether option-dependent output dtypes can stay
  unsupported initially.
- **What benchmark set and regression threshold gate replacing a hand-written implementation?**
  (9130.) This branch has no policy, only precedent: per-adopter divan benchmarks with constant and
  non-constant arms, `fastest` of two runs, and one accepted regression (1 to 3% on the
  always-overlapping `intersects` arm, taken for a 2.1x win on the disjoint arm). Worth turning that
  precedent into a stated rule.
- **Do sinks need branch-and-skip, and are nullable outputs required for v1?** (9128, 9129, 9130.)
  Sinks currently stay on dense and filter because a sink has no skipped-row representation;
  nullable outputs are step 3 above.

9130's three non-blocking follow-ups are the same items this branch recorded, so they need no
separate tracking here: the double `reduce_encoded` probe when branch execution is unsupported, the
global 75% threshold pending a second per-row-decode element, and the fallible branch loop's
inability to stop iterating at the first error. Two more of the same kind that the issues do not
mention: geo's null-tolerant decode still arrow-exports the full column, where slicing runs of valid
rows would blunt filter's sparse-validity advantage enough to retire the threshold for geo, and the
lifting's small-batch prelude cost discussed above.

## Loose ends, and issues worth filing separately

The reworked benchmark control arms in `vortex-array/benches/strict_validity.rs` and
`vortex-tensor/benches/l2_denorm.rs` have now been run in release; use the [issue #9128 follow-up comment](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802), rather than
the historical figures in this document, when quoting their results. Commit `72e01f96c` (the first version
of this file) is pushed unsigned, so GitHub shows it unverified; every commit after it is signed, and
fixing it needs a force-push, which is blocked.

Also worth an issue on its own, unrelated to this work: `Between::validity` declares the strict
three-way conjunction while its fallback execute path joins two comparisons with Kleene AND, so
with per-row nullable bounds the lazy validity and the executed result disagree (a valid `false`
reported as null). Verified on develop. Separately, `NotKernel` appears to have no implementations
and looks like dead code.

## How to work on this branch

Read `CLAUDE.md` and follow it. Invoke the `rust-style` skill before writing Rust: this branch is
written to Connor's personal style and reviewers will notice the difference. No em dashes anywhere,
in code, docs, or commit messages. Sign every commit off exactly as
`Signed-off-by: "Connor" <connor@spiraldb.com>`.

Verification that matters for changes here:

```bash
cargo nextest run -p vortex-array -p vortex-geo -p vortex-tensor
cargo clippy --all-targets --all-features -p vortex-array -p vortex-geo -p vortex-tensor
cargo +nightly fmt --all
git diff --check
cargo build -p vortex -p vortex-file -p vortex-datafusion
```

3,596 tests pass across the three crates as of the strict-trait deletion.

**`origin/develop` in this container is stale, and it will bite you.** It predates the
`is_null_sensitive` to `is_strict` rename (upstream #8930), so restoring a file from it yields a body
that does not compile against this branch. This branch is based on `e58fb5861`. To recover a
function's pre-port body, use the commit before the port instead: `24d1933e1^` for `not`,
`list_length` and `list_sum`. More generally, prefer `git log --oneline <base>..HEAD` over any
assumption about what develop contains, and `git fetch origin develop` before comparing against it.

Benchmarks are divan (aliased to `codspeed-divan-compat`, so they also run in CI); report `fastest`
and `median` from at least two runs and state the machine. The historical measurements here used a
shared 4-vCPU VM; the current 7950X comparison is recorded in the [issue #9128 follow-up comment](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802). Any new bench must be
registered as a `[[bench]]` with `harness = false` in the crate's `Cargo.toml`. Per-adopter
benchmarks with constant and non-constant arms are the convention, and where a claim is "as fast as
the hand-written version", put both arms in one binary rather than comparing across builds. Disk is tight:
build narrowly and delete `target/debug` subdirectories on ENOSPC. If cargo fails with exactly
`sccache: error: Operation not permitted`, rerun that command with a `RUSTC_WRAPPER=` prefix.

Force-pushing and remote branch deletion are both blocked by the permission classifier, so plan
history edits before pushing rather than after. Do not open a pull request unless asked.
