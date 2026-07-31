# Handoff: the row scalar-function framework

Written at the end of a long session, for whoever picks this up next. The companion document
`STRICT_SCALAR_FN_RESEARCH.md` (1,300+ lines) is the full design record with derivations and
measurements; this file is the orientation and the plan. Read this first, then that.

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
  derivation in the same commit (see verdict 2 below).
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

**Null-visible inputs (non-strict `RowFn` over `Option` elements) are designed and deliberately not
built.** The survey found no in-tree customer: 13 of 15 non-strict functions are cheap columnar
mask algebra, and a prototype row-function Kleene AND measured 250-1030x slower than the in-tree
fused word kernel. The two genuinely expensive null-visible kernels (`RowEncode`/`RowSize`) are
blocked by variadic heterogeneous arity, not by nullability. Four functions have value-dependent
output validity (`false AND null` is a valid `false`), for which no validity expression over child
validities exists even in principle, so such kernels belong to a different trait rather than a mode
of this one. Keep the survey's constraint list; do not build the tier.

## What to do next, in order

0. ~~**Fold the lifting into the row layer and drop the public strict trait.**~~ Done, in three
   commits on this branch: revert the two remaining columnar ports, take the two benchmark control
   arms off the trait, then delete it. No PR in the stack has to argue for a public trait that the
   measurements did not support.
1. **The tracking issue for the row framework.** It is a large addition and needs one. It should
   carry the layering, the measured results, the exclusion taxonomy (why `not`, the Kleene
   functions, `l2_denorm`'s constant path and geo `distance` are all correctly *not* row
   functions), the adaptive-strategy story, and the open items below as checkboxes. Draft prose in
   Connor's voice with the `connor-voice` skill and let him read it before posting.
2. **The PR stack**, cut fresh from current develop, each with the lifting private inside the row
   layer and each linking the issue: row core plus `byte_length`; then `OutputSink` plus
   `l2_denorm`; then the tensor ports; then `visit_prepared` together with the geo ports, which is
   where prepare's 9.1x justifies the API and where branch-and-skip earns its place. Land each API
   surface with its first user.
3. **Option outputs**, the one extension with demonstrated in-tree demand: `list_sum` (a valid
   empty list sums to null) and `variant_get` (missing path yields null) are strict but excluded
   from `RowFn` today only by the all-valid-output rule. Needs an `Option<T>`
   output form with a nullable element dtype, a nullability bit on the return witness, and derived
   `validity()` becoming `None` for such functions. `is_strict` stays true.

Known open items, none blocking: the branch fallback probes `reduce_encoded` twice when a dispatch
turns out unsupported; geo's null-tolerant decode still arrow-exports the full column, and slicing
runs of valid rows would blunt filter's sparse-validity advantage enough to retire the threshold for
geo; the fallible branch loop pays one `is_none` check per set row after the first error because
`for_each_set_index` cannot early-return; sinks stay on dense/filter; the 0.75 threshold is global
and should become per-element only when a second per-row-decode element exists to calibrate
against.

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

Benchmarks are divan; report the `fastest` column from two runs and state the machine, because this
environment is a shared 4-vCPU VM where filter-heavy arms have shown up to 2.3x run-to-run spread.
Per-adopter benchmarks with constant and non-constant arms are the convention. Disk is tight:
build narrowly and delete `target/debug` subdirectories on ENOSPC. If cargo fails with exactly
`sccache: error: Operation not permitted`, rerun that command with a `RUSTC_WRAPPER=` prefix.

Force-pushing and remote branch deletion are both blocked by the permission classifier, so plan
history edits before pushing rather than after. Do not open a pull request unless asked.
