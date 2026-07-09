# 4. Design: making the cost model pluggable

*Read this for the concrete proposal: the trait, where it hooks into today's code, how current
behavior becomes the default model, and what the presets look like.*

**TL;DR:** Split the three judgments currently fused inside `expected_compression_ratio` —
feasibility (stays in schemes), mechanical size estimation (stays in schemes / sampling), and
**policy (moves into a `CostModel` owned by the compressor)**. The selection loop stops
comparing ratios and starts minimizing a model-provided cost; the default model reproduces
today's ratio ordering exactly. Sampled candidates already return their full encoding tree, so
a time-based model can price cascades with *zero new scheme API*. Profiles (`max_ratio`,
`scan_speed`, `point_lookup`, `gpu_decode`) become values of one type instead of hand-curated
scheme lists.

- [Goals and non-goals](#goals-and-non-goals)
- [The three pieces](#the-three-pieces)
- [How selection changes](#how-selection-changes)
- [The models](#the-models)
- [Where it plugs in](#where-it-plugs-in)
- [Worked examples: rankings flip with profiles](#worked-examples-rankings-flip-with-profiles)
- [Alternatives considered](#alternatives-considered)
- [Compatibility and migration notes](#compatibility-and-migration-notes)

## Goals and non-goals

Goals:

1. **Default behavior unchanged** — same files, byte-for-byte, when no model is configured
   (this is testable and should be tested; see doc 5 phase 1).
2. **Policy in one place** — the delta tax, the "prefer X above Y" ordering comments, the
   CUDA exclusion list, and future speed preferences all become *data in a model*, not code
   in schemes.
3. **Cheap and deterministic** — table lookups and arithmetic only; no timing on the write
   path; same input ⇒ same output.
4. **Third-party schemes keep working** unmodified, at worst priced conservatively.

Non-goals for the first iteration (deliberately): adaptive cascade search (the cost model is a
*prerequisite* for it — see doc 5 phase 4), cross-chunk consistency, learned models, runtime
re-encoding, and moving the layout-level `DictStrategy` decision (flagged as an integration
point below).

## The three pieces

### 1. `Candidate` — the mechanical facts about one option

```rust
/// What the compressor knows about one (scheme, array) option at selection time.
pub struct Candidate<'a> {
    pub scheme: SchemeId,
    /// Estimated output size: from an analytic verdict (input/ratio) or measured on the sample.
    pub estimated_nbytes: u64,
    pub input_nbytes: u64,
    pub n_values: u64,
    /// The compressed *sample* array, when sampling ran. Its encoding tree is the best
    /// available prediction of the full-array tree — today only `.nbytes()` is read from it
    /// (`estimate.rs:218-229`); a time model walks it and prices each node.
    pub sampled: Option<&'a dyn Array>,
    pub stats: &'a ArrayAndStats,
    pub cascade: &'a [(SchemeId, usize)],   // where we are in the tree (depth, ancestry)
}
```

The single most useful pre-existing fact: **when a scheme is sampled, the compressor already
holds a real compressed array whose tree includes all (greedily chosen) descendants** —
`Dict(codes=BitPacked, values=FSST(...))`. Costing a cascade therefore requires no new
`Scheme` methods. For analytic (`Verdict::Ratio`) candidates there is no tree; the model
prices the scheme's typical output shape from its table (each built-in scheme's output
encoding is static and known), falling back to a conservative default for unknown third-party
schemes (plus a `tracing` warning).

### 2. `CostModel` — the policy

```rust
/// Prices a candidate. Implementations must be pure functions of their inputs
/// (determinism) and cheap (called O(schemes × cascade levels) per chunk).
pub trait CostModel: Send + Sync + 'static {
    /// Estimated cost of reading through this candidate. Lower is better; `None` rejects.
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost>;

    /// Cost of leaving the array canonical — the baseline every candidate must beat.
    /// (Generalizes today's `ratio > 1.0` validity rule, `estimate.rs:142-149`.)
    fn canonical_cost(&self, data: &ArrayAndStats, n_values: u64) -> Cost;

    /// Best cost this candidate could *possibly* achieve if its size estimate came back
    /// perfect (bytes → minimum plausible). Used to skip deferred sampling/callbacks that
    /// cannot win — generalizes the `best_so_far` early-exit threshold (`estimate.rs:24-45`),
    /// and can prune e.g. expensive FSST sampling under decode-heavy profiles.
    fn lower_bound(&self, scheme: SchemeId, data: &ArrayAndStats) -> Cost { Cost::ZERO }
}
```

`Cost` is a newtype over `f64` in **estimated nanoseconds per value** (units matter: they make
coefficients auditable and let "bytes" and "decode passes" be added meaningfully; see the
formula below). A strict total order over costs is preserved, so tie-breaking by registration
order and the two-pass deferred evaluation keep working unchanged.

### 3. `CostInputs` — data the model consults

- **`DecodeCostTable`** — `ns/value` coefficients keyed by (encoding id, dtype class, and one
  shape parameter such as bit width where relevant), plus a per-array fixed overhead. Compiled-
  in defaults per coarse arch class; regenerable offline from the existing criterion benches
  (doc 5 phase 0); user-overridable.
- **`CapabilityMatrix`** — derived at compressor build time from the *same session's kernel
  registry* (specialized `execute_parent` kernels are keyed by (operation, child encoding) in
  `vortex-array/src/optimizer/kernels.rs`), plus a small static annotation for random-access
  class and GPU-decodability. See the appendix for today's matrix.
- **`WorkloadProfile`** — `{ effective_bandwidth, weights: {full_scan, selective_scan,
  random_access}, hardware_class }` with per-column overrides.

## How selection changes

`choose_best_scheme` (`vortex-compressor/src/compressor.rs:412`) keeps its two-pass shape:

| Today | Proposed |
|---|---|
| `EstimateVerdict::Ratio(r)` → `EstimateScore` | build `Candidate { estimated_nbytes: input/r, sampled: None, .. }` → `model.cost(..)` |
| `DeferredEstimate::Sample` → measure bytes | measure bytes **and keep the sampled array** → `Candidate { sampled: Some(...) }` → `model.cost(..)` |
| valid iff `ratio > 1.0` | valid iff `cost < model.canonical_cost(..)` |
| `best_so_far` ratio threshold to callbacks | skip deferred work when `model.lower_bound(..) ≥ best cost so far` |
| winner = argmax ratio | winner = argmin cost |
| `AlwaysUse` short-circuits | unchanged (it's semantic normalization — decimal/temporal — not cost) |
| accept iff `after_nbytes < before_nbytes` (`compressor.rs:381`) | **keep** as a hard invariant (compression never grows bytes), *and* the cost acceptance above |

Everything else — exclusion rules, stats plumbing, cascade budget, constant handling,
tracing — is untouched. The winning-compression trace span gains `estimated_cost` next to
`estimated_ratio`, which keeps the estimator-accuracy observability story intact.

## The models

### `SizeCost` (the default — reproduces today exactly)

`cost = estimated_nbytes`, `canonical_cost = input_nbytes`. For a fixed input size, argmin
bytes ≡ argmax ratio, and `cost < canonical` ≡ `ratio > 1.0`, so selection order is identical;
golden-corpus tests should prove byte-identical output. The delta tax and `min_ratio` move
here as per-scheme priors (a multiplier table with exactly one non-1.0 entry, documented as
such) rather than living inside `DeltaScheme`.

### `TimeCost` (the point of the exercise)

For candidate `E` over `n` values, with profile weights `w`:

```
cost(E) = w.full_scan   · [ bytes(E)/bandwidth + Σ_node decode_ns(node) ]
        + w.selective   · [ bytes(E)/bandwidth + pushdown_aware_scan_ns(E, ops) ]
        + w.random      · [ ra_class_ns(E) ]                    (per expected access)
canonical_cost = w.full_scan · bytes(canonical)/bandwidth + ...  (decode = 0)
```

- `Σ_node decode_ns` walks the sampled tree (or the scheme's static shape) — this is where
  cascade depth stops being `MAX_CASCADE = 3` folklore and becomes "a fourth level must buy
  its decode pass in IO savings".
- `pushdown_aware_scan_ns` consults the `CapabilityMatrix`: an op with a specialized kernel
  costs ~kernel throughput on *encoded* data; a missing kernel costs the full decode plus the
  canonical op (the step function). This is the term that can make an encoded candidate
  *cheaper than canonical* — impossible to express today.
- `ra_class_ns` prices the O(1) / O(log n) / block-decode classes from the appendix.
- `bandwidth → 0` recovers `SizeCost` (bytes infinitely expensive): today's behavior is a
  *point in this family*, which is the cleanest statement of why this design subsumes rather
  than replaces the current model.

### Presets (the user-facing surface)

```rust
CostModel::max_ratio()                      // SizeCost — today's default
CostModel::scan_speed(Bandwidth::NVME)      // TimeCost, w = {1, 0, 0}
CostModel::query(Bandwidth::S3)             // TimeCost, w = {0.3, 0.6, 0.1}-ish
CostModel::point_lookup()                   // TimeCost, w = {0, 0.2, 0.8}
CostModel::gpu_decode()                     // TimeCost + cost=∞ for non-GPU-decodable nodes
```

`gpu_decode()` deserves emphasis: it replaces the hand-maintained
`only_cuda_compatible()` exclusion list (`vortex-btrblocks/src/builder.rs:163-196`) with a
derived property, so a new encoding with a CUDA kernel becomes GPU-eligible without anyone
editing a list.

## Where it plugs in

Existing seams (mapped in doc 1) extend naturally:

- `CascadingCompressor::new(schemes)` gains `.with_cost_model(Arc<dyn CostModel>)`
  (default `SizeCost`).
- `BtrBlocksCompressorBuilder` exposes the presets; `with_compact()` becomes sugar for
  scheme-set + `max_ratio()`.
- `WriteStrategyBuilder` (`vortex-file/src/strategy.rs:236`) passes the model through
  `with_btrblocks_builder` / `with_compressor`; per-column profiles can ride the existing
  per-field-path mechanism (`with_field_writer`, `strategy.rs:189`).
- Python: `VortexWriteOptions.default() / .compact()` grow `.scan_optimized()` /
  `.point_lookup()` siblings — presets only; the trait stays a Rust-level extension point.
- **Integration point to flag:** the layout-level `DictStrategy` decides dictionary encoding
  *before* the compressor runs and the writer's data compressor excludes `IntDictScheme`
  (`strategy.rs:254-260`). A speed-aware model that wants to punish or reward dictionaries
  must eventually inform that decision too; near-term it simply means the model's dict
  opinions apply to the stats/values compressor and to embedded uses, and #7697's "hardcode
  DictScheme logic into the compressor?" question is really a question about *which layer
  owns dictionary policy*.

## Worked examples: rankings flip with profiles

Why one static answer can't work (sizes/costs illustrative):

| Column | Candidates (est.) | `max_ratio` picks | `point_lookup` picks | why |
|---|---|---|---|---|
| u32, avg run 6, low cardinality | RunEnd 5.1×, Dict 4.8×, BitPack 2.5× | RunEnd | Dict (or BitPack) | run-end pays a binary search per access; dict is two O(1) reads |
| u64 near-monotone (timestamps) | Delta 4.0×, FoR+BP 3.2× | Delta (if enabled) | FoR+BP | delta point-read decodes a whole 1024-lane chunk |
| strings, filter-heavy | FSST 3.5×, Dict 2.9× | FSST | *workload-dependent* | dict compares on codes; but FSST has DFA LIKE without decompression — genuinely close, needs the capability matrix to price |
| any, GPU pipeline | zstd 6×, BitPack 2.5× | zstd | — | `gpu_decode()`: zstd = ∞ (no GPU decode), BitPack wins at any ratio |

The point is not the specific numbers; it's that the *same mechanical estimates* support all
four answers once policy is a parameter.

## Alternatives considered

| Alternative | Verdict |
|---|---|
| **More presets as scheme sets** (extend `only_cuda_compatible` pattern) | Cheapest; already how it's done; can't express "delta fine for scans, not for lookups", can't price cascades or IO-vs-decode, leaves policy distributed. Fine as a stopgap, dead end as an architecture. |
| **Add a speed field to `EstimateVerdict`** (e.g. ratio + speed class, combined in the selector) | Smaller diff, but hardcodes the combining function in the framework — the same problem one level up, and every scheme must now self-report speed too. |
| **Full tree search with the cost as objective** | The eventual destination (#7697's adaptive cascading); needs this cost model first anyway. Do the model, then the search. |
| **Learned selection (CodecDB-style)** | Needs the ground-truth harness (doc 5 phase 0) and a retraining story tied to kernel changes; a later layer on top of the same `CostModel` interface, not a first step. |

## Compatibility and migration notes

- `Scheme::expected_compression_ratio` keeps its name and `CompressionEstimate` its variants;
  what changes is the *interpretation* of `Ratio` (a size signal, not a score). Third-party
  schemes compile unmodified.
- Policy constants migrate gradually: delta tax → default model prior (behavior-preserving);
  cardinality/run-length gates can stay in schemes initially (they double as estimation-cost
  savers) and move later if a profile needs to vary them.
- Determinism: models are pure; the sampled tree is already deterministic (fixed seed). The
  `DecodeCostTable` is part of compressor configuration, so "same config ⇒ same file" still
  holds — but note "same file across *machines*" now requires pinning the table (default
  tables are compiled in, not measured at runtime, precisely for this).
- Selection overhead budget: model calls are table lookups; the added work per chunk is
  O(schemes × levels) ≈ dozens of lookups, invisible next to sampling costs. Verify with
  `benchmarks/compress-bench` encode-throughput before/after (doc 5).
