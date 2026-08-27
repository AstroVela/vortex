# vortex-morsel

An experimental morsel-driven scan executor for Vortex layouts — the P1 spine of the design in
`docs/developer-guide/internals/scan-execution-models/morsel-based-plan-execution.md`.

A scan is cut into *morsels* (contiguous root row ranges). Each morsel is driven by a tree of
stateful `ExecNode` state machines, inline and depth-first, on one thread that took the morsel off
a single atomic cursor. Nodes never perform IO: `next_plan` *names* the reads a morsel will make
by registering keyed uses against the IO plane, and `execute` may only wait on tickets its own
planning stream emitted.

The crate is a prototype and is not part of the public API. It supports flat, chunked and
struct layouts only; anything else is a build error rather than a fallback. It deliberately holds
no state the V1 `LayoutReader` does not have: no decoded-array cache, and a morsel's IO cells are
released when it retires — the eval counters show requests = uses = decodes everywhere.

## Measured

Against the V1 `LayoutReader` on shape-matched workloads (see
`docs/.../morsel-prototype-p1-findings.md` for the full contract and caveats): geomean 0.650 at
equal thread count, 0.238 at four threads with coalesced morsels, with every configuration
validated against V1's output before timing.

## Running the evaluation

```bash
cargo run --release -p vortex-morsel --features _test-harness --bin morsel-eval
```
