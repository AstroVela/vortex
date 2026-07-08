# Overview

The Developer Guide is for people working *on* Vortex or building *against* its lower-level
interfaces — as opposed to the [User Guide](../user-guide/index.md), which is for people querying
Vortex data with engines and dataframe libraries. It is organized by what you are trying to do.

## How this guide is organized

- **Extending** — add new capabilities to Vortex: custom [encodings](extending/writing-an-encoding.md),
  [layouts](extending/writing-a-layout.md), [compute functions](extending/writing-a-compute-fn.md),
  and [extension dtypes](extending/extension-dtypes.md). Start here if you want Vortex to understand a
  new compression scheme or logical type.
- **Embedding** — drive Vortex from another system or language: the [C FFI](embedding/ffi.md),
  [C++ binding](embedding/cxx.md), [Scan API](embedding/scan-api.md), and [GPU](embedding/gpu.md)
  paths. Start here if you are building a query-engine connector or a data source.
- **Internals** — how the implementation actually works: the [crate architecture](internals/architecture.md),
  [vtables](internals/vtables.md), [session system](internals/session.md),
  [async runtime](internals/async-runtime.md), [execution](internals/execution.md),
  [I/O subsystem](internals/io.md), and [CUDA](internals/cuda.md). Start here if you are contributing
  to the core.
- **Integrations** — implementation notes for the engine connectors
  ([DataFusion](integrations/datafusion.md), [DuckDB](integrations/duckdb.md),
  [Spark](integrations/spark.md)). These complement the user-facing how-tos in the User Guide with
  the *why* and *how* of each connector.
- **[Language Bindings](language-bindings.md)** and **[Benchmarking](benchmarking.md)** round out the
  guide.

:::{note}
Several pages in **Extending** and **Embedding** are still under construction — they currently list
their planned contents rather than full walkthroughs. The **Internals** and **Integrations** sections
are the most complete. If a page you need is a stub, the [API reference](../api/index.md) and the
source are the best fallback, and the [Slack community](https://vortex.dev/slack) can help.
:::

## A note on where the difficulty lives

A recurring theme across this guide — and a good lens for understanding Vortex's design — is that
**reading the bytes of a Vortex file is not the hard part.** As the
[Reading a File](../specification/reading-a-file.md) walkthrough shows, a correct reader is a magic
number, a postscript, a footer, the root dtype, and a layout tree: an afternoon's work.

**The hard part is getting *high performance* out of those bytes** — pruning irrelevant row ranges
from zone statistics, pushing projections and filters down so you fetch and decompress as little as
possible, and scheduling I/O so thousands of small segment reads become a few large coalesced ones.
That is where the bulk of Vortex's engineering goes, and it is why the
[concepts](../concepts/index.md) (layouts, the scan API) and the
[internals](internals/io.md) sections exist. If you only ever need to read a file, you can stop at
the [specification](../specification/index.md). If you need it to be *fast*, this guide is where the
real work is documented.
