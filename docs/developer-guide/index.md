# Developer Guide

Guide for extending, embedding, and contributing to the Vortex ecosystem. See the
[Overview](overview.md) for how the guide is organized and where to start.

```{toctree}
---
maxdepth: 1
---

overview
```

```{toctree}
---
maxdepth: 2
caption: Extending
---

extending/index
extending/writing-an-encoding
extending/writing-a-layout
extending/writing-a-compute-fn
extending/extension-dtypes
```

```{toctree}
---
maxdepth: 2
caption: Embedding
---

embedding/index
embedding/ffi
embedding/cxx
embedding/scan-api
embedding/gpu
```

```{toctree}
---
maxdepth: 2
caption: Internals
---

internals/architecture
internals/session
internals/async-runtime
internals/vtables
internals/execution
internals/stats-pruning
internals/io
internals/cuda
```

```{toctree}
---
maxdepth: 2
caption: Integrations
---

integrations/datafusion
integrations/duckdb
integrations/spark
```

```{toctree}
---
maxdepth: 2
caption: More
---

language-bindings
benchmarking
```
