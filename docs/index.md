---
sd_hide_title: true
---

# Vortex

:::{image} _static/vortex_wordmark.svg
:class: only-light vortex-wordmark
:alt: Vortex
:align: center
:::

:::{image} _static/vortex_wordmark_dark_theme.svg
:class: only-dark vortex-wordmark
:alt: Vortex
:align: center
:::

An extensible ecosystem for compressed columnar data. Vortex spans in-memory arrays,
on-disk file formats, over-the-wire protocols, and query-engine integrations — all built around
recent research from the database community.

These docs are organized by what you're trying to do. If you're **using** Vortex, start with the
guides and API reference. If you're **building on or contributing to** Vortex, start with the
concepts and the format specification.

## Using Vortex

**[Getting Started](getting-started/index)** — Install the `vx` command-line tool or a language
binding, convert a Parquet file, and run your first query in Python or Rust.

**[User Guide](user-guide/index)** — Query Vortex data with **DataFusion**, **DuckDB**, **Spark**,
or **Ray**, and move data to and from **pandas**, **Polars**, and
**PyArrow**.

**[API Reference](api/index)** — Reference for the **Python**, **Rust**, **Java**, **C**, and **C++**
interfaces. The Rust and Python APIs are the most complete; the C, C++, and Java bindings are still
evolving.

## Understanding & extending Vortex

**[Concepts](concepts/index)** — The mental model behind Vortex: how **DTypes**, **Arrays**,
**Encodings**, **Layouts**, and the **Scan API** fit together as composable building blocks.

**[Specification](specification/index)** — The on-disk file format (stable) and the over-the-wire IPC
protocol (still unstable), plus a step-by-step walkthrough of how to **read a Vortex file** from
scratch. The starting point for implementing a reader or porting Vortex to a new language.

**[Developer Guide](developer-guide/index)** — **Extend** Vortex with your own encodings, layouts,
compute functions, and types; **embed** it through the C FFI, C++ binding, or Scan API; or dig into
the **internals**.

## Highlights

- **Compressed arrays**: Operate directly on compressed data with encodings like
  [FastLanes](https://github.com/spiraldb/fastlanes),
  [FSST](https://github.com/spiraldb/fsst), and
  [ALP](https://github.com/spiraldb/alp) — no decompression needed for many operations.

- **Extensible file format**: Zero-allocation reads and FlatBuffer metadata for O(1) column access,
  with a layout system designed to evolve without breaking existing readers.

- **Query engine integration**: Filter and projection pushdown through the Scan API, with native
  integrations for DataFusion, DuckDB, Spark, and Ray.

- **Language bindings**: First-class Python (PyO3) and Rust support, with Java (JNI), C (FFI), and
  C++ (cxx) bindings evolving.

```{toctree}
---
hidden:
---

getting-started/index
user-guide/index
concepts/index
specification/index
developer-guide/index
api/index
project/index
```
