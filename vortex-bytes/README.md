# Vortex Bytes

Aligned, reference-counted byte regions. This is the untyped layer beneath `vortex-buffer`: it
owns memory and nothing else. Element types, lengths, and alignment policy belong to the layer
above; everything here is bytes.

Two handle types divide the world by exclusivity, the way `Vec` and `RawVec` divide it by
responsibility:

- `SharedBytes` is a window that may be aliased. It is `Clone`, and it only ever hands out
  `&[u8]`.
- `UniqueBytes` is a window nothing else can see. It is the only one that hands out `&mut [u8]`,
  and it is not `Clone`.

`UniqueBytes::freeze` moves a window from the second world into the first without allocating, and
`SharedBytes::try_into_unique` moves it back whenever the region turns out to have only one
handle. That round trip is the crate's reason to exist: it is what lets a buffer adopt foreign
memory — a `Vec<T>`, an Arrow buffer, a writable memory map, an FFI allocation — and later regain
mutability without a copy, which `bytes::Bytes::try_into_mut` cannot do.

## Deferred sharing

A handle that has never been shared owns its region outright and describes it inline, in one
tagged word; no refcount is allocated until a second handle actually exists. This is the same
trick `bytes` plays with its "promotable" vtables, and it keeps the common
build-freeze-read-drop path down to a single allocation.

```text
bit  63                              8   7  2   1 0
    ┌─────────────────────────────────┬───────┬─────┐
    │ size                            │ align │ 0 1 │  OWNED  - inline description
    └─────────────────────────────────┴───────┴─────┘
    ┌───────────────────────────────────────────┬───┐
    │ *mut Shared                               │ 0 │  SHARED - refcounted, 8-aligned
    └───────────────────────────────────────────┴───┘
                                              0b10    STATIC - owns nothing
```

## Why it is its own crate

The crate has no dependencies at all - not on `vortex-error`, not on anything else. Failures are
reported as `InvalidAlignment` and through panics rather than through a shared error type, so it
can be built, audited, and reused entirely on its own. `vortex-error` converts `InvalidAlignment`
into a `VortexError` behind its `vortex-bytes` feature, which is what keeps `?` working for
callers inside Vortex.

Everything here is non-generic on purpose. All of the `unsafe` that manages regions — provenance,
refcount discipline, promotion, `realloc` — compiles exactly once rather than once per element
type, and it is small enough to audit as a unit and to test exhaustively under Miri and against a
stateful property model. The crate boundary is what keeps that surface from growing: `vortex-buffer`
can only reach the handle API, never the region internals.
