# Edition golden files

Guidance for agents and humans touching anything under `vortex/goldenfiles/editions/` or the
serialized form of any edition member. The test suite lives in
`vortex/src/editions/golden_tests/`; the policy it enforces is specified in
`docs/specs/editions.md` ("How serialized objects evolve").

## What these files are

Every object reachable from any edition — `core` and `unstable` alike — has a directory
`<kind>/<id>/` holding one file per **historical serialized form** of that object:
`v001.bin`, `v002.bin`, ... Each kind pins its real durable surface:

| Kind               | Golden bytes                                                        |
| ------------------ | ------------------------------------------------------------------- |
| `arrays`           | IPC stream of a single-chunk array (dtype + id table + array body)  |
| `layouts`          | A whole tiny Vortex file whose footer tree contains the layout      |
| `aggregations`     | The aggregate function's serialized options, as stored in zone maps |
| `expressions`      | IPC stream of a scalar-fn array (how expressions persist in files)  |
| `extension_dtypes` | FlatBuffer encoding of a `DType` using the extension                |

The tests assert two things for every object:

1. **Write pinning.** The bytes the current code serializes are identical to the *newest*
   golden version. Any change to the serialized output — a new proto field, a reordered
   buffer, a new flatbuffer table — fails this check.
2. **Read forever.** *Every* golden version, however old, still deserializes to the
   fixture's logical value. This is the editions guarantee: all serialized forms that have
   ever existed stay readable.

## The rules

- **Whenever a serialized format changes — e.g. a new field is added — a new golden file
  must be added to the suite.** Run the tests with `UPDATE_GOLDENFILES=1` (for both feature
  sets, see below); the harness writes the next `vNNN.bin` for every object whose bytes
  changed. Commit the new files together with the format change.
- **Never edit or delete an existing golden file.** Old versions are the read-forever
  contract; the update mode only ever *adds* files. If an old golden stops deserializing,
  fix the deserializer — the golden is right, the code is wrong.
- **Never change a fixture that already has goldens.** The fixture's logical value is what
  every historical golden is checked against. If you need different data, that is a new
  object or a new suite, not an edit.
- **A new edition member needs a fixture and a golden in the same change.** The
  completeness tests fail if any declared member of any edition has no fixture, and if a
  golden directory exists for an id no edition declares. The single allowed gap is the
  documented exemption list in `golden_tests/mod.rs` (currently `vortex.stats`, the legacy
  read-only layout).
- **A version bump without an intended format change is still information.** If goldens
  change because a compressor library or writer default changed, add the new version and
  say so in the commit: the old bytes remain readable, and the diff documents when the
  output shifted.

## Running

```bash
# Check (default): fails if bytes drifted or any golden stopped deserializing.
cargo nextest run -p vortex --lib -E 'test(golden)'
cargo nextest run -p vortex --lib --features unstable_encodings -E 'test(golden)'

# Add new golden versions after an intentional format change:
UPDATE_GOLDENFILES=1 cargo nextest run -p vortex --lib -E 'test(golden)'
UPDATE_GOLDENFILES=1 cargo nextest run -p vortex --lib --features unstable_encodings -E 'test(golden)'
```

Run both feature sets: `unstable`-family members only have fixtures when
`unstable_encodings` is compiled in. Golden bytes must not depend on the feature set — the
layout files are written by a session with only the `core` edition enabled for exactly this
reason — so the second update pass must only ever add goldens for unstable-family ids.
