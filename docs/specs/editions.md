# Editions

Vortex defines an evergrowing set of serializable objects: once written, a file can be read
back by any future version of Vortex. **Editions** keep track of these objects and give
groups of them a name.

An edition covers every kind of object whose identifier appears in serialized Vortex data:

| Kind         | Objects                                   | Example ids                        | Where serialized                          |
| ------------ | ----------------------------------------- | ---------------------------------- | ----------------------------------------- |
| `array`      | Array encodings                           | `vortex.alp`, `fastlanes.for`      | Array segments                            |
| `layout`     | Layout encodings                          | `vortex.zoned`, `vortex.flat`      | The file's layout tree                    |
| `aggregation`| Aggregate functions                       | `vortex.sum`, `vortex.bounded_max` | Zone maps and file statistics             |
| `expression` | Scalar functions in serialized expressions| `vortex.tensor.l2_norm`            | Rarely durable; usually scan-time only    |
| `extension dtype` | Extension dtypes                     | `vortex.timestamp`, `vortex.uuid`  | Every serialized `DType`, incl. file schemas |

Object ids are unique within a kind. The same id may name different objects of different
kinds: `vortex.dict` is both an array encoding and a layout encoding.

The first edition, `core2025.05.0`, contains the stable objects that could be written by
Vortex `0.36.0`. This is the release from which the Vortex file format is considered stable.
Later `core` editions add stable objects released after that compatibility boundary.
Editions are additive: an edition contains every member of the previous edition of its
family, plus the members that join at it.

The writer can be configured with a set of different editions (for example, `core2026.07.0`
and `unstable2026.06.0` select stable objects released through July 2026 and unstable
objects released through June 2026). Editions can be used to constrain your minimum required
Vortex reader: the newest `min_vortex_version` across all enabled editions is the earliest
version of Vortex required to read the file.

## Resolving an unknown-object error

If a read failed with an unknown ID (an encoding, layout, aggregate function, expression
function, or extension dtype) and pointed you here, the reader met an object it does not
support. Find the ID in the [registry](#edition-registry) below:

1. **The ID is listed under an edition.** The file is newer than your Vortex build. Upgrade
   to at least that edition's required Vortex release and the file will read.
2. **The ID is not listed anywhere.** The file was written outside the editions system, with
   a custom, third-party, or experimental object. Ask the producer of the file how to read
   it, or register the object with your session before reading. Tools that only inspect or
   relocate data (rather than query it) can opt in to `allow_unknown`, which decodes
   unrecognised arrays into inert placeholders and disables pruning for zone maps whose
   aggregates it cannot resolve.

## Writing with an edition

By default the writer targets a `core` edition lagging the latest Vortex release by a few
versions, giving a delay before the newest objects are written to disk.
Every file you write carries the read-forever guarantee. If a file would contain an array
encoding outside the targeted editions, the write fails immediately; edition violations
never surface as someone else's read error later. (Arrays are enforced at write time today;
layout, aggregation, and extension-dtype membership is declared and validated, with
write-time enforcement staged — see [enforcement status](#enforcement-status).)

The enabled editions are stored on the writer's Vortex session. Registering an edition makes
its declaration available to the session; enabling it separately allows the writer to emit
its members. Enabling another edition from the same family replaces the earlier selection.

Two knobs exist when the default is not what you want:

- **Pin an older edition** when files must stay readable by deployments running older
  Vortex.
- **Opt in to additional edition families.** Editions come in independently versioned,
  additive families — `core` today, with families for more specialised groups (for example
  spatial objects) possible later. A writer targets at most one edition per family and may
  emit any object in their union; each object belongs to exactly one family.

Lower-level sessions without an enabled-editions store opt out of editions entirely and can
write custom or experimental objects. A raw `with_allow_encodings` writer policy is another
explicit opt-out. Either choice gives up the standardization guarantee — only readers that
know those objects can read the files.

## How editions change

A published edition is frozen — its member list never grows or shrinks, for any object kind.
New objects are staged in a **draft** edition and become guaranteed only when that draft is
frozen as the next edition; each object's registry entry records the edition it joined in.
In the future an object may be *deprecated*, meaning writers stop emitting it — but readers
keep decoding it indefinitely, so deprecation never invalidates existing files. The
`vortex.stats` layout is an example: writers moved on to `vortex.zoned`, while every file
carrying `vortex.stats` remains readable.

## How serialized objects evolve

Editions name *which* objects a file may contain. This section defines how the serialized
form of those objects — array metadata, layout metadata, aggregate options, expression
options, extension dtype metadata — is allowed to change over time.

### Reading: deserialize to the latest version

Vortex maintains exactly one in-memory representation per object: the latest one.
Deserialization always targets it, from **every serialized form that has ever existed**:

- A serialized form, once shipped in a release, stays readable **forever**. Deserializers
  accumulate historical forms; they never drop one.
- Old forms deserialize *into the latest in-memory version*, not into parallel legacy code
  paths. For example, zone maps written before aggregate descriptors existed (and whole
  `vortex.stats` layouts) deserialize into the same zone-map machinery that modern
  `vortex.zoned` layouts use; the reader upgrades on the way in.
- Consequently there is no version negotiation at read time: a reader either knows the
  object (and then reads all of its historical forms) or reports an unknown-object error
  covered [above](#resolving-an-unknown-object-error).

When changing an object's serialized form, the change must land as *additional* accepted
input, alongside a deserializer for every earlier form. Removing or repurposing existing
fields is never allowed; new fields must be optional or gated behind a new form that old
data simply does not carry.

### Writing: translate down, or convert to canonical and recompress

Writers emit the **newest serialized form permitted by the target editions**. An object (or
object version) newer than the target edition never leaks into the file; the writer resolves
the conflict in one of two ways:

1. **Translate.** If the newer in-memory version has a defined translation to a serialized
   form the target edition guarantees, the writer emits the older form. This is preferred
   when the translation is lossless, e.g. re-emitting a newer layout's zone statistics using
   an older stats schema.
2. **Convert to canonical and recompress.** Otherwise the writer decompresses the data to a
   canonical representation and recompresses it using only the configured compressors,
   filtered to the target editions. This is how arrays are handled today: the write pipeline
   normalizes each chunk (recursively executing any encoding outside the permitted set down
   to canonical) and the configured compressor — default, BtrBlocks-style, or custom — is
   restricted to choose encodings from the enabled editions.

Both paths run inside the ordinary write pipeline, so the configured compressors are always
what produces the final bytes; targeting an older edition costs compression ratio, never
correctness. If neither path can express the data inside the target editions, the write
fails immediately rather than emitting a file the target reader could not load.

### What this means per kind

- **Arrays.** Enforced at write time today. The writer's array context only permits
  encodings from the enabled editions; anything else is normalized to canonical and
  recompressed by the edition-filtered compressor.
- **Layouts.** The layout strategy decides the layout tree at write time. Layout membership
  declares which layout encodings (in their current serialized form) a target reader
  understands; strategies must degrade to older structures (e.g. plain chunked data instead
  of newer auxiliary layouts) when targeting editions that predate them.
- **Aggregations.** Zone maps and file statistics serialize aggregate function ids plus
  their options. Writers targeting an edition without a given aggregate must omit it or
  translate to an older stats schema; readers meeting an unknown aggregate disable pruning
  for that zone map (under `allow_unknown`) rather than failing the scan, since dropping
  statistics is always sound.
- **Expressions.** Expressions serialize as trees of scalar-function ids with options, but
  they are usually transient — scan predicates cross process boundaries, not storage — so
  most scalar functions never join an edition. A scalar function joins only when its
  serialized form can reach durable data (for example the tensor similarity functions used
  by vector indexes); it then carries the same guarantee as every other member.
- **Extension dtypes.** Every serialized `DType` — including every file's schema — embeds
  the ids and metadata of the extension dtypes it uses, so an extension dtype in durable
  data needs the same guarantee as an encoding. Readers resolve ids against the session's
  dtype registry; under `allow_unknown` an unrecognised extension deserializes as an opaque
  foreign dtype over its storage type, which is always readable. Extension dtypes shipped
  by opt-in crates (`vortex.st.*` spatial types, `vortex.json`) sit outside the editions
  system until their crates declare a family, exactly like their encodings.

### Enforcement status

| Kind         | Declared & validated | Enforced at write time              |
| ------------ | -------------------- | ----------------------------------- |
| `array`      | yes                  | yes                                 |
| `layout`     | yes                  | staged; strategies choose layouts   |
| `aggregation`| yes                  | staged; defaults stay in `core`     |
| `expression` | yes                  | not applicable to file writes today |
| `extension dtype` | yes             | staged; schemas come from the input |

Declared membership is validated by unit tests against the session registries (every `core`
member must be registered, layouts and aggregations included) and pinned so a frozen edition
cannot silently change. Write-time enforcement for layouts and aggregations follows the
array mechanism: strategies will consult the enabled editions and translate or degrade,
exactly as described above.

## Edition registry

The first-party declarations live in the `vortex::editions` module, one file per edition,
and are pinned by unit tests; this table is maintained alongside them. `Since` is the
edition an object joined — it is a member of that edition and every later edition of the
same family.

### `core` family

Frozen editions: `core2025.05.0` (Vortex `0.36.0`), `core2025.06.0` (`0.40.0`),
`core2025.10.0` (`0.54.0`), `core2026.07.0` (`0.65.0`), `core2026.08.0` (`0.84.0`).

| Id                                  | Kind         | Since          |
| ----------------------------------- | ------------ | -------------- |
| `fastlanes.bitpacked`               | array        | `core2025.05.0`|
| `fastlanes.for`                     | array        | `core2025.05.0`|
| `fastlanes.rle`                     | array        | `core2025.10.0`|
| `vortex.alp`                        | array        | `core2025.05.0`|
| `vortex.alprd`                      | array        | `core2025.05.0`|
| `vortex.bool`                       | array        | `core2025.05.0`|
| `vortex.bytebool`                   | array        | `core2025.05.0`|
| `vortex.chunked`                    | array        | `core2025.05.0`|
| `vortex.constant`                   | array        | `core2025.05.0`|
| `vortex.datetimeparts`              | array        | `core2025.05.0`|
| `vortex.decimal`                    | array        | `core2025.05.0`|
| `vortex.decimal_byte_parts`         | array        | `core2025.05.0`|
| `vortex.dict`                       | array        | `core2025.05.0`|
| `vortex.ext`                        | array        | `core2025.05.0`|
| `vortex.fixed_size_list`            | array        | `core2025.10.0`|
| `vortex.fsst`                       | array        | `core2025.05.0`|
| `vortex.list`                       | array        | `core2025.05.0`|
| `vortex.listview`                   | array        | `core2025.10.0`|
| `vortex.map`                        | array        | `core2026.08.0`|
| `vortex.masked`                     | array        | `core2025.10.0`|
| `vortex.null`                       | array        | `core2025.05.0`|
| `vortex.pco`                        | array        | `core2025.06.0`|
| `vortex.primitive`                  | array        | `core2025.05.0`|
| `vortex.runend`                     | array        | `core2025.05.0`|
| `vortex.sequence`                   | array        | `core2025.06.0`|
| `vortex.sparse`                     | array        | `core2025.05.0`|
| `vortex.struct`                     | array        | `core2025.05.0`|
| `vortex.varbin`                     | array        | `core2025.05.0`|
| `vortex.varbinview`                 | array        | `core2025.05.0`|
| `vortex.variant`                    | array        | `core2026.07.0`|
| `vortex.zigzag`                     | array        | `core2025.05.0`|
| `vortex.zstd`                       | array        | `core2025.06.0`|
| `vortex.chunked`                    | layout       | `core2025.05.0`|
| `vortex.dict`                       | layout       | `core2026.08.0`|
| `vortex.flat`                       | layout       | `core2025.05.0`|
| `vortex.list`                       | layout       | `core2026.08.0`|
| `vortex.stats`                      | layout       | `core2025.05.0`|
| `vortex.struct`                     | layout       | `core2025.05.0`|
| `vortex.zoned`                      | layout       | `core2026.08.0`|
| `vortex.all_nan`                    | aggregation  | `core2026.08.0`|
| `vortex.all_non_distinct`           | aggregation  | `core2026.08.0`|
| `vortex.all_non_nan`                | aggregation  | `core2026.08.0`|
| `vortex.all_non_null`               | aggregation  | `core2026.08.0`|
| `vortex.all_null`                   | aggregation  | `core2026.08.0`|
| `vortex.bounded_max`                | aggregation  | `core2026.08.0`|
| `vortex.bounded_min`                | aggregation  | `core2026.08.0`|
| `vortex.first`                      | aggregation  | `core2026.08.0`|
| `vortex.is_constant`                | aggregation  | `core2026.08.0`|
| `vortex.is_sorted`                  | aggregation  | `core2026.08.0`|
| `vortex.last`                       | aggregation  | `core2026.08.0`|
| `vortex.max`                        | aggregation  | `core2026.08.0`|
| `vortex.min`                        | aggregation  | `core2026.08.0`|
| `vortex.nan_count`                  | aggregation  | `core2026.08.0`|
| `vortex.null_count`                 | aggregation  | `core2026.08.0`|
| `vortex.sum`                        | aggregation  | `core2026.08.0`|
| `vortex.uncompressed_size_in_bytes` | aggregation  | `core2026.08.0`|
| `vortex.date`                       | extension dtype | `core2025.05.0`|
| `vortex.time`                       | extension dtype | `core2025.05.0`|
| `vortex.timestamp`                  | extension dtype | `core2025.05.0`|
| `vortex.uuid`                       | extension dtype | `core2026.07.0`|

Note that some layouts shipped long before the edition recorded here (`vortex.zoned` dates
back to the stable-format release), but their *current* serialized form is only guaranteed
readable from the recorded edition's `min_vortex_version` onwards. A membership floor moves
earlier only with compat-fixture evidence.

### `unstable` family

All `unstable` editions are drafts: they carry no minimum reader version and no guarantee
yet, and are written only when the `unstable_encodings` feature is selected.

| Id                                 | Kind         | Since              |
| ---------------------------------- | ------------ | ------------------ |
| `fastlanes.delta`                  | array        | `unstable2025.05.0`|
| `vortex.onpair`                    | array        | `unstable2026.06.0`|
| `vortex.parquet.variant`           | array        | `unstable2026.04.0`|
| `vortex.patched`                   | array        | `unstable2026.04.0`|
| `vortex.tensor.normalized`         | array        | `unstable2026.04.0`|
| `vortex.zstd_buffers`              | array        | `unstable2026.02.0`|
| `vortex.tensor.cosine_similarity`  | expression   | `unstable2026.04.0`|
| `vortex.tensor.inner_product`      | expression   | `unstable2026.04.0`|
| `vortex.tensor.l2_norm`            | expression   | `unstable2026.04.0`|
| `vortex.tensor.fixed_shape_tensor` | extension dtype | `unstable2026.04.0`|
| `vortex.tensor.vector`             | extension dtype | `unstable2026.04.0`|
