# xtask - swiss army knife builder

This crate is not published and is only used by developers.

It automates a number of tasks that a project maintainer might need to do, for example
code generation.

You can run `cargo xtask -h` to get a list of supported commands.

## Current commands

### `generate-fbs`

This will generate the `src/generated` Rust files in the `vortex-flatbuffers` crate. This
must be run every time changes are made to one of the .fbs files, or if any are added/deleted.

### `generate-proto`

This will generate the `src/generated` Rust files in the `vortex-proto` crate. This must
be run every time changes are made to one of the .fbs files, or if any are added/deleted.

### `generate-editions-docs`

This regenerates the edition registry pages under `docs/specs/editions/` — an `index.md`
listing every edition plus one page per edition — from the edition declarations in
`vortex/src/editions/`. It first dumps the declarations as a TOML manifest to
`target/editions-manifest.toml` (a build artifact, never committed) by running the
`editions_manifest` example of the `vortex` crate, then renders that manifest into the
registry pages, replacing the whole directory. The static pages (`docs/specs/editions.md`
and the docs landing page) link to the generated registry. Run it every time an edition
declaration changes; CI fails if the committed registry is stale.



