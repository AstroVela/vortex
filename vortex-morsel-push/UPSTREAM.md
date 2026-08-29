# Push-reader upstream

This crate was imported from the `vortex-morsel` tree at
`ae8b9800409a60d1ceebb2b8181a144581a0cc45` on `codex/morsel-push-optimized`.
It is deliberately separate from `vortex-morsel`, which contains the pull reader.

To refresh the imported implementation from a later push-reader commit, apply only the
upstream subtree diff here:

```bash
git diff --binary \
  ae8b9800409a60d1ceebb2b8181a144581a0cc45:vortex-morsel \
  <new-push-commit>:vortex-morsel \
  | git apply --3way --directory=vortex-morsel-push
```

Then update the commit recorded above. Conflicts should be limited to the thin integration
surface: the package and binary names, `src/executor.rs`, and the ordered-completion support in
`src/driver.rs`. Keep SQL backend selection in `vortex-morsel/src/backend.rs`; it is shared by
DuckDB and DataFusion and should not be copied from either implementation branch.
