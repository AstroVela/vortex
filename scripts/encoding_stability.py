#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Derive the STABLE Vortex encoding set from the repository, code-authoritatively.

Stability rule (the single predicate): an array encoding is *in-spec* (STABLE) iff it is registered
WITHOUT the `unstable_encodings` Cargo feature. The complementary set — encodings whose registration
is gated behind `unstable_encodings` — is UNSTABLE (out-of-spec, covered by the fail-loud reject rule,
not by a spec section).

This is the prerequisite for the spec-conformance tripwire, which imports `stable_encodings()` and
asserts each has a spec section. This module ONLY derives the set; it enforces nothing.

Source of truth (both sides derived at runtime, never a hard-coded list). The must-spec set is the
UNION of all first-party registered array encodings across THREE registration sources:
- Source 1 — the vortex-array SESSION: the `this.register(<Kind>)` calls in `impl Default for
  ArraySession` (vortex-array/src/session/mod.rs), where the CANONICAL kinds register (Null, Bool,
  Primitive, Struct, List, ListView, FixedSizeList, VarBin, VarBinView, Decimal, Constant, Chunked,
  Dict, Masked, Variant, Extension). These are in-scope encodings that need byte-layout specs. The
  `this.register(...)` scan is scoped to THAT impl block only — the identically-shaped calls in the
  sibling dtype/scalar_fn/aggregate_fn sessions register extension DTypes and scalar/aggregate fns
  (NOT array encodings), so they are deliberately not collected.
- Source 2 — the vortex-file DEFAULTS: the `arrays.register(...)` calls in `register_default_encodings`
  (vortex-file/src/lib.rs) PLUS the per-crate `initialize()` functions it calls (e.g.
  `vortex_alp::initialize` registers ALP/ALPRD; `vortex_fastlanes::initialize` registers
  BitPacked/Delta/FoR/RLE). Following the initialize() calls is what exposes the finer per-encoding
  granularity the spec cares about, rather than only the crate name.
- Source 3 — PARQUET-VARIANT: `vortex.parquet.variant` (`encodings/parquet-variant`), registered via
  `vortex_parquet_variant::initialize`. That initialize() is called from the JNI session, NOT from
  `register_default_encodings`, so it is added as an explicit third source; it is stable (not
  `unstable_encodings`-gated) and in-scope. Parsed from the crate's initialize() so it stays
  code-authoritative.
- {unstable_encodings-gated}: the `#[cfg(feature = "unstable_encodings")]` (or an `all(...)` cfg that
  contains it) attribute on a registration, OR propagation from an initialize() call that is itself so
  gated. The `unstable_encodings` feature is cross-checked as still-defined in vortex-file/Cargo.toml, so
  a rename of the predicate fails loud instead of silently classifying everything STABLE.

Three classes are produced:
- STABLE   — registered without `unstable_encodings` (this is the spec-covered set the tripwire consumes).
- UNSTABLE — registered only under `unstable_encodings`.
- PARKED   — registered only inside a runtime `use_experimental_patches()` guard (the env-var-gated
             Patched array + its ALP/BitPacked shim plugins). Treated as out-of-spec here; its env-var
             gate is intentionally not reconciled onto the Cargo feature.

Note: the base `Zstd` array is gated by `#[cfg(feature = "zstd")]` (a DEFAULT feature of the
`vortex` crate), NOT `unstable_encodings`, so it is STABLE / in-spec — it ships by default and is a
documented encoding. Only `ZstdBuffers` (the `unstable_encodings`-gated part of zstd) is UNSTABLE.

Note: `vortex_tensor::initialize` registers extension DTypes and scalar functions, NOT an `arrays`
encoding, so tensor contributes nothing to the array-encoding set (it is neither STABLE nor UNSTABLE
here) even though its initialize() call is `unstable_encodings`-gated.

Usage:
    python scripts/encoding_stability.py             # print the derived stable / unstable / parked sets
    python scripts/encoding_stability.py --self-test # prove the derivation flips when gating changes
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from functools import lru_cache
from pathlib import Path

STABLE = "stable"
UNSTABLE = "unstable"
PARKED = "parked"

# Highest wins when a name is seen more than once (defensive; encodings register in one place today).
_PRECEDENCE = {STABLE: 0, PARKED: 1, UNSTABLE: 2}

# `arrays.register(X)` (a local `arrays` binding in register_default_encodings) OR
# `session.arrays().register(X)` (the per-crate initialize() form). Anchored to `arrays` so
# `dtypes().register(...)` / `scalar_fns().register(...)` / `register_aggregate_kernel(...)` do NOT match.
_REGISTER_RE = re.compile(r"\barrays(?:\(\))?\.register\(\s*([\w:]+)\s*\)")
# `this.register(X)` — the receiver form used inside `impl Default for ArraySession` where the canonical
# kinds register. WARNING: the identically-shaped `this.register(...)` calls in the sibling
# dtype/scalar_fn/aggregate_fn sessions register NON-array items, so this regex is only ever applied to
# the ArraySession Default impl body (see `_array_session_registrations`), never repo-wide.
_SESSION_REGISTER_RE = re.compile(r"\bthis\.register\(\s*([\w:]+)\s*\)")
ARRAY_SESSION_SRC = "vortex-array/src/session/mod.rs"
# A `vortex_<crate>::initialize(session)` call — group 1 is the crate module (e.g. `alp`, `datetime_parts`).
_INIT_RE = re.compile(r"\bvortex_(\w+)::initialize\s*\(")
_CFG_RE = re.compile(r"#\[cfg\((.*)\)\]")
# The runtime experimental-patches guard body (single-statement, no nested braces) — its registrations are
# PARKED. `finditer` collects every such guard in a fn body (ALP + FastLanes each have one).
_ENV_GUARD_RE = re.compile(r"use_experimental_patches\s*\(\s*\)\s*\{([^{}]*)\}", re.DOTALL)
_UNSTABLE_CFG_RE = re.compile(r'feature\s*=\s*"unstable_encodings"')

UNSTABLE_FEATURE = "unstable_encodings"


def repo_root() -> Path:
    """The git top-level, or — on any failure to obtain it — the script's parent's parent (the script
    lives at <repo>/scripts/)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(out)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return Path(__file__).resolve().parent.parent


def _strip_rust_comments(src: str) -> str:
    """Remove `//` line comments and NESTED `/* */` block comments from Rust source, so a decoy
    registration hidden in a comment cannot be sourced. (Copied from the docs-conformance harness to keep
    this module standalone-importable without a circular dependency.)"""
    out: list[str] = []
    i, n, depth = 0, len(src), 0
    while i < n:
        two = src[i:i + 2]
        if depth == 0 and two == "//":
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif two == "/*":
            depth += 1
            i += 2
        elif two == "*/" and depth > 0:
            depth -= 1
            i += 2
        elif depth > 0:
            i += 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def _brace_body(src: str, header_re: str, label: str) -> str:
    """The brace-matched `{ ... }` body following the FIRST match of `header_re` in `src` (comments
    assumed stripped). Raises if the header or its closing brace is not found, so a moved/renamed source
    fails loud."""
    m = re.search(header_re, src)
    if not m:
        raise LookupError(f"could not find {label} — the registration source moved")
    start = src.index("{", m.start())
    depth, end = 0, None
    for j in range(start, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    if end is None:
        raise LookupError(f"{label} body is not closed")
    return src[start + 1:end]


def _fn_body(src: str, fn_name: str) -> str:
    """The brace-matched body of `fn <fn_name>(...) { ... }` in `src`, tolerating an optional
    `-> ReturnType` before the brace (comments assumed stripped). Raises if the function or its closing
    brace is not found, so a moved/renamed source fails loud."""
    return _brace_body(
        src, rf"fn {re.escape(fn_name)}\s*\([^)]*\)\s*(?:->\s*[^{{;]+?)?\{{", f"`fn {fn_name}(...)`")


@lru_cache(maxsize=None)
def _crate_dirs(root: Path) -> dict[str, Path]:
    """Map each workspace crate's package name to its directory (from every Cargo.toml `[package] name`),
    so a `vortex_<crate>::initialize` call resolves to that crate's source. Excludes build output."""
    dirs: dict[str, Path] = {}
    for cargo in root.rglob("Cargo.toml"):
        s = str(cargo)
        if "/target/" in s or "/.git/" in s:
            continue
        m = re.search(r'^\s*name\s*=\s*"([^"]+)"', cargo.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            dirs[m.group(1)] = cargo.parent
    if len(dirs) < 10:
        raise LookupError(f"found only {len(dirs)} workspace crates; the Cargo.toml scan may be broken")
    return dirs


def _env_gated_idents(body: str) -> set[str]:
    """Registered idents inside a `use_experimental_patches()` runtime guard within `body` — the PARKED
    (env-var-gated) encodings, derived from the guard rather than hard-coded to `Patched`."""
    ids: set[str] = set()
    for m in _ENV_GUARD_RE.finditer(body):
        ids |= {x.split("::")[-1] for x in _REGISTER_RE.findall(m.group(1))}
    return ids


def _walk_registrations(body: str, register_re: re.Pattern[str] = _REGISTER_RE):
    """Yield `(kind, name, cfg)` for each registration statement in a fn body, in source order.
    `kind` is 'register' (name = the registered struct ident, last path segment) or 'init' (name = the
    `vortex_<name>::initialize` crate module). `register_re` selects the receiver form: the default
    `arrays`/`arrays()` form (Sources 2/3), or `_SESSION_REGISTER_RE` (`this.register`) for the
    vortex-array session body (Source 1). `cfg` is the raw contents of the `#[cfg(...)]` attribute
    immediately preceding the statement, or None. A `#[cfg(...)]` attribute on its own line applies to the
    next statement; any other non-blank line clears a pending attribute."""
    pending: str | None = None
    for raw in body.split("\n"):
        s = raw.strip()
        if not s:
            continue
        cfgm = _CFG_RE.search(s)
        has_stmt = register_re.search(s) or _INIT_RE.search(s)
        if cfgm and not has_stmt:
            pending = cfgm.group(1)
            continue
        matched = False
        for m in register_re.finditer(s):
            yield "register", m.group(1).split("::")[-1], pending
            matched = True
        for m in _INIT_RE.finditer(s):
            yield "init", m.group(1), pending
            matched = True
        pending = None if matched or not cfgm else pending


def _cfg_is_unstable(cfg: str | None) -> bool:
    """Whether a `#[cfg(...)]` contents string gates on `unstable_encodings` (bare or within `all(...)`)."""
    return cfg is not None and _UNSTABLE_CFG_RE.search(cfg) is not None


def _classify(is_unstable: bool, is_env_gated: bool) -> str:
    """UNSTABLE wins over PARKED wins over STABLE. A non-`unstable_encodings` feature gate (e.g. `zstd`)
    does NOT make an encoding unstable — that is the deliberate rule (see the module docstring)."""
    if is_unstable:
        return UNSTABLE
    if is_env_gated:
        return PARKED
    return STABLE


def _merge(out: dict[str, str], name: str, cls: str) -> None:
    if name not in out or _PRECEDENCE[cls] > _PRECEDENCE[out[name]]:
        out[name] = cls


def stability_from_body(
    body: str, *, parent_unstable: bool = False, register_re: re.Pattern[str] = _REGISTER_RE
) -> dict[str, str]:
    """Classify the DIRECT `register(...)` calls in a single fn body (no crate recursion) — the pure core
    the self-test drives on synthetic input to prove a gating flip moves an encoding between sets.
    `parent_unstable` models an `unstable_encodings`-gated initialize() call enclosing the body.
    `register_re` selects the receiver form (default `arrays`; `_SESSION_REGISTER_RE` for `this.register`)."""
    env = _env_gated_idents(body)
    out: dict[str, str] = {}
    for kind, name, cfg in _walk_registrations(body, register_re):
        if kind != "register":
            continue
        _merge(out, name, _classify(parent_unstable or _cfg_is_unstable(cfg), name in env))
    return out


def _array_session_registrations(root: Path) -> dict[str, str]:
    """Classify the canonical array kinds registered by `impl Default for ArraySession` in vortex-array
    (Source 1 — the `this.register(<Kind>)` calls). These are in-scope encodings that need byte-layout
    specs. Scoped to that impl block so the identically-shaped `this.register(...)` calls in the sibling
    dtype/scalar_fn/aggregate_fn sessions (extension DTypes, scalar/aggregate fns — NOT array encodings)
    are not miscollected. Fails loud if implausibly few kinds are found (the parse likely broke)."""
    src = _strip_rust_comments((root / ARRAY_SESSION_SRC).read_text(encoding="utf-8"))
    body = _brace_body(src, r"impl Default for ArraySession\s*\{", "`impl Default for ArraySession`")
    out = stability_from_body(body, register_re=_SESSION_REGISTER_RE)
    if len(out) < 8:
        raise LookupError(
            f"derived only {len(out)} canonical kinds from `impl Default for ArraySession` in "
            f"{ARRAY_SESSION_SRC}; the session-registration parse likely broke")
    return out


def _initialize_body(root: Path, crate_mod: str) -> str:
    """The body of `pub fn initialize(session: ...)` for the crate behind a `vortex_<crate_mod>::initialize`
    call. `crate_mod` (e.g. `datetime_parts`) maps to package `vortex-datetime-parts`."""
    pkg = "vortex-" + crate_mod.replace("_", "-")
    d = _crate_dirs(root).get(pkg)
    if d is None:
        raise LookupError(f"initialize() references crate {pkg!r}, not found in the workspace")
    for p in sorted((d / "src").rglob("*.rs")):
        t = _strip_rust_comments(p.read_text(encoding="utf-8"))
        if re.search(r"\bpub fn initialize\s*\(", t):
            return _fn_body(t, "initialize")
    raise LookupError(f"no `pub fn initialize` found in {pkg}")


def _collect(root: Path, body: str, parent_unstable: bool, visited: frozenset[str]) -> dict[str, str]:
    """Classify every array encoding reachable from `body`: its direct `arrays.register(...)` calls, plus
    (recursively, cycle-guarded) the encodings registered by each `vortex_<crate>::initialize` it calls.
    `parent_unstable` propagates an `unstable_encodings` gate down into a called initialize()."""
    env = _env_gated_idents(body)
    out: dict[str, str] = {}
    for kind, name, cfg in _walk_registrations(body):
        gate_unstable = parent_unstable or _cfg_is_unstable(cfg)
        if kind == "register":
            _merge(out, name, _classify(gate_unstable, name in env))
        else:  # 'init' — descend into the crate's initialize()
            pkg = "vortex-" + name.replace("_", "-")
            if pkg in visited:
                continue
            child = _collect(root, _initialize_body(root, name), gate_unstable, visited | {pkg})
            for cn, cc in child.items():
                _merge(out, cn, cc)
    return out


def _assert_unstable_feature_defined(root: Path) -> None:
    """Fail loud if vortex-file/Cargo.toml no longer defines the `unstable_encodings` feature — the cfg
    gate this module keys on. Ties the two sides together: a rename would move both, and a mismatch (only
    the feature or only the cfg renamed) is caught rather than silently classifying everything STABLE."""
    data = tomllib.loads((root / "vortex-file/Cargo.toml").read_text(encoding="utf-8"))
    if UNSTABLE_FEATURE not in data.get("features", {}):
        raise LookupError(
            f"vortex-file/Cargo.toml no longer defines the `{UNSTABLE_FEATURE}` feature; the stability "
            "predicate moved — update encoding_stability.py")


@lru_cache(maxsize=None)
def classify_encodings(root: Path) -> dict[str, str]:
    """The derived classification: array-encoding name -> STABLE / UNSTABLE / PARKED, sourced from the
    UNION of the three registration sources (see the module docstring): the vortex-array session
    canonical kinds, the vortex-file `register_default_encodings` (+ the crate initialize() fns it
    calls), and parquet-variant. Fails loud if any registration source, an initialize() crate, or the
    `unstable_encodings` feature moved."""
    _assert_unstable_feature_defined(root)
    result: dict[str, str] = {}

    # Source 1 — vortex-array session: the canonical kinds (`this.register(...)`), always in the default
    # session (VortexSession::default wires `.with::<ArraySession>()`); all ungated -> STABLE.
    for name, cls in _array_session_registrations(root).items():
        _merge(result, name, cls)

    # Source 2 — vortex-file: register_default_encodings + the crate initialize() fns it calls, with
    # per-registration unstable_encodings / zstd-feature / env-patch gating classified as today.
    src = _strip_rust_comments((root / "vortex-file/src/lib.rs").read_text(encoding="utf-8"))
    body = _fn_body(src, "register_default_encodings")
    for name, cls in _collect(root, body, parent_unstable=False,
                              visited=frozenset({"vortex-file"})).items():
        _merge(result, name, cls)

    # Source 3 — parquet-variant: parsed from its own initialize() (called from the JNI session, not
    # register_default_encodings). Stable, in-scope.
    for name, cls in _collect(root, _initialize_body(root, "parquet_variant"), parent_unstable=False,
                              visited=frozenset({"vortex-parquet-variant"})).items():
        _merge(result, name, cls)

    if "ParquetVariant" not in result:
        raise LookupError(
            "expected `ParquetVariant` from vortex_parquet_variant::initialize; the parquet-variant "
            "registration source moved — update encoding_stability.py")
    if len(result) < 24:
        raise LookupError(
            f"derived only {len(result)} encodings across the array session + file defaults + "
            "parquet-variant; the parse likely broke")
    return dict(result)


def stable_encodings(root: Path) -> set[str]:
    """The spec-covered stable encoding set = registered encodings NOT gated behind `unstable_encodings`
    (and not env-gated). This is the predicate the tripwire consumes."""
    return {n for n, c in classify_encodings(root).items() if c == STABLE}


def unstable_encodings(root: Path) -> set[str]:
    """The excluded set: encodings registered only under the `unstable_encodings` Cargo feature."""
    return {n for n, c in classify_encodings(root).items() if c == UNSTABLE}


def parked_encodings(root: Path) -> set[str]:
    """Encodings registered only inside a `use_experimental_patches()` env-var guard — out-of-spec by
    fiat (Patched + its ALP/BitPacked shim plugins); their env-gate is not reconciled onto the feature."""
    return {n for n, c in classify_encodings(root).items() if c == PARKED}


def feature_gated_stable(root: Path) -> set[str]:
    """STABLE encodings that are nonetheless behind SOME (non-`unstable_encodings`) cfg feature gate —
    today just `Zstd` (behind the default `zstd` feature). Surfaced as the known divergence from the
    original 'exclude zstd' expectation. Derived, not hard-coded (see the module docstring)."""
    src = _strip_rust_comments((root / "vortex-file/src/lib.rs").read_text(encoding="utf-8"))
    body = _fn_body(src, "register_default_encodings")
    stable = stable_encodings(root)
    gated: set[str] = set()
    for kind, name, cfg in _walk_registrations(body):
        if kind == "register" and cfg is not None and not _cfg_is_unstable(cfg) and name in stable:
            gated.add(name)
    return gated


def self_test_failures(root: Path) -> list[str]:
    """Negative + positive checks proving the derivation's logic on SYNTHETIC input, then confirming the
    live derivation. Returned as a list of failure strings (empty == green), in the docs-harness style."""
    f: list[str] = []

    # --- drift: flipping the unstable_encodings gate moves an encoding between the sets --------------
    off = stability_from_body("arrays.register(Widget);\n    arrays.register(Keeper);")
    if off.get("Widget") != STABLE or off.get("Keeper") != STABLE:
        f.append("baseline: an ungated encoding must be STABLE")
    on = stability_from_body('#[cfg(feature = "unstable_encodings")]\n    arrays.register(Widget);\n'
                             "    arrays.register(Keeper);")
    if on.get("Widget") != UNSTABLE:
        f.append("drift: gating Widget behind unstable_encodings did not move it to UNSTABLE")
    if on.get("Keeper") != STABLE:
        f.append("drift: gating one encoding wrongly reclassified an unrelated one")
    if "Widget" not in {n for n, c in off.items() if c == STABLE}:
        f.append("drift: Widget should be in the STABLE set when ungated")
    if "Widget" in {n for n, c in on.items() if c == STABLE}:
        f.append("drift: Widget still in the STABLE set after being gated behind unstable_encodings")

    # --- a non-unstable feature gate (the zstd-base case) stays STABLE; a combined cfg is UNSTABLE ---
    if stability_from_body('#[cfg(feature = "zstd")]\n    arrays.register(Zstd);').get("Zstd") != STABLE:
        f.append("a non-unstable feature gate (zstd) must remain STABLE per the rule")
    combined = stability_from_body('#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]\n'
                                   "    arrays.register(ZstdBuffers);")
    if combined.get("ZstdBuffers") != UNSTABLE:
        f.append("a combined cfg containing unstable_encodings must be UNSTABLE")

    # --- env-gated (Patched-style) shim is PARKED; its else-branch encoding stays STABLE ------------
    env = stability_from_body("if use_experimental_patches() {\n        arrays.register(Shim);\n"
                              "    } else {\n        arrays.register(Real);\n    }")
    if env.get("Shim") != PARKED or env.get("Real") != STABLE:
        f.append("env-gated shim should be PARKED and its else-branch encoding STABLE")

    # --- an unstable_encodings-gated initialize() propagates UNSTABLE to its registrations ----------
    if stability_from_body("arrays.register(Inner);", parent_unstable=True).get("Inner") != UNSTABLE:
        f.append("an encoding under an unstable_encodings-gated initialize() must be UNSTABLE")

    # --- Source 1: the vortex-array session receiver form (`this.register(...)`) is parsed ----------
    sess = stability_from_body("this.register(Canon);\n    #[cfg(feature = \"unstable_encodings\")]\n"
                               "    this.register(Exp);", register_re=_SESSION_REGISTER_RE)
    if sess.get("Canon") != STABLE:
        f.append("session source: an ungated `this.register(...)` canonical kind must be STABLE")
    if sess.get("Exp") != UNSTABLE:
        f.append("session source: a `this.register(...)` gated behind unstable_encodings must be UNSTABLE")
    # scoping guarantee: the default `arrays` receiver must NOT pick up a `this.register(...)` call, so
    # the sibling dtype/scalar_fn/aggregate_fn sessions cannot leak into the array-encoding set.
    if stability_from_body("this.register(Canon);").get("Canon") is not None:
        f.append("scoping: the default `arrays` receiver wrongly matched a `this.register(...)` call")

    # --- live derivation over the real repo ---------------------------------------------------------
    try:
        cls = classify_encodings(root)
    except (LookupError, OSError) as e:
        f.append(f"live derivation raised {type(e).__name__}: {e}")
        return f
    stable = {n for n, c in cls.items() if c == STABLE}
    unstable = {n for n, c in cls.items() if c == UNSTABLE}
    parked = {n for n, c in cls.items() if c == PARKED}
    for name in ("ByteBool", "Dict", "FSST", "Pco", "ZigZag", "ALP", "ALPRD", "FoR", "Delta", "RLE",
                 "BitPacked", "RunEnd", "Sparse", "DateTimeParts", "DecimalByteParts", "Sequence",
                 # Source 1 — canonical kinds from the vortex-array session
                 "Null", "Bool", "Primitive", "Decimal", "VarBin", "VarBinView", "Struct", "List",
                 "ListView", "FixedSizeList", "Constant", "Chunked", "Masked", "Variant", "Extension",
                 # Source 3 — parquet-variant
                 "ParquetVariant"):
        if name not in stable:
            f.append(f"live: expected {name} in the STABLE set (got {sorted(stable)})")
    if unstable != {"OnPair", "ZstdBuffers"}:
        f.append(f"live: expected UNSTABLE == {{OnPair, ZstdBuffers}}, got {sorted(unstable)}")
    if "Patched" not in parked:
        f.append(f"live: expected Patched in the PARKED set (got {sorted(parked)})")
    if stable & unstable or stable & parked or unstable & parked:
        f.append("live: stable/unstable/parked sets are not disjoint")
    if feature_gated_stable(root) != {"Zstd"}:
        f.append(f"live: expected the known divergence feature_gated_stable == {{Zstd}}, "
                 f"got {sorted(feature_gated_stable(root))}")
    return f


def _print_sets(root: Path) -> int:
    cls = classify_encodings(root)
    stable = sorted(n for n, c in cls.items() if c == STABLE)
    unstable = sorted(n for n, c in cls.items() if c == UNSTABLE)
    parked = sorted(n for n, c in cls.items() if c == PARKED)
    print(f"STABLE encodings ({len(stable)}, spec-covered): {', '.join(stable)}")
    print(f"UNSTABLE encodings ({len(unstable)}, excluded via unstable_encodings): {', '.join(unstable)}")
    print(f"PARKED encodings ({len(parked)}, env-gated / out-of-spec): {', '.join(parked)}")
    print("\nSources unioned: (1) vortex-array session canonical kinds, (2) vortex-file "
          "register_default_encodings + crate initialize() fns, (3) parquet-variant.")
    diverge = sorted(feature_gated_stable(root))
    if diverge:
        print(f"\nNOTE: {', '.join(diverge)} classified STABLE by the unstable_encodings rule "
              "but gated behind a non-unstable feature (`zstd`, a default) and is in-spec; only ZstdBuffers "
              "(the unstable_encodings-gated part) is excluded (see module docstring).")
    print("NOTE: vortex_tensor::initialize registers extension DTypes/scalar-fns, not an array encoding, "
          "so tensor is absent from all three sets by construction.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive the stable Vortex encoding set from the repo")
    ap.add_argument("--self-test", action="store_true", help="prove the derivation flips on a gating change")
    ns = ap.parse_args()
    root = repo_root()
    if ns.self_test:
        failures = self_test_failures(root)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        if failures:
            print(f"\nSELF-TEST FAILED: {len(failures)} check(s).", file=sys.stderr)
            return 1
        print("SELF-TEST OK — gating flips move an encoding between the stable/unstable sets, and the live "
              "derivation matches the expected classification.")
        return 0
    return _print_sets(root)


if __name__ == "__main__":
    sys.exit(main())
