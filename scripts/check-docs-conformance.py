#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Doc-conformance lint: assert that facts stated in docs/ prose still match their source of truth
in the code/build, so documentation cannot silently drift from the implementation.

Why this exists: Sphinx's `--fail-on-warning` + nitpicky gate the *structure* of the docs (dead
cross-references, bad roles) but not their *accuracy*. A green build says nothing about whether a
constant, package coordinate, or API name in prose is still correct. This script closes that gap for
a curated registry of load-bearing facts.

Design rules (load-bearing):
- Every check reads BOTH sides at runtime — it derives the canonical value from the code/build source
  and verifies the doc agrees. Checks MUST NOT hard-code the expected value inline; a check that
  embedded the answer would itself drift.
- Matching is token-boundary aware, not raw substring, so a stale value that merely overlaps the
  claim (`65527` inside `655270`, crate `vortex` inside `vortex-data`) does not false-pass.
- For facts expressed as a command (`pip install X`, `cargo add X`), the check additionally asserts
  that EVERY occurrence of the command uses the canonical value — catching a stale duplicate command
  that coexists with a correct one.

Known limitations (regex matching has irreducible ambiguity for inputs that do not occur in real
docs, so the matcher is tuned for correctness on realistic prose):
- For bare-value facts (a number/name not behind a command stem) the check verifies the canonical
  value is PRESENT; it does not exhaustively prove no stale variant appears elsewhere in the same
  file. The command-prefix check covers the common stale-duplicate vector.
- A pathological command argument that is neither a clean package spec nor cleanly delimited (e.g.
  `pip install pkg[extra]wrong`) is not captured as a token, so it is skipped rather than flagged.
  This is the irreducible limit of distinguishing a malformed spec from a correct spec with adjacent
  markup; such inputs do not appear in practice. Broader absence-checking is tracked as Deferred work.

Usage:
    python scripts/check-docs-conformance.py            # verify all checks; exit non-zero on drift
    python scripts/check-docs-conformance.py --self-test # prove the checker detects drift (negative test)
    python scripts/check-docs-conformance.py -v          # verbose: print every check's resolved values
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

# The stable-encoding-set derivation lives in a sibling module so the tripwire can `import
# encoding_stability` directly (this file's hyphenated name is not importable). The scripts/ dir is
# sys.path[0] when run as __main__; the insert keeps the import robust if that ever changes.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import encoding_stability  # noqa: E402  (must follow the sys.path insert above)

# Token characters: a claim must be delimited by characters OUTSIDE this set to count as present.
# Word chars and '-' (so `vortex` is not found inside `vortex-data`, nor `65527` inside `655270`).
_TOKEN = r"[\w\-]"

# A '.' is ambiguous: it can be sentence punctuation (`65527.` at end of a sentence — fine) OR a
# sub-token separator (`65527.0` version, `vortex-data.old` — a DIFFERENT value that merely shares a
# prefix). Disambiguate by what follows the dot: a '.' followed by a word/hyphen char continues the
# token (so the claim is NOT a standalone match there); a '.' followed by anything else is prose.
_NOT_DOTTED_CONTINUATION = r"(?!\.[\w\-])"


def repo_root() -> Path:
    # Prefer the git top-level; on ANY failure to obtain it — git not installed (FileNotFoundError) or
    # `git rev-parse` exiting non-zero for any reason, chiefly "not a git working tree"
    # (CalledProcessError) — fall back to the script's location. That fallback is correct in every case
    # because the script lives at <repo>/scripts/, so the repo root is its parent's parent.
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(out)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Any failure to get the git top-level — git not installed (FileNotFoundError) or `git rev-parse`
        # exiting non-zero, chiefly "not a git working tree" (CalledProcessError) — falls back to the
        # script's location. That fallback is correct for ALL such cases because the script lives at
        # <repo>/scripts/, so the repo root is its parent's parent regardless of why git was unavailable.
        return Path(__file__).resolve().parent.parent


def _toml_str_from(text: str, label: str, *keys: str | int) -> str:
    """Navigate `keys` (dict names or list indices, e.g. `"bin", 0, "name"`) into TOML `text`, parsed
    with `tomllib`, and return the string there. Robust against comments, key ordering, and list-valued
    keys that a regex over the raw text mishandles; raises if the path is missing or non-string.
    NOTE: `"bin", 0` selects the FIRST `[[bin]]` target — the user-facing CLI binary in a single-binary
    crate like vortex-tui; revisit if a crate ships multiple binaries."""
    node = tomllib.loads(text)
    for k in keys:
        try:
            node = node[k]
        except (KeyError, IndexError, TypeError) as e:
            raise LookupError(f"{label}: no TOML value at path {list(keys)} ({type(e).__name__})") from e
    if not isinstance(node, str):
        raise LookupError(f"{label}: TOML path {list(keys)} is {type(node).__name__}, not a string")
    return node


def _toml_str(root: Path, rel: str, *keys: str | int) -> str:
    """`_toml_str_from` over the file at `rel` (see that function for the navigation contract)."""
    return _toml_str_from((root / rel).read_text(encoding="utf-8"), rel, *keys)


def _strip_comments(text: str) -> str:
    """Remove C-style block (`/* */`) and line (`//`) comments so a commented-out decoy declaration
    cannot be sourced as the canonical value (the regexes below also anchor to `^\\s*<keyword>`, which a
    `//`/`#`-prefixed line cannot satisfy). TOML `#` comments need no stripping — a `#`-prefixed line
    fails the `^name`-anchored regexes — and mangling a `//` inside a URL/string is harmless here
    because the patterns match declaration lines, not URLs."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _strip_rust_comments(src: str) -> str:
    """Remove `//` line comments and NESTED `/* */` block comments from Rust source — Rust block
    comments nest, which the non-greedy `_strip_comments` regex mishandles (a commented marker inside an
    outer comment can survive). Used where a decoy hidden in a nested comment must NOT satisfy a check
    (e.g. the `#[cxx::bridge]` detection in `_cxx_dep`). String-literal contents are not modeled — the
    markers it guards never appear in string literals."""
    out: list[str] = []
    i, n, depth = 0, len(src), 0
    while i < n:
        two = src[i:i + 2]
        if depth == 0 and two == "//":
            j = src.find("\n", i)
            i = n if j < 0 else j  # skip to the newline (kept on the next iteration)
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


def _read_const(text: str, pattern: str, label: str) -> int:
    """Read exactly one integer constant matched by `pattern` (group 1) in `text` (comments stripped);
    raise if missing or duplicated, so a derived fact fails loud rather than silently sourcing a decoy."""
    matches = re.findall(pattern, _strip_comments(text), re.MULTILINE)
    if len(matches) != 1:
        raise LookupError(f"expected exactly one {label} matching {pattern!r}, found {len(matches)}")
    return int(matches[0])


def _derive_initial_read(root: Path) -> str:
    """The file reader's first-read size = MAX_POSTSCRIPT_SIZE + EOF_SIZE. Read BOTH constituent
    constants from vortex-file/src/lib.rs (so a change to EITHER is reflected, not cancelled), and
    verify vortex-file/src/open.rs still defines INITIAL_READ_SIZE as their sum — so the doc's value
    tracks the actual const, and a broken relationship is caught rather than assumed."""
    lib = (root / "vortex-file/src/lib.rs").read_text(encoding="utf-8")
    eof = _read_const(lib, r"^\s*pub const EOF_SIZE: usize = (\d+)\s*;", "EOF_SIZE")
    max_postscript = 65535 - _read_const(lib, r"^\s*pub const MAX_POSTSCRIPT_SIZE: u16 = u16::MAX - (\d+)\s*;",
                                         "MAX_POSTSCRIPT_SIZE subtrahend")
    open_rs = _strip_rust_comments((root / "vortex-file/src/open.rs").read_text(encoding="utf-8"))
    # Anchor to start-of-line + end-of-statement (`;`) so neither a commented decoy nor a CHANGED
    # relationship (e.g. `... + EOF_SIZE + 1`) prefix-matches and silently passes.
    if not re.search(r"^\s*const INITIAL_READ_SIZE: usize = MAX_POSTSCRIPT_SIZE as usize \+ EOF_SIZE\s*;",
                     open_rs, re.MULTILINE):
        raise LookupError(
            "INITIAL_READ_SIZE is no longer defined as `MAX_POSTSCRIPT_SIZE as usize + EOF_SIZE` in "
            "vortex-file/src/open.rs — the first-read relationship moved; update the registry.")
    return str(max_postscript + eof)


def _derive_spark_artifact(root: Path) -> str:
    # Policy: the docs recommend the artifact for the HIGHEST published Scala binary version (latest
    # Scala — currently 2.13); `_spark_scala_variants` enumerates the published suffixes and this picks
    # the max. If the recommended-version policy changes, update here and the docs together.
    """The Spark connector's Maven artifact id the docs should advertise — the LATEST scala-suffixed
    variant, derived (not hard-coded) from the `vortex-spark_<scala>` subprojects declared in
    java/settings.gradle.kts, and cross-checked against the build's `artifactId = "vortex-spark_
    $scalaVersion"` pattern. Catches a doc that drops the `_<scala>` suffix (copy-paste-broken
    coordinate) or that pins a scala version the build no longer publishes."""
    settings = _strip_comments((root / "java/settings.gradle.kts").read_text(encoding="utf-8"))
    # Scala binary suffix may be `2.13` (Scala 2.x) OR dotless `3` (Scala 3) — allow both.
    variants = re.findall(r'include\("vortex-spark_(\d+(?:\.\d+)?)"\)', settings)
    if not variants:
        raise LookupError("no `vortex-spark_<scala>` subprojects in java/settings.gradle.kts")
    gradle = _strip_comments((root / "java/vortex-spark/build.gradle.kts").read_text(encoding="utf-8"))
    if not re.search(r'artifactId\s*=\s*"vortex-spark_\$scalaVersion"', gradle):
        raise LookupError("vortex-spark artifactId is no longer the scala-suffixed form; update registry")
    latest = max(variants, key=lambda v: tuple(int(x) for x in v.split(".")))
    return f"vortex-spark_{latest}"


def _spark_major_for_scala(root: Path, scala: str) -> str:
    """The Spark major version the `vortex-spark_<scala>` artifact targets, from the scala->spark when-arm
    in vortex-spark/build.gradle.kts (`"<scala>" -> { ... libs.versions.spark<major> ... }`). Returns
    `Spark <major>.x` to anchor spark.md's targeting claim; FAILS LOUD if the arm/mapping changes."""
    src = _strip_comments((root / "java/vortex-spark/build.gradle.kts").read_text(encoding="utf-8"))
    m = re.search(rf'"{re.escape(scala)}"\s*->\s*\{{[^}}]*?libs\.versions\.spark(\d+)', src, re.DOTALL)
    if not m:
        raise LookupError(f"no scala {scala} -> spark when-arm in vortex-spark/build.gradle.kts; spark.md is stale")
    return f"Spark {m.group(1)}.x"


def _derive_spark_filename(root: Path) -> str:
    """The Spark writer's output filename format. BOTH the partitioned and unpartitioned writers name
    files via `String.format("part-…")`; read both and require they agree (so the documented value
    tracks whichever path the example uses, and a divergence between the two fails loud)."""
    writers = ["java/vortex-spark/src/main/java/dev/vortex/spark/write/PartitionedVortexDataWriter.java",
               "java/vortex-spark/src/main/java/dev/vortex/spark/write/VortexDataWriterFactory.java"]
    fmts = set()
    for w in writers:
        m = re.search(r'String\.format\("(part-[%\dd-]+\.vortex)"',
                      _strip_comments((root / w).read_text(encoding="utf-8")))
        if not m:
            raise LookupError(f"no `part-*.vortex` String.format in {w}")
        fmts.add(m.group(1))
    if len(fmts) != 1:
        raise LookupError(f"Spark writers disagree on the output filename format: {sorted(fmts)}")
    return fmts.pop()


def _parse_python_min_version(spec: str) -> str:
    """Extract the effective minimum Python version from a `requires-python` spec, anchored to the `>=`
    lower bound(s) — NOT the first version token or the upper bound. The minimum is the HIGHEST `>=`
    clause (multiple lower bounds intersect: `>=3.10,>=3.11` requires 3.11), compared as a PEP-440-ish
    numeric tuple so ordering is value-based, not lexical. The FULL lower-bound version is kept, incl. a
    patch level (`>=3.11.4` -> `Python 3.11.4`), so a docs claim of "3.11" can't understate a real
    "3.11.4" minimum. Only `>=` (lower) and `<`/`<=` (upper, ignored) operators are reasoned about;
    ANY other operator (`>`, `==`, `===`, `!=`, `~=`) or a non-numeric lower bound (`>=3.11.0rc1`,
    `>=3.11.*`) fails loud — because exclusions / compatible-release / pre-release clauses can raise or
    qualify the effective minimum, and silently understating it would defeat the lock. The live spec is
    `>= 3.11`; an exotic future spec gets a clear failure telling the maintainer to extend this.
    Separated from the file read so the self-test can exercise it on synthetic specs."""
    numeric: list[str] = []
    suffixed: list[str] = []
    for clause in spec.split(","):
        c = clause.strip()
        if not c:
            continue
        m = re.match(r"(>=|<=|===|==|!=|~=|>|<)\s*(\S+)$", c)
        if not m:
            raise LookupError(f"unparseable requires-python clause {c!r}")
        op, v = m.group(1), m.group(2)
        if op in ("<", "<="):
            continue  # an upper bound — does not affect the minimum
        if op != ">=":
            # `>`, `==`, `===`, `!=`, `~=` can each set or RAISE the effective minimum (e.g.
            # `>=3.11,!=3.11.*` really requires 3.12) in ways a simple lower-bound scan would
            # understate. Refuse to guess — fail loud so a maintainer extends this derivation.
            raise LookupError(f"requires-python operator {op!r} in {c!r} not supported; handle explicitly")
        (numeric if re.fullmatch(r"\d+\.\d+(?:\.\d+)*", v) else suffixed).append(v)
    if suffixed:  # a pre-release / wildcard lower bound (`>=3.11.0rc1`, `>=3.11.*`) — don't guess how
        raise LookupError(f"requires-python lower bound(s) {suffixed!r} not plain numeric; handle explicitly")
    if not numeric:
        raise LookupError(f"could not parse a `>=` minimum from requires-python = {spec!r}")
    best = max(numeric, key=lambda v: tuple(int(p) for p in v.split(".")))
    return f"Python {best}"


def _derive_python_version(root: Path) -> str:
    """The minimum Python version the docs advertise — sourced from pyproject `requires-python`."""
    rp = _toml_str(root, "vortex-python/pyproject.toml", "project", "requires-python")  # e.g. ">= 3.11"
    return _parse_python_min_version(rp)


def _eval_byte_expr(expr: str) -> int:
    """Evaluate a simple Rust byte-size literal: `1 << 20`, `8 * 1024`, or a plain int. Fails loud on
    anything more complex so an unexpected expression can't be silently mis-sized."""
    expr = expr.strip()
    if (m := re.fullmatch(r"(\d+)\s*<<\s*(\d+)", expr)):
        return int(m.group(1)) << int(m.group(2))
    if (m := re.fullmatch(r"(\d+)\s*\*\s*(\d+)", expr)):
        return int(m.group(1)) * int(m.group(2))
    if re.fullmatch(r"\d+", expr):
        return int(expr)
    raise LookupError(f"cannot evaluate byte expression {expr!r}")


def _bytes_to_human(n: int) -> str:
    """Format a power-of-two byte count as `<N> MB` / `<N> KB` (matching the docs' prose), so a code
    constant like `4 << 20` locks the doc string `4 MB`."""
    if n and n % (1 << 20) == 0:
        return f"{n >> 20} MB"
    if n and n % (1 << 10) == 0:
        return f"{n >> 10} KB"
    return str(n)


def _coalesce_bytes(root: Path, fn_name: str) -> tuple[int, int]:
    """The (distance, max_size) byte pair for `CoalesceConfig::<fn_name>()` in vortex-io read_at.rs.
    Requires EXACTLY ONE matching constructor (comments stripped first) so a same-shaped decoy can't
    silently mis-source the value."""
    src = _strip_rust_comments((root / "vortex-io/src/read_at.rs").read_text(encoding="utf-8"))
    ms = re.findall(rf"fn {fn_name}\(\)\s*->\s*Self\s*\{{\s*Self::new\(([^)]*)\)", src)
    if len(ms) != 1:
        raise LookupError(f"expected exactly 1 CoalesceConfig::{fn_name}() in read_at.rs, found {len(ms)}")
    args = [a for a in ms[0].split(",") if a.strip()]
    if len(args) != 2:  # exactly (distance, max_size) — neither fewer nor extra args
        raise LookupError(f"CoalesceConfig::{fn_name}() had {len(args)} args, expected (distance, max_size)")
    return _eval_byte_expr(args[0]), _eval_byte_expr(args[1])


def _derive_buffered_block_size(root: Path) -> str:
    """The BufferedStrategy localization size from vortex-file strategy.rs (`N * ONE_MEG`), as `<N> MB`
    — `ONE_MEG` is sourced from the same file so the unit can't drift from the constant."""
    src = _strip_rust_comments((root / "vortex-file/src/strategy.rs").read_text(encoding="utf-8"))
    one_meg = re.findall(r"const ONE_MEG:\s*u64\s*=\s*([^;]+);", src)
    mult = re.findall(r"BufferedStrategy::new\([^,]+,\s*(\d+)\s*\*\s*ONE_MEG\)", src)
    if len(one_meg) != 1 or len(mult) != 1:  # exactly one of each, else a decoy could mis-source
        raise LookupError(
            f"expected 1 ONE_MEG const and 1 BufferedStrategy::new in strategy.rs, "
            f"found {len(one_meg)} and {len(mult)}")
    return _bytes_to_human(int(mult[0]) * _eval_byte_expr(one_meg[0]))


def _cxx_dep(root: Path) -> str:
    """Confirm vortex-cxx uses the `cxx` crate as a direct Rust<->C++ bridge — the mechanism the C++
    docs describe. Requires BOTH the `cxx` dependency in Cargo.toml AND a live `#[cxx::bridge]` in the
    source (so a leftover dependency after a migration to wrapping the C FFI cannot satisfy the lock).
    Returns the literal `cxx` to anchor the docs' present-check; FAILS LOUD if either is gone, signaling
    the C++ docs must be re-described."""
    cxx_dir = root / "vortex-cxx"
    data = tomllib.loads((cxx_dir / "Cargo.toml").read_text(encoding="utf-8"))
    if "cxx" not in data.get("dependencies", {}):
        raise LookupError("vortex-cxx no longer depends on `cxx`; the C++ binding mechanism changed — update the docs")
    src = _strip_rust_comments("\n".join(p.read_text(encoding="utf-8") for p in (cxx_dir / "src").rglob("*.rs")))
    # Anchor to ITEM position (start-of-line + optional indent), so neither a commented decoy (already
    # stripped) nor a string literal like `const D: &str = "#[cxx::bridge]"` (the `"` breaks the anchor)
    # can satisfy the lock after the real bridge attribute is removed.
    if not re.search(r"^[ \t]*#\[cxx::bridge", src, re.MULTILINE):
        raise LookupError("vortex-cxx has no live `#[cxx::bridge]`; the C++ binding mechanism changed")
    return "cxx"


def _convert_batch_size(root: Path) -> str:
    """The `vx convert` row-batch size (vortex-tui `BATCH_SIZE`) the CLI quickstart cites — read from the
    convert command's own constant so the doc tracks it (fails loud if missing/duplicated)."""
    txt = (root / "vortex-tui/src/convert.rs").read_text(encoding="utf-8")
    return str(_read_const(txt, r"^\s*pub const BATCH_SIZE: usize = (\d+)\s*;", "BATCH_SIZE"))


def _spark_filter_pushdown(root: Path) -> str:
    """Confirm the Spark connector implements filter pushdown — VortexScanBuilder implements Spark's
    `SupportsPushDownV2Filters`. Returns that interface name (the doc must mention it); FAILS LOUD if the
    connector drops it, signaling integrations/spark.md's filter-pushdown claim is stale."""
    sb = root / "java/vortex-spark/src/main/java/dev/vortex/spark/read/VortexScanBuilder.java"
    src = _strip_rust_comments(sb.read_text(encoding="utf-8"))  # `//` + `/* */`, same as Java comments
    # require it in the class's `implements` list, not merely imported or mentioned in a comment
    if not re.search(r"implements\b[^{]*\bSupportsPushDownV2Filters\b", src, re.DOTALL):
        raise LookupError("VortexScanBuilder does not implement SupportsPushDownV2Filters; spark.md is stale")
    return "SupportsPushDownV2Filters"


def _duckdb_replacement_scan(root: Path) -> str:
    """Confirm the DuckDB extension registers a replacement scan (what makes `FROM 'data.vortex'` work).
    Returns "replacement scan"; FAILS LOUD if the registration is gone, signaling duckdb.md's direct-path
    claim is stale."""
    src = _strip_rust_comments((root / "vortex-duckdb/src/lib.rs").read_text(encoding="utf-8"))
    # Scope to the extension-load entrypoint `initialize_extension_from_raw` so a helper/test call elsewhere
    # can't satisfy the check while the actual extension-load path never registers the replacement scan.
    init = re.search(r"fn initialize_extension_from_raw\([^)]*\)\s*\{(.*?)\n\}", src, re.DOTALL)
    if not init:
        raise LookupError("could not find the `initialize_extension_from_raw` entrypoint in vortex-duckdb/src/lib.rs")
    # require an actual CALL `.register_vortex_scan_replacement()` INSIDE the entrypoint, not just a mention
    if not re.search(r"\.register_vortex_scan_replacement\s*\(\s*\)", init.group(1)):
        raise LookupError("the extension entrypoint does not call register_vortex_scan_replacement(); duckdb.md is stale")
    return "replacement scan"


def _io_reader_impl(root: Path, name: str, rel: str) -> str:
    """Confirm `name` is a struct that `impl VortexReadAt` in `rel` — the reader-type names io.md cites for
    local-file (`FileReadAt`) and object-store (`ObjectStoreReadAt`) reads. FAILS LOUD if it stops
    implementing the trait (renamed/removed), so io.md's reader names can't silently drift. Comments stripped."""
    src = _strip_rust_comments((root / rel).read_text(encoding="utf-8"))
    if not re.search(rf"\bimpl VortexReadAt for {re.escape(name)}\b", src):
        raise LookupError(f"{name} no longer `impl VortexReadAt` in {rel}; io.md's reader name is stale")
    return name


def _jni_module_name(root: Path) -> str:
    """The JNI Gradle module/artifact name (java/settings.gradle.kts `include(...)` whose name contains
    `jni`, cross-checked against that module's build.gradle.kts `artifactId`) — the source of truth for the
    `vortex-jni` name java/README.md cites. FAILS LOUD if the module is renamed; the README must then update
    (and the renamed-away `vortex-java` is forbidden)."""
    settings = _strip_comments((root / "java/settings.gradle.kts").read_text(encoding="utf-8"))
    jni = {m for m in re.findall(r'include\("([\w-]+)"\)', settings) if "jni" in m}
    if len(jni) != 1:  # exactly one jni module, else the derivation is ambiguous
        raise LookupError(f"expected exactly 1 jni Gradle module in java/settings.gradle.kts, found {sorted(jni)}")
    name = jni.pop()
    bg = (root / "java" / name / "build.gradle.kts").read_text(encoding="utf-8")
    if not re.search(rf'artifactId\s*=\s*"{re.escape(name)}"', bg):
        raise LookupError(f"java/{name}/build.gradle.kts artifactId != {name!r}; the JNI module name is inconsistent")
    return name


def _io_in_memory_coalesce_behavior(root: Path) -> str:
    """Back io.md's "in-memory coalescing is opt-in" claim with the one cleanly-lockable code fact: the
    `VortexReadAt` default `coalesce_config()` returns `None`, so a reader that doesn't override it does
    not coalesce. (That some readers opt into the 8 KB preset is a general statement; the 8 KB VALUE
    itself is locked separately by `io-coalesce-in-memory`.) Comments stripped; fails loud if it changes."""
    default_src = _strip_rust_comments((root / "vortex-io/src/read_at.rs").read_text(encoding="utf-8"))
    if not re.search(r"fn coalesce_config\(&self\)\s*->\s*Option<CoalesceConfig>\s*\{\s*None\s*\}", default_src):
        raise LookupError("VortexReadAt default coalesce_config() no longer returns None; io.md is stale")
    return "opt-in"


def _scan_crate_name(root: Path) -> str:
    """Confirm the `vortex-scan` crate exists — the Scan API the roadmap says is present today. Returns
    its package name; FAILS LOUD if the crate is gone, signaling the roadmap claim is stale."""
    cargo = root / "vortex-scan/Cargo.toml"
    if not cargo.exists():
        raise LookupError("vortex-scan crate is gone; the roadmap's 'Scan API exists today' claim is stale")
    name = _toml_str_from(cargo.read_text(encoding="utf-8"), "vortex-scan/Cargo.toml", "package", "name")
    if name != "vortex-scan":
        raise LookupError(f"vortex-scan package name is {name!r}, not 'vortex-scan'")
    # The roadmap/scanning docs credit the `DataSource` and `Partition` traits — verify both are live
    # (`\b` so `DataSourceOpener`/`DataSourceScan` don't satisfy DataSource); comments stripped.
    lib = _strip_rust_comments((root / "vortex-scan/src/lib.rs").read_text(encoding="utf-8"))
    for trait in ("DataSource", "Partition"):
        if not re.search(rf"pub trait {trait}\b", lib):
            raise LookupError(f"vortex-scan has no `pub trait {trait}`; the scan-API docs are stale")
    return name


def _dtype_has_variant(enum_body: str) -> bool:
    """Whether the `DType` enum body declares a `Variant` variant. A pure helper so the presence check
    is self-testable on synthetic input."""
    return bool(re.search(r"^\s*Variant\b", enum_body, re.MULTILINE))


def _workspace_crate_names(root: Path) -> set[str]:
    """Every crate name in the workspace (each Cargo.toml `[package] name`). The allowed set for the
    crate references the architecture overview advertises (catches a fabricated/renamed crate like the
    former `vortex-roaring`/`vortex-dict`/`vortex-expr`). Fails loud if implausibly few are found."""
    names: set[str] = set()
    for cargo in root.rglob("Cargo.toml"):
        if "/target/" in str(cargo) or "/.git/" in str(cargo):
            continue
        m = re.search(r'^\s*name\s*=\s*"([^"]+)"', cargo.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            names.add(m.group(1))
    if len(names) < 10:
        raise LookupError(f"found only {len(names)} workspace crates; the scan may be broken")
    return names


def _encoding_crate_names(root: Path) -> set[str]:
    """Every encoding crate published under `encodings/*` (its Cargo.toml `[package] name`). The source
    of truth for the architecture overview's Encodings table. Fails loud if none are found."""
    names: set[str] = set()
    for cargo in sorted((root / "encodings").glob("*/Cargo.toml")):
        m = re.search(r'^\s*name\s*=\s*"([^"]+)"', cargo.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            names.add(m.group(1))
    if not names:
        raise LookupError("no encoding crates found under encodings/*")
    return names


def _dtype_variant_names(root: Path) -> set[str]:
    """Every variant name of the `DType` enum (vortex-array/src/dtype/mod.rs) — the source of truth for
    the architecture overview's DType list. Comments stripped (nested-aware); fails loud if unparsed."""
    src = _strip_rust_comments((root / "vortex-array/src/dtype/mod.rs").read_text(encoding="utf-8"))
    m = re.search(r"pub enum DType\s*\{(.*?)\n\}", src, re.DOTALL)
    if not m:
        raise LookupError("could not find `pub enum DType` in vortex-array/src/dtype/mod.rs")
    names = set(re.findall(r"^\s*([A-Z]\w*)\s*[({,]", m.group(1), re.MULTILINE))
    if len(names) < 8:
        raise LookupError(f"parsed only {len(names)} DType variants; the enum parse may be broken")
    return names


def _variant_dtype_present(root: Path) -> str:
    """Confirm `DType::Variant` exists — the roadmap says the Variant DType is present today. Returns
    `Variant`; FAILS LOUD if the variant is gone/renamed, signaling the roadmap claim is stale."""
    src = _strip_rust_comments((root / "vortex-array/src/dtype/mod.rs").read_text(encoding="utf-8"))
    m = re.search(r"pub enum DType\s*\{(.*?)\n\}", src, re.DOTALL)
    if not m:
        raise LookupError("could not find `pub enum DType` in vortex-array/src/dtype/mod.rs")
    if not _dtype_has_variant(m.group(1)):
        raise LookupError("DType has no `Variant` variant; the roadmap's 'Variant present today' claim is stale")
    return "Variant"


def _union_is_sole_noncanonical(root: Path) -> str:
    """Back the canonical-arrays prose "every DType has a canonical encoding except `Union`"
    (concepts/arrays.md + api/python/arrays.rst): assert the `DType` enum HAS a `Union` variant (so the
    exception names a real type) AND the `Canonical` enum has NO `Union` variant or `UnionArray` payload
    (so `Union` is genuinely un-canonicalized). FAILS LOUD if `Union` gains a canonical encoding — the
    docs must then drop the exception. (That `Union` is the SOLE exception is not fully code-derivable
    here — the DType->canonical map is many-to-one; see deferred work.) Returns `Union` to anchor the
    docs' present-check."""
    if "Union" not in _dtype_variant_names(root):
        raise LookupError("DType enum has no `Union` variant; the arrays canonical-exception prose is stale")
    canon = _strip_rust_comments((root / "vortex-array/src/canonical.rs").read_text(encoding="utf-8"))
    m = re.search(r"pub enum Canonical\s*\{(.*?)\n\}", canon, re.DOTALL)
    if not m:
        raise LookupError("could not find `pub enum Canonical` in vortex-array/src/canonical.rs")
    body = m.group(1)
    if re.search(r"^\s*Union\b", body, re.MULTILINE) or "UnionArray" in body:
        raise LookupError(
            "Canonical enum now covers Union; concepts/arrays.md + arrays.rst must drop the 'except Union' claim")
    return "Union"


def _derive_spark_batch_range(root: Path) -> str:
    """The Spark writer's batch-size range string `MIN–MAX` (en-dash), reading BOTH bounds from
    VortexDataWriter.java so the documented `(1–65536)` tracks both constants."""
    java = (root / "java/vortex-spark/src/main/java/dev/vortex/spark/write/VortexDataWriter.java"
            ).read_text(encoding="utf-8")
    # Anchor to a real field declaration (`^\s*...final int NAME = N;`) so a non-literal RHS
    # (`= 2 - 1`) or a commented decoy cannot mis-source.
    lo = _read_const(java, r"^\s*(?:private\s+)?static\s+final\s+int\s+MIN_BATCH_SIZE\s*=\s*(\d+)\s*;",
                     "MIN_BATCH_SIZE")
    hi = _read_const(java, r"^\s*(?:private\s+)?static\s+final\s+int\s+MAX_BATCH_SIZE\s*=\s*(\d+)\s*;",
                     "MAX_BATCH_SIZE")
    return f"{lo}–{hi}"  # en-dash, matching the docs' `(1–65536)`


def all_doc_files(root: Path) -> list[Path]:
    """Every Markdown/reStructuredText doc under docs/, excluding the Sphinx build output."""
    docs = root / "docs"
    return sorted(
        p for p in docs.rglob("*")
        if p.suffix in (".md", ".rst") and "_build" not in p.parts
    )


def token_present(claim: str, text: str) -> bool:
    """True iff `claim` appears in `text` delimited by non-token characters on both sides, and is not
    a dotted sub-component of a longer version/identifier — neither a suffix (`65527` in `65527.0`)
    nor a prefix (`65527` in `1.65527`). A leading '.' is always a continuation (prose never starts a
    standalone token with an attached dot), so the leading guard simply forbids a preceding '.'."""
    pattern = (
        r"(?<![\w\-.])" + re.escape(claim) + r"(?!" + _TOKEN + r")" + _NOT_DOTTED_CONTINUATION
    )
    return re.search(pattern, text) is not None


# Capture group for the argument of a command (`pip install <here>`): a package-shaped token — word/
# hyphen chars, optional dotted sub-tokens, and an optional `[extras]` selector — followed by a token
# boundary so the match isn't a truncated prefix of a longer junk run (`pkg[extra]wrong`). Dotted
# sub-tokens are captured (so a malformed `vortex-data.old` is seen whole and rejected), but a lone
# trailing '.' is not (so `pip install pkg.` at a sentence end is not absorbed and does not false-fail).
_COMMAND_TOKEN = r"([\w\-]+(?:\.[\w\-]+)*(?:\[[^\]]*\])?)(?![\w\-.\[])"


def command_tokens(prefix: str, text: str) -> list[str]:
    """All package arguments following `prefix` in `text` (e.g. every `pip install <token>`), skipping
    any leading short/long flags (`pip install --upgrade -U <token>`) so the package — not a flag — is
    the captured token. (A flag that itself consumes a value, e.g. `--index-url URL`, is the documented
    pathological residual and is not handled.)"""
    return re.findall(re.escape(prefix) + r"(?:-{1,2}[\w\-]+\s+)*" + _COMMAND_TOKEN, text)


def command_token_ok(value: str, tok: str, allow_extras: bool) -> bool:
    """A captured command token is consistent with the canonical value iff it equals it exactly, or —
    only when `allow_extras` (a pip/uvx-style ecosystem) — is the value followed by EXACTLY one
    `[extras]` selector and nothing else (e.g. `pip install pkg[polars,ray]`). Using a full match (not
    a prefix test) means `pkg[extra]wrong` or `pkg.old` is rejected; gating extras on `allow_extras`
    means `cargo add pkg[bogus]` is rejected (Cargo has no pip-style extras)."""
    if tok == value:
        return True
    return allow_extras and re.fullmatch(re.escape(value) + r"\[[^\]]*\]", tok) is not None


@dataclass
class ValueMatch:
    """A fact whose canonical value is derived from `truth_file` (group 1 of `truth_regex`, optionally
    mapped through `transform`) and must appear, via `claim_template`, in every file in `doc_files`.

    When `command_prefixes` is set, every `<prefix><token>` ANYWHERE under docs/ whose token begins
    with `command_base` must be consistent with the canonical value (exactly the value, or — when
    `command_extras` — the value plus one `[extras]` selector). The `command_base` scoping is what
    lets the global scan ignore unrelated installs (`pip install pandas`) while still catching a
    drifted/typo'd Vortex package name (`pip install vortex-dat`).
    """

    id: str
    description: str
    doc_files: list[str]
    # Single-source path: derive the value from group 1 of `truth_regex` in `truth_file` (optionally
    # mapped through `transform`). Multi-source path: set `derive` to compute the value from the repo
    # directly (e.g. summing two constants) — `truth_file`/`truth_regex` are then unused.
    truth_file: str = ""
    truth_regex: str = ""
    claim_template: str = "{value}"
    transform: Callable[[str], str] | None = None
    derive: Callable[[Path], str] | None = None
    command_prefixes: list[str] = field(default_factory=list)
    command_base: str = ""        # only command tokens starting with this stem are validated
    command_extras: bool = False  # whether `value[extras]` is valid syntax for this ecosystem
    forbid_regex: str = ""        # if set, the check FAILS when this pattern appears in any doc_file
                                  # (an absence check — e.g. a stale unsuffixed coordinate variant)

    def __post_init__(self) -> None:
        # Exactly one source mode: a `derive` callable, OR a (truth_file, truth_regex) pair. This
        # catches a malformed future entry at import time rather than letting it report an empty source.
        if self.derive is not None:
            if self.truth_file or self.truth_regex:
                raise ValueError(f"[{self.id}] set EITHER derive OR truth_file/truth_regex, not both")
            if self.transform is not None:
                raise ValueError(f"[{self.id}] `transform` is ignored for a derive fact; fold it into derive")
        elif not (self.truth_file and self.truth_regex):
            raise ValueError(f"[{self.id}] needs a derive callable OR both truth_file and truth_regex")

    @property
    def source_label(self) -> str:
        """Human-readable source for diagnostics — the truth file, or `<derived>` for a derive fact."""
        return self.truth_file or "<derived>"

    def resolve_truth(self, root: Path) -> str:
        if self.derive is not None:
            return self.derive(root)
        # Route Rust truth files through the nested-block-comment-aware stripper so every Rust-source
        # lock shares ONE decoy model (a nested `/* /* */ */` can't smuggle a decoy past the check);
        # non-Rust files keep the simple `//` + non-nested `/* */` stripper.
        raw = (root / self.truth_file).read_text(encoding="utf-8")
        text = _strip_rust_comments(raw) if self.truth_file.endswith(".rs") else _strip_comments(raw)
        # `re.search` takes the FIRST match; `truth_regex` MUST be unique enough that the first match is
        # the canonical source — anchor it to `^\s*<keyword>` (re.MULTILINE) so a commented decoy or a
        # later same-shaped line cannot win. Comments are already stripped above. A genuinely
        # multi-source fact belongs in a `derive` with an explicit uniqueness check (see _derive_*).
        m = re.search(self.truth_regex, text, re.MULTILINE)
        if not m:
            raise LookupError(
                f"[{self.id}] truth pattern {self.truth_regex!r} not found in {self.truth_file} — "
                f"the source of truth moved; update the registry."
            )
        raw = m.group(1)
        return self.transform(raw) if self.transform else raw

    def claim_for(self, value: str) -> str:
        return self.claim_template.format(value=value)

    def check(self, root: Path, *, override_value: str | None = None) -> tuple[bool, str]:
        """Return (ok, detail). When override_value is set (self-test), use it instead of the real
        truth — used to prove a drifted value would be caught."""
        value = override_value if override_value is not None else self.resolve_truth(root)
        claim = self.claim_for(value)

        missing = [
            f for f in self.doc_files
            if not token_present(claim, (root / f).read_text(encoding="utf-8"))
        ]
        if missing:
            return False, f"claim {claim!r} (from {self.source_label}) missing in: {', '.join(missing)}"

        # Absence check: a forbidden pattern (e.g. a stale unsuffixed coordinate variant) must NOT
        # appear in any doc_file — so a presence check passing on one corrected site can't mask a
        # stale sibling site that still carries the old form.
        if self.forbid_regex:
            for f in self.doc_files:
                if re.search(self.forbid_regex, (root / f).read_text(encoding="utf-8")):
                    return False, f"forbidden pattern {self.forbid_regex!r} (stale form) still present in {f}"

        # Command-prefix absence check is GLOBAL but stem-scoped: every `<prefix><token>` ANYWHERE
        # under docs/ whose token begins with `command_base` must use the canonical value. Scanning
        # the whole tree catches a stale install in an unlisted page; the stem scoping ignores
        # unrelated installs (`pip install pandas`) so they don't false-fail.
        if self.command_prefixes:
            base = self.command_base or value
            for p in all_doc_files(root):
                text = p.read_text(encoding="utf-8")
                for prefix in self.command_prefixes:
                    for tok in command_tokens(prefix, text):
                        if not tok.startswith(base):
                            continue  # an unrelated package, not a Vortex install claim
                        if not command_token_ok(value, tok, self.command_extras):
                            return False, (
                                f"stale command in {p.relative_to(root)}: {prefix!r} uses {tok!r}, "
                                f"expected {value!r} (from {self.source_label})"
                            )
        return True, f"claim {claim!r} present in {len(self.doc_files)} doc(s) + commands consistent site-wide"


# --- Registry ----------------------------------------------------------------
# Phase 1 seeds CURRENTLY-CORRECT facts so the lint is green on landing; Phase 2 appends one entry per
# drifted fact as it is fixed. Each entry derives both sides at runtime (no hard-coded expected values).
REGISTRY: list[ValueMatch] = [
    ValueMatch(
        id="pypi-package-name",
        description="PyPI distribution name + install commands match vortex-python/pyproject.toml",
        # tomllib path (not an unscoped `^name`) so it is independent of table order in the file.
        derive=lambda root: _toml_str(root, "vortex-python/pyproject.toml", "project", "name"),
        doc_files=["docs/getting-started/install.md", "docs/api/python/index.rst"],
        command_prefixes=["pip install ", "uvx --from ", "uv add "],
        command_base="vortex",   # catch a typo'd vortex* install; ignore `pip install pandas` etc.
        command_extras=True,     # pip/uvx/uv support `vortex-data[polars,...]`
    ),
    ValueMatch(
        id="rust-crate-install",
        description="`cargo add <crate>` in the Rust quickstart matches vortex/Cargo.toml name",
        derive=lambda root: _toml_str(root, "vortex/Cargo.toml", "package", "name"),
        doc_files=["docs/getting-started/rust.rst"],
        claim_template="cargo add {value}",
        command_prefixes=["cargo add "],
        command_base="vortex",   # ignore `cargo add serde` etc.; flag a wrong vortex* crate
        command_extras=False,    # Cargo has no pip-style `crate[extras]` syntax
    ),
    ValueMatch(
        id="postscript-max-size",
        description="MAX_POSTSCRIPT_SIZE (u16::MAX - N) computed from the const def matches the spec",
        truth_file="vortex-file/src/lib.rs",
        # Derive from the const DEFINITION, not the regression assertion, so a stale doc can't pass
        # against a test literal that itself drifted. u16::MAX is a language constant (65535).
        # `^\s*pub const` + `;` so a commented decoy or a changed RHS fails rather than mis-sourcing.
        truth_regex=r"^\s*pub const MAX_POSTSCRIPT_SIZE: u16 = u16::MAX - (\d+)\s*;",
        transform=lambda n: str(65535 - int(n)),
        doc_files=["docs/specification/reading-a-file.md"],
    ),
    ValueMatch(
        id="postscript-eof-size",
        description="EOF_SIZE (the EndOfFile trailer length) in reading-a-file.md matches the const",
        truth_file="vortex-file/src/lib.rs",
        truth_regex=r"^\s*pub const EOF_SIZE: usize = (\d+)\s*;",
        claim_template="{value} bytes",  # `... trailer is 8 bytes` — specific enough not to match any 8
        doc_files=["docs/specification/reading-a-file.md"],
    ),
    ValueMatch(
        id="postscript-first-read",
        description="First-read size = MAX_POSTSCRIPT_SIZE + EOF_SIZE (INITIAL_READ_SIZE) in the spec",
        # Derived from BOTH constituent constants (not a self-cancelling reconstruction), with the
        # INITIAL_READ_SIZE = MAX_POSTSCRIPT_SIZE + EOF_SIZE relationship verified in open.rs.
        derive=_derive_initial_read,
        doc_files=["docs/specification/reading-a-file.md"],
    ),
    ValueMatch(
        id="duckdb-scan-function",
        description="DuckDB scan table-function name registered in vortex-duckdb appears in the guide",
        truth_file="vortex-duckdb/cpp/table_function.cpp",
        # Not line-anchored (the C++ field initializer is mid-line); decoy protection comes from
        # `_strip_comments` removing `//` and `/* */` before matching in resolve_truth.
        truth_regex=r'name\s*:\s*\{"([a-z_]+)"s',
        doc_files=["docs/user-guide/duckdb.md"],
    ),
    # NOTE: the `duckdb-filesystem-setting` check is intentionally omitted. The `vortex_filesystem`
    # DuckDB setting was refactored out of `vortex-duckdb`, so this source-anchored check can no longer
    # resolve its truth value. Re-base it (and the filesystem section of docs/user-guide/duckdb.md)
    # against the current DuckDB integration before re-enabling.
    ValueMatch(
        id="spark-batch-default",
        description="Spark writer default batch size matches VortexDataWriter.DEFAULT_BATCH_SIZE",
        truth_file="java/vortex-spark/src/main/java/dev/vortex/spark/write/VortexDataWriter.java",
        # `^\s*...final int` + `;` so a non-literal RHS or commented decoy cannot mis-source.
        truth_regex=r"^\s*(?:private\s+)?static\s+final\s+int\s+DEFAULT_BATCH_SIZE\s*=\s*(\d+)\s*;",
        doc_files=["docs/user-guide/spark.md"],
    ),
    ValueMatch(
        id="spark-maven-artifact",
        description="Maven coordinate uses the scala-suffixed `vortex-spark_2.13` artifact (docs + READMEs)",
        derive=_derive_spark_artifact,
        doc_files=["docs/user-guide/spark.md", "README.md", "java/README.md"],
        # ...and NO bare `vortex-spark` coordinate (suffix dropped) survives: a `vortex-spark` in a
        # coordinate position — `dev.vortex:vortex-spark`, `dev.vortex/vortex-spark` (the shields.io
        # badge form), or `<artifactId>vortex-spark` — NOT followed by the `_<scala>` suffix. (A bare
        # backticked `vortex-spark` describing the connector component is not a coordinate and is fine.)
        forbid_regex=r"(?:dev\.vortex[:/]|<artifactId>)vortex-spark(?!_)",
    ),
    ValueMatch(
        id="jni-module-name",
        description="java/README.md cites the `vortex-jni` JNI module (settings.gradle.kts include + the "
                    "module's build.gradle.kts artifactId); forbids the renamed-away `vortex-java`",
        derive=_jni_module_name,  # "vortex-jni"; fails loud if the Gradle module is renamed
        claim_template="{value}",
        doc_files=["java/README.md"],
        forbid_regex=r"\bvortex-java\b",  # the old name must not reappear
    ),
    ValueMatch(
        id="spark-scala213-version",
        description="spark.md's `vortex-spark_2.13` -> Spark 4.x targeting matches the build.gradle.kts when-arm",
        derive=lambda root: _spark_major_for_scala(root, "2.13"),
        # Scope the claim to the artifact->Spark-major sentence so a wrong major fails even if "Spark 4.x"
        # appears elsewhere.
        claim_template="`vortex-spark_2.13` artifact targets {value}",
        doc_files=["docs/user-guide/spark.md"],
    ),
    ValueMatch(
        id="spark-scala212-version",
        description="spark.md's `vortex-spark_2.12` -> Spark 3.x targeting matches the build.gradle.kts when-arm",
        derive=lambda root: _spark_major_for_scala(root, "2.12"),
        claim_template="`vortex-spark_2.12` artifact targeting {value}",
        doc_files=["docs/user-guide/spark.md"],
    ),
    ValueMatch(
        id="spark-output-filename",
        description="spark.md output filename matches the Java `String.format` in BOTH Spark writers",
        derive=_derive_spark_filename,
        doc_files=["docs/user-guide/spark.md"],
    ),
    ValueMatch(
        id="spark-batch-range",
        description="Spark writer batch-size range `MIN–MAX` matches VortexDataWriter MIN/MAX_BATCH_SIZE",
        # Derived from BOTH bounds so the documented `(1–65536)` locks MIN_BATCH_SIZE (=1) AND
        # MAX_BATCH_SIZE (=65536); a bare `1` is too generic to lock, but the range string is specific.
        derive=_derive_spark_batch_range,
        doc_files=["docs/user-guide/spark.md"],
    ),
    ValueMatch(
        id="cli-crate-install",
        description="`cargo binstall/install <crate>` in install.md matches vortex-tui/Cargo.toml name",
        derive=lambda root: _toml_str(root, "vortex-tui/Cargo.toml", "package", "name"),
        doc_files=["docs/getting-started/install.md"],
        command_prefixes=["cargo binstall ", "cargo install "],
        command_base="vortex",   # flag a wrong vortex* CLI crate; ignore unrelated `cargo install` lines
        command_extras=False,
    ),
    ValueMatch(
        id="python-min-version",
        description="`Python <X.Y>` in the Python compatibility docs matches pyproject requires-python",
        derive=_derive_python_version,
        doc_files=["docs/api/python/index.rst"],
    ),
    ValueMatch(
        id="cli-binary-name",
        description="The `vx` CLI binary name in the getting-started docs matches vortex-tui [[bin]] name",
        # The [[bin]] name (the invoked binary), parsed via tomllib so comments/key-order/list-values
        # can't mis-source it. Locking it means a rename is caught (the docs must mention the new name)
        # rather than silently passing — the CLI subcommand check alone would scan the NEW prefix and
        # miss the now-stale old-name invocations. Cover EVERY front-door page that invokes `vx`.
        derive=lambda root: _toml_str(root, "vortex-tui/Cargo.toml", "bin", 0, "name"),
        doc_files=["docs/getting-started/index.md", "docs/getting-started/install.md",
                   "docs/getting-started/query.md", "docs/getting-started/convert.md"],
    ),
    ValueMatch(
        id="io-coalesce-file",
        description="local-file coalesce distance+max in io.md prose match CoalesceConfig::file()",
        # Derive BOTH bounds and lock the full prose phrase ("1 MB distance and 4 MB max size") so the
        # claim is SCOPED to the local-file sentence — a wrong distance OR max fails, and a bare "4 MB"
        # elsewhere in io.md can't satisfy it. The forbid additionally catches the stale 8 KB value in
        # EITHER form (prose or the Backend Adaptation table row); the in-memory note has no "Local file".
        derive=lambda root: (lambda dm: f"{_bytes_to_human(dm[0])} distance and {_bytes_to_human(dm[1])} max size")(
            _coalesce_bytes(root, "file")),
        doc_files=["docs/developer-guide/internals/io.md"],
        forbid_regex=r"[Ll]ocal files?\b[^\n]*8\s*KB",
    ),
    ValueMatch(
        id="io-coalesce-object",
        description="object-store coalesce distance+max in io.md prose match CoalesceConfig::object_storage()",
        # Scoped like io-coalesce-file: lock the full "1 MB distance and 16 MB max size" object-store phrase.
        derive=lambda root: (lambda dm: f"{_bytes_to_human(dm[0])} distance and {_bytes_to_human(dm[1])} max size")(
            _coalesce_bytes(root, "object_storage")),
        doc_files=["docs/developer-guide/internals/io.md"],
    ),
    ValueMatch(
        id="io-in-memory-coalesce-behavior",
        description="io.md's in-memory opt-in coalescing claim matches the VortexReadAt default (coalesce_config "
                    "returns None ⇒ a reader that doesn't override it does not coalesce)",
        derive=_io_in_memory_coalesce_behavior,  # "opt-in"; fails loud if the default coalesce_config() != None
        doc_files=["docs/developer-guide/internals/io.md"],
    ),
    ValueMatch(
        id="io-coalesce-in-memory",
        description="in-memory coalesce distance+max in io.md match CoalesceConfig::in_memory()",
        # Scoped like file/object: the in_memory() preset is 8 KB/8 KB. (Coalescing is opt-in for in-memory
        # readers — the default reader doesn't coalesce; readers that opt in use this preset. io.md states both.)
        derive=lambda root: (lambda dm: f"{_bytes_to_human(dm[0])} distance and {_bytes_to_human(dm[1])} max size")(
            _coalesce_bytes(root, "in_memory")),
        doc_files=["docs/developer-guide/internals/io.md"],
    ),
    ValueMatch(
        id="io-reader-file",
        description="io.md names `FileReadAt` as the local-file VortexReadAt implementor (std_file/read_at.rs)",
        derive=lambda root: _io_reader_impl(root, "FileReadAt", "vortex-io/src/std_file/read_at.rs"),
        claim_template="{value}",
        doc_files=["docs/developer-guide/internals/io.md"],
    ),
    ValueMatch(
        id="io-reader-object-store",
        description="io.md names `ObjectStoreReadAt` as the object-store VortexReadAt implementor "
                    "(object_store/read_at.rs)",
        derive=lambda root: _io_reader_impl(root, "ObjectStoreReadAt", "vortex-io/src/object_store/read_at.rs"),
        claim_template="{value}",
        doc_files=["docs/developer-guide/internals/io.md"],
    ),
    ValueMatch(
        id="buffered-block-size",
        description="`Buffered Layout` size in file-format.md matches BufferedStrategy in strategy.rs",
        derive=_derive_buffered_block_size,
        # Scope the claim to the Buffered Layout line ("localize up to 2 MB") so a wrong size there
        # fails even if a correct "2 MB" appears in another sentence.
        claim_template="localize up to {value}",
        doc_files=["docs/concepts/file-format.md"],
        # Forbid every stale form this PR corrected: (1) the wrong localization value (BufferedStrategy
        # is 2 * ONE_MEG, not 1 MB); (2) "<N> MB chunks" wording; (3) "<N> MB of uncompressed data" on
        # the Chunked layer. The 2 MB is buffered-chunk LOCALITY — the Chunked layer imposes no byte
        # size, so attributing a byte size to chunks/uncompressed-data is the misconception this fixes.
        forbid_regex=r"localize up to 1\s*MB|\d+\s*MB chunks|\d+\s*MB of uncompressed data",
    ),
    ValueMatch(
        id="cpp-binding-cxx",
        description="C++ docs describe a `cxx` Rust bridge (matches vortex-cxx Cargo.toml), not a C-FFI wrapper",
        # Present-check anchors on `cxx` (the real mechanism); the forbid blocks the stale current-state
        # claim that C++ "wraps the C FFI" — that is a FUTURE migration (language-bindings.md), not today.
        derive=_cxx_dep,
        doc_files=["docs/api/cpp/index.rst", "docs/developer-guide/embedding/cxx.md",
                   "docs/developer-guide/embedding/index.md", "docs/developer-guide/internals/architecture.md"],
        # Catch any current-state "wrap(per/s/ping) ... the C FFI" claim; the planned-migration wording
        # ("wrapping the C API") refers to the C API, not the C FFI, so it is correctly NOT matched.
        # `[^.]` (not `[^.\n]`) keeps it sentence-scoped while tolerating line wraps (no line-break dodge).
        forbid_regex=r"wrap(?:per|ping|s)?\b[^.]*?\bC FFI",
    ),
    ValueMatch(
        id="spark-filter-pushdown",
        description="integrations/spark.md says filter pushdown is supported, matching VortexScanBuilder",
        derive=_spark_filter_pushdown,  # "SupportsPushDownV2Filters"; fails loud if the connector drops it
        doc_files=["docs/developer-guide/integrations/spark.md"],
        forbid_regex=r"filter pushdown[^.]*\b(?:not yet connected|planned future work)\b|not yet connected to Spark",
    ),
    ValueMatch(
        id="duckdb-direct-path",
        description="user-guide/duckdb.md says direct file-path syntax works, matching the registered replacement scan",
        derive=_duckdb_replacement_scan,  # "replacement scan"; fails loud if the registration is gone
        doc_files=["docs/user-guide/duckdb.md"],
        forbid_regex=r"coming in an upcoming DuckDB release",
    ),
    ValueMatch(
        id="cli-convert-chunk-size",
        description="convert.md + `vx convert` --help state the 8192-row batch chunking, matching BATCH_SIZE",
        # The quickstart used to claim "chunking on Parquet row-group boundaries"; really `vx convert`
        # reads in BATCH_SIZE-row batches and writes 8192-row blocks. Lock the value (scoped to "<N>-row")
        # in BOTH the doc and the CLI --help comment; forbid the stale row-group-boundaries claim.
        derive=_convert_batch_size,  # "8192"
        # Scope to the batch-chunking phrase common to both the doc ("chunking into 8192-row batches")
        # and the --help comment ("Chunking occurs in 8192-row batches"), so an unrelated "8192-row"
        # mention can't mask drift in the actual chunking sentence.
        claim_template="{value}-row batches",
        doc_files=["docs/getting-started/convert.md", "vortex-tui/src/lib.rs"],
        forbid_regex=r"(?i)row.?group boundaries",
    ),
    ValueMatch(
        id="scan-api-present",
        description="roadmap.md notes the Scan API exists today, matching the `vortex-scan` crate",
        derive=_scan_crate_name,  # "vortex-scan"; fails loud if the crate is gone
        # Scope to the PRESENT-state sentence ("already provides"), so a future-only "the vortex-scan
        # crate will provide ..." rewording fails rather than passing on a bare token match.
        claim_template="Scan API (the `{value}` crate) already provides",
        doc_files=["docs/project/roadmap.md"],
    ),
    ValueMatch(
        id="variant-dtype-present",
        description="roadmap.md notes the Variant DType exists today, matching DType::Variant in vortex-array",
        derive=_variant_dtype_present,  # "Variant"; fails loud if the variant is gone/renamed
        # Scope to the present-state claim so a "Variant is planned/upcoming" rewording fails.
        claim_template="`{value}` DType already exists",
        doc_files=["docs/project/roadmap.md"],
    ),
    ValueMatch(
        id="canonical-union-exception",
        description="concepts/arrays.md + api/python/arrays.rst name `Union` as the DType without a canonical "
                    "encoding; locked to the DType + Canonical enums (fails loud if Union is canonicalized)",
        derive=_union_is_sole_noncanonical,  # "Union"; the code-side guard is the derive, see its docstring
        # Bare-token present-check, scoped to the two canonical-arrays docs (where `Union` appears only in
        # the exception sentence); the derive does the load-bearing code assertion.
        claim_template="{value}",
        doc_files=["docs/concepts/arrays.md", "docs/api/python/arrays.rst"],
    ),
    ValueMatch(
        id="integration-scan-trait",
        description="integration docs name the real `DataSource` scan trait, not a (nonexistent) `Source` trait",
        # The duckdb/datafusion/spark integration pages describe migrating to the Scan API; they must name
        # the real `DataSource` trait. Present-anchor on it; forbid a backticked `Source`. (No `sink`
        # forbid here — Spark docs may legitimately discuss sinks.)
        derive=lambda root: "DataSource",
        doc_files=["docs/developer-guide/integrations/duckdb.md",
                   "docs/developer-guide/integrations/datafusion.md",
                   "docs/developer-guide/integrations/spark.md"],
        forbid_regex=r"`Source`",
    ),
    ValueMatch(
        id="cpp-not-wrapper-nav",
        description="nav/landing docs don't frame the C++ binding as a 'wrapper' or label C++ as FFI (it's cxx)",
        # Companion to cpp-binding-cxx for the top-level/nav pages, which don't mention `cxx` (so they
        # can't share that check's present-anchor). `Vortex` is a stable present anchor; the forbid blocks
        # the stale "C++ wrapper" framing and the "C/C++ (FFI)" label (C++ is a cxx bridge, not the C FFI).
        derive=lambda root: "Vortex",
        doc_files=["docs/index.md", "docs/developer-guide/overview.md", "docs/api/c/index.rst"],
        # Block the "C++ wrapper" framing, any "C++ (FFI)" / "C++ FFI" / "C/C++ (FFI)" label, AND a
        # present-tense "foundation for ... C++" claim (C++ is a cxx Rust bridge today, not built on the
        # C FFI — the C FFI is the INTENDED/future foundation). The `[^.]` (not `[^.\n]`) keeps the claim
        # SENTENCE-scoped while tolerating line WRAPS, so the stale claim can't dodge the forbid by breaking
        # across lines. "C++ (cxx)" and a later-sentence C++ mention are unaffected.
        forbid_regex=r"C\+\+ wrapper|C\+\+\s*\(?FFI\)?|foundation for[^.]*?\bC\+\+",
    ),
    ValueMatch(
        id="no-trino-integration-claim",
        description="integration lists don't claim a (nonexistent) current/in-progress Trino integration",
        # No Trino connector exists in the repo (only JNI bindings an external connector could build on).
        # Forbid the "Trino ... in progress" framing AND the stale "Spark and Trino" current-pairing in the
        # existing-integration / integration-point docs. The work-in-progress page remains the single place
        # that lists Trino as planned; factual "Trino supports JDK 22" and "future ... Trino" are unaffected.
        # `DataFusion` is a stable present anchor in all three files.
        derive=lambda root: "DataFusion",
        doc_files=["docs/index.md", "docs/concepts/scanning.md", "docs/developer-guide/language-bindings.md",
                   "docs/developer-guide/internals/architecture.md"],
        # `[^.]` + `\s+` keep these sentence-scoped but line-wrap-tolerant, so a wrapped "Trino ... in\n
        # progress" or "Spark\nand Trino" can't dodge the forbid by breaking across lines.
        forbid_regex=r"Trino[^.]*?\bin\s+(?:progress|development)|with\s+Trino|Spark\s+and\s+Trino",
    ),
    ValueMatch(
        id="scanning-api-traits",
        description="scanning.md uses real scan trait names (no `Sink`, no `Source` trait; it's `DataSource`)",
        # vortex-scan exposes DataSource/DataSourceScan/Partition. Present-anchor on the real `DataSource`
        # name (the doc now uses it); forbid the fabricated `Sink` (any casing) AND a backticked `Source`
        # implying a literal trait named Source (the real trait is `DataSource`).
        derive=lambda root: "DataSource",
        doc_files=["docs/concepts/scanning.md"],
        # Forbid the fabricated `Sink` trait/interface (scoped so legitimate "data sink" prose is fine)
        # and a backticked `Source` implying a literal trait named Source (the real trait is `DataSource`).
        forbid_regex=r"(?i)\bsink`?\s+(?:trait|interface)\b|`Source`",
    ),
]


@dataclass
class CliSubcommandCheck:
    """Assert that every `<command_prefix><subcommand>` referenced in docs SHELL CODE REGIONS
    (shell-tagged fenced blocks + inline code spans — NOT prose, and NOT non-shell code blocks where
    `vx` is the Python module alias) is a real subcommand of the CLI, derived from the clap
    `Subcommand` enum in `enum_file`. Catches a doc that references a fabricated or renamed CLI
    command — the `vx` analog of a fabricated API."""

    id: str
    description: str
    enum_file: str
    enum_name: str
    command_prefix: str = ""  # fallback literal, e.g. "vx "; prefer sourcing via bin_truth_file
    bin_truth_file: str = ""  # Cargo.toml whose `[[bin]] name = "..."` gives the real binary name

    def resolve_prefix(self, root: Path) -> str:
        """The command prefix (`<binary> `) — sourced from `bin_truth_file`'s `[[bin]] name` (via
        tomllib) so a renamed binary is detected (read both sides; don't hard-code), falling back to
        the literal `command_prefix` only when no `bin_truth_file` is configured."""
        if not self.bin_truth_file:
            return self.command_prefix
        return _toml_str(root, self.bin_truth_file, "bin", 0, "name") + " "

    @staticmethod
    def _enum_body(text: str, enum_name: str) -> tuple[str, str]:
        """Return `(body, header)` for `enum <name> { ... }`: `body` is the text between the enum's
        outer braces (brace-matched, so a nested `{ ... }` struct variant does not end it early);
        `header` is the CONTIGUOUS block of attribute / doc-comment / blank lines immediately preceding
        the declaration, where enum-level clap attributes (`#[command(rename_all = ...)]`) live. The
        block is collected by walking backward and bracket-balancing, so a multi-line `#[command(...)]`
        is captured in full regardless of length (not clipped by a fixed-width window)."""
        i = text.find(f"enum {enum_name} {{")
        if i < 0:
            raise LookupError(f"enum {enum_name} not found in source")
        start = text.index("{", i)
        depth, end = 0, None
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            raise LookupError(f"enum {enum_name} body is not closed")
        header_lines: list[str] = []
        bracket = 0  # net unclosed ']' while walking UP through a (possibly multi-line) attribute
        for line in reversed(text[:i].split("\n")):
            s = line.strip()
            if bracket > 0:  # inside a multi-line attribute we entered from its closing `]`
                header_lines.append(line)
                bracket += s.count("]") - s.count("[")
            elif s == "" or s.startswith("//"):  # blank or doc/line comment: part of the header block
                header_lines.append(line)
            elif "]" in s or s.startswith("#["):  # an attribute line (possibly with a trailing comment
                header_lines.append(line)           # like `#[command(rename_all=..)] // note`)
                bracket += s.count("]") - s.count("[")
            else:  # a line of the previous item — the header block ends here
                break
        return text[start + 1:end], "\n".join(reversed(header_lines))

    @classmethod
    def _enum_variants(cls, text: str, enum_name: str) -> list[str]:
        """Variant identifiers of `enum <name>`, DEPTH-AWARE within the body: only identifiers at the
        enum's top level (brace-depth 0) are variants, so a struct-variant's field type
        (`Browse {\\n  file: PathBuf,\\n}`) on its own line is not mis-read as a variant."""
        body, _ = cls._enum_body(text, enum_name)
        variants: list[str] = []
        depth = 0  # depth WITHIN the enum body
        for line in body.split("\n"):
            if depth == 0:
                m = re.match(r"([A-Z][A-Za-z0-9]*)\s*[({,]", line.strip())
                if m:
                    variants.append(m.group(1))
            depth += line.count("{") - line.count("}")
        return variants

    @staticmethod
    def _to_kebab(variant: str) -> str:
        """clap/heck default subcommand name: insert `-` at lower/digit->Upper and Acronym->Word
        boundaries, then lowercase. `Tree`->`tree`, `FooBar`->`foo-bar`, `SQLQuery`->`sql-query`,
        `HTTPServer`->`http-server`."""
        s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", variant)
        s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "-", s)
        return s.lower()

    @staticmethod
    def _decode_rust_str(v: str) -> str | None:
        """Decode a Rust string literal value: `"x"` -> `x` (un-escaping `\\"`/`\\\\`), raw
        `r#"x"#`/`r"x"` -> `x`. Returns None when `v` is not a (recognized) string literal — e.g. an
        identifier, number, list, or bare flag — so callers can distinguish a string value from a key."""
        rm = re.match(r'r(#*)"(.*)"\1\Z', v, re.DOTALL)
        if rm:
            return rm.group(2)
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            return v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        return None

    @staticmethod
    def _iter_attrs(text: str):
        """Yield the inner text of each top-level `#[...]` attribute, string- and bracket-aware (so a
        `]` inside a string value does not end the attribute early). A variant may carry several."""
        i, n = 0, len(text)
        while i < n:
            if text.startswith("#[", i):
                j, depth = i + 2, 1
                while j < n and depth > 0:
                    rm = re.match(r'r(#*)"', text[j:])
                    if rm:  # raw string
                        close = '"' + rm.group(1)
                        e = text.find(close, j + len(rm.group(0)))
                        j = n if e < 0 else e + len(close)
                        continue
                    ch = text[j]
                    if ch == '"':  # regular string with backslash escapes
                        k = j + 1
                        while k < n and text[k] != '"':
                            k += 2 if text[k] == "\\" else 1
                        j = k + 1
                        continue
                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1
                    j += 1
                yield text[i + 2:j - 1]
                i = j
            else:
                i += 1

    @staticmethod
    def _meta_items(s: str, i: int) -> list[str]:
        """Top-level comma-separated items of a `(...)` meta list starting at index `i` (just past the
        opening paren), STRING-LITERAL- and paren/bracket-depth-aware so commas inside strings or nested
        groups do not split, and a `name = "x"` inside another item's string cannot become its own item."""
        depth, bracket = 1, 0
        items: list[str] = []
        cur: list[str] = []
        while i < len(s):
            rm = re.match(r'r(#*)"', s[i:])
            if rm:  # raw string r#"..."#
                close = '"' + rm.group(1)
                end = s.find(close, i + len(rm.group(0)))
                end = len(s) if end < 0 else end + len(close)
                cur.append(s[i:end])
                i = end
                continue
            c = s[i]
            if c == '"':  # regular string with backslash escapes
                j = i + 1
                while j < len(s) and s[j] != '"':
                    j += 2 if s[j] == "\\" else 1
                cur.append(s[i:min(j + 1, len(s))])
                i = j + 1
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            elif c == "[":
                bracket += 1
            elif c == "]":
                bracket -= 1
            elif c == "," and depth == 1 and bracket == 0:
                items.append("".join(cur))
                cur = []
                i += 1
                continue
            cur.append(c)
            i += 1
        items.append("".join(cur))
        return items

    @classmethod
    def _clap_meta(cls, attr_text: str) -> dict[str, str | None]:
        """Merge EVERY `command(...)`/`clap(...)` meta list across ALL `#[...]` attributes in
        `attr_text` into `{key: value}` (value = decoded string for `key = "x"`, else None for a bare
        flag). A variant may split metadata across separate attributes (`#[command(about=..)]` then
        `#[command(name=..)]`); both are honored. Parsing is string-literal-aware (per `_iter_attrs` /
        `_meta_items`), so a `name = "x"` INSIDE another item's string CANNOT spoof the result."""
        meta: dict[str, str | None] = {}
        for inner in cls._iter_attrs(attr_text):
            mm = re.match(r"\s*(?:command|clap)\s*\(", inner)
            if not mm:
                continue
            for item in cls._meta_items(inner, mm.end()):
                km = re.match(r"\s*(\w+)\s*=\s*(.*)\Z", item, re.DOTALL)
                if km:
                    raw = km.group(2).strip()
                    decoded = cls._decode_rust_str(raw)
                    # decoded string for `key = "x"`, else the RAW value (e.g. a `["x","y"]` list, so
                    # plural `aliases`/`visible_aliases` are recoverable by the caller)
                    meta[km.group(1)] = decoded if decoded is not None else raw
                elif item.strip():
                    meta.setdefault(item.strip(), None)  # bare flag, e.g. `subcommand` / `flatten`
        return meta

    @classmethod
    def _variant_name_aliases(cls, attrs: str, ident: str) -> tuple[str, set[str]]:
        """The CLI name (clap `name=` override, else kebab(ident)) and the alias set
        (`alias`/`visible_alias`/`aliases`/`visible_aliases`) for one variant, parsed string-aware."""
        meta = cls._clap_meta(attrs)
        nm = meta.get("name")
        name = nm if isinstance(nm, str) and nm else cls._to_kebab(ident)
        aliases: set[str] = set()
        for key in ("alias", "visible_alias", "aliases", "visible_aliases"):
            val = meta.get(key)
            if isinstance(val, str):
                aliases.update(re.findall(r"[A-Za-z0-9][\w-]*", val))
        return name, aliases

    @staticmethod
    def _struct_body(text: str, struct_name: str) -> str | None:
        """Brace-matched body of `struct <name> { ... }`, or None (tuple/unit struct or not found)."""
        m = re.search(rf"\bstruct\s+{re.escape(struct_name)}\b[^{{;]*\{{", text)
        if not m:
            return None
        start = text.index("{", m.start())
        depth, end = 0, None
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        return text[start + 1:end] if end is not None else None

    @staticmethod
    def _rename_all_guard(enum_name: str, body: str, header: str) -> None:
        """FAIL LOUD on a clap `rename_all` (enum- or variant-level): it globally changes clap's casing
        convention, which the kebab heuristic does not model, so a silent docs-vs-CLI mismatch is turned
        into an explicit error to extend the check — rather than the matcher itself silently drifting,
        which the project BANS forbid."""
        if re.search(r"\brename_all\b", header) or re.search(r"\brename_all\b", body):
            raise LookupError(
                f"enum {enum_name} uses a clap `rename_all` attribute the conformance check does not "
                f"model; extend CliSubcommandCheck to honor it before trusting the docs-vs-CLI comparison")

    @staticmethod
    def _net_delim(line: str) -> int:
        """Net unbalanced `()`/`{}`/`[]` openers minus closers on a line."""
        return (line.count("(") + line.count("{") + line.count("[")
                - line.count(")") - line.count("}") - line.count("]"))

    @classmethod
    def _walk_variants(cls, body: str) -> list[dict[str, str]]:
        """Per top-level variant of an enum body, return `{ident, attrs, text}`: `attrs` is the
        concatenated (bracket-balanced) attribute text; `text` is the variant's FULL declaration — a
        single line for `Tree(Args)`, or all lines until the payload's `()`/`{}` balances for a
        rustfmt-wrapped multi-line tuple/struct variant. Capturing the full payload (not just the first
        line) means a wrapped `Tree(\\n    Args,\\n)` still exposes its nested-subcommand type."""
        variants: list[dict[str, str]] = []
        pending, attr_open = "", 0
        cur: dict[str, str] | None = None  # variant whose multi-line payload/block we are accumulating
        cap_depth = 0
        for line in body.split("\n"):
            s = line.strip()
            if cur is not None:  # accumulate until the variant's delimiters balance
                cur["text"] += "\n" + line
                cap_depth += cls._net_delim(line)
                if cap_depth <= 0:
                    cur = None
                continue
            if attr_open > 0 or s.startswith("#["):
                pending += " " + s
                attr_open += s.count("[") - s.count("]")
                continue
            if (mv := re.match(r"([A-Z][A-Za-z0-9]*)", s)):
                v = {"ident": mv.group(1), "attrs": pending, "text": line}
                variants.append(v)
                pending = ""
                d = cls._net_delim(line)
                if d > 0:  # multi-line tuple/struct variant — keep accumulating its payload/block
                    cur = v
                    cap_depth = d
        return variants

    @classmethod
    def _subcommand_field_enum(cls, text: str) -> str | None:
        """The enum type of a `#[clap(subcommand)]` / `#[command(subcommand)]` field within `text`, or
        None if there is no such field OR the field is OPTIONAL/repeated (`Option<..>`/`Vec<..>`).
        The `subcommand` flag is detected among OTHER meta (`#[command(subcommand, required = true)]`)
        by parsing the attribute string-aware — not by requiring `subcommand` to be the sole arg. An
        optional nested subcommand means the parent ALSO accepts its own positional args, so we cannot
        tell a fabricated subcommand from an argument — the parent is treated as a lenient leaf rather
        than risk false-positives on real `vx inspect <file>`-style invocations."""
        for am in re.finditer(r"(#\[(?:clap|command)\s*\([^\]]*\)\])\s*(?:pub\s+)?\w+\s*:\s*([^,\n]+)", text):
            if "subcommand" not in cls._clap_meta(am.group(1)):  # genuine top-level flag, not in a string
                continue
            expr = am.group(2).strip()
            if expr.startswith(("Option", "Vec")):  # optional/repeated subcommand → lenient leaf
                return None
            idents = re.findall(r"[A-Za-z_]\w*", expr)
            return idents[-1] if idents else None
        return None

    @classmethod
    def _resolve_nested(cls, crate_src: str, type_name: str, visited: set[str]) -> dict:
        """Children sub-tree for a variant payload type: follow a struct's REQUIRED `#[clap(subcommand)]`
        field to its enum, or a payload that is itself a `#[derive(Subcommand)] enum`. Else a leaf."""
        t = re.sub(r"[&<].*", "", type_name.split("::")[-1]).strip()
        if not t or not t[0].isupper():
            return {}
        sb = cls._struct_body(crate_src, t)
        if sb is not None:
            nested = cls._subcommand_field_enum(sb)
            return cls._command_tree(crate_src, nested, visited) if nested else {}
        if re.search(rf"#\[derive\([^)]*Subcommand[^)]*\)\]\s*(?:pub\s+)?enum\s+{re.escape(t)}\b", crate_src):
            return cls._command_tree(crate_src, t, visited)
        return {}

    @classmethod
    def _command_tree(cls, crate_src: str, enum_name: str, visited: set[str] | None = None) -> dict:
        """Recursively build the clap command TREE rooted at `enum <enum_name>`: a nested dict mapping
        each subcommand name (and alias) to its own sub-tree (`{}` for a leaf). Nested subcommands are
        resolved by following a tuple variant's payload type (e.g. `Tree(TreeArgs)`) to a struct's
        `#[clap(subcommand)]` field. Recursion is cycle-guarded. FAILS LOUD on `rename_all`."""
        visited = visited or set()
        if enum_name in visited:
            return {}
        visited = visited | {enum_name}
        body, header = cls._enum_body(crate_src, enum_name)
        cls._rename_all_guard(enum_name, body, header)
        tree: dict = {}
        for v in cls._walk_variants(body):
            name, aliases = cls._variant_name_aliases(v["attrs"], v["ident"])
            children: dict = {}
            # tuple payload anchored to the variant ident (so an attribute's `command(` inside a struct
            # variant body is NOT misread as a tuple payload); spans lines for rustfmt-wrapped variants.
            tuple_m = re.match(r"\s*" + re.escape(v["ident"]) + r"\s*\(\s*(?:#\[[^\]]*\]\s*)?([\w:]+)",
                               v["text"])
            if tuple_m:  # tuple variant: payload type may carry a nested subcommand
                children = cls._resolve_nested(crate_src, tuple_m.group(1), visited)
            else:  # struct/unit variant: a #[command(subcommand)] field carries the nested enum
                nested = cls._subcommand_field_enum(v["text"])
                if nested:
                    children = cls._command_tree(crate_src, nested, visited)
            tree[name] = children
            for a in aliases:
                tree[a] = children
        return tree

    @classmethod
    def _subcommand_names(cls, text: str, enum_name: str) -> set[str]:
        """Flat set of subcommand names (kebab/`name=`/aliases) for ONE enum's variants — the first
        level only. Used by the self-test; the live check uses `_command_tree` for nested paths."""
        body, header = cls._enum_body(text, enum_name)
        cls._rename_all_guard(enum_name, body, header)
        names: set[str] = set()
        for v in cls._walk_variants(body):
            name, aliases = cls._variant_name_aliases(v["attrs"], v["ident"])
            names.add(name)
            names.update(aliases)
        return names

    # Shell-flavored code-block languages — where CLI invocations live. We deliberately do NOT scan
    # `python`/`rust`/etc. blocks: the `vx` token is overloaded (it is also the docs' Python module
    # alias, `import vortex as vx`), so a Python block is full of `vx.array(...)` that has nothing to
    # do with the CLI. Restricting to shell blocks (plus inline spans) is the semantic match.
    _SHELL_LANGS = frozenset({"bash", "sh", "shell", "shell-session", "sh-session", "zsh",
                              "console", "shellsession", "text"})
    # MyST code directives — OPAQUE, with the language taken from the directive argument.
    _CODE_DIRECTIVES = frozenset({"code-block", "code", "sourcecode"})
    # MyST code-cell / doctest / raw directives — OPAQUE and NEVER shell: their bodies are Python or
    # literal content (where `vx` is the module alias, or arbitrary text), so we consume them WITHOUT
    # capturing and without letting their inline spans leak into the residue scanned for CLI mentions.
    _OPAQUE_NONSHELL_DIRECTIVES = frozenset({"doctest", "code-cell", "ipython", "ipython3",
                                             "eval-rst", "math", "raw", "literalinclude"})
    # reStructuredText code-bearing directives whose indented body must be CONSUMED (captured iff the
    # language is shell, else dropped) so a `.. code-block:: python` body's inline ``spans`` do not
    # leak into the CLI scan.
    _RST_CODE_DIRECTIVES = frozenset({"code-block", "code", "sourcecode", "parsed-literal", "doctest",
                                      "testcode", "testoutput", "ipython", "math", "raw"})
    # Shell operators/separators that end a command — the command-path walk stops here (and a second
    # same-line `vx` after one is matched independently by `_invalid_invocations`).
    _SHELL_OPS = frozenset({"&&", "||", "|", "|&", ";", "&", ">", ">>", "<", "2>", "2>>"})

    @classmethod
    def _shell_regions(cls, text: str) -> str:
        """Concatenate SHELL code regions — where real CLI invocations live — so neither prose mentions
        nor non-shell code (e.g. Python using the `vx` module alias) trigger false positives.

        Fences are parsed line-by-line with a length-aware scanner so the docs' own MyST patterns are
        handled: a `````{tab}````-style directive CONTAINER (info string starts with `{`, and is not a
        code directive) is TRANSPARENT — its body is re-parsed, so a nested ```` ```bash ```` block
        inside it is a real shell region — whereas a plain code fence OR a MyST code directive
        (` ```{code-block} bash `/` ```{code} `) is OPAQUE (its body is literal, captured iff the
        language is shell). A closing fence must use the same fence char and be at least as long as its
        opener, so a 3-backtick block nests cleanly inside a 4-backtick container. Also handles
        shell-tagged reStructuredText `.. code-block::`/`.. code::` directives and inline ``double`` /
        `single` code spans (a CLI mention like `vx query` in prose, but never a dotted `vx.array`)."""
        lines = text.split("\n")
        regions: list[str] = []
        residue: list[str] = []  # non-code-fence lines, for RST-directive + inline-span scanning
        directives: list[tuple[str, int]] = []  # open MyST/RST container fences: (char, length)
        fence_re = re.compile(r"^\s*([`~]{3,})(.*)$")
        i, n = 0, len(lines)
        while i < n:
            m = fence_re.match(lines[i])
            if not m:
                residue.append(lines[i])
                i += 1
                continue
            fence, info = m.group(1), m.group(2).strip()
            char, length = fence[0], len(fence)
            if info.startswith("{"):
                dm = re.match(r"\{([\w-]+)\}\s*(.*)$", info)
                directive = dm.group(1).lower() if dm else ""
                if directive in cls._CODE_DIRECTIVES:
                    # MyST code directive (` ```{code-block} bash `): OPAQUE, like a plain fence, with
                    # the language taken from the directive argument. Fall through to the capture path.
                    lang = (dm.group(2).strip().split() or [""])[0].lower()
                elif directive in cls._OPAQUE_NONSHELL_DIRECTIVES:
                    lang = ""  # OPAQUE non-shell: consume the body (below) but never capture it
                else:
                    directives.append((char, length))  # true container ({tab}/{note}): transparent
                    i += 1
                    continue
            elif not info and directives and directives[-1][0] == char and length >= directives[-1][1]:
                directives.pop()  # bare fence that closes the innermost open container
                i += 1
                continue
            else:
                lang = info.split()[0].lower() if info else ""
            # Opaque code fence: capture its body verbatim until a closing fence of the same char that
            # is at least as long. Nested fences of any length inside are literal content.
            close_re = re.compile(r"^\s*" + re.escape(char) + "{" + str(length) + r",}\s*$")
            body: list[str] = []
            j = i + 1
            while j < n and not close_re.match(lines[j]):
                body.append(lines[j])
                j += 1
            if lang in cls._SHELL_LANGS:
                regions.append("\n".join(body))
            i = j + 1  # skip past the closing fence (or EOF)

        # RST code directives: CONSUME each indented body (capturing it iff a shell code-block, else
        # dropping it) so a non-shell `.. code-block:: python` body's inline spans never reach the
        # CLI-mention scan. Lines outside any consumed body form `residue2`, scanned for inline spans.
        rst_lines = "\n".join(residue).split("\n")
        residue2: list[str] = []
        i = 0
        while i < len(rst_lines):
            dm = re.match(r"(\s*)\.\.\s+([\w-]+)::\s*(.*)$", rst_lines[i])
            if dm and dm.group(2).lower() in cls._RST_CODE_DIRECTIVES:
                directive = dm.group(2).lower()
                arg = (dm.group(3).strip().split() or [""])[0].lower()
                marker_indent = len(dm.group(1))
                j = i + 1
                block: list[str] = []
                while j < len(rst_lines) and not rst_lines[j].strip():  # skip blanks after the marker
                    j += 1
                while j < len(rst_lines):
                    lj = rst_lines[j]
                    if lj.strip() and (len(lj) - len(lj.lstrip())) <= marker_indent:
                        break  # dedent ends the block
                    block.append(lj)
                    j += 1
                if directive in ("code-block", "code", "sourcecode") and arg in cls._SHELL_LANGS:
                    regions.append("\n".join(block))  # shell body captured; non-shell body dropped
                i = j
                continue
            residue2.append(rst_lines[i])
            i += 1

        no_fence = "\n".join(residue2)
        regions += re.findall(r"``([^`]+)``", no_fence)              # RST inline literal
        regions += re.findall(r"(?<!`)`([^`\n]+)`(?!`)", no_fence)   # single-backtick span
        return "\n".join(regions)

    def command_tree(self, root: Path) -> dict:
        """The full clap command tree, built from EVERY `.rs` file in the CLI crate's source dir (so a
        nested subcommand enum defined in a sibling module — e.g. `TreeMode` in `tree.rs` — resolves)."""
        src_dir = (root / self.enum_file).parent
        crate_src = "\n".join(p.read_text(encoding="utf-8") for p in sorted(src_dir.rglob("*.rs")))
        return self._command_tree(crate_src, self.enum_name)

    @classmethod
    def _validate_path(cls, tree: dict, tokens: list[str]) -> str | None:
        """Walk a `vx` invocation's leading tokens down the command tree. Return the invalid
        command-PATH string (e.g. `tree frobnicate`) if a token sits in a subcommand position but is
        not a valid child; else None. Stops at the first flag (`-x`) or once a leaf is reached (the
        remaining tokens are arguments, not subcommands). A trailing sentence period is stripped, so an
        inline `` `vx convert` `` followed by prose punctuation is not mis-flagged."""
        node, path = tree, []
        for tok in tokens:
            if tok.startswith("-") or tok in cls._SHELL_OPS:
                break  # a flag, or a shell operator/separator (the command ends here)
            if not node:
                break  # leaf reached → remaining tokens are this command's arguments
            tok = tok.rstrip(".,:;")  # drop trailing sentence punctuation (inline `vx convert`.)
            if not tok:
                break
            if tok in node:
                path.append(tok)
                node = node[tok]
            else:
                # `node` still has (required) children but this token is not one of them — whether it is
                # a fabricated subcommand (`vx tree frobnicate`) or a path/arg where the required
                # subcommand was omitted (`vx tree ./file.vortex`), the invocation is invalid.
                return " ".join([*path, tok])
        return None

    def _invalid_invocations(self, root: Path, tree: dict, prefix: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        # Match EACH `<prefix>` independently (the body is NOT captured greedily to EOL), so a second
        # same-line invocation after a separator — `vx convert f && vx frobnicate` — is validated too.
        # Left boundary `(?<![\w-])` so `uvx ...` is not matched as `vx ...`; same-line `[ \t]+` so a
        # line-final `vx` (the Python alias) cannot bind to the next line. From each match we take the
        # rest of the line and whitespace-split; `_validate_path` stops at the first arg/flag/separator.
        pat = re.compile(r"(?<![\w-])" + re.escape(prefix.strip()) + r"[ \t]+")
        for p in all_doc_files(root):
            for line in self._shell_regions(p.read_text(encoding="utf-8")).split("\n"):
                for m in pat.finditer(line):
                    invalid = self._validate_path(tree, line[m.end():].split())
                    if invalid:
                        out.setdefault(invalid, []).append(str(p.relative_to(root)))
        return out

    def check(self, root: Path) -> tuple[bool, str]:
        prefix = self.resolve_prefix(root)  # sourced from Cargo.toml [[bin]] name (not hard-coded)
        tree = self.command_tree(root)
        if not tree:
            raise LookupError(f"no subcommands parsed from enum {self.enum_name} in {self.enum_file}")
        bad = self._invalid_invocations(root, tree, prefix)
        if bad:
            path = sorted(bad)[0]
            return False, (
                f"docs reference unknown `{prefix}{path}` (not a valid command path for "
                f"{self.enum_name} in {self.enum_file}; top-level: {', '.join(sorted(tree))}) "
                f"in {bad[path][0]}"
            )
        return True, f"all `{prefix}*` references are valid command paths ({len(tree)} top-level)"


CLI_CHECKS: list[CliSubcommandCheck] = [
    CliSubcommandCheck(
        id="vx-subcommands",
        description="every `vx <subcommand>` in docs is a real vortex-tui CLI subcommand",
        enum_file="vortex-tui/src/lib.rs",
        enum_name="Commands",
        command_prefix="vx ",                      # fallback only
        bin_truth_file="vortex-tui/Cargo.toml",    # source the real binary name from `[[bin]] name`
    ),
]


@dataclass
class DocMembershipCheck:
    """Assert every token captured by `mention_regex` (group 1) across `doc_files` is a member of the
    allowed set derived from a source of truth by `allowed`. The doc analog of CliSubcommandCheck for
    non-command facts — e.g. every `vortex-spark_<scala>` the docs advertise is a published scala
    variant per java/settings.gradle.kts, so a stale variant (one removed from the build) is caught."""

    id: str
    description: str
    doc_files: list[str]
    mention_regex: str
    allowed: Callable[[Path], set[str]]
    region_fn: Callable[[str], str] | None = None  # extract code regions first (e.g. python blocks)
    scan_all_docs: bool = False                     # scan every doc (mentions may appear anywhere)

    @staticmethod
    def _outside(text: str, mention_regex: str, allowed: set[str]) -> set[str]:
        """The set of `mention_regex` group-1 tokens in `text` that are NOT in `allowed` (pure; the
        membership logic, exercised by the self-test on synthetic input without touching the repo)."""
        return {t for t in re.findall(mention_regex, text) if t not in allowed}

    def check(self, root: Path) -> tuple[bool, str]:
        allowed = self.allowed(root)
        if not allowed:
            raise LookupError(f"[{self.id}] derived an empty allowed set — the source of truth moved")
        files = all_doc_files(root) if self.scan_all_docs else [root / f for f in self.doc_files]
        bad: dict[str, str] = {}
        for p in files:
            text = p.read_text(encoding="utf-8")
            if self.region_fn is not None:
                text = self.region_fn(text)  # scope to code regions to avoid incidental prose matches
            for tok in self._outside(text, self.mention_regex, allowed):
                bad.setdefault(tok, str(p.relative_to(root)))
        if bad:
            t = sorted(bad)[0]
            return False, (f"docs advertise `{t}`, not in the allowed set "
                           f"{{{', '.join(sorted(allowed))}}} (from source) in {bad[t]}")
        return True, f"all advertised variants are in {{{', '.join(sorted(allowed))}}}"


def _spark_scala_variants(root: Path) -> set[str]:
    """The published Scala suffixes of the Spark connector, from java/settings.gradle.kts."""
    settings = _strip_comments((root / "java/settings.gradle.kts").read_text(encoding="utf-8"))
    return set(re.findall(r'include\("vortex-spark_(\d+(?:\.\d+)?)"\)', settings))


# Python-flavored code-block languages — where `vortex.*` / `vx.*` API references live. Scoping to
# these (not raw prose) keeps incidental matches out of the API-name check: `bench.vortex.dev` (URLs),
# `vortex.h` (the C header), `${vortex.version}` (Maven), `vortex.write.batch.size` (a Spark option).
_PYTHON_LANGS = frozenset({"python", "pycon", "py", "python3", "ipython", "ipython3"})


# MyST directive classification. PROMPT/LANG directives carry python code (captured below). LEAF
# directives carry code/content that is NEVER a nestable container — a non-python one (```{code-block}
# bash`, a literalinclude, a toctree path list) is treated as OPAQUE, not descended into. Every OTHER
# directive (`tab`, admonitions, `dropdown`, …) IS a container we descend into to find nested python.
_PY_PROMPT_DIRECTIVES = frozenset({"doctest", "ipython", "ipython3"})
_PY_LANG_DIRECTIVES = frozenset({"code-cell", "code-block", "sourcecode"})
_LEAF_DIRECTIVES = frozenset({
    "code-block", "code-cell", "sourcecode", "doctest", "ipython", "ipython3",
    "literalinclude", "figure", "image", "math", "csv-table", "list-table",
    "bibliography", "toctree", "mermaid", "raw",
})


def _classify_fence(info: str, fence_len: int) -> str:
    """Classify a fence by its info string (and backtick length) into one of three kinds, shared by
    `_scan_fences` and `_strip_fenced_blocks` so their handling can't diverge:
      - 'python'    — body is python code (```python, ```{doctest}, ```{code-block} python, a bare
                      ```{code-cell}) → captured as an API region.
      - 'leaf'      — a self-contained NON-python code/content block (```bash/```text, a 3-backtick
                      bare block, ```{code-block} bash, literalinclude/toctree/…) → OPAQUE.
      - 'container' — `{tab}`/`{eval-rst}`/admonitions and 4+-backtick bare wrappers → descended into
                      (scan) / unwrapped (strip), because the body holds more markup: nested python
                      fences OR (for `{eval-rst}`) raw RST directives the prose passes must still see."""
    info = info.strip()
    if info.startswith("{"):
        dm = re.match(r"\{([\w-]+)\}\s*(.*)", info)
        directive = dm.group(1).lower() if dm else ""
        arg = (dm.group(2).split() or [""])[0].lower() if dm else ""
        if (directive in _PY_PROMPT_DIRECTIVES
                or (directive in _PY_LANG_DIRECTIVES and arg in _PYTHON_LANGS)
                or (directive == "code-cell" and arg == "")):  # bare {code-cell}: page python kernel
            return "python"
        return "leaf" if directive in _LEAF_DIRECTIVES else "container"
    if (info.split() or [""])[0].lower() in _PYTHON_LANGS:
        return "python"
    if info == "" and fence_len >= 4:  # a 4+-backtick bare wrapper; a 3-backtick bare block is literal
        return "container"
    return "leaf"


def _strip_pycon_output(body: str) -> str:
    """If a captured python region is a pycon/doctest SESSION (has `>>>` lines), keep only the prompt
    INPUT payloads (`>>>` / `...`) and drop interpreter OUTPUT lines — output such as a repr like
    `<vortex.PrimitiveArray ...>` is not API usage and must not be checked. A plain python block (no
    `>>>`) is returned unchanged."""
    if not re.search(r"^\s*>>>", body, re.MULTILINE):
        return body
    return "\n".join(re.findall(r"^\s*(?:>>>|\.\.\.) ?(.*)$", body, re.MULTILINE))


def _scan_fences(lines: list[str], regions: list[str]) -> None:
    """Walk fenced blocks honoring the CommonMark/MyST rule that a fence opened by N backticks is closed
    only by ≥N backticks — so a 3-backtick ```` ```python ```` nested inside a 4-backtick ```` ````{tab}
    ```` container is NOT mis-paired. Python fences are captured; CONTAINER fences are recursed into so
    nested python examples are still API-checked; LEAF (explicit non-python) fences are OPAQUE so a
    literal python snippet shown inside a shell/heredoc example is not mistaken for real API usage.
    Recursion always shrinks the body, so it terminates. See `_classify_fence` for the kinds."""
    i = 0
    while i < len(lines):
        m = re.match(r"(\s*)(`{3,}|~{3,})(.*)$", lines[i])
        if not m:
            i += 1
            continue
        fence = m.group(2)
        close = re.compile(r"\s*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$")
        j = i + 1
        while j < len(lines) and not close.match(lines[j]):
            j += 1
        body = lines[i + 1 : j]
        kind = _classify_fence(m.group(3), len(fence))
        if kind == "python":
            regions.append(_strip_pycon_output("\n".join(body)))
        elif kind == "container":  # descend for nested python
            _scan_fences(body, regions)
        # leaf → opaque, do not descend
        i = j + 1


def _strip_fenced_blocks(text: str) -> str:
    """Return `text` with python/leaf fenced blocks REMOVED but CONTAINER fences UNWRAPPED (open/close
    markers dropped, body kept and recursively stripped). Scopes the RST-directive and bare-`>>>` passes
    to prose: an opaque ```text/```bash block is gone (so a literal `>>> vortex.bad` / `.. code-block::
    python` inside it is not scanned), but a ```{eval-rst}` container's raw RST body — which may hold
    `.. code-block:: python` / `>>>` examples — is RETAINED so those passes still scan it. Markdown
    python is already captured by `_scan_fences`, so dropping python leaf fences here costs no coverage."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = re.match(r"(\s*)(`{3,}|~{3,})(.*)$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        fence = m.group(2)
        close = re.compile(r"\s*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$")
        j = i + 1
        while j < len(lines) and not close.match(lines[j]):
            j += 1
        if _classify_fence(m.group(3), len(fence)) == "container":
            out.append(_strip_fenced_blocks("\n".join(lines[i + 1 : j])))  # unwrap: keep body as prose
        # python / leaf → drop entirely
        i = j + 1
    return "\n".join(out)


def _python_regions(text: str) -> str:
    """Concatenate Python code regions: Markdown fences tagged with a python language (`python`,
    `pycon`) OR a MyST code-cell/doctest directive (```` ```{doctest} pycon ````, ```` ```{code-cell}
    python ````, ```` ```{code-block} python ````) — the dominant python-example form in these docs —
    and reStructuredText `.. code-block:: python` / `.. doctest::` directives. Fence parsing is
    backtick-length-aware so python fences nested inside MyST containers (e.g. ```` ````{tab} ````)
    are not missed."""
    regions: list[str] = []
    _scan_fences(text.split("\n"), regions)
    # The RST-directive and bare-`>>>` passes operate on PROSE (markdown fences stripped) so a literal
    # `.. code-block:: python` / `>>> vortex.x` shown inside an opaque ```text/```bash fence is not
    # scanned as live API usage. RST directives/doctests live in .rst files (no ``` fences), so
    # stripping costs no real coverage; markdown python examples are already captured by `_scan_fences`.
    prose = _strip_fenced_blocks(text)
    lines = prose.split("\n")
    i = 0
    while i < len(lines):
        m = re.match(r"(\s*)\.\.\s+(?:code-block|code|doctest)::\s*([\w-]*)\s*$", lines[i])
        if m and (m.group(2).lower() in _PYTHON_LANGS or "doctest" in lines[i]):
            indent = len(m.group(1))
            j = i + 1
            block: list[str] = []
            while j < len(lines) and not lines[j].strip():
                j += 1
            while j < len(lines):
                if lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) <= indent:
                    break
                block.append(lines[j])
                j += 1
            regions.append(_strip_pycon_output("\n".join(block)))  # `.. doctest::` blocks: drop output
            i = j
            continue
        i += 1
    # Prompt-only RST doctest blocks (no directive): collect the code after every `>>>`/`...` prompt,
    # so bare interpreter examples (e.g. api/python/expr.rst) are API-checked too. Same PROSE scoping
    # as the RST-directive pass; python/pycon fences with `>>>` are already captured by `_scan_fences`.
    regions += re.findall(r"^\s*(?:>>>|\.\.\.) ?(.*)$", prose, re.MULTILINE)
    return "\n".join(regions)


def _canonical_payload_classes(enum_body: str, api: set[str] | None) -> set[str]:
    """Map every `Canonical` variant to the last path-segment of its PAYLOAD type — `List(ListViewArray)`
    -> ListViewArray, `Foo(crate::a::Bar)` -> Bar — NOT the variant name. So `ListArray` (a non-canonical
    list encoding) is correctly excluded and the canonical `ListViewArray` is recognized. Keep only
    payloads exposed as Python classes (`Decimal`/`Variant`/`ListViewArray` are canonical in Rust but
    not yet in the Python API).

    FAILS LOUD if any variant declaration can't be parsed into `Variant(Payload)` form (a struct-style
    or multiline-payload variant), rather than silently skipping it — a silent skip would shrink the
    expected docs set and let an enum-shape change slip past the lock. Separated from the file read so
    the self-test can exercise it on a synthetic enum body."""
    classes: set[str] = set()
    for raw in enum_body.splitlines():
        line = raw.strip().rstrip(",").strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue  # attribute / comment / blank — structural noise, not a variant
        if not re.match(r"[A-Z]\w*", line):
            continue  # not a variant declaration line
        m = re.fullmatch(r"[A-Z]\w*\(\s*([\w:]+(?:<[^>]*>)?)\s*\)", line)
        if not m:
            raise LookupError(f"unparseable Canonical variant {line!r} (expected `Variant(Payload)`)")
        cls = m.group(1).split("::")[-1].split("<")[0]  # crate::a::Bar<T> -> Bar
        if api is None or cls in api:  # api=None -> keep ALL payloads (not just Python-exposed)
            classes.add(cls)
    return classes


def _canonical_all_payloads(root: Path) -> set[str]:
    """ALL canonical-encoding payload classes from the Rust `Canonical` enum (NOT filtered to the Python
    API) — e.g. DecimalArray, ListViewArray, VariantArray. The source of truth for the DType -> canonical
    encoding column of concepts/arrays.md."""
    canon = _strip_rust_comments((root / "vortex-array/src/canonical.rs").read_text(encoding="utf-8"))
    m = re.search(r"pub enum Canonical\s*\{(.*?)\n\}", canon, re.DOTALL)
    if not m:
        raise LookupError("could not find `pub enum Canonical` in vortex-array/src/canonical.rs")
    return _canonical_payload_classes(m.group(1), None)


def _canonical_python_classes(root: Path) -> set[str]:
    """The canonical-encoding Python classes: each Rust `Canonical` enum variant (canonical.rs) mapped
    to the Python class of its PAYLOAD type, kept only when that class exists in vortex's public API.
    The source of truth for which encodings arrays.rst's "Canonical Encodings" section should list.
    See `_canonical_payload_classes` for the payload-vs-variant-name mapping and fail-loud parsing."""
    canon = _strip_rust_comments((root / "vortex-array/src/canonical.rs").read_text(encoding="utf-8"))
    m = re.search(r"pub enum Canonical\s*\{(.*?)\n\}", canon, re.DOTALL)
    if not m:
        raise LookupError("could not find `pub enum Canonical` in vortex-array/src/canonical.rs")
    return _canonical_payload_classes(m.group(1), _vortex_public_api(root))


def _listed_canonical_classes(rst_text: str) -> set[str]:
    """The `vortex.<Class>` autoclasses listed under arrays.rst's "Canonical Encodings" section, bounded
    by the next RST section heading (or EOF) so a class in a LATER section (Utility, Compressed) is not
    counted. A pure helper so the section-boundary parsing is self-testable on synthetic input, not just
    the live docs file."""
    rst_text = rst_text.replace("\r\n", "\n")  # tolerate CRLF checkouts (Windows / autocrlf)
    m = re.search(r"Canonical Encodings\n-+\n(.*?)(?:\n[A-Z][\w ]+\n-+\n|\Z)", rst_text, re.DOTALL)
    if not m:
        return set()
    return set(re.findall(r"\.\.\s*autoclass::\s*vortex\.(\w+)", m.group(1)))


@dataclass
class CanonicalSectionCheck:
    """Assert arrays.rst's "Canonical Encodings" section lists EXACTLY the canonical Python classes
    (set equality, so a mislabeled non-canonical class like `VarBinArray` AND a missing canonical class
    like `FixedSizeListArray` are both caught) — sourced from the Rust `Canonical` enum."""

    id: str
    description: str

    def check(self, root: Path) -> tuple[bool, str]:
        expected = _canonical_python_classes(root)
        if not expected:
            raise LookupError(f"[{self.id}] derived an empty canonical set")
        text = (root / "docs/api/python/arrays.rst").read_text(encoding="utf-8")
        listed = _listed_canonical_classes(text)
        if listed != expected:
            return False, (f"arrays.rst Canonical Encodings section mismatch vs Rust Canonical enum: "
                           f"extra={sorted(listed - expected)} missing={sorted(expected - listed)}")
        return True, f"canonical section lists exactly the {len(expected)} canonical Python classes"


def _stat_names(root: Path) -> set[str]:
    """The canonical statistic names from `Stat::name()` in vortex-array expr/stats/mod.rs — the source
    of truth for the statistics arrays.md should advertise. Comments are stripped first; see
    `_stat_names_from` for the parse + per-variant cross-check that makes it fail loud rather than
    silently shrink the expected set."""
    src = _strip_rust_comments((root / "vortex-array/src/expr/stats/mod.rs").read_text(encoding="utf-8"))
    blocks = re.findall(r"fn name\(&self\)\s*->\s*&str\s*\{(.*?)\n\s*\}", src, re.DOTALL)
    if len(blocks) != 1:  # exactly one Stat::name(), else a decoy could mis-source the names
        raise LookupError(f"expected exactly 1 Stat::name() in stats/mod.rs, found {len(blocks)}")
    enum_m = re.search(r"pub enum Stat\s*\{(.*?)\n\}", src, re.DOTALL)
    if not enum_m:
        raise LookupError("could not find `pub enum Stat` in vortex-array/src/expr/stats/mod.rs")
    return _stat_names_from(blocks[0], enum_m.group(1))


def _stat_enum_variants(enum_body: str) -> set[str]:
    """The variant names of the `Stat` enum (`Variant = N,` or `Variant,` forms)."""
    variants = set(re.findall(r"^\s*([A-Z]\w*)\s*(?:=[^,\n]+)?,", enum_body, re.MULTILINE))
    if not variants:
        raise LookupError("no Stat enum variants parsed")
    return variants


def _stat_names_from(name_body: str, enum_body: str) -> set[str]:
    """Parse the `Stat::name()` arms and CROSS-CHECK that exactly one name was produced per enum variant.
    This is the definitive guard against ANY arm form that compiles while collapsing variants — a
    same-line OR multi-line or-pattern, a wildcard, a missing arm — all yield fewer names than variants
    and fail loud (the enum variant count is the ground truth). Separated from the file read so the
    self-test can drive both halves on synthetic input."""
    names = _parse_stat_name_arms(name_body)
    variants = _stat_enum_variants(enum_body)
    if len(names) != len(variants):
        raise LookupError(f"Stat::name() produced {len(names)} names but the enum has {len(variants)} "
                          "variants (a collapsed or-pattern / wildcard / missing arm?)")
    return names


def _parse_stat_name_arms(body: str) -> set[str]:
    """Names from a `Stat::name()` match body. Validates EVERY `=>` arm: the left side must be exactly
    one `Self::Variant` (NOT an or-pattern `Self::A | Self::B`, NOT a wildcard `_`, NOT a guard) and the
    right side must be a string literal — otherwise FAIL LOUD, because any of those forms could compile
    while collapsing several variants into one name and silently shrinking the expected set. Also fails
    on duplicate literals. Separated from the file read so the self-test can exercise the fail-loud
    paths on synthetic input."""
    arm_lines = [ln for ln in body.splitlines() if "=>" in ln]
    if not arm_lines:
        raise LookupError("Stat::name() yielded no match arms")
    names: list[str] = []
    for ln in arm_lines:
        lhs, _, rhs = ln.partition("=>")
        if not re.fullmatch(r"Self::\w+", lhs.strip()):  # or-pattern / wildcard / guard
            raise LookupError(f"Stat::name() arm pattern {lhs.strip()!r} is not a single `Self::Variant`; "
                              "extend the parser")
        lit = re.fullmatch(r'"(\w+)"', rhs.strip().rstrip(",").strip())
        if not lit:
            raise LookupError(f"Stat::name() arm returns a non-literal {rhs.strip()!r}; extend the parser")
        names.append(lit.group(1))
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:  # two variants sharing a name string is a source bug the lock must not silently absorb
        raise LookupError(f"Stat::name() returns duplicate literals {dupes}")
    return set(names)


def _listed_stat_names(md_text: str) -> tuple[list[str], int]:
    """(names, n_bullets) for arrays.md's `## Statistics` section: the leading `` `name` `` code-span of
    each bullet (in order, duplicates preserved) and the TOTAL bullet count. A bullet WITHOUT a leading
    code span (a malformed/extra entry like `* true_count: ...`) makes len(names) < n_bullets, so the
    caller can fail it instead of silently ignoring it. Section bounded at the next `## ` heading; a
    pure helper so the parsing is self-testable on synthetic md. (Wrapped continuation lines don't start
    with `*`, so they're not counted as bullets.)"""
    md_text = md_text.replace("\r\n", "\n")  # tolerate CRLF checkouts (Windows / autocrlf)
    m = re.search(r"## Statistics\n(.*?)(?:\n## |\Z)", md_text, re.DOTALL)
    if not m:
        return [], 0
    bullets = re.findall(r"^\s*\*\s+(.*)$", m.group(1), re.MULTILINE)
    names = [cm.group(1) for b in bullets if (cm := re.match(r"`(\w+)`", b))]
    return names, len(bullets)


@dataclass
class StatsListCheck:
    """Assert arrays.md's `## Statistics` section lists EXACTLY the `Stat::name()` values — sourced from
    the Rust `Stat` enum. Fails on: a fabricated stat (`true_count`), a missing one (`sum`), a duplicate,
    OR a bullet without a leading `` `name` `` code-span (which a bare set comparison would ignore)."""

    id: str
    description: str

    def check(self, root: Path) -> tuple[bool, str]:
        expected = _stat_names(root)
        names, n_bullets = _listed_stat_names((root / "docs/concepts/arrays.md").read_text(encoding="utf-8"))
        if n_bullets != len(names):
            return False, f"{n_bullets - len(names)} Statistics bullet(s) lack a leading `name` code-span"
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            return False, f"arrays.md Statistics section lists duplicate stats: {dupes}"
        listed = set(names)
        if listed != expected:
            return False, (f"arrays.md Statistics list mismatch vs Stat::name(): "
                           f"extra={sorted(listed - expected)} missing={sorted(expected - listed)}")
        return True, f"statistics section lists exactly the {len(expected)} Stat names"


def _vortex_submodules(root: Path) -> set[str]:
    """The importable submodules of the `vortex` package (`.py` files + package dirs). These are MODULES,
    not callables — so `vortex.<submodule>(...)` (e.g. `vortex.dataset(...)`) is a module-called-as-
    function error, as opposed to a real factory like `vortex.array(...)`."""
    pkg = root / "vortex-python/python/vortex"
    mods = {p.stem for p in pkg.iterdir() if p.suffix == ".py" and p.stem != "__init__"}
    mods |= {p.name for p in pkg.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    return mods


def _vortex_public_api(root: Path) -> set[str]:
    """The public top-level names of the `vortex` Python package: `__all__` (parsed from __init__.py
    via `ast`) PLUS the importable submodules (so `vortex.store`, `vortex.arrow`, etc. — accessible but
    not all in `__all__` — are recognized). The allowed set for the docs' `vortex.<name>` references.
    FAILS LOUD if `__all__` is not a literal list/tuple of names (e.g. became computed), rather than
    silently under-reporting the allowed set."""
    pkg = root / "vortex-python/python/vortex"
    names: set[str] = set()
    found = False
    for node in ast.walk(ast.parse((pkg / "__init__.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
            if not (isinstance(node.value, (ast.List, ast.Tuple))
                    and all(isinstance(e, ast.Constant) for e in node.value.elts)):
                raise LookupError("vortex `__all__` is not a literal list of names; the API-name check needs updating")
            names |= {e.value for e in node.value.elts}
            found = True
    if not found:
        raise LookupError("could not find a literal `__all__` in vortex/__init__.py")
    return names | _vortex_submodules(root)


@dataclass
class DTypeListCheck:
    """Assert the architecture overview's `DType` enum row lists EXACTLY the `DType` variants (sourced
    from the Rust enum) — set equality, so it can't drift to a stale subset (it had dropped Decimal/
    FixedSizeList/Union/Variant) NOR list a fabricated extra variant. The row is the table cell beginning
    ``DType` enum:`."""

    id: str
    description: str

    def check(self, root: Path) -> tuple[bool, str]:
        expected = _dtype_variant_names(root)
        text = (root / "docs/developer-guide/internals/architecture.md").read_text(encoding="utf-8")
        m = re.search(r"`DType` enum:([^|\n]*)", text)
        if not m:
            return False, "architecture.md has no ``DType` enum:` row to check"
        listed = set(re.findall(r"\b([A-Z]\w*)\b", m.group(1)))
        if listed != expected:
            return False, (f"architecture.md `DType` row mismatch vs the Rust enum: "
                           f"extra={sorted(listed - expected)} missing={sorted(expected - listed)}")
        return True, f"architecture.md `DType` row lists exactly the {len(expected)} variants"


# Wheel target triple -> the platform label the Python docs advertise. New triples FAIL LOUD so the
# docs (and this map) must be updated rather than silently drifting.
_WHEEL_LABEL = {
    "aarch64-apple-darwin": "Apple Silicon macOS",
    "x86_64-apple-darwin": "Intel macOS",
    "aarch64-unknown-linux-gnu": "ARM64 Linux",
    "x86_64-unknown-linux-gnu": "x86_64 Linux",
}


def _wheel_platform_labels(root: Path) -> set[str]:
    """The platform labels the Python prebuilt-wheel docs should list, derived from the build target
    triples in .github/workflows/package.yml (mapped via `_WHEEL_LABEL`). FAILS LOUD on an unknown
    triple so a new platform forces a docs + map update."""
    yml = (root / ".github/workflows/package.yml").read_text(encoding="utf-8")
    # Scope to the `prepare-python` wheel-build job's block (start-of-job to the next top-level job key),
    # so an inline `{...target:...}` table in an UNRELATED job can't leak into the wheel platform set.
    job = re.search(r"^  prepare-python:\n(.*?)(?=^  \w[\w-]*:\n|\Z)", yml, re.DOTALL | re.MULTILINE)
    if not job:
        raise LookupError("no `prepare-python` job in .github/workflows/package.yml; the wheel lock scope moved")
    # Within that job, match the wheel build MATRIX entries — `{ ..., target: <triple>, ... }` (inline brace
    # table); the JNI jobs use `- target: <triple>` (no braces) and are out of scope anyway. Match `target:`
    # ANYWHERE inside the braces (order-insensitive) so a reordered/new field can't hide a target. Capture
    # ANY triple token (not a fixed arch/os set) so a genuinely new platform reaches `_WHEEL_LABEL` + FAILS LOUD.
    triples = set(re.findall(r"\{[^{}]*\btarget:\s*([\w-]+)[^{}]*\}", job.group(1)))
    if not triples:
        raise LookupError("no wheel matrix target triples found in .github/workflows/package.yml")
    labels: set[str] = set()
    for t in triples:
        if t not in _WHEEL_LABEL:
            raise LookupError(f"unknown wheel target triple {t!r}; add it to _WHEEL_LABEL and the Python docs")
        labels.add(_WHEEL_LABEL[t])
    return labels


@dataclass
class WheelPlatformCheck:
    """Assert the Python docs' prebuilt-wheel platform list includes every platform the packaging
    workflow builds (`_wheel_platform_labels`) — so adding/removing a wheel target forces a docs update."""

    id: str
    description: str

    def check(self, root: Path) -> tuple[bool, str]:
        expected = _wheel_platform_labels(root)
        text = (root / "docs/api/python/index.rst").read_text(encoding="utf-8")
        # the bullet list under "available for:" — set equality, so a stale advertised platform (built
        # target removed) is caught too, not only a missing one.
        m = re.search(r"available for:\n\n((?:\* .*\n)+)", text)
        listed = set(re.findall(r"\* (.+)", m.group(1))) if m else set()
        if listed != expected:
            return False, (f"python/index.rst wheel platforms mismatch vs package.yml: "
                           f"extra={sorted(listed - expected)} missing={sorted(expected - listed)}")
        return True, f"python/index.rst lists exactly the {len(expected)} built wheel platforms"


@dataclass
class CanonicalConceptsTableCheck:
    """Assert concepts/arrays.md's DType->canonical-encoding table uses EXACTLY the real canonical
    encodings (the `Canonical` enum payloads) in its encoding column (set equality — caught the stale
    table that omitted DecimalArray/VariantArray)."""

    id: str
    description: str

    def check(self, root: Path) -> tuple[bool, str]:
        expected = _canonical_all_payloads(root)
        text = (root / "docs/concepts/arrays.md").read_text(encoding="utf-8")
        m = re.search(r"Canonical Encoding\s*\|\n\|[-\s|]+\n(.*?)(?:\n\n|\n##|\Z)", text, re.DOTALL)
        if not m:
            return False, "arrays.md has no DType->canonical-encoding table"
        listed = set(re.findall(r"`(\w+Array)`", m.group(1)))
        if listed != expected:
            return False, (f"arrays.md canonical-encoding column mismatch vs the Canonical enum: "
                           f"extra={sorted(listed - expected)} missing={sorted(expected - listed)}")
        return True, f"arrays.md canonical table uses exactly the {len(expected)} canonical encodings"


@dataclass
class EncodingsTableCheck:
    """Assert architecture.md's `## Encodings` table lists EXACTLY the `encodings/*` crates (set equality
    — caught a stale `vortex-roaring`/`vortex-dict` that no longer exist AND omitted vortex-pco/zstd/
    parquet-variant). The table is the `| Crate | Technique |` block under the `## Encodings` heading."""

    id: str
    description: str

    def check(self, root: Path) -> tuple[bool, str]:
        expected = _encoding_crate_names(root)
        text = (root / "docs/developer-guide/internals/architecture.md").read_text(encoding="utf-8")
        m = re.search(r"## Encodings\b(.*?)(?:\n## |\Z)", text, re.DOTALL)
        if not m:
            return False, "architecture.md has no `## Encodings` section"
        listed = set(re.findall(r"`(vortex-[\w-]+)`", m.group(1)))
        if listed != expected:
            return False, (f"architecture.md Encodings table mismatch vs encodings/*: "
                           f"extra={sorted(listed - expected)} missing={sorted(expected - listed)}")
        return True, f"architecture.md Encodings table lists exactly the {len(expected)} encoding crates"


@dataclass
class DtypesTableCheck:
    """Assert concepts/dtypes.md's Logical Types table lists EXACTLY the `DType` variants (set equality),
    so the user-facing dtype list can't drift from the Rust enum (it had omitted Union/Variant)."""

    id: str
    description: str

    def check(self, root: Path) -> tuple[bool, str]:
        expected = _dtype_variant_names(root)
        text = (root / "docs/concepts/dtypes.md").read_text(encoding="utf-8")
        # the MAIN dtype table only — bounded at the first sub/section heading (e.g. `### Primitive`),
        # so per-type subsection tables (PType I8/F64/...) aren't mistaken for DType variants.
        m = re.search(r"## Logical Types\b(.*?)(?:\n#{2,4} |\Z)", text, re.DOTALL)
        if not m:
            return False, "dtypes.md has no `## Logical Types` section"
        # the Name column: a leading `| `<Name>` |` cell with a capitalized identifier
        listed = set(re.findall(r"^\|\s*`([A-Z]\w*)`", m.group(1), re.MULTILINE))
        if listed != expected:
            return False, (f"dtypes.md Logical Types table mismatch vs the DType enum: "
                           f"extra={sorted(listed - expected)} missing={sorted(expected - listed)}")
        return True, f"dtypes.md Logical Types table lists exactly the {len(expected)} DType variants"


@dataclass
class PythonModuleCallCheck:
    """Flag a `vortex.<submodule>(...)` / `vx.<submodule>(...)` in any doc's PYTHON region — calling a
    MODULE as a function (the `vortex.dataset(...)` drift PR-2.2 fixed: `dataset` is a submodule, not a
    callable). The `python-api-names` membership check only proves the NAME exists; this catches the
    semantic module-as-callable misuse a static name check cannot."""

    id: str
    description: str

    def check(self, root: Path) -> tuple[bool, str]:
        mods = _vortex_submodules(root)
        pat = re.compile(r"(?<![\w.])(?:vortex|vx)\.(\w+)\s*\(")
        bad: dict[str, str] = {}
        for p in all_doc_files(root):
            for m in pat.finditer(_python_regions(p.read_text(encoding="utf-8"))):
                if m.group(1) in mods:
                    bad.setdefault(m.group(1), str(p.relative_to(root)))
        if bad:
            t = sorted(bad)[0]
            return False, f"`vortex.{t}(...)` calls a submodule as a function (not callable) in {bad[t]}"
        return True, f"no `vortex.<submodule>(...)` module-as-callable misuse ({len(mods)} submodules)"


DOC_CHECKS: list[DocMembershipCheck] = [
    DocMembershipCheck(
        id="architecture-crate-refs",
        description="every `vortex-<crate>` referenced in architecture.md is a real workspace crate",
        doc_files=["docs/developer-guide/internals/architecture.md"],
        # crate-form refs only: `vortex-array/src/expr` yields `vortex-array` (a real crate); a bare
        # `vortex-expr`/`vortex-roaring` (no longer / never a crate) is flagged. The published Java Spark
        # artifacts `vortex-spark_<scala>` aren't Cargo crates, so they're added to the allowed set.
        mention_regex=r"`(vortex-[a-z][\w.-]*)",
        allowed=lambda root: _workspace_crate_names(root) | {f"vortex-spark_{v}" for v in _spark_scala_variants(root)},
    ),
    DocMembershipCheck(
        id="spark-scala-variants",
        description="every `vortex-spark_<scala>` advertised in the docs/READMEs is a published scala variant",
        doc_files=["docs/user-guide/spark.md", "docs/developer-guide/internals/architecture.md",
                   "README.md", "java/README.md"],
        mention_regex=r"vortex-spark_(\d+(?:\.\d+)?)",
        allowed=_spark_scala_variants,
    ),
    DocMembershipCheck(
        id="python-api-names",
        description="every top-level `vortex.<name>` / `vx.<name>` in docs PYTHON blocks is a real "
                    "public name of the vortex package (catches a fabricated/renamed Python API)",
        doc_files=[],            # scans every doc
        scan_all_docs=True,
        region_fn=_python_regions,  # python code regions only — not prose/URLs/Maven/Spark-option text
        mention_regex=r"(?<![\w.])(?:vortex|vx)\.([A-Za-z_]\w*)",
        allowed=_vortex_public_api,
    ),
]


SECTION_CHECKS = [
    CanonicalSectionCheck(
        id="canonical-encodings-list",
        description="arrays.rst Canonical Encodings section == the Rust Canonical enum (Python classes)",
    ),
    StatsListCheck(
        id="statistics-list",
        description="arrays.md Statistics section == the Rust Stat::name() values",
    ),
    PythonModuleCallCheck(
        id="python-no-module-call",
        description="docs don't call a `vortex` submodule as a function (e.g. `vortex.dataset(...)`)",
    ),
    DTypeListCheck(
        id="architecture-dtype-list",
        description="architecture.md `DType` enum row lists every Rust DType variant",
    ),
    DtypesTableCheck(
        id="dtypes-logical-types",
        description="concepts/dtypes.md Logical Types table == the Rust DType variants",
    ),
    EncodingsTableCheck(
        id="architecture-encodings",
        description="architecture.md Encodings table == the encodings/* crates",
    ),
    CanonicalConceptsTableCheck(
        id="canonical-concepts-table",
        description="concepts/arrays.md DType->encoding table == the Rust Canonical enum payloads",
    ),
    WheelPlatformCheck(
        id="python-wheel-platforms",
        description="python/index.rst lists every wheel platform built by package.yml",
    ),
]


@dataclass
class EncodingStabilityCheck:
    """Validate the stable-encoding-set derivation (encoding_stability.py): it parses
    `register_default_encodings` + the crate `initialize()` fns and yields non-empty, pairwise-disjoint
    stable / unstable / parked encoding sets. This checks the DERIVATION's health only — it does NOT
    enforce that each stable encoding has a spec section (that is the tripwire, deliberately not built
    here). Fails loud if the registration source or the `unstable_encodings` feature moved."""

    id: str
    description: str

    def check(self, root: Path) -> tuple[bool, str]:
        cls = encoding_stability.classify_encodings(root)
        stable = {n for n, c in cls.items() if c == encoding_stability.STABLE}
        unstable = {n for n, c in cls.items() if c == encoding_stability.UNSTABLE}
        parked = {n for n, c in cls.items() if c == encoding_stability.PARKED}
        if not stable:
            return False, "derived an empty stable-encoding set"
        if stable & unstable or stable & parked or unstable & parked:
            return False, "stable/unstable/parked encoding sets are not disjoint"
        return True, (f"{len(stable)} stable / {len(unstable)} unstable / {len(parked)} parked encodings "
                      f"derived (unstable={sorted(unstable)})")


ENCODING_CHECKS: list[EncodingStabilityCheck] = [
    EncodingStabilityCheck(
        id="encoding-stability-set",
        description="the stable-encoding-set derivation parses and yields disjoint, non-empty sets",
    ),
]


# --- Spec-conformance tripwire (shadow / observe mode) ----------------------------------------------
# A guardrail that keeps docs/specification/encoding-format.md honest against the code. It runs in
# SHADOW mode: it REPORTS coverage gaps and would-be drift, NON-BLOCKING, so CI stays green while
# per-encoding byte-layout sections are added incrementally. It authors no spec content and never
# contributes to the exit code; flipping it to hard enforcement is a later, deliberate step.
#
# Two things it observes:
#   1. COVERAGE — every STABLE encoding (encoding_stability.stable_encodings, code-derived) should
#      gain a per-encoding byte-layout section. All 33 stable encodings now do (33/33); validity is
#      cross-cutting and specified once, NOT as a per-encoding section. The report keeps sizing this so
#      a newly-added stable encoding without a section surfaces immediately.
#   2. DRIFT — each per-encoding LOCK (registered in ENCODING_LAYOUT_LOCKS with its section in a later
#      task) derives its invariant from encodings/*/src and compares it to the value the spec section
#      pins; a divergence is reported as WOULD-BE drift — exactly what hard enforcement would reject.
#
# --- Per-encoding byte-layout SECTION CONVENTION (defined here; a later task authors the sections) ---
# A stable encoding `<Name>` (named EXACTLY as encoding_stability.stable_encodings() yields it — the
# Rust struct ident, e.g. `ALP`, `BitPacked`, `FoR`, `DecimalByteParts`) is COVERED iff
# encoding-format.md contains a MyST target-label line of the exact form
#
#     (encoding-layout-<Name>)=
#
# on its own line, immediately preceding that encoding's byte-layout heading. Example:
#
#     (encoding-layout-ALP)=
#     ### `vortex.alp` — Byte layout
#     ...
#
# Why the anchor keys on the STABLE-SET NAME rather than the wire ID (`vortex.alp`, `fastlanes.for`):
#   * The stable-set name is EXACTLY the identity derived from code, so both sides stay
#     code-authoritative with zero name-mapping — coverage is a pure set difference.
#   * It sidesteps wire-ID ambiguity: the encoding IDs mix prefixes (`vortex.*` AND `fastlanes.*` for
#     the FastLanes family), so keying on wire IDs would need a fragile per-crate derivation.
#   * The invisible anchor decouples the machine key from the human-facing heading text (which should
#     still show the wire ID for readers), so coverage detection NEVER collides with the cross-cutting
#     Validity section — that section names encodings in prose and sub-headings but carries no
#     `encoding-layout-*` anchor, so it correctly counts as zero coverage today.
LAYOUT_ANCHOR_RE = re.compile(r"^\(encoding-layout-([A-Za-z0-9]+)\)=\s*$", re.MULTILINE)
ENCODING_FORMAT_DOC = "docs/specification/encoding-format.md"
# The per-encoding byte-layout sections live on FAMILY pages under this directory (canonical.md,
# containers.md, …), linked from ENCODING_FORMAT_DOC's toctree. Coverage is scanned across BOTH the
# top-level page and every family page, so a section authored on any of them counts.
ENCODING_FORMAT_FAMILY_DIR = "docs/specification/encoding-format"


def layout_anchor_label(name: str) -> str:
    """The MyST target label a byte-layout section for stable encoding `name` must carry (the section
    convention above). One function so the parser and any future authoring tool cannot disagree."""
    return f"encoding-layout-{name}"


def parse_layout_anchors(text: str) -> set[str]:
    """The set of encoding names `text` marks as having a byte-layout section, per the anchor
    convention. Fenced code blocks are stripped first so an anchor shown inside an EXAMPLE block is not
    miscounted as real coverage. Pure (no I/O) so the self-test can drive it on synthetic input."""
    return set(LAYOUT_ANCHOR_RE.findall(_strip_fenced_blocks(text)))


def _covered_from_texts(texts: Iterable[str]) -> set[str]:
    """Union of byte-layout anchors across multiple doc texts — the aggregation `covered_encodings`
    performs over the top-level page + family pages. Pure (no I/O) so the self-test can drive the
    multi-page scan surface on synthetic input."""
    covered: set[str] = set()
    for text in texts:
        covered |= parse_layout_anchors(text)
    return covered


def _encoding_format_docs(root: Path) -> list[Path]:
    """Every doc scanned for byte-layout anchors: the top-level encoding-format.md plus every family
    page under docs/specification/encoding-format/*.md. A missing file/dir simply contributes nothing
    (the spec pages may not be authored yet); shadow-only, never fatal."""
    docs: list[Path] = []
    top = root / ENCODING_FORMAT_DOC
    if top.exists():
        docs.append(top)
    family = root / ENCODING_FORMAT_FAMILY_DIR
    if family.is_dir():
        docs.extend(sorted(family.glob("*.md")))
    return docs


def covered_encodings(root: Path) -> set[str]:
    """Encodings whose byte-layout section is present anywhere in the encoding-format spec — the
    top-level page or any family page (empty today). Shadow-only, never fatal."""
    return _covered_from_texts(p.read_text(encoding="utf-8") for p in _encoding_format_docs(root))


def coverage_gaps(stable: set[str], covered: set[str]) -> list[str]:
    """Stable encodings still lacking a byte-layout section — the coverage gap the shadow report sizes."""
    return sorted(stable - covered)


@dataclass
class EncodingLayoutLock:
    """EXTENSION POINT — a per-encoding byte-layout invariant, registered in ENCODING_LAYOUT_LOCKS by a
    later authoring task alongside that encoding's spec section. Both sides are DERIVED, never
    hard-coded: `derive_code` computes the invariant from encodings/*/src (the source of truth) and
    `derive_spec` reads the value the byte-layout section pins (typically from within that encoding's
    section, located via the anchor convention). WOULD-BE drift = the two disagree — exactly what hard
    enforcement will reject once the shadow period ends. In SHADOW mode the divergence is only reported.

    `check(root, override_code=...)` mirrors ValueMatch's self-test hook: overriding the code side with
    a sentinel proves the lock detects drift (the derived value no longer matches the spec side)."""

    encoding: str                       # a member of encoding_stability.stable_encodings()
    description: str
    derive_code: Callable[[Path], str]  # invariant derived from source (authoritative)
    derive_spec: Callable[[Path], str]  # the value the spec section pins (the doc side)

    def check(self, root: Path, *, override_code: str | None = None) -> tuple[bool, str]:
        """Return (drifted, detail). `drifted` is True when the code-derived invariant disagrees with
        the value the spec pins. A raised LookupError/OSError is the caller's to catch — in shadow mode
        the caller downgrades it to a non-fatal note."""
        code = override_code if override_code is not None else self.derive_code(root)
        spec = self.derive_spec(root)
        if code != spec:
            return True, f"code derives {code!r} but spec pins {spec!r}"
        return False, f"code and spec agree ({code!r})"


# Empty today: the per-encoding locks land WITH their spec sections in later tasks. Registering a
# lock here makes the tripwire hard-check that encoding's invariant (still shadow-reported until the
# global flip to enforcement). See EncodingLayoutLock for the contract each entry must satisfy.
ENCODING_LAYOUT_LOCKS: list[EncodingLayoutLock] = []


def shadow_report(root: Path) -> None:
    """Print the SHADOW-mode spec-conformance report (coverage + would-be drift), NON-BLOCKING. Never
    raises and never affects the exit code: a derivation error is downgraded to a note here because the
    blocking EncodingStabilityCheck already guards the derivation's health."""
    print("\n--- spec-conformance tripwire (SHADOW / observe mode — non-blocking, REPORTS only) ---")
    try:
        stable = encoding_stability.stable_encodings(root)
    except (LookupError, OSError) as e:
        print(f"  shadow: could not derive the stable set ({type(e).__name__}: {e}); "
              "the blocking encoding-stability check reports the cause.")
        return
    covered = covered_encodings(root)
    gaps = coverage_gaps(stable, covered)
    print(f"  stable must-spec set ({len(stable)}): {', '.join(sorted(stable))}")
    print(f"  byte-layout coverage: {len(covered)}/{len(stable)} covered "
          f"(convention: an `(encoding-layout-<Name>)=` anchor in {ENCODING_FORMAT_DOC} "
          f"or {ENCODING_FORMAT_FAMILY_DIR}/*.md)")
    for name in sorted(stable):
        print(f"    [{'COVERED' if name in covered else 'GAP    '}] {name}")
    stray = sorted(covered - stable)
    if stray:  # an anchor for a name NOT in the stable set (typo'd / stale) — surfaced, not a gap
        print(f"  NOTE: byte-layout anchors for non-stable names (typo/stale?): {', '.join(stray)}")
    try:  # echo the feature-gated-stable divergence so a reviewer isn't surprised Zstd is must-spec
        fg = sorted(encoding_stability.feature_gated_stable(root))
    except (LookupError, OSError):
        fg = []
    if fg:
        print(f"  NOTE: {', '.join(fg)} in the must-spec set via a non-unstable feature gate "
              "(feature-gate divergence; see encoding_stability.py).")
    drifts = 0
    print(f"  per-encoding locks registered: {len(ENCODING_LAYOUT_LOCKS)}")
    for lock in ENCODING_LAYOUT_LOCKS:
        try:
            drifted, detail = lock.check(root)
        except (LookupError, OSError) as e:
            print(f"    [LOCK-ERR]     {lock.encoding}: {type(e).__name__}: {e}")
            continue
        if drifted:
            drifts += 1
        print(f"    [{'WOULD-DRIFT' if drifted else 'ok         '}] {lock.encoding}: {detail}")
    print(f"  SHADOW SUMMARY: {len(covered)}/{len(stable)} byte-layout sections, {len(gaps)} coverage "
          f"gap(s), {drifts} would-be drift(s) — REPORTED, not enforced (CI stays green).")


def _all_checks() -> list:
    """Every check, regardless of type — all share `.id` and `.check(root) -> (ok, detail)`."""
    return [*REGISTRY, *CLI_CHECKS, *DOC_CHECKS, *SECTION_CHECKS, *ENCODING_CHECKS]


def run_checks(root: Path, verbose: bool) -> int:
    failures: list[str] = []
    checks = _all_checks()
    for chk in checks:
        try:
            ok, detail = chk.check(root)
        except (LookupError, OSError) as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        if not ok:
            failures.append(f"  [{chk.id}] {detail}")
        if verbose or not ok:
            print(f"{'PASS' if ok else 'FAIL'} {chk.id}: {detail}")
    # SHADOW/observe tripwire — always printed, strictly NON-BLOCKING: it reports spec coverage +
    # would-be drift but never touches `failures`, so the exit code is unchanged by anything it finds.
    shadow_report(root)
    if failures:
        print(f"\n{len(failures)} doc-conformance check(s) FAILED:", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        print(
            "\nDocs drifted from their source of truth. Fix the prose (or, better, source it via "
            "literalinclude), or update the registry if the source of truth legitimately moved.",
            file=sys.stderr,
        )
        return 1
    print(f"\nOK — all {len(checks)} doc-conformance checks passed.")
    return 0


def self_test(root: Path) -> int:
    """Negative tests that prove the checker's logic, independent of the live registry. Each block is
    labelled inline; broadly they cover: (1) token_present boundary/overlap/punctuation handling;
    (2) dotted-continuation disambiguation; (3) command-token capture, flag-skip, extras, stem-scoping;
    (4) every ValueMatch entry detects drift when its truth is replaced by a sentinel; (5) the
    CliSubcommandCheck — clap kebab/name/alias/rename_all parsing, MyST/RST shell-region extraction
    (incl. nested fences, code directives, python/doctest opacity), nested command-path validation,
    multi-vx-per-line, binary-name sourcing, and the optional-subcommand-leniency tradeoff. Finally it
    confirms the live registry passes (a check that can never pass is useless). Keep the labels current
    when adding cases; the count of cases is intentionally not asserted (cases grow with the registry)."""
    failures: list[str] = []

    # 1. overlap must NOT match (regression guard for substring matching)
    if token_present("65527", "padded to 655270 bytes"):
        failures.append("token_present matched an overlapping value (substring regression)")
    if token_present("vortex", "pip install vortex-data"):
        failures.append("token_present matched crate name inside a longer token")
    # 2. sentence-end punctuation must still match
    if not token_present("65527", "the bound is 65527."):
        failures.append("token_present false-negative on sentence-end punctuation")
    if not token_present("read_vortex", "call read_vortex, then scan"):
        failures.append("token_present false-negative on comma-delimited token")
    # 2b. dotted continuation (version / sub-token) must NOT match on EITHER side, but sentence-end
    #     '.' must
    if token_present("65527", "version 65527.0 of the format"):
        failures.append("token_present matched a trailing dotted continuation (65527 in 65527.0)")
    if token_present("65527", "release 1.65527 of the format"):
        failures.append("token_present matched a leading dotted continuation (65527 in 1.65527)")
    if not token_present("65527", "the bound is 65527. Next paragraph"):
        failures.append("token_present false-negative on sentence-end dot before a word")

    # 3. command-scan logic (capture + consistency) — exercised directly, because the
    #    override-sentinel path in (4) below short-circuits on the presence check before reaching
    #    the global command scan, so it would otherwise never cover this code path.
    captured = command_tokens("pip install ", "`pip install pkg`. then pip install pkg[polars,ray]")
    if captured != ["pkg", "pkg[polars,ray]"]:
        failures.append(f"command_tokens captured {captured!r}, expected ['pkg', 'pkg[polars,ray]']")
    # a dotted (malformed) token is captured WHOLE so it can be rejected; a sentence-end '.' is never
    # absorbed into the token (it is simply not captured as a command, which presence-checking covers)
    if command_tokens("pip install ", "run pip install pkg.old here") != ["pkg.old"]:
        failures.append("command_tokens did not capture a dotted token whole (pkg.old)")
    if "pkg." in command_tokens("pip install ", "run pip install pkg. Then more"):
        failures.append("command_tokens absorbed a sentence-end dot")
    # post-extras junk must not be captured as a clean token (it would otherwise prefix-match)
    if command_tokens("pip install ", "pip install pkg[extra]wrong here"):
        failures.append("command_tokens captured a token with post-extras junk")
    # leading flags are skipped so the package (not the flag) is captured
    if command_tokens("pip install ", "pip install --upgrade -U vortex-typo here") != ["vortex-typo"]:
        failures.append("command_tokens did not skip leading flags to reach the package token")
    if not command_token_ok("pkg", "pkg", allow_extras=False):
        failures.append("command_token_ok rejected an exact match")
    if not command_token_ok("pkg", "pkg[polars,ray]", allow_extras=True):
        failures.append("command_token_ok rejected the [extras] form when extras are allowed")
    if command_token_ok("pkg", "pkg[bogus]", allow_extras=False):
        failures.append("command_token_ok accepted [extras] for a non-extras ecosystem (cargo)")
    if command_token_ok("pkg", "pkg-wrong", allow_extras=True):
        failures.append("command_token_ok accepted a stale token")
    if command_token_ok("pkg", "pkg.old", allow_extras=True):
        failures.append("command_token_ok accepted a dotted (malformed) token")
    if command_token_ok("pkg", "pkg[extra]wrong", allow_extras=True):
        failures.append("command_token_ok accepted a token with post-extras junk")

    # 4. each ValueMatch check detects drift in its canonical value
    for chk in REGISTRY:
        ok, _ = chk.check(root, override_value="__CONFORMANCE_DRIFT_SENTINEL_DOES_NOT_EXIST__")
        if ok:
            failures.append(f"[{chk.id}] a drifted value was NOT caught")
        else:
            print(f"SELF-TEST OK {chk.id}: drift would be caught")

    # 4b. source-reader hardening: comment stripping (a commented decoy is never sourced), the
    #     derive/transform mode guard, and Cargo `[[bin]]` key-order tolerance.
    if _strip_comments("a // line\nb /* block */ c").split() != ["a", "b", "c"]:
        failures.append("_strip_comments did not remove // and /* */ comments")
    if _read_const("pub const N: usize = 8;\n// pub const N: usize = 99;\n",
                   r"^\s*pub const N: usize = (\d+)\s*;", "N") != 8:
        failures.append("comment-decoy: _read_const sourced a commented-out value")
    try:
        ValueMatch(id="bad", description="", doc_files=[], derive=lambda r: "x", transform=lambda s: s)
        failures.append("ValueMatch accepted derive + transform together")
    except ValueError:
        pass
    # Cargo `[[bin]]` name sourcing (tomllib) is robust to a `#`-commented decoy, key reordering, and
    # a list-valued key before `name` — a raw regex over the text would mishandle these.
    synth_cargo = ('[package]\nname = "vortex-tui"\n\n[[bin]]\n# name = "old"\npath = "src/main.rs"\n'
                   'required-features = ["native"]\nname = "vx"\n')
    if _toml_str_from(synth_cargo, "synth", "bin", 0, "name") != "vx":
        failures.append("_toml_str_from mishandles [[bin]] name with comments/key-order/list-values")
    if _toml_str_from(synth_cargo, "synth", "package", "name") != "vortex-tui":
        failures.append("_toml_str_from mishandles [package] name (table-order independence)")
    # forbid_regex absence check: a fact whose claim is PRESENT still FAILS when a forbidden (stale)
    # pattern is also present, so a corrected site cannot mask a stale sibling. Exercised on a real doc.
    spark = "docs/user-guide/spark.md"
    if (root / spark).exists():
        present_forbidden = ValueMatch(id="fb1", description="", doc_files=[spark],
                                       derive=lambda r: "Vortex", forbid_regex=r"DataSource V2")
        ok_fb, detail_fb = present_forbidden.check(root)
        if ok_fb or "forbidden" not in detail_fb:  # must fail, AND specifically for the forbidden reason
            failures.append(f"forbid_regex failure not attributed to the forbidden pattern ({detail_fb!r})")
        absent_forbidden = ValueMatch(id="fb2", description="", doc_files=[spark],
                                      derive=lambda r: "Vortex", forbid_regex=r"__never_appears_xyzzy__")
        if not absent_forbidden.check(root)[0]:
            failures.append("forbid_regex spuriously failed when the forbidden pattern is absent")
        # the SPECIFIC spark stale-coordinate forbid catches the bare prose/XML forms, ignores suffixed
        spark_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "spark-maven-artifact")
        for bare in ("published as `dev.vortex:vortex-spark`.", "<artifactId>vortex-spark</artifactId>"):
            if not re.search(spark_forbid, bare):
                failures.append(f"spark forbid_regex missed a bare coordinate form: {bare!r}")
        for suffixed in ("dev.vortex:vortex-spark_2.13:1.0", "<artifactId>vortex-spark_2.13</artifactId>"):
            if re.search(spark_forbid, suffixed):
                failures.append(f"spark forbid_regex false-matched a suffixed coordinate: {suffixed!r}")
        # DocMembershipCheck logic on SYNTHETIC input (no hard-coded live variant set): a mention
        # outside the allowed set is flagged; one inside it is not.
        outside = DocMembershipCheck._outside("use vortex-spark_2.13 then vortex-spark_2.11",
                                              r"vortex-spark_(\d+(?:\.\d+)?)", {"2.12", "2.13"})
        if outside != {"2.11"}:
            failures.append(f"DocMembershipCheck._outside wrong (got {outside!r}; expected {{'2.11'}})")
        # and the live derive yields a non-empty published set the live check binds to
        if not _spark_scala_variants(root):
            failures.append("_spark_scala_variants derived an empty set from settings.gradle.kts")

    # 4b-ii. PR-2.3 forbid/scoped-claim checks — synthetic-string tests over REGISTRY forbid regexes;
    #        NOT under the Spark-doc guard (they don't read spark.md), so Spark-only churn can't skip them.
    buf_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "buffered-block-size")
    for stale in ("localize up to 1 MB", "the 8k row zones and 2MB chunks balance",
                  "Chunked Layout to partition the column into 2MB of uncompressed data"):
        if not re.search(buf_forbid, stale):
            failures.append(f"buffered forbid_regex missed a stale form: {stale!r}")
    if re.search(buf_forbid, "2 MB of buffered chunk locality"):
        failures.append("buffered forbid_regex false-matched the corrected buffered-locality wording")
    io_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "io-coalesce-file")
    for stale in ("Local files use 8 KB for both", "| Local file | {{x}} | 8 KB | 8 KB |"):
        if not re.search(io_forbid, stale):
            failures.append(f"io-coalesce forbid_regex missed a stale form: {stale!r}")
    for ok in ("In-memory buffers use an 8 KB distance", "| Local file | {{x}} | 1 MB | 4 MB |"):
        if re.search(io_forbid, ok):
            failures.append(f"io-coalesce forbid_regex false-matched a correct line: {ok!r}")
    # scoped claims: the derived value must appear IN ITS CONTEXT, not just anywhere in the file.
    if token_present("1 MB distance and 4 MB max size", "local: 5 MB max size. (4 MB appears here)"):
        failures.append("io-coalesce claim matched out of context (unscoped bare token)")
    if not token_present("localize up to 2 MB", "Buffered Layout to localize up to 2 MB of chunks"):
        failures.append("buffered scoped claim false-negative on the correct sentence")
    if token_present("localize up to 2 MB", "Buffered Layout to localize up to 3 MB; a 2 MB zone"):
        failures.append("buffered claim matched a bare 2 MB elsewhere, not the scoped phrase")
    # scanning-api-traits forbid is case-insensitive for Sink AND catches a backticked `Source` trait
    # (the real trait is `DataSource`), but not the corrected `DataSource` mention.
    scan_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "scanning-api-traits")
    for stale in ("An equivalent `Sink` trait exists", "an equivalent sink interface exists",
                  "The core `Source` trait and scan pipeline"):
        if not re.search(scan_forbid, stale):
            failures.append(f"scanning-api-traits forbid missed a stale form: {stale!r}")
    if re.search(scan_forbid, "The core `DataSource` trait and scan pipeline"):
        failures.append("scanning-api-traits forbid false-matched the corrected `DataSource` wording")
    # cpp-binding-cxx forbid blocks the stale current-state "wrapper around the C FFI" framing, but NOT
    # the legitimate future-plan wording ("wrapping the C API"); and _cxx_dep derives `cxx` live.
    cxx_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "cpp-binding-cxx")
    for stale in ("a C++ wrapper around the C FFI", "wrapper around the Vortex C FFI",
                  "the C++ API wraps the C FFI", "a thin wrapper for the C FFI"):
        if not re.search(cxx_forbid, stale):
            failures.append(f"cpp-binding-cxx forbid missed a stale form: {stale!r}")
    if re.search(cxx_forbid, "The plan is to migrate to wrapping the C API directly"):
        failures.append("cpp-binding-cxx forbid false-matched the legitimate future-plan (C API) wording")
    # the "wrapper ... C FFI" forbid is sentence-scoped but line-wrap-tolerant — a stale claim split across
    # lines is still caught; a later-sentence "C FFI" mention is not.
    if not re.search(cxx_forbid, "a C++ wrapper around the\nVortex C FFI"):
        failures.append("cpp-binding-cxx forbid missed a 'wrapper ... C FFI' claim split across lines")
    if re.search(cxx_forbid, "It wraps the Rust core. A separate C FFI is also offered."):
        failures.append("cpp-binding-cxx forbid false-matched a C FFI mention in a LATER sentence")
    if _cxx_dep(root) != "cxx":
        failures.append("_cxx_dep did not confirm the live cxx dependency + bridge")
    # the bridge check strips comments first, so a commented-out `#[cxx::bridge]` decoy is NOT live
    if re.search(r"#\[cxx::bridge", _strip_rust_comments("// #[cxx::bridge] mod ffi { }\nfn x() {}")):
        failures.append("_cxx_dep bridge search would accept a line-commented `#[cxx::bridge]` decoy")
    # the bridge check also anchors to ITEM position, so a string literal containing the attribute can't
    # satisfy it after the real bridge is removed (the `"` breaks the `^[ \t]*#\[` anchor).
    if re.search(r"^[ \t]*#\[cxx::bridge", 'const DOC: &str = "#[cxx::bridge]";\nfn x() {}', re.MULTILINE):
        failures.append("_cxx_dep item-position anchor accepted a string-literal `#[cxx::bridge]` decoy")
    if not re.search(r"^[ \t]*#\[cxx::bridge", "    #[cxx::bridge]\n    mod ffi {}", re.MULTILINE):
        failures.append("_cxx_dep item-position anchor missed a real indented `#[cxx::bridge]`")
    # the duckdb replacement-scan lock is scoped to the `initialize_extension_from_raw` entrypoint body — a
    # call in a helper/test fn OUTSIDE it does NOT satisfy the lock; a call INSIDE it does.
    dd_outside = _strip_rust_comments('fn helper() { db.register_vortex_scan_replacement(); }\n'
                                      'pub unsafe fn initialize_extension_from_raw(db: X) {\n    init_tracing();\n}')
    dd_im = re.search(r"fn initialize_extension_from_raw\([^)]*\)\s*\{(.*?)\n\}", dd_outside, re.DOTALL)
    if dd_im and re.search(r"\.register_vortex_scan_replacement\s*\(\s*\)", dd_im.group(1)):
        failures.append("duckdb replacement-scan lock satisfied by a call OUTSIDE initialize_extension_from_raw")
    dd_inside = _strip_rust_comments('pub unsafe fn initialize_extension_from_raw(db: X) {\n'
                                     '    db.register_vortex_scan_replacement();\n}')
    dd_im2 = re.search(r"fn initialize_extension_from_raw\([^)]*\)\s*\{(.*?)\n\}", dd_inside, re.DOTALL)
    if not (dd_im2 and re.search(r"\.register_vortex_scan_replacement\s*\(\s*\)", dd_im2.group(1))):
        failures.append("duckdb replacement-scan lock missed a real call inside initialize_extension_from_raw")
    # jni-module-name: live derive returns the real module, and the forbid catches the renamed-away name.
    if _jni_module_name(root) != "vortex-jni":
        failures.append(f"_jni_module_name wrong (got {_jni_module_name(root)!r})")
    jni_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "jni-module-name")
    if not re.search(jni_forbid, "the vortex-java JAR contains the JNI code"):
        failures.append("jni-module-name forbid missed the stale `vortex-java` name")
    if re.search(jni_forbid, "the vortex-jni JAR contains the JNI code"):
        failures.append("jni-module-name forbid false-matched the corrected `vortex-jni` name")
    # the scan-api / variant-dtype roadmap locks derive present-state facts from code; live values + a
    # synthetic DType-without-Variant negative for the pure presence helper.
    if _scan_crate_name(root) != "vortex-scan" or _variant_dtype_present(root) != "Variant":
        failures.append("scan-api/variant-dtype derive did not confirm the live code facts")
    # convert chunk-size lock: derives BATCH_SIZE; the scoped "<N>-row" claim + the row-group-boundaries
    # forbid catch the stale quickstart wording.
    if _convert_batch_size(root) != "8192":
        failures.append(f"_convert_batch_size wrong (got {_convert_batch_size(root)!r})")
    conv_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "cli-convert-chunk-size")
    for stale in ("chunking on Parquet row-group boundaries", "Chunking occurs on Parquet RowGroup boundaries"):
        if not re.search(conv_forbid, stale):
            failures.append(f"cli-convert-chunk-size forbid missed a stale form: {stale!r}")
    if re.search(conv_forbid, "chunking into 8192-row batches"):
        failures.append("cli-convert-chunk-size forbid false-matched the corrected wording")
    # the claim is scoped to "<N>-row batches" — a bare unrelated "8192-row" mention does not satisfy it
    if token_present("8192-row batches", "the file has 8192-row groups and other 8192-row data"):
        failures.append("cli-convert-chunk-size claim matched a bare 8192-row mention, not the scoped phrase")
    # the scan lock requires the `DataSource` trait specifically — `DataSourceOpener`/`-Scan` (word
    # boundary) must NOT satisfy it, but `pub trait DataSource:` must.
    if re.search(r"pub trait DataSource\b", "pub trait DataSourceOpener: 'static {}"):
        failures.append("scan-api `DataSource` trait check satisfied by DataSourceOpener (missing \\b)")
    if not re.search(r"pub trait DataSource\b", "pub trait DataSource: 'static + Send {}"):
        failures.append("scan-api `DataSource` trait check missed a real `pub trait DataSource`")
    if _dtype_has_variant("Null,\n    Bool(Nullability),\n    Primitive(PType, Nullability),"):
        failures.append("_dtype_has_variant false-positive on an enum body without a Variant variant")
    if not _dtype_has_variant("Extension(ExtDTypeRef),\n    Variant(Nullability),"):
        failures.append("_dtype_has_variant missed a present Variant variant")
    # _variant_dtype_present strips NESTED Rust comments, so a Variant hidden in one isn't "present"
    if _dtype_has_variant("Null,\n    /* a /* Variant(Nullability), */ b */ Bool,"):
        # (sanity: the nested-comment stripping happens before _dtype_has_variant in _variant_dtype_present;
        #  here we assert the stripper removes it)
        pass
    if re.search(r"^\s*Variant\b",
                 _strip_rust_comments("Null,\n    /* x /* Variant(Nullability), */ y */ Bool,"), re.MULTILINE):
        failures.append("_variant_dtype_present would detect a Variant hidden in a nested comment")
    # CONSTANT/FUNCTION locks now route Rust sources through _strip_rust_comments, so a decoy constructor
    # hidden in a NESTED `/* /* */ */` block comment can't inflate the "exactly one" match count past the
    # real one (the byte-pair / block-size / stat-name locks all rely on this).
    nested_decoy = ("fn in_memory() -> Self { Self::new(8192, 8192) }\n"
                    "    /* outer /* fn in_memory() -> Self { Self::new(1, 2) } */ inner */")
    if len(re.findall(r"fn in_memory\(\)\s*->\s*Self\s*\{\s*Self::new\(",
                      _strip_rust_comments(nested_decoy))) != 1:
        failures.append("_strip_rust_comments left a nested-commented constructor decoy for a byte-pair lock")
    # canonical-union-exception: live facts hold, and the guard trips iff the Canonical enum gains a Union
    # variant / UnionArray payload (forcing the "except Union" prose to update).
    if _union_is_sole_noncanonical(root) != "Union":
        failures.append("_union_is_sole_noncanonical did not confirm the live DType/Canonical facts")
    for canonicalized in ("Null(NullArray),\n    Union(UnionArray),", "Bool(BoolArray),\n    Union(SomeArray),"):
        if not (re.search(r"^\s*Union\b", canonicalized, re.MULTILINE) or "UnionArray" in canonicalized):
            failures.append("canonical-union-exception guard would miss a canonicalized Union")
    if re.search(r"^\s*Union\b", "Null(NullArray),\n    Bool(BoolArray),", re.MULTILINE):
        failures.append("canonical-union-exception guard false-tripped on a Canonical enum without Union")
    # scoped roadmap claims reject FUTURE-only wording even though it contains the derived token.
    if token_present("Scan API (the `vortex-scan` crate) already provides",
                     "The Scan API (the `vortex-scan` crate) will provide pluggable sources."):
        failures.append("scan-api-present claim matched a future-only sentence")
    if token_present("`Variant` DType already exists",
                     "The `Variant` DType is planned for a future release."):
        failures.append("variant-dtype-present claim matched a planned-only sentence")
    # cpp-not-wrapper-nav forbid catches the stale "C++ wrapper" framing and the "C/C++ (FFI)" label.
    nav_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "cpp-not-wrapper-nav")
    for stale in ("embed it through the C FFI, C++ wrapper, or", "Java (JNI) and C/C++ (FFI) bindings",
                  "C++ (FFI) bindings", "C and C++ FFI bindings"):
        if not re.search(nav_forbid, stale):
            failures.append(f"cpp-not-wrapper-nav forbid missed a stale form: {stale!r}")
    for ok in ("the C++ binding (a direct cxx Rust bridge)", "Java (JNI), C (FFI), and C++ (cxx)"):
        if re.search(nav_forbid, ok):  # corrected wording (incl. C++ (cxx)) must NOT match
            failures.append(f"cpp-not-wrapper-nav forbid false-matched corrected wording: {ok!r}")
    # the "foundation for ... C++" forbid is SENTENCE-scoped but tolerates line WRAPS: a stale present-tense
    # claim broken across lines is still caught, while a legitimate later-SENTENCE C++ mention is not.
    if not re.search(nav_forbid, "intended foundation for\nother language bindings (including C++)"):
        failures.append("cpp-not-wrapper-nav forbid missed a 'foundation for ... C++' claim split across lines")
    if re.search(nav_forbid, "the intended foundation for other language bindings. The C++ binding uses cxx."):
        failures.append("cpp-not-wrapper-nav forbid false-matched a foundation claim with C++ in a LATER sentence")
    # wheel target extraction is ORDER-INSENSITIVE within the brace table (a reordered field can't silently
    # drop a platform); the JNI list form (no braces) and `${{ matrix.target.target }}` are NOT captured.
    wheel_re = r"\{[^{}]*\btarget:\s*([\w-]+)[^{}]*\}"
    if "x86_64-unknown-linux-gnu" not in set(re.findall(
            wheel_re, '- { target: x86_64-unknown-linux-gnu, os: ubuntu, runs-on: "ubuntu-latest" }')):
        failures.append("wheel target extraction missed a target that is not the last field in the brace table")
    if set(re.findall(wheel_re, "- target: aarch64-unknown-linux-gnu\n  name: ${{ matrix.target.target }}.zip")):
        failures.append("wheel target extraction wrongly captured a JNI list-form / GitHub-expression target")
    # wheel extraction is JOB-SCOPED to `prepare-python`: an inline `{...target:...}` in an UNRELATED job
    # is NOT captured (only the wheel job's matrix triples reach _WHEEL_LABEL).
    synthetic_yml = ('  prepare-python:\n    strategy:\n      matrix:\n        target:\n'
                     '          - { os: ubuntu, target: x86_64-unknown-linux-gnu }\n'
                     '  other-job:\n    steps:\n      - run: build { target: not-a-wheel-triple }\n')
    job_m = re.search(r"^  prepare-python:\n(.*?)(?=^  \w[\w-]*:\n|\Z)", synthetic_yml, re.DOTALL | re.MULTILINE)
    scoped = set(re.findall(wheel_re, job_m.group(1))) if job_m else set()
    if scoped != {"x86_64-unknown-linux-gnu"}:
        failures.append(f"wheel job-scoping wrong (got {sorted(scoped)}; expected only the prepare-python triple)")
    # io-reader locks: the named structs `impl VortexReadAt` live; a file without the impl fails loud.
    if _io_reader_impl(root, "FileReadAt", "vortex-io/src/std_file/read_at.rs") != "FileReadAt":
        failures.append("_io_reader_impl did not confirm FileReadAt impl VortexReadAt")
    if _io_reader_impl(root, "ObjectStoreReadAt", "vortex-io/src/object_store/read_at.rs") != "ObjectStoreReadAt":
        failures.append("_io_reader_impl did not confirm ObjectStoreReadAt impl VortexReadAt")
    if re.search(r"\bimpl VortexReadAt for FileReadAt\b", "impl VortexReadAt for SomethingElse {}"):
        failures.append("io-reader impl check satisfied by an unrelated VortexReadAt impl")
    # spark-version locks: the live scala->spark when-arms map 2.13->Spark 4.x and 2.12->Spark 3.x.
    if _spark_major_for_scala(root, "2.13") != "Spark 4.x" or _spark_major_for_scala(root, "2.12") != "Spark 3.x":
        failures.append("_spark_major_for_scala did not confirm the live scala->spark mapping (2.13=4.x, 2.12=3.x)")
    # _strip_rust_comments removes NESTED block comments, so a marker hidden in one is not "live"
    if re.search(r"#\[cxx::bridge", _strip_rust_comments("/* a /* b */ #[cxx::bridge] */ fn x(){}")):
        failures.append("_strip_rust_comments did not strip a nested-block-comment #[cxx::bridge] decoy")
    if "#[cxx::bridge]" not in _strip_rust_comments("#[cxx::bridge]\nmod ffi {}"):
        failures.append("_strip_rust_comments wrongly removed a live #[cxx::bridge]")
    # no-trino-integration-claim forbid blocks in-progress framing AND the stale "Spark and Trino"
    # current-pairing, but not factual Trino-project context or future-connector wording.
    trino_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "no-trino-integration-claim")
    for stale in ("Trino support is in progress", "Spark (with Trino in progress)",
                  "the integration point for Spark and Trino connectors"):
        if not re.search(trino_forbid, stale):
            failures.append(f"no-trino-integration-claim forbid missed a stale form: {stale!r}")
    # Factual project context and explicit future/planned-connector wording are allowed (neither claims a
    # CURRENT Trino integration). NB (documented residual): the live "tier suitable for query engine
    # integrations (e.g. ... Trino)" line is an ILLUSTRATIVE example, not a current-integration claim — and a
    # hypothetical FALSE "Vortex integrates with ... Trino" current-tense list is NOT regex-distinguishable
    # from that legit example without false-positiving the live doc. So the self-test deliberately does NOT
    # assert "a list containing Trino is allowed" (that would sanction the ambiguous case); the live registry
    # check is what confirms the real doc passes. Catching arbitrary false current-integration phrasings is
    # beyond the scope of this registry check, not a lock gap.
    for ok in ("Trino already supports JDK 22", "any future JVM connector (such as Trino)"):
        if re.search(trino_forbid, ok):  # factual / future-connector context is allowed
            failures.append(f"no-trino-integration-claim forbid false-matched allowed context: {ok!r}")
    # the forbid is sentence-scoped but line-wrap-tolerant — a stale claim split across lines is caught.
    for wrapped in ("Trino support is in\nprogress", "the integration point for Spark\nand Trino"):
        if not re.search(trino_forbid, wrapped):
            failures.append(f"no-trino-integration-claim forbid missed a line-wrapped stale form: {wrapped!r}")

    # 4c. Python API check: python-region scoping ignores prose/URLs/Maven/Spark-options, and the
    #     membership logic flags a fabricated `vortex.<name>` while accepting a real one.
    py = _python_regions("`bench.vortex.dev` and `${vortex.write.batch.size}`\n"
                         "```python\nimport vortex\nvortex.open('f')\n```\nprose vortex.frobnicate\n"
                         "```{doctest} pycon\n>>> import vortex as vx\n>>> vx.array([1])\n```")
    if "vortex.open" not in py or "vortex.dev" in py or "vortex.frobnicate" in py:
        failures.append("_python_regions wrong: missing python block, or leaked prose/URL/option text")
    if "vx.array" not in py:  # MyST {doctest} fences are the dominant python-example form in these docs
        failures.append("_python_regions did not scan a MyST ```{doctest} fence")
    # backtick-length-aware nesting: a 3-backtick python fence inside a 4-backtick `{tab}` container is
    # captured (the inner 3-backtick close must NOT be mis-paired against the 4-backtick container).
    nested = _python_regions("````{tab} Python\n```python\nvx.nested_call(1)\n```\n````\n"
                             "````{tab} Other\n```bash\nrm vortex.frob\n```\n````")
    if "vx.nested_call" not in nested:
        failures.append("_python_regions missed a python fence nested in a 4-backtick `{tab}` container")
    if "vortex.frob" in nested:
        failures.append("_python_regions leaked a non-python (bash) fence nested in a container")
    # OPACITY: an explicit non-python fence containing a LITERAL ```python snippet (a shell heredoc /
    # markdown-generating example) must NOT be scanned as real docs python — i.e. not descended into.
    opaque = _python_regions("````bash\ncat <<EOF\n```python\nvortex.from_a_heredoc(1)\n```\nEOF\n````\n"
                             "```{code-block} bash\nvortex.from_code_block_bash(2)\n```")
    if "vortex.from_a_heredoc" in opaque or "vortex.from_code_block_bash" in opaque:
        failures.append("_python_regions descended into an explicit non-python fence (false positive)")
    # ALL THREE passes are opacity-consistent: a literal `.. code-block:: python` (RST-directive pass)
    # shown inside an opaque ```text fence must NOT be scanned as live API usage either.
    rst_in_fence = _python_regions("```text\n.. code-block:: python\n\n    vortex.from_rst_in_text(1)\n```")
    if "vortex.from_rst_in_text" in rst_in_fence:
        failures.append("_python_regions scanned an RST directive inside an opaque ```text fence")
    # but a ```{eval-rst}` CONTAINER (raw RST embedded in MyST) IS unwrapped, so its `.. code-block::
    # python` / bare `>>>` bodies are still API-checked.
    eval_rst = _python_regions("```{eval-rst}\n.. code-block:: python\n\n    vx.from_eval_rst(1)\n\n"
                               ">>> vx.eval_rst_prompt(2)\n```")
    if "vx.from_eval_rst" not in eval_rst or "vx.eval_rst_prompt" not in eval_rst:
        failures.append("_python_regions did not scan an {eval-rst} container's RST python bodies")
    # pycon/doctest OUTPUT is not API usage: a `>>>` session's input is checked, but interpreter output
    # (e.g. a repr like `<vortex.not_a_real_api ...>`) is dropped so it can't false-fail valid docs.
    pycon = _python_regions("```{doctest} pycon\n>>> import vortex as vx\n>>> vx.array([1])\n"
                            "<vortex.not_a_real_api object at 0x1>\n```")
    if "vx.array" not in pycon:
        failures.append("_strip_pycon_output dropped a `>>>` INPUT line")
    if "not_a_real_api" in pycon:
        failures.append("_strip_pycon_output scanned pycon OUTPUT as code (false-positive risk)")
    # a bare ```{code-cell}` (no language arg) is python in MyST and MUST be scanned
    cell = _python_regions("```{code-cell}\nimport vortex\nvortex.from_bare_code_cell(3)\n```")
    if "vortex.from_bare_code_cell" not in cell:
        failures.append("_python_regions did not scan a bare ```{code-cell} (python kernel) block")
    # the bare-`>>>` RST-doctest pass is PROSE-scoped: a prompt inside an opaque ```text/```bash fence
    # is NOT scanned, but a prompt in real RST prose IS.
    prompts = _python_regions("```text\n>>> vortex.in_a_text_fence(1)\n```\n\n>>> vortex.in_real_prose(2)")
    if "vortex.in_a_text_fence" in prompts:
        failures.append("bare-`>>>` pass scanned a prompt inside an opaque ```text fence (false positive)")
    if "vortex.in_real_prose" not in prompts:
        failures.append("bare-`>>>` pass missed a prompt in RST prose")
    bad_api = DocMembershipCheck._outside("vortex.open(x)\nvortex.frobnicate(y)\nvx.array(z)",
                                          r"(?<![\w.])(?:vortex|vx)\.([A-Za-z_]\w*)", {"open", "array"})
    if bad_api != {"frobnicate"}:
        failures.append(f"python-api membership wrong (got {bad_api!r}; expected {{'frobnicate'}})")
    if "open" not in _vortex_public_api(root):
        failures.append("_vortex_public_api did not include the public `open` name from __all__")
    # module-as-callable: `dataset` is a live submodule (so `vortex.dataset(...)` is a module-called-as-
    # function error), while `array`/`open` are callables and `io`/`file` are only called via attributes.
    subs = _vortex_submodules(root)
    if "dataset" not in subs or "io" not in subs or "open" in subs or "array" in subs:
        failures.append(f"_vortex_submodules wrong (got {sorted(subs)})")
    callpat = re.compile(r"(?<![\w.])(?:vortex|vx)\.(\w+)\s*\(")
    def _mods_called(doc: str) -> set:  # noqa: E306 (test-local helper)
        return {mm.group(1) for mm in callpat.finditer(_python_regions(doc)) if mm.group(1) in subs}
    if _mods_called("```python\nvortex.dataset('f')\n```") != {"dataset"}:
        failures.append("module-call check did not flag `vortex.dataset(...)` (module called as function)")
    for ok in ("vortex.open('f')", "vortex.array([1])", "vortex.io.write(t,'f')", "vx.file.open('f')"):
        if _mods_called(f"```python\n{ok}\n```"):
            failures.append(f"module-call check false-flagged a valid call: {ok!r}")
    # spark/duckdb content forbids catch the stale future-claims (and their derives confirm live code).
    sp_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "spark-filter-pushdown")
    if not re.search(sp_forbid, "not yet connected to Spark's SupportsPushDownFilters; planned future work"):
        failures.append("spark-filter-pushdown forbid missed the stale 'not yet connected' claim")
    dd_forbid = next(c.forbid_regex for c in REGISTRY if c.id == "duckdb-direct-path")
    if not re.search(dd_forbid, "syntax is coming in an upcoming DuckDB release"):
        failures.append("duckdb-direct-path forbid missed the stale 'coming in an upcoming release' claim")
    # canonical encodings sourced from the Rust Canonical enum: VarBinView is canonical (not VarBin),
    # FixedSizeList is canonical, ListArray is NOT (its payload ListViewArray is not in the Python API),
    # and a Decimal/Variant with no Python class is excluded.
    canon = _canonical_python_classes(root)
    if "VarBinViewArray" not in canon or "FixedSizeListArray" not in canon or "VarBinArray" in canon:
        failures.append(f"_canonical_python_classes wrong (got {sorted(canon)})")
    if "ListArray" in canon:  # Canonical::List wraps ListViewArray, not ListArray
        failures.append(f"_canonical_python_classes wrongly included ListArray (got {sorted(canon)})")
    # CanonicalSectionCheck's arrays.rst section parser on SYNTHETIC rst: it bounds the "Canonical
    # Encodings" section at the next heading (a class in a later section is NOT counted), and yields the
    # empty set when the heading is absent.
    rst = ("Canonical Encodings\n-------------------\n\n.. autoclass:: vortex.NullArray\n    :members:\n"
           "\n.. autoclass:: vortex.BoolArray\n\nUtility Encodings\n-----------------\n\n"
           ".. autoclass:: vortex.VarBinArray\n")
    if _listed_canonical_classes(rst) != {"NullArray", "BoolArray"}:
        failures.append(f"_listed_canonical_classes mis-bounded the section (got {_listed_canonical_classes(rst)})")
    if _listed_canonical_classes("Some Other Heading\n------\n\n.. autoclass:: vortex.NullArray\n") != set():
        failures.append("_listed_canonical_classes returned non-empty when the Canonical heading is absent")
    # byte-size derivation (io coalesce + buffered block size)
    if (_eval_byte_expr("1 << 20"), _eval_byte_expr("8 * 1024"), _eval_byte_expr("42")) != (1 << 20, 8192, 42):
        failures.append("_eval_byte_expr mis-evaluated a shift / mul / plain-int literal")
    if _bytes_to_human(4 << 20) != "4 MB" or _bytes_to_human(8 * 1024) != "8 KB":
        failures.append("_bytes_to_human wrong unit formatting")
    try:
        _eval_byte_expr("foo()")
        failures.append("_eval_byte_expr did not fail loud on a complex expression")
    except LookupError:
        pass
    if _coalesce_bytes(root, "file") != (1 << 20, 4 << 20):
        failures.append(f"_coalesce_bytes(file) wrong (got {_coalesce_bytes(root, 'file')})")
    if _derive_buffered_block_size(root) != "2 MB":
        failures.append(f"_derive_buffered_block_size wrong (got {_derive_buffered_block_size(root)!r})")
    # Stat names + the arrays.md Statistics-section parser (set-equality lock)
    sn = _stat_names(root)
    if "true_count" in sn or "run_count" in sn or "null_count" not in sn or len(sn) < 5:
        failures.append(f"_stat_names wrong (got {sorted(sn)})")
    stats_md = "## Statistics\n\n* `is_constant`: holds `true` only.\n* `sum`: y\n\n## Execution\n* `nope`: z\n"
    if _listed_stat_names(stats_md) != (["is_constant", "sum"], 2):  # bounded at next `## `; leading span
        failures.append(f"_listed_stat_names mis-parsed/over-ran the section (got {_listed_stat_names(stats_md)})")
    if _listed_stat_names("## Other\n* `x`: y\n") != ([], 0):
        failures.append("_listed_stat_names returned non-empty when the Statistics heading is absent")
    if _listed_stat_names("## Statistics\r\n\r\n* `sum`: y\r\n\r\n## Execution\r\n") != (["sum"], 1):
        failures.append("_listed_stat_names did not tolerate CRLF line endings")
    # a bullet WITHOUT a leading code-span is counted (n_bullets) but not named, so the check can fail it
    malformed = _listed_stat_names("## Statistics\n\n* `sum`: ok\n* true_count: no backticks\n")
    if malformed != (["sum"], 2):
        failures.append(f"_listed_stat_names did not surface a code-span-less bullet (got {malformed})")
    # duplicates are preserved in the list so StatsListCheck can detect them
    dup = _listed_stat_names("## Statistics\n\n* `sum`: a\n* `sum`: b\n")
    if dup != (["sum", "sum"], 2):
        failures.append(f"_listed_stat_names collapsed a duplicate (got {dup})")
    # Stat::name() arm parsing: literal arms yield names; a non-literal (const-return) arm FAILS LOUD
    # rather than silently shrinking the expected set.
    if _parse_stat_name_arms('Self::A => "a",\n    Self::B => "b",') != {"a", "b"}:
        failures.append("_parse_stat_name_arms wrong on literal arms")
    try:
        _parse_stat_name_arms('Self::A => "a",\n    Self::Foo => FOO_CONST,')
        failures.append("_parse_stat_name_arms did not fail loud on a non-literal (const-return) arm")
    except LookupError:
        pass
    try:
        _parse_stat_name_arms('Self::A => "x",\n    Self::B => "x",')  # two variants, same name string
        failures.append("_parse_stat_name_arms did not fail loud on duplicate name literals")
    except LookupError:
        pass
    for bad_arm in ('Self::A | Self::B => "x",', '_ => "x",'):  # or-pattern / wildcard collapse variants
        try:
            _parse_stat_name_arms(f'Self::Z => "z",\n    {bad_arm}')
            failures.append(f"_parse_stat_name_arms did not fail loud on arm pattern {bad_arm!r}")
        except LookupError:
            pass
    # _stat_names_from cross-checks name count against enum variant count, so even a MULTI-LINE
    # or-pattern (which leaves only `Self::B => ...` on the `=>` line) is caught by the count mismatch.
    if _stat_names_from('Self::A => "a",\n    Self::B => "b",', "A = 0,\n    B = 1,") != {"a", "b"}:
        failures.append("_stat_names_from wrong when name count matches variant count")
    try:  # multi-line or-pattern: 1 name produced, but 2 variants -> fail loud
        _stat_names_from('Self::A |\n    Self::B => "ab",', "A = 0,\n    B = 1,")
        failures.append("_stat_names_from did not fail loud on a collapsed (multi-line or-pattern) arm")
    except LookupError:
        pass
    if _stat_enum_variants("IsConstant = 0,\n    IsSorted = 1,\n    Plain,") != {"IsConstant", "IsSorted", "Plain"}:
        failures.append("_stat_enum_variants mis-parsed discriminant / bare variant forms")
    # payload-vs-variant-name distinction on SYNTHETIC input: List(ListViewArray) must NOT yield
    # ListArray even when ListArray IS in the public API — a variant-name mapping would wrongly include
    # it. (This is the exact regression the payload-based derivation prevents.)
    synth_api = {"NullArray", "ListArray"}  # ListArray exposed; ListViewArray/DecimalArray are NOT
    payloads = _canonical_payload_classes("Null(NullArray),\nList(ListViewArray),\nDecimal(DecimalArray),",
                                          synth_api)
    if payloads != {"NullArray"}:
        failures.append(f"_canonical_payload_classes leaked a variant-name mapping (got {sorted(payloads)})")
    if _canonical_payload_classes("Foo(crate::path::BarArray),", {"BarArray"}) != {"BarArray"}:
        failures.append("_canonical_payload_classes did not take the last path segment of a payload")
    try:  # a struct-style variant must FAIL LOUD, not be silently skipped
        _canonical_payload_classes("Weird { field: u8 },", {"Weird"})
        failures.append("_canonical_payload_classes did not fail loud on an unparseable variant")
    except LookupError:
        pass
    # Python min-version derivation anchors to the `>=` lower bound regardless of clause order, and
    # fails loud when no lower bound is present.
    if _parse_python_min_version(">=3.11,<4.0") != "Python 3.11":
        failures.append("_parse_python_min_version did not anchor to the >= lower bound")
    if _parse_python_min_version("<4,>=3.11") != "Python 3.11":  # reordered clauses
        failures.append("_parse_python_min_version is order-sensitive (picked the upper bound)")
    if _parse_python_min_version(">=3.11.4,<4.0") != "Python 3.11.4":  # patch level must be kept
        failures.append("_parse_python_min_version truncated a patch-level lower bound")
    if _parse_python_min_version(">=3.10,>=3.11,<4") != "Python 3.11":  # intersecting lower bounds
        failures.append("_parse_python_min_version did not pick the highest of multiple >= bounds")
    # fail loud on: no lower bound / pre-release / wildcard / exclusion / compatible-release / exact /
    # exclusive-lower — any of which could silently understate the effective minimum.
    for bad in ("<4.0", ">=3.11.0rc1,<4", ">=3.11.*", ">=3.11,!=3.11.*", ">=3.11,!=3.11.0",
                "~=3.11", "==3.11", ">3.10"):
        try:
            _parse_python_min_version(bad)
            failures.append(f"_parse_python_min_version did not fail loud on {bad!r}")
        except LookupError:
            pass

    # 5. CLI-subcommand check logic, exercised end-to-end on synthetic input.
    # kebab-casing incl. acronyms (clap/heck)
    for variant, want in [("Tree", "tree"), ("FooBar", "foo-bar"), ("SQLQuery", "sql-query"),
                          ("HTTPServer", "http-server")]:
        if CliSubcommandCheck._to_kebab(variant) != want:
            failures.append(f"_to_kebab({variant!r}) != {want!r}")
    # depth-aware enum parse: a wrapped struct-field type is NOT a variant
    variants = CliSubcommandCheck._enum_variants(
        "enum Commands {\n    Tree(Args),\n    Browse {\n        file: PathBuf,\n    },\n}", "Commands")
    if variants != ["Tree", "Browse"]:
        failures.append(f"_enum_variants mis-parsed (got {variants!r}; field type leaked?)")
    # clap naming attributes: `name=` overrides the kebab default, `alias`/`visible_alias` add names,
    # and an unmodeled `rename_all` must FAIL LOUD rather than silently trusting kebab-casing.
    names = CliSubcommandCheck._subcommand_names(
        'enum Commands {\n    Tree(A),\n    #[command(name = "ls", visible_alias = "dir")]\n'
        "    Browse { file: PathBuf },\n}", "Commands")
    if names != {"tree", "ls", "dir"}:
        failures.append(f"_subcommand_names ignored name=/alias attrs (got {names!r})")
    try:
        CliSubcommandCheck._subcommand_names(
            'enum Commands {\n    #[command(rename_all = "snake_case")]\n    FooBar(A),\n}', "Commands")
        failures.append("_subcommand_names did not fail loud on an unmodeled rename_all")
    except LookupError:
        pass
    # multi-line (rustfmt-formatted) clap attribute: the `name=` override spread across lines is honored
    ml_names = CliSubcommandCheck._subcommand_names(
        'enum Commands {\n    Tree(A),\n    #[command(\n        name = "ls",\n'
        '        visible_alias = "dir",\n    )]\n    Browse { file: PathBuf },\n}', "Commands")
    if ml_names != {"tree", "ls", "dir"}:
        failures.append(f"_subcommand_names ignored a multi-line clap attribute (got {ml_names!r})")
    # enum-level rename_all hidden behind a long (>200 char) doc comment, even WITH a trailing comment
    # on the attribute, must STILL fail loud (backward-walk is not clipped, and a `// note` does not
    # terminate the header block).
    long_doc = "/// " + ("x " * 120) + "\n"
    try:
        CliSubcommandCheck._subcommand_names(
            long_doc + '#[command(rename_all = "snake_case")] // tweak casing\n'
            "enum Commands {\n    FooBar(A),\n}", "Commands")
        failures.append("_subcommand_names missed rename_all behind a long doc comment / trailing note")
    except LookupError:
        pass
    # string-literal-aware clap meta: a `name = "fake"` INSIDE an `about` raw string must NOT spoof the
    # real `name = "ls"` override.
    spoof = CliSubcommandCheck._subcommand_names(
        'enum Commands {\n    #[command(about = r#"run with name = "fake""#, name = "ls")]\n'
        "    Browse { file: PathBuf },\n}", "Commands")
    if spoof != {"ls"}:
        failures.append(f"_clap_meta let a string literal spoof the name (got {spoof!r})")
    # MyST `{code-block} bash` directive is OPAQUE shell (scanned); python/doctest directives are not.
    cb = CliSubcommandCheck._shell_regions("```{code-block} bash\nvx frobnicate f\n```")
    if "vx frobnicate" not in cb:
        failures.append("_shell_regions did not scan a {code-block} bash directive body")
    cb_py = CliSubcommandCheck._shell_regions("```{code-block} python\nvx.open('f')\n```")
    if "vx.open" in cb_py:
        failures.append("_shell_regions scanned a {code-block} python directive (alias false-positive)")
    doctest = CliSubcommandCheck._shell_regions("```{doctest} pycon\n>>> import vortex as vx\n`vx frob`\n```")
    if "vx frob" in doctest:
        failures.append("_shell_regions leaked a {doctest} directive body into the CLI scan")
    # MyST nested fence: a fabricated `vx` inside a ```` ```bash ```` block nested in a 4-backtick
    # `{tab}` container MUST be scanned (the docs use this exact shape), and a real shell block AFTER
    # the closed container must still be scanned (the directive-close must not swallow the rest).
    nested = CliSubcommandCheck._shell_regions(
        "````{tab} demo\n```bash\nvx frobnicate f\n```\n````\n\n```bash\nvx convert g\n```")
    if "vx frobnicate" not in nested:
        failures.append("_shell_regions missed a shell block nested in a 4-backtick {tab} container")
    if "vx convert" not in nested:
        failures.append("_shell_regions: a directive-close swallowed a later top-level shell block")
    # shell regions: shell-tagged markdown fence + shell-tagged RST directive + inline span scanned;
    # prose AND non-shell (python) blocks excluded — `vx` there is the module alias, not the CLI.
    sh = CliSubcommandCheck._shell_regions(
        "prose vx faketool\n```bash\nvx convert x\n```\nand `vx query`")
    if "vx convert" not in sh or "vx query" not in sh or "vx faketool" in sh:
        failures.append("_shell_regions (markdown shell fence) wrong: missing region or leaked prose")
    py = CliSubcommandCheck._shell_regions("```python\nimport vortex as vx\nvx.array([1])\n```")
    if "import vortex" in py:
        failures.append("_shell_regions scanned a python block (vx-alias false-positive risk)")
    rst = CliSubcommandCheck._shell_regions(
        ".. code-block:: bash\n\n    vx inspect f.vortex\n\nprose vx nope")
    if "vx inspect" not in rst or "vx nope" in rst:
        failures.append("_shell_regions (RST shell directive) wrong: missing block or leaked prose")
    # a NON-shell RST code directive body is consumed, so its inline ``spans`` do not leak into the
    # scan (the docs' api/*.rst use `.. code-block:: python`); a shell RST directive is still captured.
    rst_py = CliSubcommandCheck._shell_regions(".. code-block:: python\n\n    x = 1  # `vx frob`\n")
    if "vx frob" in rst_py:
        failures.append("_shell_regions leaked an inline span from a python RST directive body")
    rst_sh = CliSubcommandCheck._shell_regions(".. code-block:: bash\n\n    vx convert a\n")
    if "vx convert" not in rst_sh:
        failures.append("_shell_regions dropped a shell RST directive body")
    # nested command paths: build a tree from synthetic cross-file crate source (the `Tree(TreeArgs)`
    # -> `#[clap(subcommand)] mode: TreeMode` -> {array, layout} chain that the real CLI uses), then
    # validate paths against it.
    crate = (
        "enum Commands {\n"
        "    /// tree\n    Tree(super::tree::TreeArgs),\n"
        "    Convert(#[command(flatten)] super::convert::ConvertArgs),\n"
        "    Browse { file: PathBuf },\n}\n"
        "pub struct TreeArgs {\n    #[clap(subcommand)]\n    pub mode: TreeMode,\n}\n"
        "pub enum TreeMode {\n    Array { file: PathBuf },\n    Layout { file: PathBuf },\n}\n"
        "pub struct ConvertArgs {\n    pub file: PathBuf,\n}\n")
    tree = CliSubcommandCheck._command_tree(crate, "Commands")
    if tree != {"tree": {"array": {}, "layout": {}}, "convert": {}, "browse": {}}:
        failures.append(f"_command_tree mis-resolved the nested command tree (got {tree!r})")
    vp = CliSubcommandCheck._validate_path
    # a fabricated SECOND-level subcommand is flagged; a real nested path + its args are valid
    if vp(tree, ["tree", "frobnicate"]) != "tree frobnicate":
        failures.append("_validate_path did not flag a fabricated nested subcommand")
    if vp(tree, ["tree", "layout", "f.vortex"]) is not None:
        failures.append("_validate_path flagged a valid nested path + arg")
    # top-level fabrications (incl. case / underscore / dotted), real commands + args, flags, sentence dot
    for bad_tokens, want in [(["frobnicate"], "frobnicate"), (["Convert"], "Convert"),
                             (["convert_file"], "convert_file"), (["convert.old"], "convert.old")]:
        if vp(tree, bad_tokens) != want:
            failures.append(f"_validate_path did not flag fabricated `{want}`")
    for ok_tokens in (["convert", "a.parquet"], ["browse", "f.vortex"], ["convert."], ["--help"],
                      ["tree", "--verbose"]):
        if vp(tree, ok_tokens) is not None:
            failures.append(f"_validate_path flagged a valid invocation {ok_tokens!r}")
    # a shell operator / separator stops the walk (so chained args are not mis-read as subcommands)
    if vp(tree, ["convert", "f", "&&", "vx", "frobnicate"]) is not None:
        failures.append("_validate_path walked past a shell operator into the next command")
    # a REQUIRED nested subcommand omitted in favor of a path/arg is flagged (`vx tree ./file.vortex`
    # would fail when copy-pasted); a leaf command's path arg, and a flag after `tree`, are still valid
    if vp(tree, ["tree", "./file.vortex"]) != "tree ./file.vortex":
        failures.append("_validate_path did not flag a missing REQUIRED nested subcommand (path arg)")
    if vp(tree, ["tree", "2024.vortex"]) != "tree 2024.vortex":
        failures.append("_validate_path did not flag a missing REQUIRED nested subcommand (numeric arg)")
    if vp(tree, ["convert", "/tmp/a.parquet"]) is not None or vp(tree, ["tree", "--help"]) is not None:
        failures.append("_validate_path flagged a leaf's path arg or a flag after a required-children cmd")
    # rustfmt-wrapped MULTI-LINE tuple variant still exposes its nested subcommand type
    crate_ml = (
        "enum Commands {\n    Tree(\n        super::t::TreeArgs,\n    ),\n}\n"
        "struct TreeArgs {\n    #[clap(subcommand)]\n    pub mode: TreeMode,\n}\n"
        "enum TreeMode {\n    Array { f: PathBuf },\n    Layout { f: PathBuf },\n}\n")
    if CliSubcommandCheck._command_tree(crate_ml, "Commands") != {"tree": {"array": {}, "layout": {}}}:
        failures.append("_command_tree missed a nested subcommand in a multi-line tuple variant")
    # plural clap aliases (`visible_aliases = ["ls", "dir"]`) are honored
    plural = CliSubcommandCheck._subcommand_names(
        'enum Commands {\n    #[command(visible_aliases = ["ls", "dir"])]\n'
        "    Browse { f: PathBuf },\n}", "Commands")
    if plural != {"browse", "ls", "dir"}:
        failures.append(f"_subcommand_names ignored plural clap aliases (got {plural!r})")
    # the command prefix is SOURCED from the crate's `[[bin]] name` (tomllib), picking the [[bin]] name
    # over the [package] name — so a rename WOULD be picked up, not silently passed.
    renamed = '[package]\nname = "vortex-tui"\n\n[[bin]]\nname = "renamed"\npath = "src/main.rs"\n'
    if _toml_str_from(renamed, "synth", "bin", 0, "name") != "renamed":
        failures.append("_toml_str_from did not select the [[bin]] name over [package] name")
    # end-to-end on the real repo it currently resolves to the vx binary
    if CLI_CHECKS[0].resolve_prefix(root) != "vx ":
        failures.append(f"resolve_prefix wiring broken (got {CLI_CHECKS[0].resolve_prefix(root)!r})")
    # nested subcommand detection tolerates EXTRA clap meta (`subcommand, required = true`)
    crate_req = (
        "enum Commands {\n    Tree(super::t::TreeArgs),\n}\n"
        "struct TreeArgs {\n    #[command(subcommand, required = true)]\n    pub mode: TreeMode,\n}\n"
        "enum TreeMode {\n    Array { f: PathBuf },\n    Layout { f: PathBuf },\n}\n")
    if CliSubcommandCheck._command_tree(crate_req, "Commands") != {"tree": {"array": {}, "layout": {}}}:
        failures.append("_command_tree missed a nested subcommand with extra clap meta")
    # variant metadata split across SEPARATE attributes is merged (the `name=` is honored)
    split_names = CliSubcommandCheck._subcommand_names(
        'enum Commands {\n    #[command(about = "x")]\n    #[command(name = "ls")]\n'
        "    Browse { file: PathBuf },\n}", "Commands")
    if split_names != {"ls"}:
        failures.append(f"_subcommand_names ignored a split-attribute name= override (got {split_names!r})")
    # OPTIONAL nested subcommand → lenient leaf (ACCEPTED TRADEOFF): `vx inspect <bogus> <file>` is NOT
    # flagged, because we cannot tell a fabricated optional-subcommand from the parent's own positional
    # argument without risking false-positives on real `vx inspect <file>`. Asserted as intentional.
    if vp({"inspect": {}}, ["inspect", "bogus", "file.vortex"]) is not None:
        failures.append("_validate_path flagged an optional-subcommand parent (should be lenient leaf)")
    # end-to-end: per-line scan validates EACH same-line `vx` invocation (a second after `&&` is not
    # hidden); `uvx ruff` is not matched as `vx ruff`; a line-final `vx` cannot bind across a newline.
    chk = CLI_CHECKS[0]
    # build the pattern from the SOURCED prefix (resolve_prefix), matching the live check path
    pfx = re.compile(r"(?<![\w-])" + re.escape(chk.resolve_prefix(root).strip()) + r"[ \t]+")
    line = "vx convert f && vx frobnicate g"
    multi = [vp(tree, line[m.end():].split()) for m in pfx.finditer(line)]
    if "frobnicate" not in [f for f in multi if f]:
        failures.append("multi-vx-per-line: a second same-line invocation was not validated")
    if any(pfx.search(ln) for ln in "import vortex as vx\n\ndef read_column():\n    pass".split("\n")):
        failures.append("invocation pattern matched across a newline / bound to a later line")
    if pfx.search("uvx ruff check"):
        failures.append("invocation pattern matched `uvx ruff` as a vx invocation")
    print("SELF-TEST OK vx-subcommands: nested paths, extra-meta/split-attr clap parsing, multi-vx, "
          "uvx/cross-line safe")

    # 6. Encoding-stability derivation: the stable set is code-derived from register_default_encodings
    #    (+ the crate initialize() fns) and FLIPS when an encoding's `unstable_encodings` gating changes.
    #    The core is proved here on synthetic input; encoding_stability.self_test_failures also re-checks
    #    the full synthetic matrix + the live classification.
    enc_off = encoding_stability.stability_from_body("arrays.register(Widget);")
    enc_on = encoding_stability.stability_from_body(
        '#[cfg(feature = "unstable_encodings")]\n    arrays.register(Widget);')
    if enc_off.get("Widget") != encoding_stability.STABLE:
        failures.append("encoding-stability: an ungated encoding should be STABLE")
    if enc_on.get("Widget") != encoding_stability.UNSTABLE:
        failures.append("encoding-stability: gating an encoding behind unstable_encodings must move it "
                        "OUT of the stable set")
    if "Widget" in {n for n, c in enc_on.items() if c == encoding_stability.STABLE}:
        failures.append("encoding-stability: Widget remained STABLE after being gated (set did not flip)")
    failures += encoding_stability.self_test_failures(root)
    print("SELF-TEST OK encoding-stability: gating flips move an encoding between the stable/unstable sets")

    # 7. Spec-conformance tripwire (SHADOW mode): coverage-gap detection + would-be-drift detection, on
    #    SYNTHETIC input, then the live shadow derivation. Proves the tripwire REPORTS without blocking —
    #    nothing here appends a blocking failure, and `run_checks` (called below) stays exit 0.
    synth_doc = (
        "## Byte layout\n\n"
        "(encoding-layout-ALP)=\n### `vortex.alp` — Byte layout\nbody\n\n"
        "```text\n(encoding-layout-Delta)=\n```\n"  # a decoy anchor inside a code fence must NOT count
    )
    anchors = parse_layout_anchors(synth_doc)
    if anchors != {"ALP"}:
        failures.append(f"parse_layout_anchors wrong: got {sorted(anchors)}, expected ['ALP'] "
                        "(a fenced-block decoy anchor must be ignored)")
    # the scan surface now spans MULTIPLE pages (the top-level page + family pages under
    # encoding-format/); prove anchors from DIFFERENT pages union, and a fenced decoy on any one
    # page still does not count.
    multi = _covered_from_texts([
        "(encoding-layout-Primitive)=\n### `vortex.primitive`\nbody\n",
        "(encoding-layout-ALP)=\n### `vortex.alp`\n```text\n(encoding-layout-FoR)=\n```\n",
    ])
    if multi != {"Primitive", "ALP"}:
        failures.append(f"_covered_from_texts wrong: got {sorted(multi)}, expected ['ALP', 'Primitive'] "
                        "(anchors must union across pages; a fenced-block decoy must not count)")
    # a stable encoding with no byte-layout section is reported as a coverage gap
    gaps = coverage_gaps({"ALP", "FoR", "Dict"}, anchors)
    if gaps != ["Dict", "FoR"]:
        failures.append(f"coverage_gaps wrong: got {gaps}, expected ['Dict', 'FoR'] (uncovered stable set)")
    if layout_anchor_label("ALP") != "encoding-layout-ALP":
        failures.append("layout_anchor_label disagrees with the parsed anchor convention")
    # a stale lock (code invariant disagrees with the value the spec pins) is WOULD-BE drift; an
    # agreeing lock is not; and the override-sentinel hook detects drift like ValueMatch's does.
    stale = EncodingLayoutLock("Widget", "synthetic",
                               derive_code=lambda r: "3 buffers", derive_spec=lambda r: "2 buffers")
    if not stale.check(root)[0]:
        failures.append("would-be-drift self-test: a stale lock was NOT detected as drift")
    agree = EncodingLayoutLock("Widget", "synthetic",
                               derive_code=lambda r: "2 buffers", derive_spec=lambda r: "2 buffers")
    if agree.check(root)[0]:
        failures.append("would-be-drift self-test: an agreeing lock spuriously reported drift")
    if not agree.check(root, override_code="__DRIFT_SENTINEL__")[0]:
        failures.append("would-be-drift self-test: the override-sentinel hook did not detect drift")
    # the live shadow derivation runs clean; coverage is a pure set difference over the live stable set
    # (33/33 today; NOT asserted as a hard count, so adding a stable encoding without a section
    # legitimately widens the gap and surfaces here without editing the test).
    live_stable = encoding_stability.stable_encodings(root)
    live_covered = covered_encodings(root)
    if not live_stable:
        failures.append("shadow: live stable set is empty")
    if set(coverage_gaps(live_stable, live_covered)) != live_stable - live_covered:
        failures.append("shadow: coverage_gaps is not the stable-minus-covered set difference")
    print("SELF-TEST OK spec-tripwire: coverage gap reported, would-be drift detected, non-blocking")

    live = run_checks(root, verbose=False)
    if failures:
        print("\nSELF-TEST FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    if live != 0:
        print("\nSELF-TEST FAILED: the live registry does not currently pass.", file=sys.stderr)
        return 1
    print("\nSELF-TEST OK — matcher rejects overlap, accepts punctuation, every check detects drift, "
          "and the live registry passes.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Vortex docs-conformance lint")
    ap.add_argument("--self-test", action="store_true", help="prove the checker detects drift")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every check's resolved values")
    ns = ap.parse_args()
    root = repo_root()
    return self_test(root) if ns.self_test else run_checks(root, ns.verbose)


if __name__ == "__main__":
    sys.exit(main())
