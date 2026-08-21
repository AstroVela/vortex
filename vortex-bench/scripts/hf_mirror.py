#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Mirror Hugging Face datasets to Vortex and compare against their Parquet originals.

Every public dataset on the Hub is auto-converted to Parquet under the
``refs/convert/parquet`` branch, so any dataset -- not just the ones already published as
Parquet -- can be mirrored through the same path.

Typical use:

    # Show the candidate lists (trending / most-liked / most-downloaded / curated).
    ./hf_mirror.py list --sort trending

    # Mirror a couple of datasets and report Parquet vs Vortex sizes.
    ./hf_mirror.py mirror HuggingFaceFW/fineweb-edu wikimedia/wikipedia:20231101.en

    # Re-print the report for everything mirrored so far.
    ./hf_mirror.py report

Set ``HF_TOKEN`` to reach gated or rate-limited datasets.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

HF_API = "https://huggingface.co/api"
DATASETS_SERVER = "https://datasets-server.huggingface.co"

# Datasets picked because they are popular *and* exercise a distinct part of the encoding
# space. The key is `dataset[:config[:split]]`; the value is what we expect Vortex to do well
# (or badly) on. Configs are pinned where the alphabetically-first one is unrepresentative --
# `wikipedia` would otherwise resolve to Abkhazian and `c4` to Afrikaans.
CURATED: dict[str, str] = {
    "HuggingFaceFW/fineweb-edu": "huge text blobs plus low-cardinality metadata; FSST + dict vs zstd",
    "HuggingFaceFW/finepdfs:eng_Latn": "same shape as fineweb-edu with more metadata columns",
    "allenai/c4:en": "text + timestamp + URL; classic web-crawl layout",
    "wikimedia/wikipedia:20231101.en": "long-form text, very high raw:parquet ratio",
    "yaak-ai/L2D": "driving telemetry: numeric lists and timeseries, ALP/FoR territory",
    "jat-project/jat-dataset-tokenized": "tokenized RL trajectories, dense integer sequences",
    "mteb/results": "narrow benchmark table: low-cardinality strings + floats, dict-friendly",
    "Open-Orca/OpenOrca": "instruction tuning; system_prompt repeats heavily, dict-friendly",
    "HuggingFaceTB/smoltalk:all": "nested list-of-struct chat messages, exercises nested layouts",
    "PleIAs/common_corpus": "mixed metadata and text, many low-cardinality categoricals",
    "lmsys/chatbot_arena_conversations": "nested conversations plus categorical model names",
    "nyu-mll/glue:mnli": "small classic NLP table, quick smoke test",
}

# Config and split names to prefer when a dataset has several and the caller did not pick one.
PREFERRED_CONFIGS = ("default", "all", "main", "en", "train")
PREFERRED_SPLITS = ("train", "test", "validation")

SORTS = {
    "trending": "trendingScore",
    "likes": "likes",
    "downloads": "downloads",
}

SIZE_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
}


def parse_size(text: str) -> int:
    """Parse a human size such as ``512MiB`` or ``2GB`` into bytes."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", text)
    if match is None:
        raise argparse.ArgumentTypeError(f"cannot parse size: {text!r}")
    value, unit = float(match.group(1)), match.group(2).lower() or "b"
    if unit not in SIZE_UNITS:
        raise argparse.ArgumentTypeError(f"unknown size unit: {unit!r}")
    return int(value * SIZE_UNITS[unit])


def format_size(size_bytes: float) -> str:
    """Format bytes as a human-readable size."""
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if size_bytes >= scale:
            return f"{size_bytes / scale:.2f} {unit}"
    return f"{size_bytes:.0f} B"


def get_json(url: str) -> object:
    """GET a JSON document from the Hub, attaching ``HF_TOKEN`` when present."""
    request = urllib.request.Request(url, headers=auth_headers())
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def auth_headers() -> dict[str, str]:
    """Authorization headers for the Hub, empty when no token is configured."""
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


@dataclass
class Candidate:
    """A dataset as advertised by the Hub listing APIs."""

    id: str
    likes: int
    downloads: int
    formats: list[str]
    modalities: list[str]
    size_category: str
    note: str = ""


def to_candidate(entry: dict, note: str = "") -> Candidate:
    """Build a [`Candidate`] from a raw Hub dataset record."""
    tags = entry.get("tags", [])

    def tagged(prefix: str) -> list[str]:
        return [tag.split(":", 1)[1] for tag in tags if tag.startswith(prefix)]

    return Candidate(
        id=entry["id"],
        likes=entry.get("likes", 0),
        downloads=entry.get("downloads", 0),
        formats=tagged("format:"),
        modalities=tagged("modality:"),
        size_category=next(iter(tagged("size_categories:")), ""),
        note=note,
    )


def list_candidates(sort: str, limit: int, tabular_only: bool) -> list[Candidate]:
    """Fetch one of the Hub "hot" lists, or the curated shortlist."""
    if sort == "curated":
        return [
            to_candidate(
                get_json(f"{HF_API}/datasets/{urllib.parse.quote(DatasetRef.parse(ref).dataset)}"),
                note,
            )
            for ref, note in CURATED.items()
        ]

    query = [("sort", SORTS[sort]), ("direction", "-1"), ("limit", str(limit))]
    if tabular_only:
        query += [("filter", "format:parquet"), ("filter", "modality:tabular")]
    entries = get_json(f"{HF_API}/datasets?{urllib.parse.urlencode(query)}")
    return [to_candidate(entry) for entry in entries]


def dataset_size(dataset: str) -> dict | None:
    """Row and byte counts from the dataset viewer, or `None` when unavailable."""
    url = f"{DATASETS_SERVER}/size?dataset={urllib.parse.quote(dataset)}"
    try:
        return get_json(url)["size"]["dataset"]
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError):
        return None


@dataclass(frozen=True)
class DatasetRef:
    """A dataset id with an optional config and split, spelled `dataset[:config[:split]]`."""

    dataset: str
    config: str | None = None
    split: str | None = None

    @classmethod
    def parse(cls, text: str) -> "DatasetRef":
        """Parse a `dataset[:config[:split]]` reference."""
        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError(f"expected dataset[:config[:split]], got {text!r}")
        return cls(*parts)

    def __str__(self) -> str:
        return ":".join(part for part in (self.dataset, self.config, self.split) if part)


def pick(available: list[str], requested: str | None, preferred: tuple[str, ...], what: str) -> str:
    """Resolve a config or split name, preferring `requested`, then `preferred`, then the first."""
    if requested is not None:
        if requested not in available:
            raise RuntimeError(f"unknown {what} {requested!r}; available: {', '.join(available)}")
        return requested
    return next((name for name in preferred if name in available), available[0])


def shard_urls(ref: DatasetRef) -> tuple[list[str], DatasetRef]:
    """URLs of the auto-converted Parquet shards, plus the config and split actually used.

    The Hub converts every public dataset to Parquet regardless of its source format, so
    this works for CSV-, JSON-, and Arrow-backed datasets too.
    """
    tree = get_json(f"{HF_API}/datasets/{urllib.parse.quote(ref.dataset)}/parquet")
    if not tree:
        raise RuntimeError(f"{ref.dataset}: no auto-converted Parquet available")
    config = pick(list(tree), ref.config, PREFERRED_CONFIGS, "config")
    splits = tree[config]
    split = pick(list(splits), ref.split, PREFERRED_SPLITS, "split")
    return list(splits[split]), DatasetRef(ref.dataset, config, split)


def download(url: str, target: Path, attempts: int = 3) -> int:
    """Idempotently download `url` to `target`, returning its size in bytes.

    The Hub redirects to a CDN that occasionally closes a connection early. A short read
    leaves a Parquet file that still has its `PAR1` header but a truncated footer, so verify
    the transferred length against `Content-Length` and retry rather than converting garbage.
    """
    if target.exists():
        return target.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(url, headers=auth_headers())

    for attempt in range(1, attempts + 1):
        with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
            expected = response.headers.get("Content-Length")
        written = partial.stat().st_size
        if expected is None or written == int(expected):
            partial.rename(target)
            return written
        print(
            f"    short read ({written} of {expected} bytes), attempt {attempt}/{attempts}",
            file=sys.stderr,
        )

    partial.unlink(missing_ok=True)
    raise RuntimeError(f"{url}: truncated after {attempts} attempts")


@dataclass
class MirrorResult:
    """Parquet-versus-Vortex sizes for one mirrored dataset, per compression strategy."""

    dataset: str
    shards: int
    rows: int | None
    parquet_bytes: int
    #: Strategy name to `(vortex bytes, convert wall seconds)`.
    vortex: dict[str, list[float]] = field(default_factory=dict)
    note: str = ""
    errors: list[str] = field(default_factory=list)

    def ratio(self, strategy: str) -> float:
        """Parquet bytes over Vortex bytes: above 1.0 means Vortex is smaller."""
        entry = self.vortex.get(strategy)
        if not entry or not entry[0]:
            return float("nan")
        return self.parquet_bytes / entry[0]

    def best_ratio(self) -> float:
        """The best ratio across strategies, for ordering the report."""
        ratios = [self.ratio(name) for name in self.vortex]
        return max((r for r in ratios if r == r), default=float("-inf"))


def vx_command(vx: str | None) -> list[str]:
    """The `vx` invocation to use, falling back to a release build from the workspace."""
    if vx:
        return [vx]
    found = shutil.which("vx")
    if found:
        return [found]
    return [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "-p",
        "vortex-tui",
        "--features",
        "native",
        "--bin",
        "vx",
        "--",
    ]


def convert(vx: list[str], parquet: Path, strategy: str) -> tuple[Path, float]:
    """Convert `parquet` to Vortex with `vx convert`, returning the output and wall time.

    `vx convert` always writes `<input>.vortex`, so the result is renamed per strategy to let
    several strategies coexist for the same shard.
    """
    vortex = parquet.with_suffix(f".{strategy}.vortex")
    if vortex.exists():
        return vortex, float("nan")
    start = time.perf_counter()
    subprocess.run(
        [*vx, "convert", "--quiet", "--strategy", strategy, str(parquet)],
        check=True,
    )
    elapsed = time.perf_counter() - start
    written = parquet.with_suffix(".vortex")
    if not written.exists():
        raise RuntimeError(f"vx convert did not produce {written}")
    written.rename(vortex)
    return vortex, elapsed


def mirror(
    ref: DatasetRef,
    data_dir: Path,
    max_bytes: int,
    max_shards: int,
    strategies: list[str],
    vx: list[str],
) -> MirrorResult:
    """Download a bounded prefix of a dataset's Parquet shards and convert each to Vortex."""
    urls, resolved = shard_urls(ref)
    target_dir = data_dir / str(resolved).replace("/", "__").replace(":", "-")
    parquet_bytes = 0
    vortex: dict[str, list[float]] = {name: [0.0, 0.0] for name in strategies}
    shards = 0
    errors: list[str] = []

    for index, url in enumerate(urls[:max_shards]):
        parquet = target_dir / f"{index:05d}.parquet"
        try:
            size = download(url, parquet)
            for strategy in strategies:
                output, elapsed = convert(vx, parquet, strategy)
                vortex[strategy][0] += output.stat().st_size
                if elapsed == elapsed:  # not NaN, i.e. we actually ran the conversion
                    vortex[strategy][1] += elapsed
        except (urllib.error.HTTPError, urllib.error.URLError) as err:
            errors.append(f"shard {index}: {err}")
            break
        except (subprocess.CalledProcessError, RuntimeError) as err:
            errors.append(f"shard {index}: {err}")
            break
        parquet_bytes += size
        shards += 1
        if parquet_bytes >= max_bytes:
            break

    size_info = dataset_size(resolved.dataset)
    return MirrorResult(
        dataset=str(resolved),
        shards=shards,
        rows=size_info["num_rows"] if size_info else None,
        parquet_bytes=parquet_bytes,
        vortex=vortex if shards else {},
        note=CURATED.get(str(ref), CURATED.get(ref.dataset, "")),
        errors=errors,
    )


def render_report(results: list[MirrorResult]) -> str:
    """Render mirrored results as a GitHub-flavored Markdown table."""
    strategies = sorted({name for result in results for name in result.vortex})
    head = "| Dataset | Shards | Parquet |"
    rule = "| --- | ---: | ---: |"
    for strategy in strategies:
        head += f" {strategy} | {strategy} ratio |"
        rule += " ---: | ---: |"
    lines = [head + " Notes |", rule + " --- |"]

    for result in sorted(results, key=MirrorResult.best_ratio, reverse=True):
        if not result.shards:
            cells = " – |" * (2 * len(strategies) + 1)
            lines.append(f"| `{result.dataset}` | 0 |{cells} {'; '.join(result.errors)} |")
            continue
        row = f"| `{result.dataset}` | {result.shards} | {format_size(result.parquet_bytes)} |"
        for strategy in strategies:
            size, _ = result.vortex.get(strategy, [0, 0])
            row += f" {format_size(size)} | {result.ratio(strategy):.2f}x |"
        lines.append(f"{row} {result.note} |")
    return "\n".join(lines)


def cmd_list(args: argparse.Namespace) -> int:
    """Print a Hub listing (or the curated shortlist) as a table."""
    candidates = list_candidates(args.sort, args.limit, args.tabular)
    header = f"{'dataset':52} {'likes':>7} {'downloads':>11} {'size':14} formats / modalities"
    print(header)
    print("-" * len(header))
    for candidate in candidates:
        formats = ",".join(candidate.formats) or "-"
        modalities = ",".join(candidate.modalities) or "-"
        print(
            f"{candidate.id[:52]:52} {candidate.likes:>7} {candidate.downloads:>11} "
            f"{candidate.size_category or '-':14} {formats} / {modalities}"
        )
        if candidate.note:
            print(f"{'':52} ↳ {candidate.note}")
    return 0


def cmd_mirror(args: argparse.Namespace) -> int:
    """Mirror the requested datasets and write a report next to the mirrored data."""
    refs = [DatasetRef.parse(text) for text in (args.datasets or list(CURATED))]
    vx = vx_command(args.vx)
    results = []
    for ref in refs:
        print(f"==> {ref}", file=sys.stderr)
        try:
            result = mirror(
                ref,
                args.data_dir,
                args.max_bytes,
                args.max_shards,
                args.strategies,
                vx,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as err:
            result = MirrorResult(str(ref), 0, None, 0, errors=[str(err)])
        results.append(result)
        write_results(args.data_dir, results)

    print(render_report(results))
    return 0 if any(r.shards for r in results) else 1


def results_path(data_dir: Path) -> Path:
    """Location of the accumulated mirror results."""
    return data_dir / "hf_mirror_results.json"


def write_results(data_dir: Path, results: list[MirrorResult]) -> None:
    """Merge `results` into the on-disk results file, keyed by dataset."""
    path = results_path(data_dir)
    merged = {entry["dataset"]: entry for entry in read_results(data_dir)}
    merged.update({result.dataset: asdict(result) for result in results})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(merged.values()), indent=2) + "\n")


def read_results(data_dir: Path) -> list[dict]:
    """Previously recorded mirror results, or an empty list."""
    path = results_path(data_dir)
    return json.loads(path.read_text()) if path.exists() else []


def cmd_report(args: argparse.Namespace) -> int:
    """Re-render the report from previously mirrored datasets."""
    entries = read_results(args.data_dir)
    if not entries:
        print(f"no results in {results_path(args.data_dir)}; run `hf_mirror.py mirror` first")
        return 1
    print(render_report([MirrorResult(**entry) for entry in entries]))
    return 0


def default_data_dir() -> Path:
    """Default mirror location, alongside the other benchmark data."""
    return Path(__file__).resolve().parents[1] / "data" / "hf"


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="where mirrored Parquet and Vortex files are written",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    lister = subparsers.add_parser("list", help="show Hub dataset rankings")
    lister.add_argument("--sort", choices=[*SORTS, "curated"], default="trending")
    lister.add_argument("--limit", type=int, default=20)
    lister.add_argument(
        "--tabular",
        action="store_true",
        help="restrict to datasets tagged as Parquet and tabular",
    )
    lister.set_defaults(func=cmd_list)

    mirrorer = subparsers.add_parser("mirror", help="mirror datasets and compare sizes")
    mirrorer.add_argument(
        "datasets",
        nargs="*",
        metavar="DATASET[:CONFIG[:SPLIT]]",
        help="dataset references; defaults to the curated set",
    )
    mirrorer.add_argument("--max-bytes", type=parse_size, default=parse_size("512MiB"))
    mirrorer.add_argument("--max-shards", type=int, default=8)
    mirrorer.add_argument(
        "--strategies",
        nargs="+",
        choices=["btrblocks", "compact"],
        default=["btrblocks", "compact"],
        help="compression strategies to convert with; each gets its own column",
    )
    mirrorer.add_argument("--vx", help="path to the vx binary; defaults to PATH then cargo run")
    mirrorer.set_defaults(func=cmd_mirror)

    reporter = subparsers.add_parser("report", help="re-render the report from disk")
    reporter.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
