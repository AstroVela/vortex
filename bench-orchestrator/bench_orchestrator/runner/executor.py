# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Benchmark binary execution."""

import os
import selectors
import subprocess
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import final

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import Benchmark, Engine, Format

console = Console()
_HEAP_PROFILE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "bench-heap-profile.sh"


@final
class BenchmarkExecutor:
    """Executes benchmark binaries and captures output."""

    def __init__(self, binary_path: Path, backend: Engine, verbose: bool = False):
        self.binary_path = binary_path
        self.backend = backend
        self.verbose = verbose

    def build_command(
        self,
        benchmark: Benchmark,
        formats: list[Format],
        queries: list[int] | None = None,
        exclude_queries: list[int] | None = None,
        iterations: int = 5,
        options: dict[str, str] | None = None,
        track_memory: bool = False,
        samply: bool = False,
        sample_rate: int | None = None,
        tracing: bool = False,
        runner: str | None = None,
        ingest_output: Path | None = None,
    ) -> list[str]:
        """Build the command used to execute a benchmark binary."""
        cmd = [
            str(self.binary_path),
            benchmark.value,
            "--display-format",
            "gh-json",
            "--iterations",
            str(iterations),
            "--hide-progress-bar",
        ]

        if self.backend in {Engine.DATAFUSION, Engine.DUCKDB}:
            cmd.extend(["--formats", ",".join(fmt.value for fmt in formats)])
        if self.backend == Engine.DUCKDB:
            cmd.append("--delete-duckdb-database")

        if queries:
            cmd.extend(["--queries", ",".join(map(str, queries))])
        if exclude_queries:
            cmd.extend(["--exclude-queries", ",".join(map(str, exclude_queries))])
        if track_memory:
            cmd.append("--track-memory")
        if tracing:
            cmd.append("--tracing")
        if runner:
            cmd.extend(["--runner", runner])
        if ingest_output is not None:
            cmd.extend(["--ingest-jsonl", str(ingest_output)])
        if options:
            for key, value in options.items():
                cmd.extend(["--opt", f"{key}={value}"])

        if samply:
            cmd = ["--"] + cmd
            cmd_prefix = ["samply", "record"]
            if sample_rate:
                cmd = cmd_prefix + ["--rate", str(sample_rate)] + cmd
            else:
                cmd = cmd_prefix + cmd

        if samply and self.backend == Engine.DUCKDB:
            # Re-use the same DuckDB instance across runs to keep samply output readable.
            cmd.append("--reuse")

        return cmd

    def run(
        self,
        benchmark: Benchmark,
        formats: list[Format],
        queries: list[int] | None = None,
        exclude_queries: list[int] | None = None,
        iterations: int = 5,
        options: dict[str, str] | None = None,
        track_memory: bool = False,
        samply: bool = False,
        sample_rate: int | None = None,
        tracing: bool = False,
        runner: str | None = None,
        ingest_output: Path | None = None,
        on_result: Callable[[str], None] | None = None,
    ) -> list[str]:
        """
        Run benchmark and return results as JSON lines.

        Args:
            benchmark: The benchmark suite to run
            formats: Data formats to benchmark
            queries: Specific queries to run (None for all)
            exclude_queries: Queries to skip
            iterations: Number of runs per query
            options: Additional options (e.g., scale_factor)
            track_memory: Enable memory tracking
            on_result: Callback for each result line (for streaming)

        Returns:
            List of JSON lines from the benchmark output
        """
        heap_profiling = bool(os.environ.get("POLARSIGNALS_CLOUD_TOKEN"))
        format_groups = [[fmt] for fmt in formats] if heap_profiling else [formats]
        results: list[str] = []
        ingest_parts: list[Path] = []

        with ExitStack() as stack:
            ingest_temp_dir = None
            if ingest_output is not None and len(format_groups) > 1:
                ingest_temp_dir = Path(stack.enter_context(TemporaryDirectory(prefix="vx-bench-profile-ingest-")))

            for index, command_formats in enumerate(format_groups):
                command_ingest_output = ingest_output
                if ingest_temp_dir is not None:
                    command_ingest_output = ingest_temp_dir / f"{index:02d}-{command_formats[0].value}.jsonl"
                    ingest_parts.append(command_ingest_output)

                cmd = self.build_command(
                    benchmark=benchmark,
                    formats=command_formats,
                    queries=queries,
                    exclude_queries=exclude_queries,
                    iterations=iterations,
                    options=options,
                    track_memory=track_memory,
                    samply=samply,
                    sample_rate=sample_rate,
                    tracing=tracing,
                    runner=runner,
                    ingest_output=command_ingest_output,
                )
                process_env = None
                if heap_profiling:
                    process_env = os.environ.copy()
                    process_env["HEAP_PROFILE_ENGINE"] = self._profile_engine()
                    process_env["HEAP_PROFILE_FORMAT"] = command_formats[0].value
                    cmd = [str(_HEAP_PROFILE_SCRIPT), *cmd]

                results.extend(
                    self._run_command(
                        cmd,
                        benchmark=benchmark,
                        formats=command_formats,
                        process_env=process_env,
                        on_result=on_result,
                    )
                )

            if ingest_output is not None and ingest_parts:
                self._combine_ingest_output(ingest_output, ingest_parts)

        return results

    def _profile_engine(self) -> str:
        if self.backend == Engine.LANCE:
            return Engine.DATAFUSION.value
        return self.backend.value

    @staticmethod
    def _combine_ingest_output(output_path: Path, input_paths: list[Path]) -> None:
        if output_path.parent != Path():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as output:
            for input_path in input_paths:
                if not input_path.exists():
                    raise RuntimeError(f"ingest output was not written by profiled benchmark: {input_path}")
                with input_path.open(encoding="utf-8") as input_file:
                    for line in input_file:
                        _ = output.write(line)

    def _run_command(
        self,
        cmd: list[str],
        *,
        benchmark: Benchmark,
        formats: list[Format],
        process_env: dict[str, str] | None,
        on_result: Callable[[str], None] | None,
    ) -> list[str]:
        if self.verbose:
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

        results: list[str] = []
        diagnostic_lines: deque[str] = deque(maxlen=200)
        target = f"{self._profile_engine()}:{','.join(fmt.value for fmt in formats)}"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            _task = progress.add_task(f"Running {target} {benchmark.value}...", total=None)

            # Merge stderr into stdout so verbose benchmark logs cannot fill a separate pipe and
            # block the child process before it emits JSON results.
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=process_env,
            )

            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            _ = selector.register(process.stdout, selectors.EVENT_READ)

            try:
                while selector.get_map():
                    for _key, _mask in selector.select(timeout=0.1):
                        line = process.stdout.readline()
                        if line == "":
                            _ = selector.unregister(process.stdout)
                            continue

                        line = line.rstrip()
                        if not line:
                            continue

                        if line.startswith("{"):
                            results.append(line)
                            if on_result:
                                on_result(line)
                        else:
                            diagnostic_lines.append(line)
                            console.print(line, markup=False)
            finally:
                selector.close()

            ret_code = process.wait()

            if ret_code != 0:
                console.print(f"[red]Benchmark failed with code {process.returncode}[/red]")
                diagnostics = "\n".join(diagnostic_lines)
                if diagnostics:
                    console.print(f"[red]{diagnostics}[/red]")
                raise RuntimeError(f"Benchmark {target} {benchmark.value} failed: {diagnostics}")

        return results
