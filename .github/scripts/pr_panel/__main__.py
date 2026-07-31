#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

"""CLI used by the PR control panel workflows.

Subcommands:

* ``render`` — emit the body of a brand-new panel comment.
* ``apply``  — read an edited panel comment, emit the redrawn body, and report which
  buttons were pressed via ``$GITHUB_OUTPUT``.
* ``report`` — render a state blob as a table (run by the downstream workflow).
* ``demo``   — render the panel locally and optionally simulate clicks, so the layout
  can be eyeballed without pushing to CI.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .panel import apply_edit, initial_state, parse_checkboxes, render, render_report


def _write_output(pairs: dict[str, str]) -> None:
    """Append ``key=value`` pairs to ``$GITHUB_OUTPUT`` if we are inside Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        for key, value in pairs.items():
            print(f"{key}={value}", file=sys.stderr)
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in pairs.items():
            if "\n" in value:
                raise ValueError(f"output {key!r} must be single-line")
            handle.write(f"{key}={value}\n")


def _cmd_render(args: argparse.Namespace) -> int:
    Path(args.out).write_text(render(initial_state()), encoding="utf-8")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    body = Path(args.body).read_text(encoding="utf-8")
    result = apply_edit(body, actor=args.actor, timestamp=args.timestamp)
    Path(args.out).write_text(result.body, encoding="utf-8")
    _write_output(
        {
            "changed": "true" if result.changed else "false",
            "dispatch": "true" if result.dispatched else "false",
            "actions": ",".join(result.dispatched),
            "state": result.state.to_json(),
        }
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    report = render_report(args.state, run_url=args.run_url, actions=args.actions)
    Path(args.out).write_text(report, encoding="utf-8")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    body = render(initial_state())
    for click in args.click:
        checked = parse_checkboxes(body)
        if click not in checked:
            raise SystemExit(f"unknown control {click!r}; known: {', '.join(sorted(checked))}")
        body = _toggle_line(body, click)
        result = apply_edit(body, actor="demo", timestamp="2026-01-01T00:00:00Z")
        print(
            f"--- click {click} -> pressed={result.pressed} changed={result.changed}",
            file=sys.stderr,
        )
        body = result.body
    print(body, end="")
    return 0


def _toggle_line(body: str, control_id: str) -> str:
    """Flip one checkbox, mimicking what GitHub writes when a user clicks it."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip().endswith(f"<!--c:{control_id}-->"):
            if "- [ ]" in line:
                lines[i] = line.replace("- [ ]", "- [x]", 1)
            else:
                lines[i] = line.replace("- [x]", "- [ ]", 1)
            break
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pr_panel")
    sub = parser.add_subparsers(dest="command", required=True)

    render_cmd = sub.add_parser("render", help="write a fresh panel body")
    render_cmd.add_argument("--out", required=True)
    render_cmd.set_defaults(func=_cmd_render)

    apply_cmd = sub.add_parser("apply", help="interpret an edited panel body")
    apply_cmd.add_argument("--body", required=True, help="file holding the current comment body")
    apply_cmd.add_argument("--out", required=True, help="file to write the redrawn body to")
    apply_cmd.add_argument("--actor", default="")
    apply_cmd.add_argument("--timestamp", default="")
    apply_cmd.set_defaults(func=_cmd_apply)

    report_cmd = sub.add_parser("report", help="render received state as a table")
    report_cmd.add_argument("--state", required=True)
    report_cmd.add_argument("--actions", default="")
    report_cmd.add_argument("--run-url", default="")
    report_cmd.add_argument("--out", required=True)
    report_cmd.set_defaults(func=_cmd_report)

    demo_cmd = sub.add_parser("demo", help="render locally, optionally simulating clicks")
    demo_cmd.add_argument("--click", action="append", default=[], metavar="CONTROL_ID")
    demo_cmd.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
