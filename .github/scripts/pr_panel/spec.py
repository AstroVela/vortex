#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Declarative definition of the PR control panel.

The panel is a single bot-authored PR comment whose markdown task list doubles as a
form. Every control owns a stable id that is embedded in the comment as an invisible
HTML comment, so labels can be reworded without breaking state.

This module contains *only* the description of the UI. Rendering, parsing, and click
semantics live in ``panel.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONTROL_ID = str


@dataclass(frozen=True)
class Toggle:
    """A persistent on/off control. Clicking it flips the stored value."""

    id: CONTROL_ID
    label: str
    default: bool = False
    help: str = ""


@dataclass(frozen=True)
class RadioOption:
    """One choice within a `Radio` group."""

    id: CONTROL_ID
    label: str
    help: str = ""


@dataclass(frozen=True)
class Radio:
    """A pick-one group. Checking an option unchecks its siblings on the next render.

    Markdown has no `<select>`, so a radio group is emulated: the panel accepts the
    newly checked option and rewrites the rest of the group as unchecked.
    """

    id: CONTROL_ID
    label: str
    options: tuple[RadioOption, ...]
    default: CONTROL_ID = ""
    help: str = ""

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(f"radio {self.id!r} needs at least one option")
        if self.default and self.default not in {o.id for o in self.options}:
            raise ValueError(f"radio {self.id!r} default {self.default!r} is not an option")

    @property
    def resolved_default(self) -> CONTROL_ID:
        return self.default or self.options[0].id


@dataclass(frozen=True)
class Button:
    """A momentary control. Checking it triggers an action, then it clears itself."""

    id: CONTROL_ID
    label: str
    help: str = ""
    # Action name handed to the downstream workflow.
    action: str = ""
    # Whether pressing it dispatches the downstream workflow, or only edits panel state.
    dispatches: bool = True
    # Whether pressing it restores every control to its default.
    resets: bool = False

    @property
    def resolved_action(self) -> str:
        return self.action or self.id


Control = Toggle | Radio | Button


@dataclass(frozen=True)
class Section:
    """A group of controls, rendered as a collapsible `<details>` block when closed."""

    id: CONTROL_ID
    title: str
    controls: tuple[Control, ...]
    # Sections rendered open are always visible; closed ones start folded.
    open: bool = True
    help: str = ""


@dataclass(frozen=True)
class Panel:
    """The whole panel: a title, some prose, and an ordered list of sections."""

    title: str
    blurb: str
    sections: tuple[Section, ...] = field(default_factory=tuple)

    def controls(self):
        for section in self.sections:
            yield from section.controls

    def control(self, control_id: CONTROL_ID) -> Control | None:
        for control in self.controls():
            if control.id == control_id:
                return control
        return None

    def buttons(self) -> tuple[Button, ...]:
        return tuple(c for c in self.controls() if isinstance(c, Button))

    def default_state(self) -> dict[str, object]:
        """The state a freshly posted panel starts in."""
        state: dict[str, object] = {}
        for control in self.controls():
            if isinstance(control, Toggle):
                state[control.id] = control.default
            elif isinstance(control, Radio):
                state[control.id] = control.resolved_default
        return state


# The demo panel. Nothing here is wired to a real benchmark yet: it exists to prove the
# click -> re-render -> dispatch loop with one control of every kind.
PANEL = Panel(
    title="Benchmark Control Panel",
    blurb=(
        "Configure a benchmark run by clicking the checkboxes below, then press "
        "**Run** to dispatch it. Your click is applied by a workflow, so the panel "
        "takes a few seconds to redraw."
    ),
    sections=(
        Section(
            id="suites",
            title="Suites",
            help="Which benchmark suites to include in the run.",
            controls=(
                Toggle(id="suite.random_access", label="Random access", default=True),
                Toggle(id="suite.compression", label="Compression", default=True),
                Toggle(id="suite.sql", label="SQL (TPC-H / Clickbench)"),
                Toggle(id="suite.gpu_compression", label="GPU compression"),
            ),
        ),
        Section(
            id="runner",
            title="Runner",
            open=False,
            help="Where the run executes.",
            controls=(
                Radio(
                    id="runner.machine",
                    label="Machine",
                    default="c6id.metal",
                    options=(
                        RadioOption(id="c6id.metal", label="`c6id.metal` (bare metal, default)"),
                        RadioOption(id="c7i.8xlarge", label="`c7i.8xlarge`"),
                        RadioOption(id="g5.xlarge", label="`g5.xlarge` (GPU)"),
                    ),
                ),
            ),
        ),
        Section(
            id="options",
            title="Options",
            open=False,
            help="Extra knobs applied to every selected suite.",
            controls=(
                Radio(
                    id="options.preset",
                    label="Matrix preset",
                    default="pr",
                    options=(
                        RadioOption(id="pr", label="`pr` — quick subset"),
                        RadioOption(id="pr-full", label="`pr-full` — everything"),
                    ),
                ),
                Toggle(id="options.profile", label="Capture a continuous profile"),
                Toggle(id="options.unstable_encodings", label="Enable unstable encodings"),
            ),
        ),
        Section(
            id="actions",
            title="Actions",
            help="Momentary buttons. They clear themselves once the click is handled.",
            controls=(
                Button(
                    id="action.run",
                    label="**Run** — dispatch a run with the settings above",
                    action="run",
                ),
                Button(
                    id="action.reset",
                    label="**Reset** — restore the default settings",
                    action="reset",
                    dispatches=False,
                    resets=True,
                ),
            ),
        ),
    ),
)
