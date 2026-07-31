#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Render, parse, and apply clicks for the PR control panel.

The comment body is the entire persistence layer. It carries:

* one task-list line per control, each tagged with an invisible ``<!--c:id-->`` marker,
* a trailing ``<!--vortex-pr-panel:state:{...}-->`` blob holding the canonical state as
  of the last render.

A click rewrites a single ``- [ ]`` into ``- [x]`` (or back), which GitHub delivers as an
``issue_comment.edited`` event. Diffing the parsed checkboxes against the embedded state
tells us exactly which control the user touched, which is what makes momentary buttons
and pick-one radio groups expressible in plain markdown.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .spec import PANEL, Button, Control, Panel, Radio, Toggle

# Identifies a panel comment. Kept on its own line so `grep`/`contains()` in a workflow
# `if:` expression can cheaply pre-filter events.
MARKER = "<!-- vortex-pr-panel -->"

STATE_VERSION = 1

_CONTROL_RE = re.compile(
    r"^\s*- \[(?P<checked>[ xX])\]\s*(?P<label>.*?)\s*<!--c:(?P<id>[A-Za-z0-9_.:-]+)-->\s*$"
)
_STATE_RE = re.compile(r"<!--vortex-pr-panel:state:(?P<payload>.*?)-->", re.DOTALL)


@dataclass
class PanelState:
    """Canonical state of the panel, as embedded in the comment."""

    controls: dict[str, Any] = field(default_factory=dict)
    revision: int = 0
    last: dict[str, Any] = field(default_factory=dict)
    # Free-form banner shown above the controls, e.g. "a run is in progress". Survives
    # clicks, so whoever set it is the one who clears it.
    notice: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "v": STATE_VERSION,
                "rev": self.revision,
                "controls": self.controls,
                "last": self.last,
                "notice": self.notice,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str) -> PanelState:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("panel state must be a JSON object")
        return cls(
            controls=dict(raw.get("controls") or {}),
            revision=int(raw.get("rev") or 0),
            last=dict(raw.get("last") or {}),
            notice=str(raw.get("notice") or ""),
        )


@dataclass
class ApplyResult:
    """Outcome of interpreting a comment edit."""

    body: str
    state: PanelState
    # Control ids the user just clicked.
    pressed: list[str]
    # Action names for the pressed buttons that ask for a downstream run.
    dispatched: list[str]
    # Whether the redrawn body differs from what is already on the PR.
    changed: bool


def marker_id(control: Control, option_id: str = "") -> str:
    """The stable id embedded in the comment for a control (or a radio option)."""
    return f"{control.id}:{option_id}" if option_id else control.id


def parse_checkboxes(body: str) -> dict[str, bool]:
    """Extract ``marker id -> checked`` for every tagged task-list line."""
    checked: dict[str, bool] = {}
    for line in body.splitlines():
        match = _CONTROL_RE.match(line)
        if match:
            checked[match.group("id")] = match.group("checked").lower() == "x"
    return checked


def parse_state(body: str, panel: Panel = PANEL) -> PanelState:
    """Read the embedded state blob, falling back to defaults when it is absent."""
    match = _STATE_RE.search(body)
    if match:
        try:
            return PanelState.from_json(match.group("payload"))
        except (ValueError, json.JSONDecodeError):
            pass
    return PanelState(controls=panel.default_state())


def canonicalize(
    previous: dict[str, Any],
    checked: dict[str, bool],
    panel: Panel = PANEL,
) -> tuple[dict[str, Any], list[str]]:
    """Fold the raw checkbox reading into canonical state plus any pressed buttons.

    ``previous`` is what the panel last rendered, so ``checked`` differing from it is
    precisely the user's click.
    """
    state: dict[str, Any] = {}
    pressed: list[str] = []

    for control in panel.controls():
        if isinstance(control, Toggle):
            prior = bool(previous.get(control.id, control.default))
            state[control.id] = bool(checked.get(control.id, prior))
        elif isinstance(control, Radio):
            state[control.id] = _resolve_radio(control, previous, checked)
        elif isinstance(control, Button):
            # Momentary: a check is an edge, never stored.
            if checked.get(control.id, False):
                pressed.append(control.id)

    return state, pressed


def _resolve_radio(radio: Radio, previous: dict[str, Any], checked: dict[str, bool]) -> str:
    option_ids = [option.id for option in radio.options]
    prior = previous.get(radio.id, radio.resolved_default)
    if prior not in option_ids:
        prior = radio.resolved_default

    now_checked = [oid for oid in option_ids if checked.get(marker_id(radio, oid), False)]
    newly_checked = [oid for oid in now_checked if oid != prior]
    if newly_checked:
        # A click landed on a sibling: it wins, and the rest are cleared on re-render.
        return newly_checked[-1]
    if prior in now_checked or not now_checked:
        # Unchecking the selected option would leave the group empty, so restore it.
        return prior
    return now_checked[0]


def apply_edit(
    body: str,
    *,
    actor: str = "",
    timestamp: str = "",
    notice: str | None = None,
    panel: Panel = PANEL,
) -> ApplyResult:
    """Interpret an edited panel comment and produce the redrawn body.

    ``notice`` replaces the banner; leave it as ``None`` to carry the existing one over.
    """
    previous = parse_state(body, panel)
    controls, pressed = canonicalize(previous.controls, parse_checkboxes(body), panel)

    buttons = [c for c in (panel.control(pid) for pid in pressed) if isinstance(c, Button)]
    if any(button.resets for button in buttons):
        controls = panel.default_state()
    dispatched = [button.resolved_action for button in buttons if button.dispatches]

    last = dict(previous.last)
    if buttons:
        last = {
            "actions": [button.resolved_action for button in buttons],
            "by": actor,
            "at": timestamp,
        }

    state = PanelState(
        controls=controls,
        revision=previous.revision + 1,
        last=last,
        notice=previous.notice if notice is None else notice,
    )
    new_body = render(state, panel)
    unchanged = _normalize(new_body) == _normalize(body)
    if unchanged:
        # Nothing to write back; keep the revision the user is already looking at.
        state.revision = previous.revision
        new_body = render(state, panel)

    return ApplyResult(
        body=new_body,
        state=state,
        pressed=pressed,
        dispatched=dispatched,
        changed=not unchanged,
    )


def initial_state(panel: Panel = PANEL, notice: str = "") -> PanelState:
    """State for a panel that has never been clicked."""
    return PanelState(controls=panel.default_state(), revision=1, notice=notice)


def _normalize(body: str) -> str:
    """Compare bodies ignoring the revision counter and trailing whitespace."""
    stripped = _STATE_RE.sub("", body)
    return "\n".join(line.rstrip() for line in stripped.strip().splitlines())


def render(state: PanelState, panel: Panel = PANEL) -> str:
    """Render canonical state as the full comment body."""
    out: list[str] = [MARKER, "", f"## {panel.title}", "", panel.blurb, ""]
    if state.notice:
        # Collapse whitespace so a notice wrapped across lines still renders as one
        # blockquote rather than breaking out of it.
        out.extend([f"> [!IMPORTANT]\n> {' '.join(state.notice.split())}", ""])
    out.extend([_status_line(state), ""])

    for section in panel.sections:
        out.extend(_render_section(section, state))

    out.extend(
        [
            "<sub>Checkboxes are editable by anyone with write access. Every click is "
            "applied by the <code>PR Control Panel</code> workflow, which redraws this "
            "comment; the buttons clear themselves once handled.</sub>",
            "",
            f"<!--vortex-pr-panel:state:{state.to_json()}-->",
        ]
    )
    return "\n".join(out) + "\n"


def _status_line(state: PanelState) -> str:
    # The revision counter is deliberately *not* shown: `_normalize` compares rendered
    # bodies to decide whether a write-back is needed, and a monotonic counter in the
    # visible text would make every edit look like a change.
    last = state.last
    if last.get("actions"):
        actions = ", ".join(f"`{a}`" for a in last["actions"])
        who = f" by @{last['by']}" if last.get("by") else ""
        when = f" at {last['at']}" if last.get("at") else ""
        return f"> [!NOTE]\n> Last dispatched {actions}{who}{when}."
    return "> [!NOTE]\n> Nothing dispatched yet."


def _render_section(section, state: PanelState) -> list[str]:
    out: list[str] = []
    if section.open:
        out.append(f"### {section.title}")
        out.append("")
        if section.help:
            out.append(section.help)
            out.append("")
    else:
        out.append("<details>")
        out.append(f"<summary><b>{section.title}</b></summary>")
        out.append("")
        if section.help:
            out.append(section.help)
            out.append("")

    for control in section.controls:
        out.extend(_render_control(control, state))

    # A blank line keeps the next heading (or `</details>`) out of the task list.
    if out[-1] != "":
        out.append("")
    if not section.open:
        out.extend(["</details>", ""])
    return out


def _render_control(control: Control, state: PanelState) -> list[str]:
    if isinstance(control, Radio):
        selected = state.controls.get(control.id, control.resolved_default)
        out = [f"**{control.label}** — pick one:", ""]
        for option in control.options:
            out.append(
                _checkbox(marker_id(control, option.id), _label(option), option.id == selected)
            )
        out.append("")
        return out

    if isinstance(control, Toggle):
        checked = bool(state.controls.get(control.id, control.default))
        return [_checkbox(control.id, _label(control), checked)]

    # Buttons always render unchecked: they are edges, not state.
    return [_checkbox(control.id, _label(control), False)]


def _label(control: Control) -> str:
    return f"{control.label} — <sub>{control.help}</sub>" if control.help else control.label


def _checkbox(cid: str, label: str, checked: bool) -> str:
    box = "x" if checked else " "
    return f"- [{box}] {label} <!--c:{cid}-->"


def render_report(
    state_json: str,
    *,
    panel: Panel = PANEL,
    run_url: str = "",
    actions: str = "",
) -> str:
    """Render received state as a table.

    This is what the downstream workflow prints: it proves the panel's state crosses the
    workflow boundary intact, and stands in for whatever the run will eventually do.
    """
    raw = json.loads(state_json) if state_json.strip() else {}
    controls = dict(raw.get("controls") or {}) if isinstance(raw, dict) else {}

    out = ["<!-- vortex-pr-panel-report -->", "", "## Panel run report", ""]
    if actions:
        out.append(f"Dispatched by: {', '.join(f'`{a}`' for a in actions.split(',') if a)}")
        out.append("")
    if run_url:
        out.append(f"Rendered by [this workflow run]({run_url}).")
        out.append("")

    out.extend(["| Control | Value |", "| --- | --- |"])
    for section in panel.sections:
        for control in section.controls:
            if isinstance(control, Button):
                continue
            value = controls.get(control.id)
            out.append(f"| {section.title} / `{control.id}` | {_format_value(control, value)} |")

    out.extend(["", f"<sub>Panel revision {raw.get('rev', '?')}.</sub>", ""])
    return "\n".join(out) + "\n"


def _format_value(control: Control, value: Any) -> str:
    if isinstance(control, Toggle):
        return "✅ on" if value else "⬜ off"
    return f"`{value}`" if value is not None else "_unset_"
