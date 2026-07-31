#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the PR control panel render/parse/apply cycle."""

from __future__ import annotations

import json

import pytest

from pr_panel.panel import (
    MARKER,
    apply_edit,
    initial_state,
    marker_id,
    parse_checkboxes,
    parse_state,
    render,
    render_report,
)
from pr_panel.spec import PANEL, Button, Radio, Toggle


def click(body: str, control_id: str) -> str:
    """Flip one checkbox the way GitHub rewrites the body when a user clicks it."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip().endswith(f"<!--c:{control_id}-->"):
            if "- [ ]" in line:
                lines[i] = line.replace("- [ ]", "- [x]", 1)
            else:
                lines[i] = line.replace("- [x]", "- [ ]", 1)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"no control line for {control_id!r}")


@pytest.fixture
def fresh() -> str:
    return render(initial_state())


def test_control_ids_are_unique():
    ids = [c.id for c in PANEL.controls()]
    assert len(ids) == len(set(ids))


def test_fresh_panel_carries_marker_and_state(fresh: str):
    assert fresh.startswith(MARKER)
    assert parse_state(fresh).controls == PANEL.default_state()


def test_every_control_is_rendered_and_parseable(fresh: str):
    found = parse_checkboxes(fresh)
    for control in PANEL.controls():
        if isinstance(control, Radio):
            for option in control.options:
                assert marker_id(control, option.id) in found
        else:
            assert control.id in found


def test_render_is_idempotent(fresh: str):
    assert apply_edit(fresh).body == fresh
    assert apply_edit(fresh).changed is False


def test_toggle_click_is_persisted(fresh: str):
    result = apply_edit(click(fresh, "suite.sql"))
    assert result.state.controls["suite.sql"] is True
    # The user's own click already left the comment correct, so there is nothing to
    # write back to GitHub.
    assert result.changed is False

    result = apply_edit(click(result.body, "suite.sql"))
    assert result.state.controls["suite.sql"] is False


def test_radio_selection_unchecks_siblings(fresh: str):
    result = apply_edit(click(fresh, "runner.machine:g5.xlarge"))

    assert result.state.controls["runner.machine"] == "g5.xlarge"
    assert result.changed is True

    checked = parse_checkboxes(result.body)
    selected = [k for k, v in checked.items() if k.startswith("runner.machine:") and v]
    assert selected == ["runner.machine:g5.xlarge"]


def test_radio_cannot_be_emptied(fresh: str):
    """Unchecking the selected option restores it: a group always has a selection."""
    result = apply_edit(click(fresh, "runner.machine:c6id.metal"))

    assert result.state.controls["runner.machine"] == "c6id.metal"
    assert parse_checkboxes(result.body)["runner.machine:c6id.metal"] is True


def test_two_clicks_landing_in_one_event_pick_the_last(fresh: str):
    """A user can out-click the workflow; the newest selection wins."""
    body = click(click(fresh, "runner.machine:c7i.8xlarge"), "runner.machine:g5.xlarge")
    result = apply_edit(body)

    assert result.state.controls["runner.machine"] == "g5.xlarge"


def test_button_press_dispatches_and_clears(fresh: str):
    result = apply_edit(click(fresh, "action.run"), actor="ada", timestamp="2026-01-01T00:00:00Z")

    assert result.pressed == ["action.run"]
    assert result.dispatched == ["run"]
    assert result.changed is True
    # The button must not stay pressed, or the next edit would re-dispatch it.
    assert parse_checkboxes(result.body)["action.run"] is False
    assert result.state.last == {
        "actions": ["run"],
        "by": "ada",
        "at": "2026-01-01T00:00:00Z",
    }
    assert "Last dispatched `run` by @ada" in result.body


def test_button_press_carries_current_settings(fresh: str):
    body = apply_edit(click(fresh, "suite.sql")).body
    body = apply_edit(click(body, "options.preset:pr-full")).body
    result = apply_edit(click(body, "action.run"))

    assert result.state.controls["suite.sql"] is True
    assert result.state.controls["options.preset"] == "pr-full"


def test_reset_restores_defaults_without_dispatching(fresh: str):
    body = apply_edit(click(fresh, "suite.sql")).body
    body = apply_edit(click(body, "runner.machine:g5.xlarge")).body

    result = apply_edit(click(body, "action.reset"))

    assert result.state.controls == PANEL.default_state()
    assert result.dispatched == []
    assert result.changed is True


def test_state_survives_a_full_round_trip(fresh: str):
    body = apply_edit(click(fresh, "options.profile")).body
    reparsed = parse_state(body)

    assert reparsed.controls["options.profile"] is True
    assert json.loads(reparsed.to_json())["v"] == 1


def test_corrupt_state_blob_falls_back_to_the_checkboxes(fresh: str):
    blob = fresh[fresh.index("<!--vortex-pr-panel:state:") :]
    body = click(fresh.replace(blob, "<!--vortex-pr-panel:state:not json-->\n"), "suite.sql")

    result = apply_edit(body)

    assert result.state.controls["suite.sql"] is True


def test_prose_edits_do_not_lose_state(fresh: str):
    """A human editing the comment text is repaired on the next apply."""
    mangled = fresh.replace("### Suites", "### Suites (edited by hand)")
    result = apply_edit(click(mangled, "suite.sql"))

    assert result.state.controls["suite.sql"] is True
    assert "### Suites (edited by hand)" not in result.body
    assert result.changed is True


def test_revision_advances_only_on_visible_change(fresh: str):
    assert parse_state(fresh).revision == 1
    assert apply_edit(fresh).state.revision == 1
    assert apply_edit(click(fresh, "action.run")).state.revision == 2


def test_report_renders_every_non_button_control(fresh: str):
    state = parse_state(click(fresh, "suite.sql")).to_json()
    report = render_report(state, run_url="https://example.invalid/run/1", actions="run")

    for control in PANEL.controls():
        if isinstance(control, Button):
            assert f"`{control.id}`" not in report
        else:
            assert f"`{control.id}`" in report
    assert "https://example.invalid/run/1" in report


def test_report_shows_toggle_and_radio_values(fresh: str):
    report = render_report(parse_state(fresh).to_json())

    assert "✅ on" in report
    assert "⬜ off" in report
    assert "`c6id.metal`" in report


def test_report_tolerates_empty_state():
    assert "Panel run report" in render_report("")


@pytest.mark.parametrize("control", [c for c in PANEL.controls() if isinstance(c, Toggle)])
def test_each_toggle_round_trips(fresh: str, control: Toggle):
    result = apply_edit(click(fresh, control.id))

    assert result.state.controls[control.id] is (not control.default)
    assert result.dispatched == []
