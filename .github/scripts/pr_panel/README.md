# PR control panel

An interactive form, rendered as a bot comment on every pull request. Clicking a
checkbox changes a setting; clicking a **button** dispatches a separate workflow run
with the whole settings blob attached.

This exists to replace the current "add a label, the bot runs, the bot removes the
label" pattern for benchmarks, where the combinations outgrew what a label list can
show. Nothing here is wired to a real benchmark yet — the downstream workflow only
renders the state it received.

## How it works

GitHub gives markdown exactly one clickable control: the task list checkbox. Ticking
one in a comment rewrites that comment's body and delivers an `issue_comment: edited`
webhook. Everything else is built on top of that single primitive.

```
user ticks a box
      │
      ▼
issue_comment: edited ──► pr-panel.yml (apply)
                              │  parse checkboxes, diff against embedded state
                              │  canonicalize (radios, momentary buttons)
                              ├─► PATCH the comment  ──► panel redraws
                              └─► workflow_dispatch  ──► pr-panel-run.yml
                                                            renders state back to the PR
```

The comment body is the only storage. It holds one tagged task-list line per control:

```markdown
- [x] Random access <!--c:suite.random_access-->
```

plus a trailing state blob that records what the panel last rendered:

```markdown
<!--vortex-pr-panel:state:{"controls":{...},"rev":3,"v":1}-->
```

Diffing the checkboxes against that blob is what identifies the click, and that is what
makes controls richer than a plain checkbox possible:

| Control | Behaviour |
| --- | --- |
| `Toggle` | Persistent on/off. The click is the new value. |
| `Radio` | Pick-one. Checking an option unchecks its siblings on redraw; unchecking the selected one restores it, so a group is never empty. |
| `Button` | Momentary. A tick is an edge: it dispatches an action and is cleared on redraw. |
| `Section` | Grouping. A closed section renders as a folded `<details>` block. |

Add or change controls in `spec.py`; `panel.py` needs no edits.

## Properties worth knowing

- **Authorization is GitHub's.** Only users with write access can tick a checkbox on
  someone else's comment, so no separate permission check is needed.
- **No feedback loop.** The redraw is written with `GITHUB_TOKEN`, and edits made with
  that token do not trigger workflows.
- **Races are serialized.** `pr-panel.yml` uses a per-PR concurrency group that never
  cancels, and re-reads the comment body from the API rather than trusting the event
  payload, so clicks that queue up are all observed.
- **Hand edits self-heal.** The panel is redrawn from canonical state, so prose someone
  typed into the comment is replaced on the next click.
- **Redraws are skipped when they would be a no-op.** Ticking a toggle already leaves the
  comment correct, so no write-back happens; only radios and buttons force a redraw.
- **Deploys lag.** `issue_comment` workflows always run from the default branch, so
  changes to this panel only take effect on PRs once merged.

## Working on it locally

```bash
cd .github/scripts

# Render the panel as it would first appear.
python3 -m pr_panel demo

# Simulate a click sequence; the trailing state blob shows the result.
python3 -m pr_panel demo --click suite.sql --click runner.machine:g5.xlarge --click action.run

# Tests (pure stdlib, no install needed).
uv run --no-project --with pytest pytest tests/test_pr_panel.py
```
