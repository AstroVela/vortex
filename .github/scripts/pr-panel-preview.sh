#!/usr/bin/env bash
#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Bounded stand-in for the `issue_comment: edited` webhook.
#
# GitHub only ever runs `issue_comment` workflows from the default branch, so a pull
# request that changes the panel cannot exercise a real click through the normal path.
# This script closes that gap by polling the panel comment for a while and applying
# whatever it finds, using the PR's own version of the code. Everything downstream of
# "we noticed the body changed" is identical to `pr-panel.yml`; only the trigger differs.
#
# Usage: pr-panel-preview.sh <owner/repo> <pr-number> [window-seconds] [poll-seconds] [notice]
# Requires: gh (authenticated via GH_TOKEN), jq, python3.

set -Eeuo pipefail

repo="$1"
pr="$2"
window="${3:-1800}"
interval="${4:-15}"
notice="${5:-}"

panel_marker="<!-- vortex-pr-panel -->"
report_marker="<!-- vortex-pr-panel-report -->"
scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

current="${work}/current.md"
next="${work}/next.md"
report="${work}/report.md"
outputs="${work}/outputs"

gh api "/repos/${repo}/issues/${pr}/comments" --paginate >"${work}/comments.json"
comment_id="$(jq -r --arg marker "${panel_marker}" \
  '.[] | select(.body | contains($marker)) | .id' "${work}/comments.json" | head -n 1)"

if [ -z "${comment_id}" ]; then
  echo "::error::no panel comment found on #${pr}"
  exit 1
fi

fetch_body() {
  gh api "/repos/${repo}/issues/comments/${comment_id}" --jq .body >"${current}"
}

patch_body() {
  jq -n --rawfile body "${next}" '{body: $body}' |
    gh api --silent --method PATCH "/repos/${repo}/issues/comments/${comment_id}" --input -
}

# Runs `pr_panel apply` and exports its GitHub-Actions outputs as shell variables.
apply() {
  : >"${outputs}"
  (
    cd "${scripts_dir}"
    GITHUB_OUTPUT="${outputs}" python3 -m pr_panel apply \
      --body "${current}" \
      --out "${next}" \
      --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$@"
  )
  changed="$(sed -n 's/^changed=//p' "${outputs}")"
  dispatch="$(sed -n 's/^dispatch=//p' "${outputs}")"
  actions="$(sed -n 's/^actions=//p' "${outputs}")"
  state="$(sed -n 's/^state=//p' "${outputs}")"
}

echo "Watching panel comment ${comment_id} on #${pr} for ${window}s (every ${interval}s)"

# Tell whoever is looking at the PR that their clicks are live, and for how long.
if [ -n "${notice}" ]; then
  fetch_body
  apply --notice "${notice}"
  patch_body
fi

deadline=$((SECONDS + window))
applied=0

while [ "${SECONDS}" -lt "${deadline}" ]; do
  fetch_body
  apply

  if [ "${changed}" = "true" ]; then
    patch_body
    applied=$((applied + 1))
    echo "redrew the panel (dispatch=${dispatch}, actions=${actions:-none})"
  fi

  if [ "${dispatch}" = "true" ]; then
    # The production path dispatches `pr-panel-run.yml`, which is only dispatchable
    # from the default branch. Rendering the report here exercises the same code.
    (
      cd "${scripts_dir}"
      python3 -m pr_panel report \
        --state "${state}" \
        --actions "${actions}" \
        --run-url "${RUN_URL:-}" \
        --out "${report}"
    )
    bash "${scripts_dir}/upsert-comment.sh" "${repo}" "${pr}" "${report_marker}" "${report}"
  fi

  sleep "${interval}"
done

fetch_body
apply --notice "Preview window closed. Push to this branch to reopen it."
patch_body

echo "Preview finished after applying ${applied} edit(s)."
