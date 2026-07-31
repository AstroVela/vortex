#!/usr/bin/env bash
#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Create or update a marker-tagged comment on an issue or pull request, so repeated
# runs edit one comment instead of piling up new ones.
#
# Usage: upsert-comment.sh <owner/repo> <issue-number> <marker> <body-file>
# Requires: gh (authenticated via GH_TOKEN), jq.

set -Eeuo pipefail

repo="$1"
issue="$2"
marker="$3"
body_file="$4"

comments="$(mktemp)"
gh api "/repos/${repo}/issues/${issue}/comments" --paginate >"${comments}"

# `--paginate` emits one JSON array per page, which jq reads as a stream.
existing="$(jq -r --arg marker "${marker}" \
  '.[] | select(.body | contains($marker)) | .id' "${comments}" | head -n 1)"

payload="$(jq -n --rawfile body "${body_file}" '{body: $body}')"

if [ -n "${existing}" ]; then
  printf '%s' "${payload}" |
    gh api --silent --method PATCH "/repos/${repo}/issues/comments/${existing}" --input -
  echo "Updated comment ${existing}"
else
  printf '%s' "${payload}" |
    gh api --silent --method POST "/repos/${repo}/issues/${issue}/comments" --input -
  echo "Created a new comment on #${issue}"
fi
