#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Upload debuginfo for every binary in $PATHS to a Parca-compatible store.
#
# Binaries are uploaded concurrently, one background job each, because the uploads are
# network-bound and independent. Each job's output is buffered to its own log and replayed in
# input order once all jobs finish, so interleaved output never mangles the step log.

set -Eeu -o pipefail

: "${PATHS:?PATHS must be set}"
: "${POLARSIGNALS_CLOUD_TOKEN:?POLARSIGNALS_CLOUD_TOKEN must be set}"
: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${STORE_ADDRESS:?STORE_ADDRESS must be set}"
FORCE="${FORCE:-false}"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

token_file="$work_dir/token"
(
  umask 077
  printf "%s" "$POLARSIGNALS_CLOUD_TOKEN" > "$token_file"
)

# Uploads a single binary, writing all output to $2. Returns 0 for the benign races where the
# store already has (or is still ingesting) debuginfo for this build ID.
upload_one() {
  local binary="$1"
  local log="$2"

  local cmd=(
    parca-debuginfo
    upload
    "--store-address=$STORE_ADDRESS"
    "--bearer-token-file=$token_file"
    "--grpc-headers=projectID=$PROJECT_ID"
    "$binary"
  )
  if [[ "$FORCE" == "true" ]]; then
    cmd+=(--force)
  fi

  local status=0
  "${cmd[@]}" > "$log" 2>&1 || status=$?

  if [[ "$status" -ne 0 ]]; then
    local benign="upload id mismatch|already exists|AlreadyExists"
    benign+="|previous upload is still in-progress|not stale yet|only stale uploads can be retried"
    if grep -Eiq "$benign" "$log"; then
      echo "::notice::Debuginfo upload already exists or is in progress for $binary; continuing" >> "$log"
      status=0
    fi
  fi

  return "$status"
}

binaries=()
while IFS= read -r binary; do
  binary="${binary#"${binary%%[![:space:]]*}"}"
  binary="${binary%"${binary##*[![:space:]]}"}"
  if [[ -z "$binary" ]]; then
    continue
  fi

  if [[ ! -f "$binary" ]]; then
    echo "::error::Debuginfo upload target does not exist: $binary"
    exit 1
  fi

  binaries+=("$binary")
done <<< "$PATHS"

if [[ "${#binaries[@]}" -eq 0 ]]; then
  echo "::error::No debuginfo upload targets were provided"
  exit 1
fi

pids=()
logs=()
for i in "${!binaries[@]}"; do
  logs+=("$work_dir/upload-$i.log")
  upload_one "${binaries[$i]}" "${logs[$i]}" &
  pids+=("$!")
done

exit_status=0
for i in "${!pids[@]}"; do
  status=0
  wait "${pids[$i]}" || status=$?

  echo "::group::parca-debuginfo upload ${binaries[$i]}"
  cat "${logs[$i]}"
  echo "::endgroup::"

  if [[ "$status" -ne 0 ]]; then
    echo "::error::Debuginfo upload failed for ${binaries[$i]} (exit $status)"
    exit_status="$status"
  fi
done

exit "$exit_status"
