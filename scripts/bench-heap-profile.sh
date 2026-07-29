#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -Eeu -o pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 2
fi

if [[ -z "${POLARSIGNALS_CLOUD_TOKEN:-}" ]]; then
    exec "$@"
fi

: "${POLARSIGNALS_PROJECT_ID:?POLARSIGNALS_PROJECT_ID is required when heap profiling}"
: "${HEAP_PROFILE_BENCHMARK:?HEAP_PROFILE_BENCHMARK is required when heap profiling}"
: "${HEAP_PROFILE_ENGINE:?HEAP_PROFILE_ENGINE is required when heap profiling}"
: "${HEAP_PROFILE_FORMAT:?HEAP_PROFILE_FORMAT is required when heap profiling}"

parca_version="0.28.0"
commit_sha="${HEAP_PROFILE_COMMIT_SHA:-${GITHUB_SHA:-unknown}}"
branch="${GITHUB_REF_NAME:-unknown}"
run_id="${GITHUB_RUN_ID:-local}"
job="${GITHUB_JOB:-benchmark}"
engine="$HEAP_PROFILE_ENGINE"
format="$HEAP_PROFILE_FORMAT"

branch="${branch//;/_}"
branch="${branch//=/_}"
job="${job//;/_}"
job="${job//=/_}"
engine="${engine//;/_}"
engine="${engine//=/_}"
format="${format//;/_}"
format="${format//=/_}"

profile_tmp_root="${RUNNER_TEMP:-/tmp}"
profile_tmp_dir="$(mktemp -d "$profile_tmp_root/vortex-heap-profile.XXXXXX")"
config_path="$profile_tmp_dir/parca.yaml"
token_path="$profile_tmp_dir/token"
log_path="$profile_tmp_dir/parca.log"
archive_path="$profile_tmp_dir/parca.tar.gz"
data_path="$profile_tmp_dir/data"

# shellcheck disable=SC2329  # Invoked by the EXIT trap.
cleanup() {
    if [[ -n "${scraper_pid:-}" ]]; then
        kill "$scraper_pid" 2>/dev/null || true
        for _ in {1..20}; do
            if ! kill -0 "$scraper_pid" 2>/dev/null; then
                break
            fi
            sleep 0.25
        done
        kill -KILL "$scraper_pid" 2>/dev/null || true
        wait "$scraper_pid" 2>/dev/null || true
    fi
    rm -r -- "$profile_tmp_dir" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf "%s" "$POLARSIGNALS_CLOUD_TOKEN" > "$token_path"
chmod 0400 "$token_path"
mkdir "$data_path"
project_id="$POLARSIGNALS_PROJECT_ID"
unset POLARSIGNALS_CLOUD_TOKEN POLARSIGNALS_PROJECT_ID

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Parca heap scraping is supported only on Linux" >&2
    exit 1
fi

case "$(uname -m)" in
    x86_64)
        parca_arch="x86_64"
        parca_sha256="f08a2ecbe4490b539107d4aa4b9aae8362bf88e09e9e099c264c734eff9d4dd4"
        ;;
    aarch64 | arm64)
        parca_arch="arm64"
        parca_sha256="e2d4a43daf1f6050fd0f3334016a750b93f1013f333b5e138abf79e6955250f6"
        ;;
    *)
        echo "Unsupported architecture for Parca heap scraping: $(uname -m)" >&2
        exit 1
        ;;
esac

parca_asset="parca_${parca_version}_Linux_${parca_arch}.tar.gz"
parca_url="https://github.com/parca-dev/parca/releases/download/v${parca_version}/${parca_asset}"
parca_cache_dir="$profile_tmp_root/vortex-parca-cache"
cached_archive_path="$parca_cache_dir/$parca_asset"

mkdir -p "$parca_cache_dir"
if [[ ! -f "$cached_archive_path" ]] \
    || ! printf '%s  %s\n' "$parca_sha256" "$cached_archive_path" \
        | sha256sum --check --status -
then
    curl --fail --location --retry 3 --retry-all-errors \
        --output "$archive_path" \
        "$parca_url"
    printf '%s  %s\n' "$parca_sha256" "$archive_path" | sha256sum --check --status -
    mv "$archive_path" "$cached_archive_path"
fi

archive_path="$cached_archive_path"
tar -xzf "$archive_path" -C "$profile_tmp_dir"
parca_path="$(find "$profile_tmp_dir" -type f -name parca -print -quit)"
if [[ -z "$parca_path" ]]; then
    echo "The Parca release archive did not contain a parca executable" >&2
    exit 1
fi
chmod +x "$parca_path"

cat > "$config_path" <<YAML
object_storage:
  bucket:
    type: "FILESYSTEM"
    config:
      directory: "$data_path"
scrape_configs:
  - job_name: "vortex-benchmark-heap"
    scrape_interval: "5s"
    static_configs:
      - targets:
          - "127.0.0.1:6060"
    profiling_config:
      pprof_config:
        memory:
          enabled: true
          path: "/debug/pprof/allocs"
          keep_sample_type:
            - type: "inuse_space"
              unit: "bytes"
        process_cpu:
          enabled: false
        block:
          enabled: false
        goroutine:
          enabled: false
        mutex:
          enabled: false
YAML

external_labels="branch=$branch;gh_run_id=$run_id;commit_sha=$commit_sha;benchmark=$HEAP_PROFILE_BENCHMARK;engine=$engine;format=$format;gh_job=$job"

parca_command=("$parca_path")
if [[ -f /tmp/vortex-benchmark.env ]]; then
    # shellcheck disable=SC1091
    source /tmp/vortex-benchmark.env
    if [[ -n "${HOUSEKEEPING_CPUS:-}" ]] && command -v taskset >/dev/null 2>&1; then
        parca_command=(taskset --cpu-list "$HOUSEKEEPING_CPUS" "$parca_path")
    fi
fi

"${parca_command[@]}" \
    --config-path="$config_path" \
    --store-address=grpc.polarsignals.com:443 \
    --bearer-token-file="$token_path" \
    "--grpc-headers=projectID=$project_id" \
    "--external-label=$external_labels" \
    --mode=scraper-only \
    --http-address=127.0.0.1:17071 \
    > "$log_path" 2>&1 &
scraper_pid=$!

for _ in {1..40}; do
    if ! kill -0 "$scraper_pid" 2>/dev/null; then
        wait "$scraper_pid" || status=$?
        scraper_pid=
        cat "$log_path" >&2
        if [[ "${status:-0}" -eq 0 ]]; then
            status=1
        fi
        exit "${status:-1}"
    fi
    if curl --silent --output /dev/null --connect-timeout 1 \
        http://127.0.0.1:17071/
    then
        scraper_ready=true
        break
    fi
    sleep 0.25
done

if [[ "${scraper_ready:-false}" != "true" ]]; then
    echo "Parca heap scraper did not start" >&2
    cat "$log_path" >&2
    exit 1
fi

set +e
"$@"
benchmark_status=$?
set -e

if ! kill -0 "$scraper_pid" 2>/dev/null; then
    set +e
    wait "$scraper_pid"
    scraper_status=$?
    set -e
    scraper_pid=
    cat "$log_path" >&2
    if [[ "$benchmark_status" -eq 0 ]]; then
        if [[ "$scraper_status" -eq 0 ]]; then
            scraper_status=1
        fi
        exit "$scraper_status"
    fi
fi

exit "$benchmark_status"
