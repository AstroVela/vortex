#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Run a command and record peak Linux resource usage for its process tree."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class ForwardedSignal(Exception):
    def __init__(self, signum):
        self.signum = signum


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=1.0, help="sample interval in seconds")
    parser.add_argument("--output", type=Path, required=True, help="final JSON summary")
    parser.add_argument("--samples", type=Path, help="optional JSON Lines samples")
    parser.add_argument("--log", type=Path,
                        help="child stdout/stderr log (default: next to --output with .log suffix)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    return args


def process_stats(root_pid):
    processes = {}
    parents = {}
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().split()
            pid = int(stat[0])
            parents[pid] = int(stat[3])
            processes[pid] = stat
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True

    clock_ticks = os.sysconf("SC_CLK_TCK")
    page_size = os.sysconf("SC_PAGE_SIZE")
    cpu_seconds = 0.0
    rss_bytes = 0
    read_bytes = 0
    write_bytes = 0
    live_pids = []
    for pid in descendants:
        stat = processes.get(pid)
        if stat is None:
            continue
        live_pids.append(pid)
        cpu_seconds += (int(stat[13]) + int(stat[14])) / clock_ticks
        try:
            rss_bytes += int((proc / str(pid) / "statm").read_text().split()[1]) * page_size
        except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError, ValueError):
            pass
        try:
            io = dict(
                line.split(":", 1) for line in (proc / str(pid) / "io").read_text().splitlines()
            )
            read_bytes += int(io.get("read_bytes", 0))
            write_bytes += int(io.get("write_bytes", 0))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            pass
    return {
        "cpu_seconds": cpu_seconds,
        "rss_bytes": rss_bytes,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "processes": len(live_pids),
    }


def network_bytes():
    received = 0
    transmitted = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        interface, counters = line.split(":", 1)
        if interface.strip() == "lo":
            continue
        fields = counters.split()
        received += int(fields[0])
        transmitted += int(fields[8])
    return received, transmitted


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def signal_handler(signum, _frame):
    raise ForwardedSignal(signum)


def safe_print(*values, **kwargs):
    try:
        print(*values, **kwargs)
    except BrokenPipeError:
        sys.stdout = open(os.devnull, "w")


def main():
    args = parse_args()
    args.output = args.output.resolve()
    if args.log is None:
        args.log = args.output.with_suffix(".log")
    else:
        args.log = args.log.resolve()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.samples:
        args.samples = args.samples.resolve()
        args.samples.parent.mkdir(parents=True, exist_ok=True)

    started_wall = time.time()
    started = time.monotonic()
    # Keep the child in its own process group so this wrapper can always write
    # the final summary before forwarding an interactive interruption.
    log_file = args.log.open("a", buffering=1)
    log_file.write(
        f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started_wall))}] "
        f"starting: {subprocess.list2cmdline(args.command)}\n"
    )
    process = subprocess.Popen(
        args.command,
        start_new_session=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    previous_handlers = {
        signum: signal.signal(signum, signal_handler)
        for signum in (signal.SIGHUP, signal.SIGTERM)
    }
    previous_time = started
    previous_process = process_stats(process.pid)
    previous_network = network_bytes()
    peaks = {
        "cpu_percent": 0.0,
        "rss_bytes": previous_process["rss_bytes"],
        "disk_read_bytes_per_second": 0.0,
        "disk_write_bytes_per_second": 0.0,
        "network_receive_bytes_per_second": 0.0,
        "network_transmit_bytes_per_second": 0.0,
        "processes": previous_process["processes"],
    }
    sample_count = 0
    samples_file = args.samples.open("w") if args.samples else None
    interrupted = False
    received_signal = None
    try:
        while process.poll() is None:
            time.sleep(args.interval)
            now = time.monotonic()
            elapsed = max(now - previous_time, 1e-9)
            current_process = process_stats(process.pid)
            current_network = network_bytes()
            sample = {
                "elapsed_seconds": now - started,
                "cpu_percent": max(
                    0.0,
                    (current_process["cpu_seconds"] - previous_process["cpu_seconds"])
                    / elapsed
                    * 100,
                ),
                "rss_bytes": current_process["rss_bytes"],
                "disk_read_bytes_per_second": max(
                    0.0,
                    (current_process["read_bytes"] - previous_process["read_bytes"]) / elapsed,
                ),
                "disk_write_bytes_per_second": max(
                    0.0,
                    (current_process["write_bytes"] - previous_process["write_bytes"]) / elapsed,
                ),
                "network_receive_bytes_per_second": max(
                    0.0, (current_network[0] - previous_network[0]) / elapsed
                ),
                "network_transmit_bytes_per_second": max(
                    0.0, (current_network[1] - previous_network[1]) / elapsed
                ),
                "processes": current_process["processes"],
            }
            for key, value in sample.items():
                if key != "elapsed_seconds":
                    peaks[key] = max(peaks[key], value)
            if samples_file:
                samples_file.write(json.dumps(sample, sort_keys=True) + "\n")
                samples_file.flush()
            sample_count += 1
            atomic_json(
                args.output,
                {
                    "command": args.command,
                    "running": True,
                    "exit_code": None,
                    "started_at_unix_seconds": started_wall,
                    "elapsed_seconds": now - started,
                    "sample_interval_seconds": args.interval,
                    "sample_count": sample_count,
                    "cpu_percent_scale": "100 percent per fully utilized logical CPU",
                    "network_scope": "host interfaces excluding loopback",
                    "peaks": peaks,
                },
            )
            previous_time = now
            previous_process = current_process
            previous_network = current_network
    except (KeyboardInterrupt, ForwardedSignal) as error:
        interrupted = True
        received_signal = error.signum if isinstance(error, ForwardedSignal) else signal.SIGINT
        if process.poll() is None:
            os.killpg(process.pid, received_signal)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
    finally:
        if samples_file:
            samples_file.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        log_file.close()

    if process.poll() is None:
        process.wait()

    result = {
        "command": args.command,
        "running": False,
        "exit_code": process.returncode,
        "interrupted": interrupted,
        "received_signal": received_signal,
        "log": str(args.log),
        "started_at_unix_seconds": started_wall,
        "elapsed_seconds": time.monotonic() - started,
        "sample_interval_seconds": args.interval,
        "sample_count": sample_count,
        "cpu_percent_scale": "100 percent per fully utilized logical CPU",
        "network_scope": "host interfaces excluding loopback",
        "peaks": peaks,
    }
    atomic_json(args.output, result)
    safe_print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
