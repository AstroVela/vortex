#!/usr/bin/env python3
"""Summarize self-paced executor trace logs and flag control-plane overhead."""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")


def fields(line: str) -> dict[str, str]:
    return {key: value.rstrip(",") for key, value in KV_RE.findall(line)}


def integer(values: dict[str, str], key: str) -> int:
    return int(values.get(key, "0"))


def stats_ns(values: list[int]) -> str:
    if not values:
        return "n=0"
    ordered = sorted(values)
    p90 = ordered[round((len(ordered) - 1) * 0.90)]
    return (
        f"n={len(values)} total_us={sum(values) / 1_000:.1f} "
        f"p50_us={statistics.median(values) / 1_000:.1f} "
        f"p90_us={p90 / 1_000:.1f} max_us={max(values) / 1_000:.1f}"
    )


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    events: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    task_latency: dict[str, list[int]] = defaultdict(list)
    wait_latency: list[int] = []
    advance_transitions: list[int] = []
    advance_updates: list[int] = []
    summary: dict[str, str] = {}
    elapsed_us = 0.0

    with args.log.open(encoding="utf-8") as source:
        for line in source:
            values = fields(line)
            if line.startswith("trace_end "):
                summary = values
                continue
            event = values.get("event")
            if event is None:
                continue
            events[event] += 1
            if "t_us" in values:
                elapsed_us = max(elapsed_us, float(values["t_us"]))
            operation = values.get("operation")
            if event == "claim" and operation:
                operations[operation] += 1
            if "task_latency_ns" in values and operation:
                task_latency[operation].append(integer(values, "task_latency_ns"))
            if event == "wait_end" and values.get("reason") == "task_completion":
                wait_latency.append(integer(values, "wait_latency_ns"))
            if event == "advance":
                advance_transitions.append(integer(values, "transitions"))
                advance_updates.append(integer(values, "updates"))

    if not summary:
        print("No trace_end summary found.")
        return 1

    completed = integer(summary, "tasks_completed")
    advances = integer(summary, "advance_calls")
    transitions = integer(summary, "transitions")
    inspected = integer(summary, "nodes_inspected")
    wake_candidates = integer(summary, "completion_wake_candidates_inspected")
    passes = integer(summary, "scheduler_passes")
    considered = integer(summary, "scheduler_tasks_considered")
    admitted = integer(summary, "scheduler_tasks_admitted")
    batches = integer(summary, "completion_batches")
    drained = integer(summary, "completions_drained")

    print(f"elapsed_us\t{elapsed_us:.1f}")
    print(f"tasks\toffered={summary.get('tasks_offered', '?')} claimed={summary.get('tasks_claimed', '?')} completed={completed}")
    print(
        "reactor\t"
        f"advances={advances} transitions={transitions} inspected={inspected} "
        f"completion_wake_candidates={wake_candidates} "
        f"advances/task={ratio(advances, completed):.2f} "
        f"inspected/transition={ratio(inspected, transitions):.2f}"
    )
    print(
        "scheduler\t"
        f"passes={passes} considered={considered} admitted={admitted} "
        f"considered/admitted={ratio(considered, admitted):.2f}"
    )
    print(
        "completions\t"
        f"batches={batches} drained={drained} max_batch={summary.get('max_completion_batch', '?')} "
        f"completions/batch={ratio(drained, batches):.2f}"
    )
    print(f"waits\t{stats_ns(wait_latency)}")

    print("operations")
    for operation, count in operations.most_common():
        print(f"  {operation}\tclaims={count} latency={stats_ns(task_latency[operation])}")

    if advance_transitions:
        print(
            "advance_work\t"
            f"transitions_p50={statistics.median(advance_transitions):.1f} "
            f"updates_p50={statistics.median(advance_updates):.1f}"
        )

    signals = []
    if drained > 1 and integer(summary, "max_completion_batch") <= 1:
        signals.append("completions are consumed one at a time")
    if admitted and considered >= admitted * 4:
        signals.append("the scheduler rescans at least 4x more tasks than it admits")
    if completed and advances >= completed * 3:
        signals.append("reactor advances are at least 3x completed tasks")
    if transitions and inspected >= transitions * 4:
        signals.append("resource inspection is at least 4x productive transitions")
    if wait_latency and sum(wait_latency) > elapsed_us * 1_000 * 0.5:
        signals.append("recorded task waits exceed half of elapsed wall time")

    print("signals")
    if signals:
        for signal in signals:
            print(f"  - {signal}")
    else:
        print("  - no configured control-plane threshold fired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
