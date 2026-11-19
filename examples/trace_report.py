"""Summarize a FuryNo5/fxbX plan trace; diagnostic only, never acceptance."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def summarize(plan, trace, entry_name="forward"):
    steps = plan["entries"][entry_name]["steps"]
    prefix = "execution_plan::" + entry_name + "::"
    expected = {
        prefix + str(i) + "::" + step["kernel"]: (i, step)
        for i, step in enumerate(steps)
    }
    preparation = {
        label for label, (_, step) in expected.items()
        if (not step["inputs"] or plan["kernels"][step["kernel"]].get(
            "execution_kind") == "static_value_preparation")
    }
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    if not isinstance(events, list):
        raise ValueError("trace must be a list or contain traceEvents")
    spans = Counter()
    observed = Counter()
    duration_by_kernel = defaultdict(float)
    task_count_by_kernel = Counter()
    invalid_timing = []
    unexpected = set()
    cpu_routes = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        name = event.get("name", "")
        if not isinstance(name, str):
            continue
        if "run_by_cpu" in name:
            cpu_routes += 1
        if name in expected:
            spans[name] += 1
        marker = "::Task::"
        if marker not in name:
            continue
        label = name.split(marker, 1)[1]
        if not label.startswith(prefix):
            continue
        if label not in expected:
            unexpected.add(label)
            continue
        observed[label] += 1
        duration = event.get("dur")
        if (event.get("ph") != "X"
                or type(duration) not in (int, float)
                or not math.isfinite(duration) or duration <= 0):
            invalid_timing.append(label)
            continue
        key = expected[label][1]["kernel"]
        duration_by_kernel[key] += duration
        task_count_by_kernel[key] += 1
    uses = Counter(step["kernel"] for step in steps)
    work = [
        {
            "kernel": key,
            "planned_calls": uses[key],
            "observed_timed_tasks": task_count_by_kernel[key],
            "sum_task_us": duration,
        }
        for key, duration in duration_by_kernel.items()
    ]
    work.sort(key=lambda row: (-row["sum_task_us"], row["kernel"]))
    return {
        "schema": "furyno5-handbook.trace-diagnostic.v1",
        "entry": entry_name,
        "status": "diagnostic_only",
        "is_numerical_or_hardware_acceptance": False,
        "time_unit_assumption": "Chrome trace dur in microseconds; verify the adapter",
        "task_sum_is_wall_latency": False,
        "expected_steps": len(expected),
        "observed_step_task_labels": len(observed),
        "missing_host_spans": sorted(set(expected) - set(spans)),
        "missing_compute_task_labels": sorted(set(expected) - preparation - set(observed)),
        "preparation_labels_without_task": sorted(preparation - set(observed)),
        "unexpected_task_labels": sorted(unexpected),
        "invalid_timing_labels": sorted(set(invalid_timing)),
        "observed_cpu_route_markers": cpu_routes,
        "kernel_task_work": work,
        "limitations": [
            "Names are adapter-specific and can change across versions or merged modes.",
            "Events outside the selected entry are not attributed here.",
            "Missing labels may mean missing instrumentation; this tool cannot decide.",
            "No tensor, EDF integrity, state, or numerical checks are performed.",
            "Overlapping tasks contribute independently to task work.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--entry", default="forward")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.plan.read_text()),
        json.loads(args.trace.read_text()),
        args.entry,
    )
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "kernel_task_work"}, ensure_ascii=False, indent=2))
    for row in result["kernel_task_work"][:10]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

