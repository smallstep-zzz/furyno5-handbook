#!/usr/bin/env python3
"""Educational length-bucket comparison; no hardware timing is predicted."""
import argparse
import json
from bisect import bisect_left

def compare(lengths, buckets):
    candidates = sorted(set(buckets))
    if not candidates or candidates[0] <= 0:
        raise ValueError("buckets must contain positive integers")
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("lengths must contain positive integers")
    rows = []
    for n in lengths:
        i = bisect_left(candidates, n)
        bucket = candidates[i] if i < len(candidates) else None
        rows.append({"length": n, "bucket": bucket,
                     "padding": None if bucket is None else bucket - n})
    covered = [r for r in rows if r["bucket"] is not None]
    total_slots = sum(r["bucket"] for r in covered)
    return {"buckets": candidates, "rows": rows,
            "uncovered_requests": len(rows) - len(covered),
            "padding_fraction_for_covered": (
                sum(r["padding"] for r in covered) / total_slots
                if total_slots else None),
            "warning": "Illustrative length-only model; not an RNGD runtime or benchmark."}

def positive_list(value):
    try:
        result = [int(x.strip()) for x in value.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError("use comma-separated integers") from e
    if not result or any(n <= 0 for n in result):
        raise argparse.ArgumentTypeError("integers must be positive")
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=positive_list, default=[120, 129, 500, 900, 2049])
    parser.add_argument('--buckets', type=positive_list, default=[128, 256, 512, 1024, 2048])
    args = parser.parse_args()
    print(json.dumps(compare(args.lengths, args.buckets), indent=2))
