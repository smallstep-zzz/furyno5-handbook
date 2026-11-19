#!/usr/bin/env python3
"""입력 길이에 맞는 버킷과 패딩 수를 비교하는 학습 예제.

모양의 낭비만 계산하며 실제 RNGD 실행 시간은 예측하지 않는다.
"""
import argparse
import json
from bisect import bisect_left

def compare(lengths, buckets):
    # 오름차순의 고유 버킷만 남긴다. 이 순서가 이진 탐색의 전제다.
    candidates = sorted(set(buckets))
    if not candidates or candidates[0] <= 0:
        raise ValueError("buckets must contain positive integers")
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("lengths must contain positive integers")
    rows = []
    for n in lengths:
        # n 이상인 첫 버킷의 위치를 찾는다. 입력보다 작은 버킷에는 넣을 수 없다.
        i = bisect_left(candidates, n)
        bucket = candidates[i] if i < len(candidates) else None
        rows.append({"length": n, "bucket": bucket,
                     "padding": None if bucket is None else bucket - n})
    covered = [r for r in rows if r["bucket"] is not None]  # 처리 가능한 요청만 집계한다.
    # 실제 길이의 합이 아니라, 선택한 버킷이 제공한 전체 칸 수다.
    total_slots = sum(r["bucket"] for r in covered)
    return {"buckets": candidates, "rows": rows,
            "uncovered_requests": len(rows) - len(covered),
            "padding_fraction_for_covered": (
                sum(r["padding"] for r in covered) / total_slots
                if total_slots else None),
            "warning": "Illustrative length-only model; not an RNGD runtime or benchmark."}

def positive_list(value):
    # CLI 문자열 '128,256,512'를 양의 정수 목록 [128,256,512]로 바꾼다.
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
