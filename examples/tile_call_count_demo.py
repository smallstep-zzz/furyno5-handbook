row_tiles = [list(range(start, start + 2)) for start in range(0, 8, 2)]
candidate_tiles = [list(range(start, start + 2)) for start in range(0, 6, 2)]
separate_calls = [(rows, candidates) for rows in row_tiles for candidates in candidate_tiles]
# 한 행 타일에서 후보 타일 3개의 내용을 이어서 다루는 논리적 호출 목록
all_candidates = [c for tile in candidate_tiles for c in tile]
grouped_calls = [(rows, all_candidates) for rows in row_tiles]

def comparison_count(calls):
    return sum(len(rows) * len(candidates) for rows, candidates in calls)

assert len(separate_calls) == 12 and len(grouped_calls) == 4
assert comparison_count(separate_calls) == comparison_count(grouped_calls) == 48
print(len(separate_calls), len(grouped_calls), comparison_count(grouped_calls))  # 12 4 48
