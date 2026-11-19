from math import exp, isclose, sqrt

def transpose(matrix):
    return [list(column) for column in zip(*matrix)]  # [행,열] → [열,행]

def matmul(left, right):
    assert len(left[0]) == len(right)  # 곱하며 더할 축의 길이가 같아야 한다.
    columns = transpose(right)
    return [
        [sum(a * b for a, b in zip(row, column)) for column in columns]
        for row in left
    ]

def softmax_row(scores):
    maximum = max(scores)
    positive = [exp(score - maximum) for score in scores]  # 모든 항에 같은 기준 사용
    total = sum(positive)
    return [value / total for value in positive]  # key 축 전체의 가중치 합을 1로 만듦

q = [[1.0]]  # [Lq=1,d=1]
k = [[1.0], [0.0]]  # [Lk=2,d=1]
v = [[10.0], [20.0]]  # [Lk=2,dv=1]
raw_scores = matmul(q, transpose(k))  # [1,1] @ [1,2] → [1,2]
scores = [[value / sqrt(len(q[0])) for value in row] for row in raw_scores]
weights = [softmax_row(row) for row in scores]  # 질의마다 key 축으로 정규화
full_output = matmul(weights, v)[0][0]  # [1,2] @ [2,1] → [1,1]의 한 값

def online_one_row(score_values):
    maximum, total, weighted = float("-inf"), 0.0, 0.0
    for score, value in score_values:
        new_maximum = max(maximum, score)
        old_scale = 0.0 if total == 0.0 else exp(maximum - new_maximum)
        new_weight = exp(score - new_maximum)
        total = total * old_scale + new_weight  # 이전 분모를 새 최대값 기준으로 환산
        weighted = weighted * old_scale + new_weight * value  # 분자에도 같은 배율
        maximum = new_maximum
    if total == 0.0:
        raise ValueError("이 예제에는 최소 한 key가 필요합니다.")
    return weighted / total

pairs = list(zip(scores[0], [row[0] for row in v]))
forward = online_one_row(pairs)  # 점수 1 다음 0: 최대값 유지
reversed_order = online_one_row(list(reversed(pairs)))  # 점수 0 다음 1: 최대값 변경
assert isclose(full_output, forward, rel_tol=1e-12)
assert isclose(full_output, reversed_order, rel_tol=1e-12)
assert isclose(sum(weights[0]), 1.0, rel_tol=1e-12)
print(scores, [round(x, 6) for x in weights[0]])  # [[1.0,0.0]], [0.731059,0.268941]
print(*(round(x, 6) for x in (full_output, forward, reversed_order)))
# 12.689414 12.689414 12.689414
