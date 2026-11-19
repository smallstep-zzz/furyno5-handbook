from math import isclose

observed = [-0.9, -0.2, 0.3, 0.9]  # 설명용 calibration activation, shape [4]
qmin, qmax = -4, 3  # 두 후보에서 같은 표현 범위를 사용
results = []
for scale in (0.25, 0.5):
    rounded = [round(x / scale) for x in observed]  # zero point=0
    codes = [min(qmax, max(qmin, q)) for q in rounded]  # 끝값으로 제한
    restored = [scale * q for q in codes]  # 같은 실수 단위로 되돌림
    clipped = sum(q != original_q for q, original_q in zip(codes, rounded))
    mse = sum((got - ref) ** 2 for got, ref in zip(restored, observed)) / len(observed)
    results.append((scale, mse))
    print(scale, clipped, round(mse, 6))
best_scale, _ = min(results, key=lambda item: item[1])  # 두 후보 중 작은 MSE 선택
assert best_scale == 0.25
assert isclose(results[0][1], 0.009375)
assert isclose(results[1][1], 0.025)
# 0.25 1 0.009375
# 0.5 0 0.025
