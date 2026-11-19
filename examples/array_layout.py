"""02장 reshape·transpose·stride·broadcasting 기초. NumPy CPU 예제.

실행: python examples/array_layout.py
예시 배열의 값·모양·저장 공간을 직접 출력하고 확인한다.
"""

import numpy as np  # CPU에서 숫자 배열과 메모리 표현을 살펴보기 위해 사용한다.

# arange(start, stop)는 start부터 stop 직전까지 만든다. 따라서 값은 1~6이다.
# float32는 원소 하나가 4바이트인 부동소수점 형식이다.
flat = np.arange(1, 7, dtype=np.float32)

# 6개 값을 2행×3열로 해석한다. 이 연속 배열에서는 원래 저장 공간을 공유한다.
x = flat.reshape(2, 3)

# 같은 6개를 3행×2열로 해석할 수도 있다. 원소의 총수 6은 유지된다.
reshaped = flat.reshape(3, 2)
print(x)          # [[1., 2., 3.], [4., 5., 6.]]
print(reshaped)   # [[1., 2.], [3., 4.], [5., 6.]]

# 앞 절의 x는 [[1,2,3],[4,5,6]], shape는 (2, 3)이다.
xt = x.T  # 행·열의 역할을 교환한다: (2, 3) → (3, 2).
print(xt)
# [[1., 4.],
#  [2., 5.],
#  [3., 6.]]

# ascontiguousarray는 xt의 '논리적 값'을 유지하며 C 순서로 연속 배치한다.
# 이 xt는 C 순서 연속 배열이 아니므로 여기서는 실제 복사가 필요하다.
packed = np.ascontiguousarray(xt)

for name, value in [("x", x), ("xt", xt), ("packed", packed)]:
    # shape: 축 길이, strides: 축별 주소 증가 바이트 수,
    # c_contiguous: 행부터 읽는 C 순서로 연속 배치됐는지 확인한다.
    print(name, value.shape, value.strides, value.flags.c_contiguous)

# 값은 같아야 한다. 배열의 저장 방법만 달라지는 실험이다.
np.testing.assert_array_equal(xt, packed)
# 이 예에서 xt는 x의 저장 공간을 공유하지만 packed는 별도 공간을 사용한다.
print(np.shares_memory(x, xt), np.shares_memory(x, packed))  # True, False

bias = np.array([10, 20, 30], dtype=np.float32)  # 열마다 더할 값, shape (3,).
y = x + bias  # 두 행에 같은 bias를 적용한다. 결과 shape는 (2, 3)이다.
print(y)      # [[11., 22., 33.], [14., 25., 36.]]

# shape가 (3, 2)로 같아도 reshape와 전치는 서로 다른 값을 배치한다.
np.testing.assert_array_equal(reshaped, [[1, 2], [3, 4], [5, 6]])
np.testing.assert_array_equal(xt, [[1, 4], [2, 5], [3, 6]])
assert not np.array_equal(reshaped, xt)

# float32는 4바이트다. x에서 다음 행은 3개, 다음 열은 1개 원소를 건너뛴다.
assert x.strides == (12, 4)
assert xt.strides == (4, 12)
assert packed.strides == (8, 4)
assert np.shares_memory(x, xt)
assert not np.shares_memory(x, packed)
np.testing.assert_array_equal(y, [[11, 22, 33], [14, 25, 36]])

# -1은 원소 수로부터 길이를 계산한다. 총 6개를 두 열로 나누므로 세 행이다.
inferred = flat.reshape(-1, 2)
np.testing.assert_array_equal(inferred, reshaped)

# reshape는 원소 수를 보존한다. 여섯 개를 여덟 칸에 넣는 요청은 실패해야 한다.
try:
    flat.reshape(4, 2)
except ValueError:
    print("예상한 오류: 6개 값을 shape (4, 2)의 8칸으로 바꿀 수 없습니다.")
else:
    raise AssertionError("원소 수가 다른 reshape가 예상과 달리 성공했습니다.")
print("배열 값·stride·메모리 공유·broadcasting 결과를 확인했습니다.")
