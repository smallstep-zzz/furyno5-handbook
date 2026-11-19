"""01장 행렬곱·einsum 기초. NumPy만 사용하는 CPU 학습 예제.

실행: python examples/tensor_basics.py
NPU 컴파일·실행·성능 측정은 수행하지 않는다.
"""

import numpy as np  # 숫자 배열을 계산하는 NumPy를 np라는 짧은 이름으로 사용한다.

# 두 입력 행, 각 행에 세 값: shape는 (2, 3)이다.
# float32는 값 하나를 32비트 부동소수점으로 저장한다는 뜻이다.
A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float32)
# 각 열이 출력 하나를 만드는 계수다. 첫 열은 첫 값+셋째 값,
# 둘째 열은 둘째 값+셋째 값을 선택하므로 shape는 (3, 2)다.
W = np.array([[1, 0],
              [0, 1],
              [1, 1]], dtype=np.float32)
C = A @ W  # 행렬곱: (2, 3) @ (3, 2) → (2, 2). 각 칸은 행·열의 내적이다.
print(C)
# [[ 4.  5.]
#  [10. 11.]]

M, K = A.shape       # M=2: 입력 행 수, K=3: 한 행에서 곱해 더할 항 수.
K_of_W, N = W.shape  # W의 행 수는 K와 같아야 한다. N=2는 출력 열 수다.
assert K == K_of_W   # 내적의 두 벡터 길이가 일치하는지 확인한다.
C_loop = np.zeros((M, N), dtype=np.float32)  # 결과 (2, 2)를 0으로 채워 준비한다.

for m in range(M):             # 입력 행을 하나 고른다: m=0, 1.
    for n in range(N):         # 만들 출력 열을 고른다: n=0, 1.
        total = 0.0           # 출력 칸마다 합계를 새로 시작한다.
        for k in range(K):    # 같은 위치끼리 곱할 세 쌍을 순서대로 방문한다.
            total += A[m, k] * W[k, n]  # 곱 하나를 현재 합계에 더한다.
        C_loop[m, n] = total  # 세 항을 모두 더한 뒤 출력 한 칸을 저장한다.

np.testing.assert_allclose(C_loop, C)  # @ 결과와 부동소수점 허용오차 안에서 같은지 검사한다.

# m,n은 결과에 남기고 k는 곱한 뒤 합산한다. 결과 shape는 (2, 2)다.
C_einsum = np.einsum("mk,kn->mn", A, W)
np.testing.assert_allclose(C_einsum, C)  # 문자열로 적은 계산도 앞의 @와 같은지 확인한다.

# *는 같은 위치끼리 곱한다. 행렬곱과 달리 출력 shape는 그대로 (2, 3)이다.
elementwise = A * A
np.testing.assert_array_equal(elementwise, [[1, 4, 9], [16, 25, 36]])

# k가 출력 표기에서 빠졌으므로 한 행의 세 값을 더한다. 결과 shape는 (2,)다.
row_sums = np.einsum("mk->m", A)
np.testing.assert_array_equal(row_sums, [6, 15])

# 두 입력에 k가 반복돼도 출력에 k를 남기면 그 축을 합산하지 않는다.
kept_axis = np.einsum("mk,mk->mk", A, A)
np.testing.assert_array_equal(kept_axis, elementwise)

# [:, :2]는 모든 행에서 앞의 두 열을 고른다. [2:, :]는 2번 행부터 끝까지다.
# K를 앞의 두 항과 마지막 항으로 나누면 두 결과 모두 shape (2, 2)의 부분합이다.
partial_first = A[:, :2] @ W[:2, :]
partial_last = A[:, 2:] @ W[2:, :]
np.testing.assert_allclose(partial_first + partial_last, C)
print("K를 나눈 첫 부분합:", partial_first, sep="\n")
print("K를 나눈 마지막 부분합:", partial_last, sep="\n")
print("행렬곱·세 반복문·einsum·부분합의 결과가 일치합니다.")
