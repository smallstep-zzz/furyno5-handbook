# kernels.py — FuryNo5의 2026.4 계열 SDK 사용 방식에 맞춘 예제
from __future__ import annotations
import furiosa.tcl as tcl

@tcl.tensor_operation
def bf16_dot(left, right, output_axes):
    # 출력에 남길 축을 전달한다. matmul_smoke에서는 K축의 곱을 합산한다.
    wide = tcl.operation.contract(left, right, to=output_axes)
    # 계산 결과를 BF16 출력 저장 형식으로 변환하는 표현을 만든다.
    return tcl.commit.typecast(wide, tcl.bf16)

@tcl.kernel
def matmul_smoke(
    left: tcl.Tensor[tcl.bf16, [M, K]],   # 왼쪽: M행 K열
    right: tcl.Tensor[tcl.bf16, [K, N]],  # 오른쪽: K행 N열, 같은 K를 공유
) -> tcl.Tensor[tcl.bf16, [M, N]]:
    return bf16_dot(left, right, [M, N]) # 출력: M행 N열, K는 축약됨
