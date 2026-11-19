# kernels.py — FuryNo5의 2026.4 계열 SDK 사용 방식에 맞춘 예제
from __future__ import annotations
import furiosa.tcl as tcl

@tcl.tensor_operation
def bf16_dot(left, right, output_axes):
    wide = tcl.operation.contract(left, right, to=output_axes)
    return tcl.commit.typecast(wide, tcl.bf16)

@tcl.kernel
def matmul_smoke(
    left: tcl.Tensor[tcl.bf16, [M, K]],
    right: tcl.Tensor[tcl.bf16, [K, N]],
) -> tcl.Tensor[tcl.bf16, [M, N]]:
    return bf16_dot(left, right, [M, N])
