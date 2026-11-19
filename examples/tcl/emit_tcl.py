# emit_tcl.py — NPU 실행을 하지 않고 TCL 소스를 생성한다.
from pathlib import Path
import furiosa.tcl as tcl
from kernels import matmul_smoke

# 이 실습의 계산 계약: (32,128) @ (128,64) -> (32,64)
sizes = {"M": 32, "K": 128, "N": 64}
# 축 이름에 실제 길이를 대입한다. 입력 값 자체는 아직 필요하지 않다.
specialized = tcl.specialize_kernel(matmul_smoke, sizes)
kernel = specialized if isinstance(specialized, tcl.Kernel) else tcl.Kernel.from_edsl(specialized)
# 컴파일러가 읽을 TCL 텍스트를 저장한다. 아직 EDF도 결과 텐서도 아니다.
Path("matmul.tc").write_text(kernel.dsl)
print("wrote matmul.tc", sizes)
