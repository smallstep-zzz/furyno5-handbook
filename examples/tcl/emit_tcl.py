# emit_tcl.py — NPU 실행을 하지 않고 TCL 소스를 생성한다.
from pathlib import Path
import furiosa.tcl as tcl
from kernels import matmul_smoke

sizes = {"M": 32, "K": 128, "N": 64}
specialized = tcl.specialize_kernel(matmul_smoke, sizes)
kernel = specialized if isinstance(specialized, tcl.Kernel) else tcl.Kernel.from_edsl(specialized)
Path("matmul.tc").write_text(kernel.dsl)
print("wrote matmul.tc", sizes)
