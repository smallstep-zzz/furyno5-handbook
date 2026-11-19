# run_edf.py — NPU와 호환 SDK가 필요하다.
from pathlib import Path
import torch
import furiosa.torch as sdk

# 검증 중에는 다른 실행 경로가 실패를 가리지 않도록 CPU 폴백을 끈다.
sdk.config.set_cpu_fallback(False)
# 이 예제는 8-PE(Processing Element) 장치 구성으로 컴파일했다.
# 여기의 fusion 수는 아래 8절에서 설명하는 연산 결합과 구분한다.
sdk.set_fusion(8)
sdk.set_mesh_kind(sdk.MeshKind.Single())

# EDF 바이트를 프로그램 표현으로 복원한다. 이 호출만으로 NPU 계산이 끝나지 않는다.
ir = sdk.ir.Runnable.from_serialized_edf(Path("matmul.edf").read_bytes())
if ir.expected_num_fusion != 8:
    raise RuntimeError("EDF fusion contract does not match the requested device")
if len(ir.input_dram_shapes) != 2 or len(ir.output_dram_shapes) != 1:
    raise RuntimeError("Unexpected EDF input/output arity")
print("ABI input shapes:", ir.input_dram_shapes)
print("ABI output shapes:", ir.output_dram_shapes)

# 실제 장치에서 호출할 객체를 만든다. 같은 객체를 반복 호출에 재사용할 수 있다.
program = sdk.Runnable(ir).to("furiosa:0")
# 컴파일 때 정한 shape와 dtype에 맞춘다. 출력의 각 원소는 1을 128번 더한 128이다.
left = torch.ones((32, 128), dtype=torch.bfloat16)
right = torch.ones((128, 64), dtype=torch.bfloat16)
# CPU 텐서의 값을 장치 텐서로 준비한다. 전송 비용은 계산 비용과 따로 본다.
left_device = left.to("furiosa:0")
right_device = right.to("furiosa:0")
with torch.inference_mode():
    # 입력 두 개를 선언 순서대로 전달한다. 실제 계산 결과의 shape는 (32,64)다.
    value = program(left_device, right_device)
if isinstance(value, (tuple, list)):
    if len(value) != 1:
        raise RuntimeError("Expected one result")
    value = value[0]
# CPU에서 결과를 읽을 수 있게 가져온 뒤 기준값과 비교한다.
actual = value.cpu()
torch.testing.assert_close(actual, left @ right, rtol=0, atol=0)
print("single-shape all-ones smoke check passed")
