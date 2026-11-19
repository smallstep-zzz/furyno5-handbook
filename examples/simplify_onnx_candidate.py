from importlib.metadata import version
from pathlib import Path
import onnx
from onnxsim import simplify

model = onnx.load("input.onnx")  # 원본은 읽고 후보는 다른 파일에 저장한다.
onnx.checker.check_model(model)  # 그래프 형식·연결 검사는 수치 정답 검증과 별개다.
weights = {tensor.name for tensor in model.graph.initializer}
for value in model.graph.input:
    if value.name in weights:
        continue  # 여기는 외부 입력의 shape 검사이며 저장된 가중치는 건너뛴다.
    tensor = value.type.tensor_type
    if tensor.elem_type != onnx.TensorProto.FLOAT:  # 이 작은 실습은 FP32 입력으로 한정
        raise ValueError("This example requires FP32 tensor inputs")
    if any(not d.HasField("dim_value") or d.dim_value <= 0
           for d in tensor.shape.dim):
        raise ValueError("Keep dynamic-shape experiments separate")

output = Path("candidate-constant-folded.onnx")
if output.exists():
    raise FileExistsError(output)
candidate, checked = simplify(
    model, check_n=5, perform_optimization=False  # 표본 비교 5회, 일반 최적화 pass는 끈다.
)
if not checked:
    raise RuntimeError("Numerical checks did not pass")
onnx.checker.check_model(candidate)
onnx.save(candidate, output)
print({name: version(name) for name in ["onnx", "onnxsim", "onnxruntime"]})
print({"before_nodes": len(model.graph.node),
       "after_nodes": len(candidate.graph.node)})
