from importlib.metadata import version
from pathlib import Path
import onnx
from onnxsim import simplify

model = onnx.load("input.onnx")
onnx.checker.check_model(model)
weights = {tensor.name for tensor in model.graph.initializer}
for value in model.graph.input:
    if value.name in weights:
        continue
    tensor = value.type.tensor_type
    if tensor.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("This example requires FP32 tensor inputs")
    if any(not d.HasField("dim_value") or d.dim_value <= 0
           for d in tensor.shape.dim):
        raise ValueError("Keep dynamic-shape experiments separate")

output = Path("candidate-constant-folded.onnx")
if output.exists():
    raise FileExistsError(output)
candidate, checked = simplify(
    model, check_n=5, perform_optimization=False
)
if not checked:
    raise RuntimeError("Numerical checks did not pass")
onnx.checker.check_model(candidate)
onnx.save(candidate, output)
print({name: version(name) for name in ["onnx", "onnxsim", "onnxruntime"]})
print({"before_nodes": len(model.graph.node),
       "after_nodes": len(candidate.graph.node)})
