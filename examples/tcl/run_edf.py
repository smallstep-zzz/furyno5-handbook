# run_edf.py — NPU와 호환 SDK가 필요하다.
from pathlib import Path
import torch
import furiosa.torch as sdk

sdk.config.set_cpu_fallback(False)
sdk.set_fusion(8)
sdk.set_mesh_kind(sdk.MeshKind.Single())

ir = sdk.ir.Runnable.from_serialized_edf(Path("matmul.edf").read_bytes())
if ir.expected_num_fusion != 8:
    raise RuntimeError("EDF fusion contract does not match the requested device")
if len(ir.input_dram_shapes) != 2 or len(ir.output_dram_shapes) != 1:
    raise RuntimeError("Unexpected EDF input/output arity")
print("ABI input shapes:", ir.input_dram_shapes)
print("ABI output shapes:", ir.output_dram_shapes)

program = sdk.Runnable(ir).to("furiosa:0")
left = torch.ones((32, 128), dtype=torch.bfloat16)
right = torch.ones((128, 64), dtype=torch.bfloat16)
left_device = left.to("furiosa:0")
right_device = right.to("furiosa:0")
with torch.inference_mode():
    value = program(left_device, right_device)
if isinstance(value, (tuple, list)):
    if len(value) != 1:
        raise RuntimeError("Expected one result")
    value = value[0]
actual = value.cpu()
torch.testing.assert_close(actual, left @ right, rtol=0, atol=0)
print("single-shape all-ones smoke check passed")
