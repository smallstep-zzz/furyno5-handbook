"""같은 Linear→ReLU 모델을 각 업체의 SDK 환경에서 CPU 기준값과 비교한다."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def fixture():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU())
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(8))
        model[0].bias.fill_(1)
    model.eval().requires_grad_(False)
    x = torch.arange(-16, 16, dtype=torch.float32).reshape(4, 8)
    return model, x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("export", "amd", "intel", "furiosa"))
    parser.add_argument("--amd-config", type=Path)
    args = parser.parse_args()
    model, x = fixture()
    with torch.inference_mode():
        reference = model(x).numpy()
        if args.backend == "export":
            torch.onnx.export(
                model, (x,), "tiny-linear.onnx", input_names=["x"],
                output_names=["y"], opset_version=17, dynamo=False,
            )
            print(torch.export.export(model, (x,)).graph_module.code)
            print("CPU reference:", reference)
            return
        if args.backend == "amd":
            import onnxruntime as ort
            if "VitisAIExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("Use the Ryzen AI ONNX Runtime installation")
            if args.amd_config is None:
                parser.error("amd requires --amd-config from your Ryzen AI SDK")
            options = ort.SessionOptions()
            options.enable_profiling = True
            # Development SDK: compile supported regions or reuse a compatible cache.
            session = ort.InferenceSession(
                "tiny-linear.onnx", sess_options=options,
                providers=["VitisAIExecutionProvider"],
                provider_options=[{"config_file": str(args.amd_config)}],
            )
            actual = session.run(["y"], {"x": x.numpy()})[0]
            profile = session.end_profiling()
            events = json.loads(Path(profile).read_text())
            assignments = sorted({
                event.get("args", {}).get("provider")
                for event in events
                if event.get("args", {}).get("provider")
            })
            print("Observed providers:", assignments, "profile:", profile)
            if "VitisAIExecutionProvider" not in assignments:
                raise RuntimeError("No Vitis AI execution observed; inspect the SDK analyzer")
        elif args.backend == "intel":
            import openvino as ov
            core = ov.Core()
            if "NPU" not in core.available_devices:
                raise RuntimeError("Intel NPU and compatible driver are required")
            ir = ov.convert_model(model, example_input=x)
            compiled = core.compile_model(ir, "NPU")
            actual = compiled([x.numpy()])[compiled.output(0)]
            print("Execution devices:", compiled.get_property("EXECUTION_DEVICES"))
        else:
            # FuryNo5 supplies binding of exported parameters to SDK Runnable inputs.
            from tclmachine import prepare_model_direct
            with prepare_model_direct(
                model, (x,), artifact_root="./tiny-furiosa", device="furiosa:0"
            ) as prepared:
                actual, = prepared(x)
                actual = actual.numpy()
        np.testing.assert_allclose(actual, reference, atol=0.02, rtol=0.02)
        print("Output matches the CPU reference:", actual.shape)


if __name__ == "__main__":
    main()
