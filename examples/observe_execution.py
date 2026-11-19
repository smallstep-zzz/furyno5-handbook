# observe_execution.py — 앞의 small-mlp.fxbx를 새 프로세스에서 한 번 관찰한다.
import json
import os
from pathlib import Path
import sys

# SDK/fxbX를 불러오기 전에 설정한다. 각 step의 이름을 trace에 남기는 경로를 고른다.
os.environ["FXBX_TRACE_STEPS"] = "1"
import torch
from fxbx import GraphFXB
from fxbex.execution_fxb import read_execution_plan
from furiosa.torch.profiler import Profiler

def main(package_path, trace_path):
    trace_path = Path(trace_path)
    if trace_path.exists():
        raise FileExistsError("Use a new trace path to preserve the previous observation")
    plan = read_execution_plan(package_path)
    entry = plan["entries"]["forward"]
    expected = {
        f"execution_plan::forward::{i}::{step['kernel']}"
        for i, step in enumerate(entry["steps"])
    }
    # 입력 없는 '정적 값 준비'로 계획에 명시된 단계만 계산 Task의 필수 대상에서 뺀다.
    prepared = {
        f"execution_plan::forward::{i}::{step['kernel']}"
        for i, step in enumerate(entry["steps"])
        if not step["inputs"] and
        plan["kernels"][step["kernel"]].get("execution_kind") == "static_value_preparation"
    }
    required = expected - prepared
    if not required:
        raise RuntimeError("This example requires at least one NPU compute step")

    torch.manual_seed(42)                 # 5절 생산자와 같은 가중치·입력을 만든다.
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 12), torch.nn.ReLU(), torch.nn.Linear(12, 8)
    ).bfloat16().eval().requires_grad_(False)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    with GraphFXB(package_path, weights=model.state_dict()) as session:
        # 이 consumer는 load 때 CPU fallback을 끈다. 적재 실패는 예외로 남는다.
        with Profiler(str(trace_path)):
            actual, = session.predict((x,), output_device="cpu")
    torch.testing.assert_close(actual, model(x), atol=0.02, rtol=0.02)

    # 공식 profiler와 고정 worker에서 확인한 필드만 읽는다. 누락되면 실패한다.
    events = json.loads(trace_path.read_text())["traceEvents"]
    names = [event.get("name", "") for event in events]
    host_spans = {name for name in names if name.startswith("execution_plan::")}
    task_steps = {
        name.split("::Task::", 1)[1] for name in names
        if "::Task::execution_plan::" in name
    }
    cpu_routes = sum("run_by_cpu" in name for name in names)
    for name in sorted(expected):
        print(name, "host=", name in host_spans, "NPU Task=", name in task_steps)
    print("cpu_routes:", cpu_routes, "trace:", trace_path)
    missing = sorted(required - task_steps)
    if cpu_routes or host_spans != expected or missing or not task_steps <= expected:
        raise RuntimeError(f"Execution evidence incomplete; missing NPU steps={missing}")
    # 한 번의 호출·이 collector 범위의 관측이다. 전체 모델 지원의 인증이 아니다.
    print("numerical comparison passed; NPU tasks observed for required steps")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
