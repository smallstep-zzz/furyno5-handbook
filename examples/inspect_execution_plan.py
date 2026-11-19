# inspect_execution_plan.py — 고정 버전과 호환되는 fxbX 설치가 필요하다.
import sys
from fxbex.execution_fxb import read_execution_plan

def main(path):
    plan = read_execution_plan(path)      # schema·연결·저장 EDF 해시를 검사하고 dict 반환
    entry = plan["entries"]["forward"]
    tensors = plan["tensors"]
    # 값 ID -> 이 값을 준비한 곳. 매 호출 입력과 외부 가중치·상태를 먼저 등록한다.
    origin = {name: "호출자 입력" for name in entry["inputs"]}
    for name, spec in tensors.items():
        if spec["kind"] in {"parameter", "buffer", "state"}:
            origin[name] = f"{spec['kind']}:{spec.get('target', name)}"

    print("entry inputs:", entry["inputs"])
    for number, step in enumerate(entry["steps"]):
        kernel_id = step["kernel"]
        kernel = plan["kernels"][kernel_id]
        print(f"step {number}: kernel={kernel_id}, EDF={kernel['member']}")
        for name in step["inputs"]:
            spec = tensors[name]
            # 같은 ID를 쓰는 소비자가 누구에게서 값을 받는지 보인다.
            print(f"  read {name} <- {origin[name]}, shape={spec['shape']}, dtype={spec['dtype']}")
        for name in step["outputs"]:
            origin[name] = f"step {number}의 출력"
            print(f"  write {name}, shape={tensors[name]['shape']}")
    for name in entry["outputs"]:
        print(f"return {name} <- {origin[name]}") # 마지막 step 출력이라고 가정하지 않는다.
    print("state updates:", entry["state_updates"]) # 이 MLP에는 지속 상태 갱신이 없다.

if __name__ == "__main__":
    main(sys.argv[1])
