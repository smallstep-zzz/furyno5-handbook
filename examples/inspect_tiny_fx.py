import torch

def main():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 12),  # 입력 [4,8] → [4,12]
        torch.nn.ReLU(),        # 음수만 0으로 바꾸며 [4,12] 유지
        torch.nn.Linear(12, 8), # [4,12] → [4,8]
    ).eval().requires_grad_(False)
    x = torch.randn(4, 8)  # CPU FP32 대표 입력; BF16 장치 컴파일 예제와 구분한다.
    ep = torch.export.export(model, (x,), strict=False)

    for spec in ep.graph_signature.input_specs:
        # arg.name은 그래프의 입력 자리, target은 연결할 parameter 이름이다.
        # USER_INPUT의 target은 보통 None이며, PARAMETER는 '0.weight' 등이다.
        name = getattr(spec.arg, "name", repr(spec.arg))
        print("INPUT", name, spec.kind.name, "target=", spec.target)

    for node in ep.graph.nodes:
        value = node.meta.get("val")  # 계산 결과가 아니라 캡처 중 추적한 tensor 정보
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        print("NODE", node.op, node.name, node.target, "shape=", shape)
        print("  args=", node.args, "kwargs=", node.kwargs)  # 선행 node 참조와 상수

    x_second = x + 0.25  # shape [4,8]은 그대로 두고 입력의 숫자만 바꾼다.
    with torch.no_grad():
        actual = ep.module()(x_second)  # parameter를 다시 연결한 CPU 실행 module
        expected = model(x_second)     # 같은 가중치의 원본 Python 모델
    torch.testing.assert_close(actual, expected)
    print("changed-input CPU comparison passed")  # 위 비교가 실제 통과한 경우에만 출력

if __name__ == "__main__":
    main()
