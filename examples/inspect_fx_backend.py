"""torch.compile이 backend에 넘기는 그래프를 살펴보는 CPU 실습.

NPU compiler를 연결하지 않으며 원래 GraphModule.forward로 계산한다.
"""
from __future__ import annotations

import torch


def main() -> None:
    torch.manual_seed(5)  # 매번 같은 난수 입력과 모델 가중치로 비교한다.
    compile_events: list[int] = []

    def inspect_backend(graph_module, example_inputs):
        # 그래프는 함수 호출의 연결, example_inputs는 준비할 입력의 예다.
        compile_events.append(len(list(graph_module.graph.nodes)))
        print(graph_module.code)
        # 실제 가속기 backend라면 여기서 컴파일한 실행기를 반환한다.
        # 이 학습 예제는 기존 그래프의 CPU 함수를 그대로 돌려준다.
        return graph_module.forward

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 12),  # [4,8] @ weight.T[8,12] + bias[12] → [4,12]
        torch.nn.ReLU(),        # 원소별 max(0,x). shape [4,12]는 유지한다.
        torch.nn.Linear(12, 8), # 특징 12개를 가중합 8개로 바꾼다: [4,8]
    ).eval()  # 추론 모드를 선택한다.
    compiled = torch.compile(
        model, backend=inspect_backend, fullgraph=True, dynamic=False
    )
    x = torch.randn(4, 8)  # 입력 사례 4개, 각 사례의 특징값 8개. CPU FP32다.
    with torch.inference_mode():  # 학습용 역전파 기록을 만들지 않는다.
        # 같은 입력에서 원래 모델과 backend 경유 모델의 수치가 같은지 비교한다.
        torch.testing.assert_close(compiled(x), model(x))
        # shape는 유지하고 모든 입력 값에 1을 더해 새 데이터도 확인한다.
        torch.testing.assert_close(compiled(x + 1), model(x + 1))
    print("Graph node counts received by backend:", compile_events)
    print("Two CPU numerical comparisons passed. No NPU execution performed.")


if __name__ == "__main__":
    main()
