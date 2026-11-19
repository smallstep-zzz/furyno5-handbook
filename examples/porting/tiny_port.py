# tiny_port.py — 호환 SDK와 RNGD가 있는 환경의 생산 스크립트
import torch
import tclmachine


def main():
    torch.manual_seed(42)  # 무작위 초기 가중치와 입력을 다시 만들 수 있게 고정한다.
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 12),  # x[4,8] @ W0.T[8,12] + b0[12] → h[4,12]
        torch.nn.ReLU(),        # 음수를 0으로 바꾸며 shape [4,12]는 유지한다.
        torch.nn.Linear(12, 8), # a[4,12] @ W2.T[12,8] + b2[8] → y[4,8]
    ).bfloat16().eval().requires_grad_(False)  # BF16 가중치, 추론 모드, 학습 제외
    x = torch.randn(4, 8, dtype=torch.bfloat16)  # CPU의 입력 4행 × 특징 8개
    x_second = x + torch.full_like(x, 0.25)  # 모든 입력 원소에 0.25를 더한 두 번째 사례
    inputs = [x, x_second]  # 두 사례는 동일한 shape·dtype을 유지한다.
    with torch.no_grad():  # 참조 계산에 미분 그래프를 만들지 않는다.
        expected = [model(value) for value in inputs]  # 각각 CPU 출력 [4,8]

    # 패키지가 별도로 요구할 가중치와, 동일 값 비교에 쓸 입력·정답을 보존한다.
    torch.save(model.state_dict(), "tiny-state.pt")
    torch.save({"inputs": inputs, "expected": expected}, "tiny-cases.pt")

    preflight = tclmachine.export_model(
        model, (x,), "tiny-preflight.fxbx",  # (x,)는 입력 tensor 하나를 담은 tuple
        preflight_only=True,  # 지원 진단만 수행하며 EDF 컴파일·실행은 하지 않는다.
    )
    print("preflight:", preflight)  # 사전 분석 보고서이며 추론 출력이 아니다.
    report = tclmachine.export_model(
        model, (x,), "tiny.fxbx",  # 계산 구조·입력 계약으로 패키지를 생성한다.
        compile_workers=4,  # CPU 컴파일 작업의 병렬 worker 수; NPU 개수가 아니다.
    )
    print("export:", report)  # 생성 상태·region·kernel 수·소요 시간을 확인한다.


if __name__ == "__main__":  # spawn된 worker가 전체 생산 작업을 다시 시작하지 않게 한다.
    main()
