# benchmark_mlp.py — 5장에서 만든 small-mlp.fxbx와 같은 디렉터리에서 실행한다.
from math import ceil
from statistics import median
from time import perf_counter_ns
import torch
from fxbx import GraphFXB


def main():
    torch.manual_seed(42)  # 5장 예제와 같은 외부 가중치·입력을 재생성한다.
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 12), torch.nn.ReLU(), torch.nn.Linear(12, 8)
    ).bfloat16().eval().requires_grad_(False)
    x = torch.randn(4, 8, dtype=torch.bfloat16)  # 입력 계약: shape (4,8), BF16(BFloat16)
    with torch.inference_mode():
        reference = model(x)                  # CPU 정답 계산은 측정 구간 밖이다.
        prepare_start = perf_counter_ns()
        with GraphFXB("./small-mlp.fxbx", weights=model.state_dict()) as session:
            prepare_ms = (perf_counter_ns() - prepare_start) / 1e6

            first_start = perf_counter_ns()
            first, = session.predict((x,), output_device="cpu")
            first_ms = (perf_counter_ns() - first_start) / 1e6
            torch.testing.assert_close(first, reference, atol=0.02, rtol=0.02)

            # 첫 호출의 지연 준비·캐시 영향을 반복 측정과 구분하기 위해 예열한다.
            for _ in range(10):
                session.predict((x,), output_device="cpu")

            elapsed_ms = []
            for _ in range(100):
                started = perf_counter_ns()
                last, = session.predict((x,), output_device="cpu")
                elapsed_ms.append((perf_counter_ns() - started) / 1e6)
            # 정답 비교 비용은 timed 구간에 넣지 않는다.
            torch.testing.assert_close(last, reference, atol=0.02, rtol=0.02)

    ordered = sorted(elapsed_ms)
    # p95는 정렬된 측정값 중 95% 위치를 선택하는 nearest-rank 정의를 사용한다.
    p95_ms = ordered[ceil(0.95 * len(ordered)) - 1]
    print({"scope": "CPU input to CPU output, warm, sequential",
           "session_prepare_ms": prepare_ms, "first_call_ms": first_ms,
           "median_ms": median(ordered), "p95_ms": p95_ms,
           "samples": len(ordered)})


if __name__ == "__main__":
    main()
