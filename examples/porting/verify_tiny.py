# verify_tiny.py — 위에서 생성한 세 파일을 같은 작업 디렉터리에서 읽는다.
import torch
from fxbx import GraphFXB


def main():
    # CPU에 가중치 tensor를 적재한다. 이것 자체는 CPU 모델 추론이 아니다.
    weights = torch.load("tiny-state.pt", map_location="cpu", weights_only=True)
    cases = torch.load("tiny-cases.pt", map_location="cpu", weights_only=True)
    with GraphFXB("tiny.fxbx", weights=weights) as runner:
        for index, (x, expected) in enumerate(zip(cases["inputs"], cases["expected"])):
            actual, = runner.predict((x,))  # 입력 tuple → 출력 tuple; 출력 하나 [4,8]을 꺼낸다.
            torch.testing.assert_close(
                actual, expected, atol=0.02, rtol=0.02,  # 작은 BF16 예제의 비교 기준
            )
            print("numerical comparison passed:", index)
    # with 블록을 나가면 runner가 보유한 실행 자원을 정리한다.


if __name__ == "__main__":
    main()
