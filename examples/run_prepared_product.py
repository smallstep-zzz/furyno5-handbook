import torch
import furiosa.torch as ft  # furiosa 장치 backend와 완료 대기 API를 제공한다.
from tclmachine import optimize

class Product(torch.nn.Module):
    def forward(self, a, b):
        return torch.mm(a, b)  # [2,3] @ [3,2] → [2,2]

def main():
    model = Product().eval().requires_grad_(False)
    a = torch.tensor([[1, -2, 3], [-4, 5, -6]], dtype=torch.bfloat16)
    b = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.bfloat16)
    with torch.no_grad():
        expected = model(a, b)  # CPU 기준값: [[4,1],[-10,-1]], shape [2,2]

    prepared = optimize(
        model,
        example_inputs=(a, b),  # CPU 대표 입력으로 shape·dtype·stride와 graph를 준비한다.
        target="furiosa:0",     # 실제 호출 입력도 이 장치에 있어야 한다.
        fallback="error",       # 이 facade가 허용하는 정책; CPU 모델 재실행 옵션이 아니다.
    )
    try:
        a_device = a.to("furiosa:0")  # 이번 호출의 실제 원소 값을 장치로 복사한다.
        b_device = b.to("furiosa:0")
        with torch.no_grad():
            output_device = prepared(a_device, b_device)  # Product의 반환형대로 tensor 하나
            actual = output_device.cpu()                # 비교할 CPU tensor [2,2]로 받는다.
        ft.current_stream(torch.device("furiosa:0")).synchronize()  # 완료를 확인한 뒤 해제
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        print(actual)  # 위 비교를 실제 통과한 결과만 출력한다.
    finally:
        prepared.close()  # 소유한 custom-op 등록을 해제한다. 이후 prepared 호출은 오류다.

if __name__ == "__main__":
    main()
