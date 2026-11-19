import torch

class OriginalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 3, bias=False)  # weight [3,8]

    def forward(self, *, features, mask):
        # * 뒤의 인수는 keyword로 받는다. 두 입력 모두 BF16 [2,8]이다.
        masked = features * mask  # 같은 위치끼리 곱함; mask=0인 위치의 값을 0으로 만든다.
        logits = self.projection(masked)  # [2,8] → [2,3]
        return {"logits": logits, "masked_features": masked}  # 원래 Python 반환 구조

class TensorAdapter(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original  # 원본 module을 등록된 자식으로 보관한다.

    def forward(self, features, mask):
        result = self.original(features=features, mask=mask)  # 원본 keyword 호출 복원
        return result["logits"]  # 장치 계약에는 [2,3] tensor 하나만 내보내기로 선택한다.

def representative_component():
    original = OriginalModel().bfloat16().eval().requires_grad_(False)
    adapter = TensorAdapter(original).eval().requires_grad_(False)
    features = torch.ones(2, 8, dtype=torch.bfloat16)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]],
        dtype=torch.bfloat16,
    )  # features와 같은 [2,8]; broadcasting을 요구하지 않는 작은 계약
    return adapter, (features, mask), {}  # 위치 인수 순서를 고정하고 kwargs는 비운다.

def main():
    adapter, args, kwargs = representative_component()
    features, mask = args
    with torch.no_grad():
        expected = adapter.original(features=features, mask=mask)["logits"]
        actual = adapter(*args, **kwargs)  # 원본과 동일 weight·입력으로 wrapper를 비교한다.
    torch.testing.assert_close(actual, expected)
    print("output shape:", tuple(actual.shape))  # (2,3), dict에서 선택한 logits만 남는다.
    print("original weights:", list(adapter.original.state_dict()))
    print("adapter weights:", list(adapter.state_dict()))

if __name__ == "__main__":
    main()
