import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

def main():
    torch.manual_seed(7)
    original = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),  # 입력 채널 3 → 출력 채널 8
        nn.BatchNorm2d(8),  # 출력의 8개 채널마다 통계·gamma·beta를 가진다.
    )

    # 합성 실습용 statistics. 실제 모델에서는 학습된 buffer를 사용한다.
    original.train()  # 이 모드에서 아래 입력들이 BN의 누적 통계를 갱신한다.
    with torch.no_grad():  # 역전파 기록만 끈다. train 모드의 통계 갱신은 계속된다.
        for _ in range(8):
            original(torch.randn(4, 3, 24, 24))  # N,C,H,W = 이미지 4개, 채널 3, 높이·너비 24
    original.eval().requires_grad_(False)  # 통계를 고정하고 추론용 가중치로 준비한다.
    folded = fuse_conv_bn_eval(original[0], original[1])  # Conv+BN을 새 Conv 계수로 접는다.
    folded.eval().requires_grad_(False)

    max_error = 0.0
    with torch.inference_mode():
        for shape in [(1, 3, 24, 24), (3, 3, 19, 27)]:  # batch·공간 크기를 달리해 확인
            for scale in [0.1, 1.0, 5.0]:
                x = scale * torch.randn(shape)
                expected, actual = original(x), folded(x)  # 두 결과 shape는 모두 [N,8,H,W]
                torch.testing.assert_close(
                    actual, expected, rtol=1e-5, atol=1e-5  # 절대오차+기준값 비례 오차 한도
                )
                max_error = max(
                    max_error, (actual - expected).abs().max().item()
                )
    print({"torch": torch.__version__, "max_abs_error": max_error})

if __name__ == "__main__":
    main()
