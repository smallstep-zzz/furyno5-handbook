import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

def main():
    torch.manual_seed(7)
    original = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
    )

    # 합성 실습용 statistics. 실제 모델에서는 학습된 buffer를 사용한다.
    original.train()
    with torch.no_grad():
        for _ in range(8):
            original(torch.randn(4, 3, 24, 24))
    original.eval().requires_grad_(False)
    folded = fuse_conv_bn_eval(original[0], original[1])
    folded.eval().requires_grad_(False)

    max_error = 0.0
    with torch.inference_mode():
        for shape in [(1, 3, 24, 24), (3, 3, 19, 27)]:
            for scale in [0.1, 1.0, 5.0]:
                x = scale * torch.randn(shape)
                expected, actual = original(x), folded(x)
                torch.testing.assert_close(
                    actual, expected, rtol=1e-5, atol=1e-5
                )
                max_error = max(
                    max_error, (actual - expected).abs().max().item()
                )
    print({"torch": torch.__version__, "max_abs_error": max_error})

if __name__ == "__main__":
    main()
