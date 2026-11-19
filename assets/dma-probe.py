from statistics import median
from time import perf_counter

import torch
import furiosa.torch as ft

device = torch.device("furiosa:0")
host = torch.arange(1024 * 1024, dtype=torch.int32)
samples = [[], []]

# Warm up the SDK, allocator and transfer path.
for _ in range(3):
    warm = host.to(device).cpu()
    ft.current_stream(device).synchronize()

for _ in range(10):
    ft.current_stream(device).synchronize()
    start = perf_counter()
    npu = host.to(device)
    ft.current_stream(device).synchronize()
    uploaded = perf_counter()

    result = npu.cpu()
    ft.current_stream(device).synchronize()
    downloaded = perf_counter()

    torch.testing.assert_close(result, host, atol=0, rtol=0)
    samples[0].append((uploaded - start) * 1000)
    samples[1].append((downloaded - uploaded) * 1000)

print(f"{host.numel() * host.element_size():,} bytes each way")
print(f"H2D median: {median(samples[0]):.3f} ms")
print(f"D2H median: {median(samples[1]):.3f} ms")
