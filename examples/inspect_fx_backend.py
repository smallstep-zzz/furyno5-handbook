"""Inspect torch.compile's backend boundary on CPU; this does not enable an NPU."""
from __future__ import annotations

import torch


def main() -> None:
    torch.manual_seed(5)
    compile_events: list[int] = []

    def inspect_backend(graph_module, example_inputs):
        compile_events.append(len(list(graph_module.graph.nodes)))
        print(graph_module.code)
        # An actual accelerator backend would compile and return its executor here.
        return graph_module.forward

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)
    ).eval()
    compiled = torch.compile(
        model, backend=inspect_backend, fullgraph=True, dynamic=False
    )
    x = torch.randn(2, 16)
    with torch.inference_mode():
        torch.testing.assert_close(compiled(x), model(x))
        torch.testing.assert_close(compiled(x + 1), model(x + 1))
    print("Graph node counts received by backend:", compile_events)
    print("Two CPU numerical comparisons passed. No NPU execution performed.")


if __name__ == "__main__":
    main()
