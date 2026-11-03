"""Measure the prototype on CPU; this is not a GPU or stock QuantLib benchmark."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from quantlibtorch import european_call


def measure(mode, steps, nodes, repeats):
    def run(record=False):
        saved = []

        def pack(tensor):
            saved.append(tensor.numel() * tensor.element_size())
            return tensor

        coefficients = torch.full((steps, nodes), 0.2, dtype=torch.float64, requires_grad=True)
        start = time.perf_counter()
        if record:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                price = european_call(coefficients, mode=mode)
        else:
            price = european_call(coefficients, mode=mode)
        middle = time.perf_counter()
        gradient, = torch.autograd.grad(price, coefficients)
        end = time.perf_counter()
        return {
            "forward_seconds": middle - start,
            "backward_seconds": end - middle,
            "price": price.item(),
            "parallel_vol_shift_derivative": gradient.sum().item(),
            "saved_tensor_count": len(saved),
            "saved_tensor_bytes_with_repeated_saves": sum(saved),
        }

    run()  # warmup excluded
    results = [run() for _ in range(repeats)]
    memory = run(record=True)  # hooks not enabled during timing
    return {
        "forward_seconds_median": statistics.median(r["forward_seconds"] for r in results),
        "backward_seconds_median": statistics.median(r["backward_seconds"] for r in results),
        "forward_seconds_samples": [r["forward_seconds"] for r in results],
        "backward_seconds_samples": [r["backward_seconds"] for r in results],
        "price": memory["price"],
        "parallel_vol_shift_derivative": memory["parallel_vol_shift_derivative"],
        "saved_tensor_count": memory["saved_tensor_count"],
        "saved_tensor_bytes_with_repeated_saves": memory["saved_tensor_bytes_with_repeated_saves"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=31)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.nodes < 2 or args.repeats < 1:
        parser.error("steps/repeats must be positive, nodes >= 2")
    torch.set_num_threads(1)
    result = {
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "os": platform.platform(), "cpu": platform.processor(),
                        "torch_threads": torch.get_num_threads(), "device": "cpu"},
        "problem": {"steps": args.steps, "nodes": args.nodes, "repeats": args.repeats,
                    "dtype": "float64", "scheme": "backward Euler"},
        "notes": "Saved bytes count repeated saves; not peak memory or unique storage. No GPU measurement.",
        "results": {mode: measure(mode, args.steps, args.nodes, args.repeats)
                    for mode in ("full", "adjoint")},
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
