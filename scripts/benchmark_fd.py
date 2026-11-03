"""Measure CPU engine scaling with both VJPs and a separate stock QuantLib oracle.

Reported tensor bytes count repeated autograd saves, not peak process memory.
Stock forward timings include market and engine construction, like TorchReal
forward timings; subprocess startup/import are excluded from the measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from quantlibtorch import fd_vanilla


def measure(mode, steps, nodes, repeats):
    def run(record=False):
        saved = []

        def pack(tensor):
            saved.append(tensor.numel() * tensor.element_size())
            return tensor

        sigma = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        start = time.perf_counter()
        if record:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                price = fd_vanilla(sigma, mode=mode, time_steps=steps, grid_points=nodes)
        else:
            price = fd_vanilla(sigma, mode=mode, time_steps=steps, grid_points=nodes)
        middle = time.perf_counter()
        gradient, = torch.autograd.grad(price, sigma)
        end = time.perf_counter()
        return {
            "forward_seconds": middle-start,
            "backward_seconds": end-middle,
            "price": price.item(),
            "parallel_vol_shift_derivative": gradient.item(),
            "saved_tensor_count": len(saved),
            "saved_tensor_bytes_with_repeated_saves": sum(saved),
        }

    run()
    samples = [run() for _ in range(repeats)]
    memory = run(record=True)
    result = {key: memory[key] for key in (
        "price", "parallel_vol_shift_derivative", "saved_tensor_count",
        "saved_tensor_bytes_with_repeated_saves")}
    for field in ("forward_seconds", "backward_seconds"):
        values = [sample[field] for sample in samples]
        result[field + "_samples"] = values
        result[field + "_median"] = statistics.median(values)
    return result


def stock(steps, nodes, repeats):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "stock_quantlib_reference.py")],
        input=json.dumps(dict(time_steps=steps, grid_points=nodes, repeats=repeats, bump=1e-5)),
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", default="10x31,30x61", help="comma-separated time_steps x grid_points")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-stock", action="store_true", help="omit the independent QuantLib==1.40 reference")
    args = parser.parse_args()
    try:
        grids = [tuple(map(int, grid.lower().split("x"))) for grid in args.grids.split(",")]
        if args.repeats < 1 or not grids or any(len(g) != 2 or g[0] < 1 or g[1] < 5 for g in grids):
            raise ValueError
    except ValueError:
        parser.error("use positive repeats and grids such as 10x31,30x61 (nodes >= 5)")
    torch.set_num_threads(1)
    results = []
    for steps, nodes in grids:
        entry = {"time_steps": steps, "grid_points": nodes,
                 "modes": {mode: measure(mode, steps, nodes, args.repeats) for mode in ("full", "adjoint")}}
        if not args.no_stock:
            entry["stock_quantlib"] = stock(steps, nodes, args.repeats)
        results.append(entry)
    payload = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "quantlib_source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party" / "QuantLib"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "os": platform.platform(), "cpu": platform.processor(),
                        "torch_threads": torch.get_num_threads(), "device": "cpu"},
        "problem": {"engine": "FdBlackScholesVanillaEngine", "scheme": "Douglas",
                    "damping_steps": 2, "dtype": "float64", "volatility": 0.2,
                    "mesh_volatility": 0.2, "repeats": args.repeats},
        "notes": [
            "Warmup excluded; tensor-save hooks run separately from timing.",
            "Saved bytes count all forward saves, including repeated and transient saves; not live storage, graph nodes, or peak RSS.",
            "Stock QuantLib is measured in a separate process and is never linked to the TorchReal extension.",
            "CPU only. Tiny scalar tensor dispatch remains a major cost; no GPU result.",
        ],
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
