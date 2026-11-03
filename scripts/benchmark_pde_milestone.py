"""Isolated 2D pricing measurements and frozen-binary 1D regression controls.

Timing, graph hooks, profiling, and cold process-memory observations run in
different workers. No performance policy is changed by the library itself.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(package_root):
    native = package_root / "python/quantlibtorch/_native/quantlibtorch_native.dll"
    if not native.exists():
        native = next((package_root / "python/quantlibtorch/_native").glob("*native.*"))
    result = {"native_binary_sha256": hashlib.sha256(native.read_bytes()).hexdigest()}
    baseline = package_root / "manifest.json"
    if baseline.exists():
        result["baseline"] = json.loads(baseline.read_text())
    else:
        files = [ROOT / "CMakeLists.txt", ROOT / "cmake/quantlib_sources.cmake",
                 ROOT / "scripts/prepare_quantlib.py", Path(__file__).resolve()]
        files += list((ROOT / "cpp").rglob("*.cpp")) + list((ROOT / "cpp").rglob("*.hpp"))
        files += list((ROOT / "python/quantlibtorch").glob("*.py"))
        result["source_sha256"] = {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}
    return result


def worker(case):
    package_root = Path(case["package_root"])
    if case["kind"] == "vanilla":
        import investigate_fd
        investigate_fd.ROOT = package_root
        result = investigate_fd.worker(case)
        result["fingerprint"] = fingerprint(package_root)
        return result

    from runtime_controls import worker_qos
    qos = worker_qos(case["windows_qos"], case["affinity"])
    import torch
    from investigate_fd import process_memory
    sys.path.insert(0, str(package_root / "python"))
    from quantlibtorch import fd_basket
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dtype = torch.float64
    coordinates = torch.tensor([[t, math.log(s / 100.)]
        for t in (0., .4, 1.) for s in (50., 80., 110., 160.)], dtype=dtype)
    # One shared network with two volatility outputs and an independent
    # transformed correlation; all 23 parameters remain graph-connected.
    base = torch.linspace(-.25, .35, 23, dtype=dtype)
    axes = [math.log(s) for s in (50., 80., 110., 160.)]

    def price(parameters):
        hidden = torch.tanh(coordinates @ parameters[:8].reshape(2, 4)
                            + parameters[8:12])
        volatility = .1 + .25 * torch.sigmoid(
            hidden @ parameters[12:20].reshape(4, 2) + parameters[20:22])
        return fd_basket(volatility[:, 0].reshape(3, 4),
                         volatility[:, 1].reshape(3, 4),
                         .75 * torch.tanh(parameters[22]),
                         times1=[0., .4, 1.], times2=[0., .4, 1.],
                         log_spots1=axes, log_spots2=axes,
                         grid_points=(case["nx"], case["ny"]),
                         time_steps=case["steps"], scheme=case["scheme"],
                         mode=case["mode"], kernels="auto")

    def run():
        parameters = base.clone().requires_grad_()
        start = time.perf_counter()
        value = price(parameters)
        loss = (value - 7.).square()
        middle = time.perf_counter()
        gradient, = torch.autograd.grad(loss, parameters)
        end = time.perf_counter()
        result = dict(price=value.item(), loss=loss.item(), gradient=gradient.tolist(),
                      forward_seconds=middle-start, backward_seconds=end-middle)
        cleanup = time.perf_counter()
        del parameters, value, loss, gradient
        result["cleanup_seconds"] = time.perf_counter()-cleanup
        result["total_seconds"] = end-start+result["cleanup_seconds"]
        return result

    result = dict(case=case, worker_qos=qos, torch_version=torch.__version__,
                  fingerprint=fingerprint(package_root),
                  gradient_target="23 network/correlation parameters, squared basket-price loss")
    task = case["task"]
    if task == "timing":
        result["warmup"] = [run() for _ in range(case["warmup"])]
        result["samples"] = samples = [run() for _ in range(case["repeats"])]
        result["median"] = {k: statistics.median(s[k] for s in samples)
            for k in ("forward_seconds", "backward_seconds", "cleanup_seconds", "total_seconds")}
    elif task == "memory":
        parameters = base.clone().requires_grad_()
        gc.collect()
        result["before"] = process_memory()
        value = price(parameters)
        loss = (value-7.).square()
        result["after_forward"] = process_memory()
        gradient, = torch.autograd.grad(loss, parameters)
        result["after_backward"] = process_memory()
        result.update(price=value.item(), gradient=gradient.tolist())
        del value, loss, gradient, parameters
        gc.collect()
        result["after_cleanup"] = process_memory()
        result["peak_rss_growth_bytes"] = max(0,
            result["after_cleanup"]["peak_rss_bytes"]-result["before"]["peak_rss_bytes"])
    elif task == "graph":
        saved = []

        def pack(tensor):
            saved.append(tensor.numel()*tensor.element_size())
            return tensor

        parameters = base.clone().requires_grad_()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            value = price(parameters)
            loss = (value-7.).square()
        seen, pending, counts = set(), [loss.grad_fn], Counter()
        while pending:
            node = pending.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            counts[type(node).__name__] += 1
            pending.extend(child for child, _ in node.next_functions)
        gradient, = torch.autograd.grad(loss, parameters)
        result.update(price=value.item(), gradient=gradient.tolist(),
                      graph_nodes=len(seen), node_types=dict(counts),
                      saved_tensor_count=len(saved),
                      saved_tensor_bytes_with_repeated_saves=sum(saved))
    elif task == "profile":
        for _ in range(case["warmup"]):
            run()
        parameters = base.clone().requires_grad_()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            with torch.profiler.record_function("milestone::forward"):
                value = price(parameters)
                loss = (value-7.).square()
            with torch.profiler.record_function("milestone::backward"):
                gradient, = torch.autograd.grad(loss, parameters)
        result.update(price=value.item(), gradient=gradient.tolist())
        result["scopes"] = [dict(name=e.key, calls=e.count,
            self_cpu_us=e.self_cpu_time_total, inclusive_cpu_us=e.cpu_time_total)
            for e in prof.key_averages() if e.key.startswith(("qlt::", "milestone::"))]
        result["aten_calls"] = sum(e.count for e in prof.key_averages()
                                   if e.key.startswith("aten::"))
    else:
        raise ValueError("unknown measurement task")
    return result


def validate(results):
    # A fresh process and graph per sample may change timing but must not change
    # the represented numerical function. Compare all warmups/timed samples,
    # both differentiation modes, and any frozen/current 1D controls.
    references, counts = {}, Counter()
    maximum_price = maximum_gradient = 0.
    for result in results:
        case = result["case"]
        key = tuple(case.get(k) for k in
                    ("kind", "steps", "nodes", "nx", "ny", "surface", "scheme"))
        samples = result.get("warmup", []) + result.get("samples", [])
        if "price" in result and not samples:
            samples = [result]
        for sample in samples:
            price, gradient = sample["price"], sample.get("gradient")
            if not math.isfinite(price):
                raise AssertionError("non-finite benchmark price")
            if key not in references:
                references[key] = (price, gradient)
            ref_price, ref_grad = references[key]
            dp = abs(price-ref_price)
            if dp > 2e-9 * (1+abs(ref_price)):
                raise AssertionError(f"price mismatch: {key}: {dp}")
            maximum_price = max(maximum_price, dp)
            if gradient is not None and ref_grad is None:
                references[key] = (ref_price, gradient)
            elif gradient is not None:
                def flatten(value):
                    if isinstance(value, list):
                        return [n for item in value for n in flatten(item)]
                    return [value]
                actual, expected = flatten(gradient), flatten(ref_grad)
                if len(actual) != len(expected):
                    raise AssertionError("gradient shape mismatch")
                for a, b in zip(actual, expected):
                    if not math.isfinite(a) or abs(a-b) > 2e-8 * (1+abs(b)):
                        raise AssertionError(f"gradient mismatch: {key}: {a-b}")
                    maximum_gradient = max(maximum_gradient, abs(a-b))
                counts["gradient_comparisons"] += 1
            counts["price_comparisons"] += 1
    return dict(counts, max_price_difference=maximum_price,
                max_gradient_difference=maximum_gradient)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=["vanilla", "basket"], default="basket")
    parser.add_argument("--grids", default="3x7x5,6x15x11",
                        help="basket steps x nx x ny; vanilla steps x nodes")
    parser.add_argument("--modes", default="full,adjoint")
    parser.add_argument("--tasks", default="timing,graph,memory")
    parser.add_argument("--scheme", default="hundsdorfer",
                        help="basket ADI scheme; the frozen 1D controls use Douglas")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--affinity", type=int, default=8)
    parser.add_argument("--windows-qos", choices=["system", "high"], default="high")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker)), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        parser.error("output already exists; preserve prior evidence and choose a new path")
    modes, tasks = args.modes.split(","), args.tasks.split(",")
    if set(modes)-{"full", "adjoint", "inference"} or set(tasks)-{"timing", "graph", "memory", "profile"}:
        parser.error("unknown mode or task")
    if args.kind == "basket" and "inference" in modes:
        parser.error("basket measurements currently compare full and adjoint")
    roots = [ROOT] if args.baseline_root is None else [args.baseline_root.resolve(), ROOT]
    if args.baseline_root and args.kind != "vanilla":
        parser.error("frozen baseline is a 1D control only")
    cases = []
    for grid in args.grids.split(","):
        dimensions = [int(n) for n in grid.split("x")]
        for package_root in roots:
            for mode in modes:
                for surface in (["surface", "neural"] if args.kind == "vanilla" else ["neural"]):
                    for task in tasks:
                        c = dict(kind=args.kind, package_root=str(package_root),
                                 mode=mode, surface=surface, task=task,
                                 scheme="douglas" if args.kind == "vanilla" else args.scheme,
                                 warmup=args.warmup, repeats=args.repeats,
                                 affinity=args.affinity, windows_qos=args.windows_qos)
                        if args.kind == "vanilla":
                            c.update(steps=dimensions[0], nodes=dimensions[1],
                                     variant="optimized", kernels="auto", contract="european")
                        else:
                            c.update(steps=dimensions[0], nx=dimensions[1], ny=dimensions[2])
                        cases.append(c)
    report = dict(measured_at_utc=datetime.now(timezone.utc).isoformat(),
                  command=sys.argv, results=[], notes=[
        "Workers execute sequentially in fresh processes with one intra/inter-op thread.",
        "Warmups and samples retain every price and coefficient/parameter VJP.",
        "Timing, graph hooks, profiling, and cold process-memory tasks are isolated.",
        "Saved-tensor bytes count repeated saves, not unique allocation or peak live memory.",
        "Process high-water growth includes allocator/native/interpreter effects.",
        "Baseline workers load only the preserved package and binary; current workers load the workspace build.",
        "Financial engine batching and GPU execution are not measured or claimed."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        env = os.environ.copy()
        # Ensure benchmark evidence describes defaults, independent of a user's
        # feature-ablation environment. Child-specific settings are explicit.
        for name in list(env):
            if name.startswith("QLT_"):
                env.pop(name)
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                              "--worker", json.dumps(case)],
                             env=env, text=True, capture_output=True, check=True)
        result = json.loads(run.stdout)
        report["results"].append(result)
        report["validation"] = validate(report["results"])
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(dict(case=case, median=result.get("median"),
                              graph_nodes=result.get("graph_nodes"),
                              peak_rss_growth_bytes=result.get("peak_rss_growth_bytes"))), flush=True)


if __name__ == "__main__":
    main()
