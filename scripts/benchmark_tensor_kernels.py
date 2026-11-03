"""Isolated batch/RHS throughput measurements; not a batched pricing benchmark.

Compare one tensor invocation over B systems against a Python loop of B
tensor invocations. Each backend/grad-mode timing gets its own process with
resident inputs, one CPU thread, preserved warmups, and synchronized CUDA wall
times. Validation and memory run in separate workers to avoid allocator state
from a large loop graph contaminating batched timing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import itertools
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from investigate_fd import native_fingerprint, process_memory

ROOT = Path(__file__).resolve().parents[1]


def worker(case):
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from quantlibtorch import band_axpyb, tridiagonal_apply

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = case["device"]
    batch, n, nrhs = case["batch"], case["nodes"], case["rhs"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    shapes = [(batch, n), (3, n), (batch, 3, n), (batch, 1), (batch, n, nrhs)]
    generator = torch.Generator().manual_seed(271)
    cpu = tuple(torch.randn(s, generator=generator, dtype=torch.float64) for s in shapes)
    inputs = tuple(t.to(device).requires_grad_() for t in cpu)
    weights = torch.linspace(-.7, 1.1, batch*n*nrhs, dtype=torch.float64).reshape(batch, n, nrhs).to(device)

    def operation(args, backend):
        a, x, y, b, rhs = args
        if backend == "batched":
            bands = band_axpyb(a, x, y, b)
            return tridiagonal_apply(bands[..., 0, 1:], bands[..., 1, :], bands[..., 2, :-1], rhs)
        values = []
        for i in range(batch):
            bands = band_axpyb(a[i], x, y[i], b[i])
            values.append(tridiagonal_apply(bands[0, 1:], bands[1], bands[2, :-1], rhs[i]))
        return torch.stack(values)

    def run(backend, inference=False):
        sync()
        start = time.perf_counter()
        with torch.no_grad() if inference else torch.enable_grad():
            value = operation(inputs, backend)
        sync()
        middle = time.perf_counter()
        grads = None if inference else torch.autograd.grad(value, inputs, weights)
        sync()
        end = time.perf_counter()
        del value, grads
        sync()
        stop = time.perf_counter()
        return dict(forward_seconds=middle-start, backward_seconds=end-middle,
                    cleanup_seconds=stop-end, total_seconds=stop-start)

    result = {"case": case}
    if case["task"] == "memory":
        gc.collect()
        sync()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            result["cuda_before_allocated_bytes"] = torch.cuda.memory_allocated()
        result["before"] = process_memory()
        value = operation(inputs, case["backend"])
        grads = torch.autograd.grad(value, inputs, weights)
        sync()
        result["after"] = process_memory()
        result["peak_rss_growth_bytes"] = max(0, result["after"]["peak_rss_bytes"]-result["before"]["peak_rss_bytes"])
        if device == "cuda":
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            result["cuda_peak_growth_bytes"] = result["cuda_peak_allocated_bytes"]-result["cuda_before_allocated_bytes"]
        return result

    if case["task"] == "validation":
        # A separate worker checks every RHS and coefficient, including the
        # shared stencil. Its large loop graph cannot affect timing workers.
        vector = operation(inputs, "batched")
        gv = torch.autograd.grad(vector, inputs, weights)
        loop = operation(inputs, "loop")
        gl = torch.autograd.grad(loop, inputs, weights)
        torch.testing.assert_close(vector, loop, rtol=1e-12, atol=1e-12)
        for g, ref in zip(gv, gl):
            torch.testing.assert_close(g, ref, rtol=1e-11, atol=2e-10)
        result["validation"] = dict(max_value_difference=(vector-loop).abs().max().item(),
            max_vjp_difference=max((g-ref).abs().max().item() for g, ref in zip(gv, gl)))
        return result

    inference, backend = case["inference"], case["backend"]
    warmups = [run(backend, inference) for _ in range(case["warmup"])]
    samples = [run(backend, inference) for _ in range(case["repeats"])]
    medians = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
    result["measurements"] = [dict(backend=backend, inference=inference,
        warmup=warmups, samples=samples, median=medians,
        systems_per_second=batch/medians["total_seconds"],
        rhs_vectors_per_second=batch*nrhs/medians["total_seconds"])]

    # Separate cost for pageable H2D + batched inference + D2H output. Includes
    # transfers, allocations, and synchronization, excludes random generation.
    if device == "cuda" and case["inference"] and case["backend"] == "batched":
        def transfer_run():
            sync()
            start = time.perf_counter()
            with torch.no_grad():
                local = tuple(t.to(device) for t in cpu)
                value = operation(local, "batched").cpu()
            sync()
            elapsed = time.perf_counter()-start
            del local, value
            return elapsed
        result["transfer_inference"] = dict(warmup=[transfer_run() for _ in range(case["warmup"])],
                                           samples=[transfer_run() for _ in range(case["repeats"])])
    return result


def bridge_worker(n, repeats, warmup):
    """Measure scalar/vector boundary cost without a whole engine graph."""
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from quantlibtorch import tridiagonal_apply
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bands = torch.linspace(-.2, .8, 3*n, dtype=torch.float64).reshape(3, n).requires_grad_()
    rhs = torch.linspace(-1., 1., n, dtype=torch.float64).requires_grad_()
    # Endpoints zero for equivalence with the compact kernel.
    mask = torch.ones_like(bands)
    mask[0, 0] = mask[2, -1] = 0.
    bands = (bands.detach()*mask).requires_grad_()
    funcs = {
        "scalar_hook": lambda: torch.ops.quantlibtorch._hook_apply(bands, rhs, 0),
        "vector_hook": lambda: torch.ops.quantlibtorch._hook_apply(bands, rhs, 1),
        "tensor_resident": lambda: tridiagonal_apply(bands[0, 1:], bands[1], bands[2, :-1], rhs[:, None]),
        "array_roundtrip": lambda: torch.ops.quantlibtorch._array_roundtrip(rhs),
    }
    results = []
    for name, fn in funcs.items():
        for inference in (True, False):
            def run():
                start = time.perf_counter()
                with torch.no_grad() if inference else torch.enable_grad():
                    out = fn()
                middle = time.perf_counter()
                grad = None if inference else torch.autograd.grad(out.sum(), (bands, rhs), allow_unused=True)
                end = time.perf_counter()
                del out, grad
                return dict(forward_seconds=middle-start, backward_seconds=end-middle,
                            total_seconds=time.perf_counter()-start)
            warmups = [run() for _ in range(warmup)]
            samples = [run() for _ in range(repeats)]
            results.append(dict(name=name, inference=inference, warmup=warmups, samples=samples,
                median={k: statistics.median(s[k] for s in samples) for k in samples[0]}))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batches", default="1,16,256")
    parser.add_argument("--nodes", default="61,257")
    parser.add_argument("--rhs", default="1,8")
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--tasks", default="timing")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=False)
    args = parser.parse_args()
    if args.worker:
        case = json.load(sys.stdin)
        result = bridge_worker(case["nodes"], case["repeats"], case["warmup"]) if case["task"] == "bridge" else worker(case)
        print(json.dumps(result))
        return
    batches, nodes, rhss = ([int(v) for v in x.split(",")] for x in (args.batches, args.nodes, args.rhs))
    devices, tasks = args.devices.split(","), args.tasks.split(",")
    if (min(batches+rhss) < 1 or min(nodes) < 2 or args.warmup < 1 or args.repeats < 1
        or set(devices)-{"cpu", "cuda"} or set(tasks)-{"timing", "memory"}):
        parser.error("invalid batch/node/RHS/device/task/repeat specification")
    import torch
    payload = dict(measured_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(), torch=torch.__version__, os=platform.platform(),
            cpu=platform.processor(), cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            torch_threads=1, torch_interop_threads=1, dtype="float64"),
        native_source_sha256=native_fingerprint(),
        notes=["Kernel pipeline only: row-band assembly followed by application; no solve, rollback, or pricing.",
               "B independent numerical systems, one shared stencil, R right-hand sides; no contract loop in batched path.",
               "Inputs device-resident; synchronized wall times include Python dispatch, forward, backward and cleanup.",
               "Reference is a loop of the same tensor kernels, not stock QuantLib; dense-reference checks live in tests.",
               "Each timing backend/grad mode has a fresh process; validation and memory also use separate workers.",
               "Memory workers have no warmups or correctness graphs.",
               "CUDA transfer inference includes pageable host inputs and returning all output values, measured separately.",
               "Bridge tests include constructing a test QuantLib operator; roundtrip isolates unbind plus stack, not the complete bridge."],
        results=[], validation_results=[], bridge=[])

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")

    def call(case):
        response = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(case), text=True, capture_output=True)
        if response.returncode:
            raise RuntimeError(f"{case}: {response.stderr}\n{response.stdout}")
        return json.loads(response.stdout)

    for device, batch, n, nrhs, task in itertools.product(devices, batches, nodes, rhss, tasks):
        if task == "timing":
            validation = call(dict(device=device, batch=batch, nodes=n, rhs=nrhs, task="validation"))
            payload["validation_results"].append(validation)
            save()
        choices = ([(b, False) for b in ("batched", "loop")] if task == "memory"
                   else list(itertools.product(("loop", "batched"), (True, False))))
        for backend, inference in choices:
            case = dict(device=device, batch=batch, nodes=n, rhs=nrhs, task=task, backend=backend,
                        inference=inference, warmup=args.warmup, repeats=args.repeats)
            result = call(case)
            payload["results"].append(result)
            save()
            summary = (result["measurements"][0]["median"] if "measurements" in result
                       else result["peak_rss_growth_bytes"])
            print(case, summary, flush=True)
    if "timing" in tasks and "cpu" in devices:
        for n in nodes:
            payload["bridge"].append(dict(nodes=n, results=call(dict(task="bridge", nodes=n,
                warmup=args.warmup, repeats=args.repeats))))
    save()
    if not args.output:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
