"""Record Heston refinement, neural training and isolated CPU measurements.

Run from the repository root. Numerical unit tests remain the stronger field-
by-field VJP references; this script records application-level measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def worker(case):
    from runtime_controls import worker_qos
    qos = worker_qos(case.get("windows_qos", "system"), case.get("affinity"))
    import torch
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT / "examples"))
    from quantlibtorch import fd_latent, state_grid
    from test_fd_latent import heston_fields, stock, AXES, PARAMS
    from neural_latent_sde import NeuralDiffusion, run
    from investigate_fd import process_memory
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    kind = case["kind"]
    if kind == "training":
        return run(case["mode"], case["latent_dimensions"], 5)
    if kind == "heston":
        p = torch.tensor(PARAMS, dtype=torch.float64)
        oracle = stock(dict(analytic=True, parameters=PARAMS))["price"]
        samples = []
        with torch.no_grad():
            for nx, ny, steps in [(21, 13, 15), (41, 25, 30), (81, 49, 60)]:
                axes = [torch.linspace(math.log(20.), math.log(400.), nx, dtype=torch.float64).tolist(),
                        (torch.linspace(0., math.sqrt(.5), ny, dtype=torch.float64)**2).tolist()]
                b, l = heston_fields(p, axes)
                start = time.perf_counter()
                value = float(fd_latent(b, l, p[4:], axes=axes, time_steps=steps, mode="adjoint"))
                elapsed = time.perf_counter()-start
                reference = stock(dict(axes=axes, parameters=PARAMS, time_steps=steps))["price"]
                samples.append(dict(grid=[nx, ny], steps=steps, price=value, stock_fd=reference,
                                    analytic_error=abs(value-oracle), stock_fd_error=abs(value-reference),
                                    inference_seconds=elapsed))
        assert samples[-1]["analytic_error"] < .04
        assert samples[-1]["analytic_error"] < samples[0]["analytic_error"]/3
        assert max(s["stock_fd_error"] for s in samples) < 2e-10
        return dict(analytic_heston=oracle, refinement=samples, runtime=qos)
    if kind == "heston_vjp":
        p = torch.tensor(PARAMS, dtype=torch.float64, requires_grad=True)
        b, l = heston_fields(p)
        value = fd_latent(b, l, p[4:], axes=AXES, time_steps=5, mode=case["mode"])
        grad, = torch.autograd.grad(value, p)
        reference = stock(dict(axes=AXES, parameters=PARAMS, time_steps=5, bump=1e-5))
        return dict(mode=case["mode"], price=float(value.detach()), gradients=grad.tolist(), reference=reference,
                    max_gradient_error=max(abs(a-b) for a, b in zip(grad.tolist(), reference["gradients"])))

    torch.manual_seed(17)
    m, nx, ny = case["latent_dimensions"], case["nx"], case["ny"]
    axes = [torch.linspace(math.log(40.), math.log(230.), nx, dtype=torch.float64).tolist()]
    axes += [torch.linspace(-1., 1., ny, dtype=torch.float64).tolist()]*m
    grid = state_grid(axes)
    features = torch.stack([torch.cat((torch.full((len(grid), 1), t, dtype=torch.float64),
        grid[:, :1]-math.log(100.), grid[:, 1:]), -1) for t in [0., .5, 1.]])
    network = NeuralDiffusion(m)
    z = torch.tensor([math.log(101.)]+[.08]*m, dtype=torch.float64, requires_grad=True)

    def workload():
        start = time.perf_counter()
        b, l = network(features)
        value = fd_latent(b, l, z, axes=axes, times=[0., .5, 1.], time_steps=case["steps"], mode=case["mode"])
        loss = (value-9.).square()
        middle = time.perf_counter()
        gradients = torch.autograd.grad(loss, (*network.parameters(), z))
        end = time.perf_counter()
        return value.detach(), [g.detach() for g in gradients], [(middle-start)*1000, (end-middle)*1000, (end-start)*1000]

    result = dict(case=case, runtime=qos, nodes=len(grid))
    if kind == "memory":
        before = process_memory()
        value, grads, _ = workload()
        after = process_memory()
        result.update(before=before, after=after, price=float(value), gradient_norm=float(torch.cat([g.flatten() for g in grads]).norm()))
    elif kind == "structure":
        saves = []
        def pack(t):
            saves.append(t.numel()*t.element_size())
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            b, l = network(features)
            value = fd_latent(b, l, z, axes=axes, times=[0., .5, 1.], time_steps=case["steps"], mode=case["mode"])
            loss = (value-9.).square()
            seen, pending = set(), [loss.grad_fn]
            while pending:
                fn = pending.pop()
                if fn is None or fn in seen:
                    continue
                seen.add(fn)
                pending.extend(v for v, _ in fn.next_functions)
            result.update(graph_nodes=len(seen), saved_tensor_events=len(saves), saved_tensor_bytes=sum(saves))
            torch.autograd.grad(loss, (*network.parameters(), z))
    else:
        for _ in range(2):
            workload()
            gc.collect()
        samples = []
        for _ in range(5):
            value, grads, times = workload()
            samples.append(times)
            gc.collect()
        result.update(samples_ms=samples, median_ms=[statistics.median(s[i] for s in samples) for i in range(3)],
                      price=float(value), gradients=[g.tolist() for g in grads])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmarks/neural_sde_phase21.json")
    parser.add_argument("--windows-qos", choices=["system", "high"], default="system")
    parser.add_argument("--affinity", type=int)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    cases = [dict(kind="heston")]
    for mode in ["full", "adjoint"]:
        cases.append(dict(kind="heston_vjp", mode=mode))
        for m in [1, 2]:
            cases.append(dict(kind="training", mode=mode, latent_dimensions=m))
            for kind in ["timing", "memory", "structure"]:
                cases.append(dict(kind=kind, mode=mode, latent_dimensions=m, nx=9, ny=5, steps=4))
        for kind in ["timing", "memory", "structure"]:
            cases.append(dict(kind=kind, mode=mode, latent_dimensions=1, nx=21, ny=13, steps=8))
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  baseline_git="4e90e37", quantlib_pin=subprocess.check_output(
                      ["git", "-C", str(ROOT / "third_party/QuantLib"), "rev-parse", "HEAD"], text=True).strip(),
                  python_version=sys.version, torch_version=version("torch"), platform=platform.platform(),
                  notes="Fresh sequential CPU float64 workers; timing excludes warmup. Memory and graph hooks use separate workers. No GPU or cross-milestone speed claim.", results=[])
    for case in cases:
        case.update(windows_qos=args.windows_qos, affinity=args.affinity)
        print(json.dumps(case), flush=True)
        completed = subprocess.run([sys.executable, __file__, "--worker"], input=json.dumps(case),
                                   text=True, capture_output=True, check=True, cwd=ROOT)
        result["results"].append(json.loads(completed.stdout))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    paths = [ROOT / "cpp/coefficient_pde.cpp", ROOT / "python/quantlibtorch/latent.py", Path(__file__),
             ROOT / "examples/neural_latent_sde.py", ROOT / "scripts/prepare_quantlib.py", ROOT / "CMakeLists.txt"]
    paths += list((ROOT / "tests").glob("*latent*.py"))
    result["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    native = next(p for p in (ROOT / "python/quantlibtorch/_native").glob("*native.*")
                  if p.suffix in (".dll", ".so", ".dylib"))
    result["binary_sha256"] = hashlib.sha256(native.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
