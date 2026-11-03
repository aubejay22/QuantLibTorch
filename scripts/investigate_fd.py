"""Isolated CPU timing, process memory, graph census, and operator profiling.

Each case/task runs in a fresh process. Profiling and save hooks NEVER wrap
timing or process-memory measurements. Set --variants to reproduce ablations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import date, datetime, timezone
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"eager": (0, 0), "constants": (1, 0), "cache": (0, 1), "optimized": (1, 1)}


def process_memory():
    """OS process working set and high water, including native allocations."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
                    "nonpaged", "pagefile", "peak_pagefile", "private")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"rss_bytes": counters.rss, "peak_rss_bytes": counters.peak_rss,
                "private_bytes": counters.private}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    rss = None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])*1024
    return {"rss_bytes": rss, "peak_rss_bytes": peak}


def worker(case):
    from runtime_controls import worker_qos
    qos = worker_qos(case.get("windows_qos", "system"), case.get("affinity"))
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from quantlibtorch import fd_vanilla

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # This affects only this worker's backward execution, not the pricing API
    # or any caller's process-wide thread configuration.
    torch.autograd.set_multithreading_enabled(not case.get("single_thread_backward", False))
    actual = list(torch.ops.quantlibtorch._performance_config())
    if actual != [bool(x) for x in VARIANTS[case["variant"]]]:
        raise RuntimeError("native performance switches do not match the requested case")
    inference = case["mode"] == "inference"
    kwargs = dict(time_steps=case["steps"], grid_points=case["nodes"],
                  mode="full" if inference else case["mode"], kernels=case.get("kernels", "scalar"))
    if case.get("contract", "european") == "bermudan":
        kwargs.update(exercise="bermudan", option_type="put", spot=96.,
                      exercise_dates=[date(2025, 7, 1), date(2026, 1, 1)],
                      cash_dividends=[(date(2025, 7, 1), 1.7)])
    if case["surface"] in {"surface", "neural"}:
        base = (.21 + .025*torch.sin(torch.arange(15, dtype=torch.float64))).reshape(3, 5)
        kwargs.update(times=[0., .37, 1.], log_spots=[math.log(s) for s in (40., 70., 100., 140., 250.)])
        if case["surface"] == "neural":
            base = torch.linspace(-.3, .4, 17, dtype=torch.float64)
            ts, xs = torch.meshgrid(torch.tensor(kwargs["times"], dtype=torch.float64),
                                   torch.tensor(kwargs["log_spots"], dtype=torch.float64)-math.log(100.), indexing="ij")
            coordinates = torch.stack((ts, xs), dim=-1)
    else:
        base = torch.tensor(.2, dtype=torch.float64)

    def price(sigma):
        with torch.no_grad() if inference else nullcontext():
            if case["surface"] == "neural":
                theta = sigma
                hidden = torch.tanh(coordinates @ theta[:8].reshape(2, 4) + theta[8:12])
                sigma = .1 + .3*torch.sigmoid(hidden @ theta[12:16] + theta[16])
            return fd_vanilla(sigma, **kwargs)

    def objective(value):
        return (value-9.).square() if case["surface"] == "neural" else value

    def run():
        sigma = base.clone().requires_grad_()
        start_cpu = time.process_time()
        start = time.perf_counter()
        value = price(sigma)
        loss = objective(value)
        middle = time.perf_counter()
        middle_cpu = time.process_time()
        gradient = None if inference else torch.autograd.grad(loss, sigma)[0]
        end = time.perf_counter()
        end_cpu = time.process_time()
        result = dict(forward_seconds=middle-start, backward_seconds=end-middle,
                      forward_cpu_seconds=middle_cpu-start_cpu, backward_cpu_seconds=end_cpu-middle_cpu,
                      price=value.item(), loss=loss.item(), gradient=None if gradient is None else gradient.tolist())
        # Graph destruction is outside forward/backward, but belongs in training
        # cost. Report it rather than allowing it to contaminate the next sample.
        cleanup_cpu = time.process_time()
        cleanup = time.perf_counter()
        del value, loss, gradient, sigma
        result["cleanup_seconds"] = time.perf_counter()-cleanup
        result["cleanup_cpu_seconds"] = time.process_time()-cleanup_cpu
        result["total_seconds"] = end-start+result["cleanup_seconds"]
        result["total_cpu_seconds"] = end_cpu-start_cpu+result["cleanup_cpu_seconds"]
        return result

    task = case["task"]
    result = {"case": case, "performance_config": actual, "worker_qos": qos,
              "gradient_target": "17 neural parameters, squared price loss" if case["surface"] == "neural" else "local volatility, NPV"}
    if task == "timing":
        result["warmup"] = [run() for _ in range(case["warmup"])]
        samples = [run() for _ in range(case["repeats"])]
        result["samples"] = samples
        result["median"] = {key: statistics.median(s[key] for s in samples)
                            for key in ("forward_seconds", "backward_seconds", "cleanup_seconds", "total_seconds", "total_cpu_seconds")}
        result.update(price=samples[-1]["price"], gradient=samples[-1]["gradient"])
    elif task == "memory":
        sigma = base.clone().requires_grad_()
        gc.collect()
        result["before"] = process_memory()
        value = price(sigma)
        loss = objective(value)
        result["after_forward"] = process_memory()
        gradient = None if inference else torch.autograd.grad(loss, sigma)[0]
        result["after_backward"] = process_memory()
        result.update(price=value.item(), gradient=None if gradient is None else gradient.tolist())
        del value, loss, gradient, sigma
        gc.collect()
        result["after_cleanup"] = process_memory()
        result["peak_rss_growth_bytes"] = max(0, result["after_cleanup"]["peak_rss_bytes"]
                                               - result["before"]["peak_rss_bytes"])
    elif task == "graph":
        saved = []

        def pack(tensor):
            saved.append(tensor.numel()*tensor.element_size())
            return tensor

        sigma = base.clone().requires_grad_()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            value = price(sigma)
            loss = objective(value)
        seen, counts, pending = set(), Counter(), [loss.grad_fn]
        while pending:
            node = pending.pop()
            if node is None or node in seen:
                continue
            seen.add(node)  # Hold the objects, not recycled Python id() values.
            counts[type(node).__name__] += 1
            pending.extend(child for child, _ in node.next_functions)
        result.update(saved_tensor_count=len(saved), saved_tensor_bytes_with_repeated_saves=sum(saved),
                      graph_nodes=len(seen), node_types=dict(counts.most_common()))
        if not inference:
            torch.autograd.grad(loss, sigma)
    elif task == "profile":
        # Warm the engine and autograd before enabling profiling. Profiler
        # overhead is substantial for tiny scalar ops; never treat this as a
        # substitute for the uninstrumented wall-clock timing task.
        for _ in range(case["warmup"]):
            run()
        sigma = base.clone().requires_grad_()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            with torch.profiler.record_function("phase3::forward"):
                value = price(sigma)
                loss = objective(value)
            if not inference:
                with torch.profiler.record_function("phase3::backward"):
                    torch.autograd.grad(loss, sigma)
        scopes = []
        for item in prof.key_averages():
            if item.key.startswith(("qlt::", "phase3::")):
                scopes.append(dict(name=item.key, calls=item.count, self_cpu_us=item.self_cpu_time_total,
                                   inclusive_cpu_us=item.cpu_time_total))
        groups = defaultdict(lambda: Counter(calls=0, self_cpu_us=0.))
        outer_scopes = defaultdict(lambda: Counter(calls=0, inclusive_cpu_us=0.))
        for event in prof.events():
            if not event.name.startswith(("aten::", "qlt::")):
                continue
            ancestor, phase, recursive_scope = event.cpu_parent, "unattributed", False
            while ancestor:
                if ancestor.name == event.name:
                    recursive_scope = True
                if ancestor.name.startswith("phase3::"):
                    phase = ancestor.name.removeprefix("phase3::")
                    break
                ancestor = ancestor.cpu_parent
            if event.name.startswith("qlt::"):
                if not recursive_scope:
                    group = outer_scopes[(phase, event.name)]
                    group["calls"] += 1
                    group["inclusive_cpu_us"] += event.cpu_time_total
                continue
            group = groups[(phase, event.name)]
            group["calls"] += 1
            group["self_cpu_us"] += event.self_cpu_time_total
        result["scopes"] = scopes
        result["outermost_scopes"] = [dict(phase=phase, name=name, **counts)
                                     for (phase, name), counts in outer_scopes.items()]
        result["operators"] = [dict(phase=phase, name=name, **counts) for (phase, name), counts
                               in sorted(groups.items(), key=lambda pair: -pair[1]["self_cpu_us"])]
        result["aten_calls"] = dict(Counter({phase: sum(c["calls"] for (p, _), c in groups.items() if p == phase)
                                             for phase, _ in groups}))
        if case.get("trace"):
            trace = Path(case["trace"])
            trace.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(trace))
            result["trace"] = str(trace)
    return result


def native_fingerprint():
    paths = [ROOT / "CMakeLists.txt", ROOT / "scripts/prepare_quantlib.py"]
    paths += sorted((ROOT / "cpp").rglob("*.cpp")) + sorted((ROOT / "cpp").rglob("*.hpp"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_results(payload):
    """Reject price/VJP regressions across modes, ablations and task processes."""
    references, comparisons = {}, 0
    max_price, max_gradient = 0., 0.

    def flatten(value):
        if isinstance(value, list):
            return [x for item in value for x in flatten(item)]
        return [value]

    for result in payload["results"]:
        if "price" not in result:
            continue
        case = result["case"]
        key = (case["steps"], case["nodes"], case["surface"], case.get("contract", "european"))
        reference = references.setdefault(key, {})
        if "price" in reference:
            error = abs(result["price"]-reference["price"])
            max_price = max(max_price, error)
            if not math.isfinite(error) or error > 2e-10:
                raise RuntimeError(f"price regression in {case}: {error}")
            comparisons += 1
        reference.setdefault("price", result["price"])
        gradient = result.get("gradient")
        if gradient is not None:
            if "gradient" in reference:
                values, expected = flatten(gradient), flatten(reference["gradient"])
                if len(values) != len(expected):
                    raise RuntimeError(f"gradient shape changed in {case}")
                for x, y in zip(values, expected):
                    error = abs(x-y)
                    max_gradient = max(max_gradient, error)
                    if not math.isfinite(error) or error > 2e-9+1e-9*abs(y):
                        raise RuntimeError(f"VJP regression in {case}: {error}")
                comparisons += 1
            reference.setdefault("gradient", gradient)
    for result in payload["stock"]:
        reference = references.get((result["steps"], result["nodes"], "constant", "european"))
        if not reference:
            continue
        if abs(result["price"]-reference["price"]) > 2e-10:
            raise RuntimeError("stock price mismatch")
        if "gradient" in reference and abs(reference["gradient"]-result["parallel_vol_shift_derivative"]) > 2e-6+3e-6*abs(reference["gradient"]):
            raise RuntimeError("stock bumped VJP mismatch")
        comparisons += 1
    return dict(comparisons=comparisons, max_price_difference=max_price,
                max_gradient_difference=max_gradient)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--grids", default="10x31,30x61")
    parser.add_argument("--variants", default="eager,optimized")
    parser.add_argument("--kernels", default="scalar", help="scalar,apply,vectorized,pcr,auto; pcr requires full/inference")
    parser.add_argument("--contracts", default="european", help="european,bermudan (coincident dividend/exercise)")
    parser.add_argument("--modes", default="inference,full,adjoint")
    parser.add_argument("--surfaces", default="constant,surface", help="constant,surface,neural (includes network and squared loss)")
    parser.add_argument("--tasks", default="timing,memory,graph")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--no-stock", action="store_true")
    parser.add_argument("--single-thread-backward", action="store_true",
                        help="diagnose autograd worker scheduling separately from intra/inter-op threads")
    parser.add_argument("--windows-qos", choices=("system", "high"), default="system",
                        help="optional worker-only opt-out of Windows execution-speed throttling")
    parser.add_argument("--native-adjoint", type=int, choices=(0, 1), default=1,
                        help="native-value execution inside the solve VJP boundary")
    parser.add_argument("--packed-arrays", type=int, choices=(0, 1), default=1)
    parser.add_argument("--lazy-arrays", type=int, choices=(0, 1), default=1)
    parser.add_argument("--vector-sampling", type=int, choices=(0, 1), default=1)
    parser.add_argument("--array-kernels", type=int, choices=(0, 1), default=1)
    parser.add_argument("--price-only", type=int, choices=(0, 1), default=1)
    parser.add_argument("--spline-kernels", type=int, choices=(0, 1), default=1)
    parser.add_argument("--tridiagonal-adjoint", type=int, choices=(0, 1), default=1)
    parser.add_argument("--pcr-roll", type=int, choices=(0, 1), default=1)
    parser.add_argument("--spatial-cache", type=int, choices=(0, 1), default=1)
    parser.add_argument("--operator-adjoint", type=int, choices=(0, 1), default=1)
    parser.add_argument("--step-adjoint", type=int, choices=(0, 1), default=1)
    parser.add_argument("--assembly-adjoint", type=int, choices=(0, 1), default=1)
    parser.add_argument("--pcr-fused", type=int, choices=(0, 1), default=0)
    parser.add_argument("--pcr-native-stage", type=int, choices=(0, 1), default=1)
    parser.add_argument("--native-solve-checks", type=int, choices=(0, 1), default=1)
    parser.add_argument("--assembly-arithmetic", type=int, choices=(0, 1), default=1)
    parser.add_argument("--operator-arithmetic", type=int, choices=(0, 1), default=1)
    parser.add_argument("--arithmetic-native-forward", type=int, choices=(0, 1), default=1)
    parser.add_argument("--thomas-arithmetic", type=int, choices=(0, 1), default=0)
    parser.add_argument("--thomas-vector-backward", type=int, choices=(0, 1), default=0)
    parser.add_argument("--adjoint-arithmetic-forward", type=int, choices=(0, 1), default=0)
    parser.add_argument("--step-native-arithmetic", type=int, choices=(0, 1), default=0)
    parser.add_argument("--basis-cache", type=int, choices=(0, 1), default=0)
    parser.add_argument("--full-tensor-step", type=int, choices=(0, 1), default=0)
    parser.add_argument("--time-variance-arithmetic", type=int, choices=(0, 1), default=0)
    parser.add_argument("--step-preparation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--assembled-bands", type=int, choices=(0, 1), default=0)
    parser.add_argument("--adjoint-assembled-bands", type=int, choices=(0, 1), default=0)
    parser.add_argument("--diffusion-selected-grads", type=int, choices=(0, 1), default=0)
    parser.add_argument("--affinity", type=int, help="pin only each benchmark worker to this logical CPU")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    variants, modes = args.variants.split(","), args.modes.split(",")
    surfaces, tasks = args.surfaces.split(","), args.tasks.split(",")
    kernels, contracts = args.kernels.split(","), args.contracts.split(",")
    try:
        grids = [tuple(map(int, g.split("x"))) for g in args.grids.split(",")]
        if (args.repeats < 1 or args.warmup < 1 or any(len(g) != 2 or g[0] < 1 or g[1] < 5 for g in grids)
                or (args.affinity is not None and not 0 <= args.affinity < (os.cpu_count() or 1))
                or set(variants)-VARIANTS.keys() or set(modes)-{"inference", "full", "adjoint"}
                or set(kernels)-{"scalar", "apply", "vectorized", "pcr", "auto"} or set(contracts)-{"european", "bermudan"}
                or ("pcr" in kernels and "adjoint" in modes)
                or set(surfaces)-{"constant", "surface", "neural"} or set(tasks)-{"timing", "memory", "graph", "profile"}):
            raise ValueError
    except ValueError:
        parser.error("invalid grid, variant, mode, surface, task, warmup, or repeat count")
    import torch
    payload = dict(measured_at_utc=datetime.now(timezone.utc).isoformat(),
                   environment=dict(python=platform.python_version(), torch=torch.__version__,
                                    os=platform.platform(), cpu=platform.processor(), torch_threads=1,
                                    torch_interop_threads=1, dtype="float64", device="cpu"),
                   native_source_sha256=native_fingerprint(),
                   windows_qos=args.windows_qos,
                   optimization_flags={"QLT_NATIVE_ADJOINT": args.native_adjoint,
                                       "QLT_PACKED_ARRAYS": args.packed_arrays,
                                       "QLT_LAZY_ARRAYS": args.lazy_arrays,
                                       "QLT_VECTOR_SAMPLING": args.vector_sampling,
                                       "QLT_ARRAY_KERNELS": args.array_kernels,
                                       "QLT_PRICE_ONLY": args.price_only,
                                       "QLT_SPLINE_KERNELS": args.spline_kernels,
                                       "QLT_TRIDIAGONAL_ADJOINT": args.tridiagonal_adjoint,
                                       "QLT_PCR_ROLL": args.pcr_roll,
                                       "QLT_SPATIAL_CACHE": args.spatial_cache,
                                       "QLT_OPERATOR_ADJOINT": args.operator_adjoint,
                                       "QLT_STEP_ADJOINT": args.step_adjoint,
                                       "QLT_ASSEMBLY_ADJOINT": args.assembly_adjoint,
                                       "QLT_PCR_FUSED": args.pcr_fused,
                                       "QLT_PCR_NATIVE_STAGE": args.pcr_native_stage,
                                       "QLT_NATIVE_SOLVE_CHECKS": args.native_solve_checks,
                                       "QLT_ASSEMBLY_ARITHMETIC": args.assembly_arithmetic,
                                       "QLT_OPERATOR_ARITHMETIC": args.operator_arithmetic,
                                       "QLT_ARITHMETIC_NATIVE_FORWARD": args.arithmetic_native_forward,
                                       "QLT_THOMAS_ARITHMETIC": args.thomas_arithmetic,
                                       "QLT_THOMAS_VECTOR_BACKWARD": args.thomas_vector_backward,
                                       "QLT_ADJOINT_ARITHMETIC_FORWARD": args.adjoint_arithmetic_forward,
                                       "QLT_STEP_NATIVE_ARITHMETIC": args.step_native_arithmetic,
                                       "QLT_BASIS_CACHE": args.basis_cache,
                                       "QLT_FULL_TENSOR_STEP": args.full_tensor_step,
                                       "QLT_TIME_VARIANCE_ARITHMETIC": args.time_variance_arithmetic,
                                       "QLT_STEP_PREPARATION": args.step_preparation,
                                       "QLT_ASSEMBLED_BANDS": args.assembled_bands,
                                       "QLT_ADJOINT_ASSEMBLED_BANDS": args.adjoint_assembled_bands,
                                       "QLT_DIFFUSION_SELECTED_GRADS": args.diffusion_selected_grads},
                   quantlib_source_commit=subprocess.check_output(
                       ["git", "-C", str(ROOT / "third_party/QuantLib"), "rev-parse", "HEAD"], text=True).strip(),
                   variants={name: dict(passive_constants=p, sample_cache=c) for name, (p, c) in VARIANTS.items()},
                   notes=["Fresh subprocess for every case/task; process startup and imports excluded from timing.",
                          "Timing samples exclude warmup. Memory, graph census and profiler are separate processes.",
                          "Process CPU times include all worker threads and may have coarse platform clock resolution; they supplement elapsed wall time.",
                          "Optional high Windows QoS affects only each benchmark worker; no machine power plan or pricing-library policy changes.",
                          "Memory is a cold first price after import; OS peak working set includes interpreter, LibTorch, allocator retention and all native allocations.",
                          "Peak growth subtracts the pre-price process high water; it is not graph memory or unique tensor storage.",
                          "Graph census counts reachable autograd nodes; saved bytes count repeated forward saves, including unused engine Greeks.",
                          "Profiler scopes are nested; do not add inclusive durations. Profiler timings include instrumentation overhead.",
                          "Stock construction plus forward uses LocalConstantVol in an isolated QuantLib process; no stock gradient timing.",
                          "Surface workload is a deterministic 3x5 nonuniform field; stock oracle only covers constant workload.",
                          "Neural workload includes a deterministic 2-4-1 tanh network, bounded sigmoid volatility, squared NPV loss, and all 17 parameter VJPs in timed work."],
                   results=[], stock=[])

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")

    for steps, nodes in grids:
        if not args.no_stock and "constant" in surfaces:
            response = subprocess.run([sys.executable, str(ROOT / "tests/stock_quantlib_reference.py")],
                input=json.dumps(dict(time_steps=steps, grid_points=nodes, repeats=max(7, args.repeats), bump=1e-5)),
                text=True, capture_output=True, check=True)
            payload["stock"].append(dict(steps=steps, nodes=nodes, **json.loads(response.stdout)))
        for surface in surfaces:
            for variant in variants:
                for mode, kernel, contract in itertools.product(modes, kernels, contracts):
                    for task in tasks:
                        case = dict(steps=steps, nodes=nodes, surface=surface, variant=variant,
                                    kernels=kernel, contract=contract,
                                    mode=mode, task=task, repeats=args.repeats, warmup=args.warmup,
                                    windows_qos=args.windows_qos,
                                    affinity=args.affinity,
                                    single_thread_backward=args.single_thread_backward)
                        label = f"{steps}x{nodes}-{surface}-{variant}-{kernel}-{contract}-{mode}-{task}"
                        if task == "profile" and args.trace_dir:
                            case["trace"] = str((args.trace_dir / (label+".json")).resolve())
                        env = os.environ.copy()
                        env.update(QLT_PASSIVE_CONSTANTS=str(VARIANTS[variant][0]),
                                   QLT_SAMPLE_CACHE=str(VARIANTS[variant][1]),
                                   QLT_NATIVE_ADJOINT=str(args.native_adjoint),
                                   QLT_PACKED_ARRAYS=str(args.packed_arrays),
                                   QLT_LAZY_ARRAYS=str(args.lazy_arrays),
                                   QLT_VECTOR_SAMPLING=str(args.vector_sampling),
                                   QLT_ARRAY_KERNELS=str(args.array_kernels),
                                   QLT_PRICE_ONLY=str(args.price_only),
                                   QLT_SPLINE_KERNELS=str(args.spline_kernels),
                                   QLT_TRIDIAGONAL_ADJOINT=str(args.tridiagonal_adjoint),
                                   QLT_PCR_ROLL=str(args.pcr_roll),
                                   QLT_SPATIAL_CACHE=str(args.spatial_cache),
                                   QLT_OPERATOR_ADJOINT=str(args.operator_adjoint),
                                   QLT_STEP_ADJOINT=str(args.step_adjoint),
                                   QLT_ASSEMBLY_ADJOINT=str(args.assembly_adjoint),
                                   QLT_PCR_FUSED=str(args.pcr_fused),
                                   QLT_PCR_NATIVE_STAGE=str(args.pcr_native_stage),
                                   QLT_NATIVE_SOLVE_CHECKS=str(args.native_solve_checks),
                                   QLT_ASSEMBLY_ARITHMETIC=str(args.assembly_arithmetic),
                                   QLT_OPERATOR_ARITHMETIC=str(args.operator_arithmetic),
                                   QLT_ARITHMETIC_NATIVE_FORWARD=str(args.arithmetic_native_forward),
                                   QLT_THOMAS_ARITHMETIC=str(args.thomas_arithmetic),
                                   QLT_THOMAS_VECTOR_BACKWARD=str(args.thomas_vector_backward),
                                   QLT_ADJOINT_ARITHMETIC_FORWARD=str(args.adjoint_arithmetic_forward),
                                   QLT_STEP_NATIVE_ARITHMETIC=str(args.step_native_arithmetic),
                                   QLT_BASIS_CACHE=str(args.basis_cache),
                                   QLT_FULL_TENSOR_STEP=str(args.full_tensor_step),
                                   QLT_TIME_VARIANCE_ARITHMETIC=str(args.time_variance_arithmetic),
                                   QLT_STEP_PREPARATION=str(args.step_preparation),
                                   QLT_ASSEMBLED_BANDS=str(args.assembled_bands),
                                   QLT_ADJOINT_ASSEMBLED_BANDS=str(args.adjoint_assembled_bands),
                                   QLT_DIFFUSION_SELECTED_GRADS=str(args.diffusion_selected_grads))
                        response = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                            input=json.dumps(case), capture_output=True, text=True, env=env)
                        if response.returncode:
                            raise RuntimeError(f"{label}: {response.stderr}\n{response.stdout}")
                        result = json.loads(response.stdout)
                        payload["results"].append(result)
                        save()
                        summary = result.get("median", result.get("peak_rss_growth_bytes", result.get("graph_nodes", result.get("aten_calls"))))
                        print(label, summary, flush=True)
    payload["validation"] = validate_results(payload)
    save()
    if not args.output:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
