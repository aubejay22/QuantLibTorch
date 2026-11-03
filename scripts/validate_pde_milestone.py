"""Write checked, reproducible numerical evidence for structured QuantLib PDE AD.

Dense matrices are restricted to small independent validation references.
The output must be a new path; an existing report is never overwritten.
Stock basket references, when requested, run in a separate Python process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

from bootstrap import QUANTLIB_COMMIT
from investigate_fd import ROOT, native_fingerprint


TOLERANCES = {
    "directional_solution": {"rtol": 2e-12, "atol": 2e-13},
    "directional_vjp": {"rtol": 2e-11, "atol": 2e-12},
    "normalized_residual": 5e-13,
    "adi_solution": {"rtol": 2e-11, "atol": 2e-12},
    "adi_vjp": {"rtol": 2e-10, "atol": 2e-12},
    "manufactured_solution": {"rtol": 2e-12, "atol": 2e-12},
    "manufactured_vjp": {"rtol": 2e-11, "atol": 2e-12},
    "stock_price": {"rtol": 2e-12, "atol": 3e-10},
    "stock_vjp": {"rtol": 5e-6, "atol": 3e-6},
}


def hashes(paths):
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def fingerprints():
    libraries = [p for p in (ROOT/"python/quantlibtorch/_native").glob("quantlibtorch_native.*")
                 if p.suffix in {".dll", ".dylib", ".so"}]
    if not libraries:
        raise RuntimeError("No compiled QuantLibTorch library found")
    references = [ROOT/"tests"/name for name in
                  ("test_structured_ops.py", "test_pde_adjoint.py", "stock_basket_reference.py")]
    scripts = [ROOT/"scripts"/name for name in
               ("validate_pde_milestone.py", "investigate_fd.py", "bootstrap.py")]
    overlay = ROOT/"build/native/quantlib-overlay"
    if not overlay.is_dir():
        raise RuntimeError("No generated QuantLib overlay found")
    return {
        "native_source_sha256": native_fingerprint(),
        "native_binary_sha256": hashes(libraries),
        "python_and_reference_sha256": hashes(
            [*sorted((ROOT/"python/quantlibtorch").glob("*.py")), *references, *scripts]),
        "build_configuration_sha256": hashes(
            [ROOT/"CMakeLists.txt", ROOT/"pyproject.toml", *sorted((ROOT/"cmake").glob("*.cmake"))]),
        "generated_overlay_sha256": hashes(p for p in overlay.rglob("*") if p.is_file()),
    }


def git(directory, *arguments):
    return subprocess.check_output(["git", "-C", str(directory), *arguments],
                                   text=True, encoding="utf-8").strip()


def checked_difference(torch, actual, expected, tolerance):
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError("A numerical comparison contains nonfinite values")
    torch.testing.assert_close(actual, expected, **TOLERANCES[tolerance])
    return (actual-expected).abs().max().item()


def checked_residual(torch, matrix, solution, rhs):
    numerator = torch.linalg.vector_norm(matrix @ solution-rhs).item()
    denominator = (torch.linalg.matrix_norm(matrix)*torch.linalg.vector_norm(solution)
                   + torch.linalg.vector_norm(rhs)).item()
    result = numerator/max(denominator, torch.finfo(rhs.dtype).tiny)
    if not math.isfinite(result) or result > TOLERANCES["normalized_residual"]:
        raise AssertionError(f"Normalized residual {result} exceeds its tolerance")
    return result


def numbers(tensor):
    return tensor.detach().tolist()


def family_summary(records):
    result = {
        "cases": len(records),
        "max_solution_difference": max(r["solution_difference"] for r in records),
        "max_vjp_difference": max(max(r["vjp_differences"].values()) for r in records),
    }
    for name in ("normalized_primal_residual", "normalized_adjoint_residual"):
        if name in records[0]:
            result["max_"+name] = max(r[name] for r in records)
    return result


def directional_evidence(torch, reference):
    records = []
    for dimensions, direction in reference.DIRECTIONS:
        if len(dimensions) < 2:
            continue
        for adjoint in (False, True):
            for negative in (False, True):
                _, inputs = reference.directional_inputs(math.prod(dimensions), negative)
                bands, rhs, scales = inputs
                original = tuple(value.detach().clone() for value in inputs)
                seed = torch.linspace(-.7, .9, rhs.numel(), dtype=torch.float64)
                actual = torch.ops.quantlibtorch._directional_solve_probe(
                    *inputs, dimensions, direction, adjoint)
                expected = reference.dense_directional_equation(*inputs, dimensions, direction)
                gradients = torch.autograd.grad(actual, inputs, seed)
                targets = torch.autograd.grad(expected, inputs, seed)
                difference = checked_difference(torch, actual, expected, "directional_solution")
                vjps = {name: checked_difference(torch, value, target, "directional_vjp")
                        for name, value, target in zip(("bands", "rhs", "scales"), gradients, targets)}
                with torch.no_grad():
                    order = reference.elimination_order(dimensions, direction)
                    matrix = torch.diag(scales[0]*bands[1]+scales[1])
                    matrix.index_put_((order[1:], order[:-1]), scales[0]*bands[0, order[1:]],
                                      accumulate=True)
                    matrix.index_put_((order[:-1], order[1:]), scales[0]*bands[2, order[:-1]],
                                      accumulate=True)
                    primal = checked_residual(torch, matrix, actual, rhs)
                    # The RHS VJP is the native adjoint solution in original layout.
                    adjoint_residual = checked_residual(torch, matrix.T, gradients[1], seed)
                for value, saved in zip(inputs, original):
                    torch.testing.assert_close(value, saved, rtol=0, atol=0)
                records.append({
                    "dimensions": dimensions, "direction": direction,
                    "mode": "adjoint" if adjoint else "full", "negative_views": negative,
                    "inputs": dict(zip(("bands", "rhs", "scales"), map(numbers, inputs))),
                    "seed": numbers(seed), "solution": numbers(actual), "reference_solution": numbers(expected),
                    "vjps": dict(zip(("bands", "rhs", "scales"), map(numbers, gradients))),
                    "reference_vjps": dict(zip(("bands", "rhs", "scales"), map(numbers, targets))),
                    "solution_difference": difference, "vjp_differences": vjps,
                    "normalized_primal_residual": primal,
                    "normalized_adjoint_residual": adjoint_residual,
                })
    return {"summary": family_summary(records), "cases": records}


def adi_evidence(torch, reference):
    records = []
    for x, y in reference.GRIDS:
        for scheme in reference.SCHEMES:
            for steps in (1, 3):
                for adjoint in (False, True):
                    inputs = tuple(value.requires_grad_() for value in reference.problem(x, y))
                    original = tuple(value.detach().clone() for value in inputs)
                    seed = reference.weights_like(inputs[1])
                    actual = reference.native(*inputs, x, y, horizon=.13, steps=steps,
                                              scheme=scheme, adjoint=adjoint)
                    expected = reference.dense_rollback(*inputs, x, y, .13, steps, scheme)
                    gradients = torch.autograd.grad(actual, inputs, seed)
                    targets = torch.autograd.grad(expected, inputs, seed)
                    difference = checked_difference(torch, actual, expected, "adi_solution")
                    names = ("drift_x", "drift_y", "diffusion_x", "diffusion_y", "mixed", "killing")
                    vjps = {name: checked_difference(torch, value, target, "adi_vjp")
                            for name, value, target in zip(names, gradients[0], targets[0])}
                    vjps["terminal"] = checked_difference(torch, gradients[1], targets[1], "adi_vjp")
                    for value, saved in zip(inputs, original):
                        torch.testing.assert_close(value, saved, rtol=0, atol=0)
                    records.append({
                        "x": x, "y": y, "scheme": scheme, "steps": steps, "horizon": .13,
                        "mode": "adjoint" if adjoint else "full",
                        "inputs": dict(zip(("coefficients", "terminal"), map(numbers, inputs))),
                        "seed": numbers(seed), "solution": numbers(actual),
                        "reference_solution": numbers(expected),
                        "vjps": dict(zip(("coefficients", "terminal"), map(numbers, gradients))),
                        "reference_vjps": dict(zip(("coefficients", "terminal"), map(numbers, targets))),
                        "solution_difference": difference, "vjp_differences": vjps,
                    })
    return {"summary": family_summary(records), "cases": records}


def manufactured_evidence(torch, reference):
    records = []
    x, y = reference.GRIDS[0]
    yy, xx = torch.meshgrid(torch.tensor(y, dtype=torch.float64),
                            torch.tensor(x, dtype=torch.float64), indexing="ij")
    zero, seed = torch.zeros_like(xx), reference.weights_like(xx)
    for scheme in reference.SCHEMES:
        for adjoint in (False, True):
            amplitude = torch.tensor(.05, dtype=torch.float64, requires_grad=True)
            coefficients = torch.stack((zero, zero, zero+.12, zero+.08,
                                        amplitude.expand_as(xx), zero))
            actual = reference.native(coefficients, xx*yy, x, y, horizon=.19, steps=4,
                                      scheme=scheme, adjoint=adjoint)
            expected = xx*yy+.19*amplitude
            gradient, = torch.autograd.grad(actual, amplitude, seed)
            target = .19*seed.sum()
            records.append({
                "x": x, "y": y, "scheme": scheme, "mode": "adjoint" if adjoint else "full",
                "horizon": .19, "steps": 4, "mixed_amplitude": .05,
                "diffusion_x": .12, "diffusion_y": .08, "terminal": "x*y",
                "seed": numbers(seed), "solution": numbers(actual), "reference_solution": numbers(expected),
                "vjp": gradient.item(), "reference_vjp": target.item(),
                "solution_difference": checked_difference(torch, actual, expected, "manufactured_solution"),
                "vjp_differences": {"mixed_amplitude": checked_difference(
                    torch, gradient, target, "manufactured_vjp")},
            })
    return {"summary": family_summary(records), "cases": records}


def stock_evidence(torch, executable):
    from quantlibtorch import fd_basket

    records = []
    for scheme in ("douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer"):
        request = dict(volatility1=.22, volatility2=.28, correlation=.35, scheme=scheme,
                       bump=1e-5, time_steps=4, grid_points=[11, 9])
        command = [executable, str(ROOT/"tests/stock_basket_reference.py")]
        result = subprocess.run(command, input=json.dumps(request), text=True,
                                capture_output=True, check=True)
        expected = json.loads(result.stdout)
        if expected.get("quantlib_version") != "1.40":
            raise RuntimeError("Stock reference did not report pinned QuantLib 1.40")
        targets = torch.tensor(expected["coefficient_derivatives"], dtype=torch.float64)
        if targets.shape != (3,):
            raise RuntimeError("Stock reference did not return all three coefficient derivatives")
        for mode in ("full", "adjoint"):
            values = torch.tensor([.22, .28, .35], dtype=torch.float64, requires_grad=True)
            actual = fd_basket(values[0], values[1], values[2], scheme=scheme, mode=mode,
                               time_steps=4, grid_points=(11, 9))
            gradient, = torch.autograd.grad(actual, values)
            price = torch.tensor(expected["price"], dtype=torch.float64)
            records.append({
                "scheme": scheme, "mode": mode, "oracle_command": command, "oracle_request": request,
                "oracle_result": expected, "solution": actual.item(), "vjps": numbers(gradient),
                "solution_difference": checked_difference(torch, actual, price, "stock_price"),
                "vjp_differences": {name: checked_difference(torch, value, target, "stock_vjp")
                                    for name, value, target in zip(
                                        ("volatility1", "volatility2", "correlation"), gradient, targets)},
            })
    return {"enabled": True, "summary": family_summary(records), "cases": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stock-basket", action="store_true",
                        help="Require isolated stock QuantLib basket price and derivative checks")
    parser.add_argument("--stock-python", default=sys.executable,
                        help="Python executable containing stock QuantLib 1.40 (used only in a subprocess)")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Evidence output already exists: {args.output}")
    quantlib = ROOT/"third_party/QuantLib"
    pin = git(quantlib, "rev-parse", "HEAD")
    dirty = git(quantlib, "status", "--porcelain")
    if pin != QUANTLIB_COMMIT or dirty:
        raise RuntimeError("Numerical evidence requires the clean pinned QuantLib checkout")
    before = fingerprints()

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(ROOT/"tests"))
    sys.path.insert(0, str(ROOT/"python"))
    import test_structured_ops as structured
    import test_pde_adjoint as pde

    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
        "git_commit": git(ROOT, "rev-parse", "HEAD"),
        "git_branch": git(ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_worktree": git(ROOT, "status", "--porcelain"),
        "quantlib_commit": pin, "quantlib_worktree": dirty,
        "python_version": sys.version, "torch_version": torch.__version__,
        "platform": platform.platform(), "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "environment": {key: value for key, value in sorted(os.environ.items())
                        if key.startswith(("QLT_", "OMP_", "MKL_"))},
        **before, "tolerances": TOLERANCES,
        "directional": directional_evidence(torch, structured),
        "adi": adi_evidence(torch, pde),
        "manufactured": manufactured_evidence(torch, pde),
        "stock_basket": (stock_evidence(torch, args.stock_python) if args.stock_basket else
                         {"enabled": False, "reason": "--stock-basket was not requested"}),
        "notes": [
            "Every reported comparison is finite and checked against its stated tolerance.",
            "Directional residual norm is ||Ax-b||2/(||A||F ||x||2+||b||2); transpose uses the RHS VJP.",
            "Dense matrices occur only in small independent validation references.",
            "ADI schemes: 0 Douglas, 1 Hundsdorfer, 2 modified Craig-Sneyd, 3 Craig-Sneyd.",
            "Axes, mesh, horizon and step counts are fixed; coefficient fields retain original graphs.",
            "No timing or peak-memory claim is made by this numerical validation.",
        ],
    }
    if fingerprints() != before or git(quantlib, "rev-parse", "HEAD") != pin or git(quantlib, "status", "--porcelain"):
        raise RuntimeError("Source, native binary, or pinned QuantLib changed during validation")
    report["validated"] = True
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    encoded = json.dumps(report, indent=2, allow_nan=False)+"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps({name: report[name].get("summary", {"enabled": False})
                      for name in ("directional", "adi", "manufactured", "stock_basket")}, indent=2))


if __name__ == "__main__":
    main()
