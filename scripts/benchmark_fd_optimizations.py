"""Reproduce incremental engine optimizations with isolated timing workers.

All extra options are forwarded to investigate_fd.py. Each level and each
case/task still runs in fresh processes; only raw JSON is combined here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from investigate_fd import ROOT, validate_results

FLAGS = ("native-adjoint", "packed-arrays", "lazy-arrays", "vector-sampling", "array-kernels", "price-only",
         "tridiagonal-adjoint", "spline-kernels", "pcr-roll", "spatial-cache", "operator-adjoint", "step-adjoint",
         "assembly-adjoint", "pcr-fused", "pcr-native-stage", "native-solve-checks", "assembly-arithmetic",
         "operator-arithmetic", "arithmetic-native-forward", "thomas-arithmetic", "thomas-vector-backward",
         "adjoint-arithmetic-forward", "step-native-arithmetic", "basis-cache", "full-tensor-step",
         "time-variance-arithmetic", "step-preparation", "assembled-bands", "adjoint-assembled-bands",
         "diffusion-selected-grads")
LEVELS = {
    "baseline": (0, 0, 0, 0, 0, 0, 0, 0, 0),
    "solve": (1, 0, 0, 0, 0, 0, 0, 0, 0),
    "packed": (1, 1, 0, 0, 0, 0, 0, 0, 0),
    "sampled": (1, 1, 0, 1, 0, 0, 0, 0, 0),
    "arrays": (1, 1, 0, 1, 1, 0, 0, 0, 0),
    "lazy": (1, 1, 1, 1, 1, 0, 0, 0, 0),
    "final": (1, 1, 1, 1, 1, 1, 0, 0, 0),
    "spline_solve": (1, 1, 1, 1, 1, 1, 1, 0, 0),
    "spline": (1, 1, 1, 1, 1, 1, 1, 1, 0),
    "pcr_rolled": (1, 1, 1, 1, 1, 1, 1, 1, 1),
    "current": (1, 1, 1, 1, 1, 1, 1, 1, 1),
}
# Preserve historical configurations as new independently ablated switches are added.
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["spatial"] = LEVELS["current"][:-1]+(1,)
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["operators"] = LEVELS["spatial"][:-1]+(1,)
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["steps"] = LEVELS["operators"][:-1]+(1,)
LEVELS = {name: flags+(0, 0) for name, flags in LEVELS.items()}
LEVELS["assembly"] = LEVELS["steps"][:-2]+(1, 0)
LEVELS["phase7"] = LEVELS["assembly"]
LEVELS["pcr_fused"] = LEVELS["assembly"][:-1]+(1,)
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["pcr_stage"] = LEVELS["phase7"][:-1]+(1,)
LEVELS["phase8"] = LEVELS["pcr_stage"]
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["native_checks"] = LEVELS["phase8"][:-1]+(1,)
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["assembly_arithmetic"] = LEVELS["native_checks"][:-1]+(1,)
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["operator_arithmetic"] = LEVELS["assembly_arithmetic"][:-1]+(1,)
LEVELS["apply_arithmetic"] = LEVELS["native_checks"][:-1]+(1,)
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["native_arithmetic"] = LEVELS["operator_arithmetic"][:-1]+(1,)
LEVELS["native_assembly"] = LEVELS["assembly_arithmetic"][:-1]+(1,)
LEVELS["native_apply"] = LEVELS["apply_arithmetic"][:-1]+(1,)
LEVELS["phase9"] = LEVELS["native_arithmetic"]
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["thomas_arithmetic"] = LEVELS["phase9"][:-1]+(1,)
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["thomas_vector"] = LEVELS["thomas_arithmetic"][:-1]+(1,)
LEVELS["phase10"] = LEVELS["thomas_vector"]
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["adjoint_arithmetic"] = LEVELS["phase10"][:-1]+(1,)
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["step_arithmetic"] = LEVELS["adjoint_arithmetic"][:-1]+(1,)
LEVELS["step_arithmetic_only"] = LEVELS["phase10"][:-1]+(1,)
LEVELS["phase11"] = LEVELS["step_arithmetic"]
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["basis_cache"] = LEVELS["phase11"][:-1]+(1,)
LEVELS["phase13"] = LEVELS["basis_cache"]
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["full_tensor_step"] = LEVELS["phase13"][:-1]+(1,)
LEVELS["phase14"] = LEVELS["full_tensor_step"]
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["time_variance"] = LEVELS["phase14"][:-1]+(1,)
LEVELS["phase15"] = LEVELS["time_variance"]
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["step_preparation"] = LEVELS["phase15"][:-1]+(1,)
LEVELS["phase16"] = LEVELS["step_preparation"]
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["assembled_bands"] = LEVELS["phase16"][:-1]+(1,)
LEVELS["phase17"] = LEVELS["assembled_bands"]
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["adjoint_bands"] = LEVELS["phase17"][:-1]+(1,)
LEVELS["phase18"] = LEVELS["adjoint_bands"]
LEVELS = {name: flags+(0,) for name, flags in LEVELS.items()}
LEVELS["diffusion_grads"] = LEVELS["phase18"][:-1]+(1,)
LEVELS["phase19"] = LEVELS["diffusion_grads"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="baseline,solve,packed,sampled,arrays,lazy,final")
    parser.add_argument("--kernel-overrides", default="", help="optional level=kernel pairs, comma separated")
    parser.add_argument("--output", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    levels = args.levels.split(",")
    if not levels or set(levels)-LEVELS.keys():
        parser.error(f"levels must be selected from {tuple(LEVELS)}")
    try:
        overrides = dict(pair.split("=") for pair in args.kernel_overrides.split(",") if pair)
        if set(overrides)-set(levels) or set(overrides.values())-{"scalar", "apply", "vectorized", "pcr", "auto"}:
            raise ValueError
    except ValueError:
        parser.error("kernel-overrides requires selected_level=valid_kernel pairs")
    if any(arg.split("=")[0] in {"--"+flag for flag in FLAGS} for arg in forwarded):
        parser.error("optimization flags are controlled by --levels")
    directory = ROOT / "build" / "fd_optimization_workers" / args.output.stem
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"command": sys.argv, "levels": {}, "results": [], "stock": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")

    for level in levels:
        temporary = directory / (level+".json")
        flags = [arg for name, value in zip(FLAGS, LEVELS[level]) for arg in ("--"+name, str(value))]
        if level in overrides:
            flags += ["--kernels", overrides[level]]
        command = [sys.executable, str(ROOT / "scripts/investigate_fd.py"), *forwarded,
                   *flags, "--output", str(temporary)]
        print(f"Optimization level: {level}", flush=True)
        subprocess.run(command, check=True)
        part = json.loads(temporary.read_text(encoding="utf-8"))
        fingerprint = part["native_source_sha256"]
        if "native_source_sha256" in payload and fingerprint != payload["native_source_sha256"]:
            raise RuntimeError("Native sources changed during the ablation")
        payload["native_source_sha256"] = fingerprint
        payload["levels"][level] = {key: value for key, value in part.items()
                                    if key not in {"results", "stock", "native_source_sha256"}}
        for result in part["results"]:
            result["case"]["optimization_level"] = level
            payload["results"].append(result)
        payload["stock"].extend(part["stock"])
        payload["validation"] = validate_results(payload)
        save()


if __name__ == "__main__":
    main()
