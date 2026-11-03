"""Execute notebooks in fresh kernels using this interpreter and fail on errors.

Install the ``tutorials`` extra first. By default all notebooks in examples/ run
and executed copies plus JSON evidence go to build/validated_examples/. Source
notebooks are only updated with --in-place. No kernel is installed globally.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path,
                        help="Notebook paths; default: all examples/*.ipynb")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "validated_examples")
    parser.add_argument("--report", type=Path, help="JSON evidence path; default: output-dir/report.json")
    parser.add_argument("--timeout", type=int, default=300, help="Maximum seconds per cell")
    parser.add_argument("--in-place", action="store_true", help="Store executed output in source notebooks")
    parser.add_argument("--scripts", action="store_true", help="Also smoke-test all Python examples on small workloads")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")

    try:
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        import torch
        import quantlibtorch
    except ImportError as exc:
        parser.exit(2, f"Notebook validation requires a working quantlibtorch install and tutorials extra: {exc}\n")

    notebooks = sorted(path.resolve() for path in (args.notebooks or (ROOT / "examples").glob("*.ipynb")))
    if not notebooks:
        parser.error("no notebooks found")
    if any(path.suffix != ".ipynb" or not path.is_file() for path in notebooks):
        parser.error("every notebook must be an existing .ipynb file")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (args.report or output_dir / "report.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "quantlibtorch": quantlibtorch.__version__,
        "python_executable": sys.executable, "torch_location": torch.__file__,
        "quantlibtorch_location": quantlibtorch.__file__,
        "tutorial_dependencies": {name: version(name) for name in
                                  ("numpy", "matplotlib", "nbformat", "nbclient", "ipykernel")},
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "runtime_environment": {key: os.environ.get(key) for key in
                                ("MKL_THREADING_LAYER", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "notebooks": [], "scripts": [], "passed": False,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="kernel-", dir=output_dir) as temp:
            temporary = Path(temp)
            kernel_root = temporary / "kernels"
            kernel_dir = kernel_root / "quantlibtorch-examples"
            kernel_dir.mkdir(parents=True)
            (kernel_dir / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "QuantLibTorch example validation", "language": "python",
            }), encoding="utf-8")
            environment = dict(os.environ, IPYTHONDIR=str(temporary / "ipython"),
                               MPLCONFIGDIR=str(temporary / "matplotlib"))
            for source in notebooks:
                print(f"Executing {source.name} ...", flush=True)
                notebook = nbformat.read(source, as_version=4)
                nbformat.validate(notebook)
                # This transient cell captures optional numerical results without
                # imposing a convention on the pre-existing notebooks.
                notebook.cells.append(nbformat.v4.new_code_cell(
                    "import json as _validation_json\n"
                    "print(_validation_json.dumps(globals().get('RESULT', {}), allow_nan=False))"))
                manager = KernelManager(
                    kernel_name="quantlibtorch-examples",
                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
                    connection_file=str(temporary / "connection.json"))
                client = NotebookClient(notebook, km=manager, timeout=args.timeout,
                                        allow_errors=False, record_timing=True)
                started = time.perf_counter()
                item = {"notebook": source.name, "passed": False}
                report["notebooks"].append(item)
                try:
                    # An empty working directory verifies that notebooks do not
                    # depend on research files or sibling example modules.
                    client.execute(cwd=str(temporary), env=environment, cleanup_kc=True)
                    result_text = "".join(output.get("text", "") for output in notebook.cells[-1].outputs
                                          if output.output_type == "stream")
                    item["results"] = json.loads(result_text)
                    item["passed"] = True
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    item["seconds"] = time.perf_counter() - started
                    notebook.cells.pop()
                    destination = source if args.in_place else output_dir / source.name
                    with destination.open("w", encoding="utf-8", newline="\n") as handle:
                        nbformat.write(notebook, handle)
                    item["executed_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
                    item["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
                    print(f"  {'PASS' if item['passed'] else 'FAIL'} ({item['seconds']:.2f}s)", flush=True)
            if args.scripts:
                cases = [
                    ("neural_local_vol.py", ["--mode", mode]) for mode in ("full", "adjoint")
                ] + [
                    ("neural_fd_engine.py", ["--mode", mode]) for mode in ("full", "adjoint")
                ] + [
                    ("pricing_layer.py", ["--engine", "analytic"]),
                    ("pricing_layer.py", ["--engine", "fd", "--mode", "adjoint"]),
                    ("neural_latent_sde.py", ["--mode", "full", "--iterations", "2"]),
                    ("neural_latent_sde.py", ["--mode", "adjoint", "--iterations", "2"]),
                    ("multi_asset_neural_sde.py", ["--device", "cpu", "--assets", "2", "--contracts", "3", "--iterations", "2"]),
                    ("calibrate_neural_sde.py", [str(ROOT / "docs" / "benchmarks" / "calibration_phase22_market.json"),
                        "--mesh", "diagnostic", "--iterations", "2", "--execution", "batched",
                        "--output", str(output_dir / "calibration_smoke.json")]),
                ]
                if torch.cuda.is_available():
                    cases.append(("multi_asset_neural_sde.py", ["--device", "cuda", "--assets", "2",
                                  "--contracts", "3", "--iterations", "2"]))
                for index, (name, arguments) in enumerate(cases):
                    print(f"Running {name} {' '.join(arguments)} ...", flush=True)
                    started = time.perf_counter()
                    completed = subprocess.run([sys.executable, str(ROOT / "examples" / name), *arguments],
                                               cwd=temporary, env=environment, capture_output=True, text=True,
                                               timeout=args.timeout)
                    log = output_dir / f"script_{index:02d}_{Path(name).stem}.log"
                    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
                    report["scripts"].append({"script": name, "arguments": arguments,
                        "passed": completed.returncode == 0, "exit_code": completed.returncode,
                        "source_sha256": hashlib.sha256((ROOT / "examples" / name).read_bytes()).hexdigest(),
                        "seconds": time.perf_counter()-started,
                        "output_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
                    if completed.returncode:
                        raise RuntimeError(f"{name} failed; see {log}")
        report["passed"] = True
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Executed {len(notebooks)} notebooks; evidence: {report_path}")


if __name__ == "__main__":
    main()
