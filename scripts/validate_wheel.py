"""Install a wheel and exact Torch variant in a fresh external venv, then price/backpropagate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--torch-index", default="https://download.pytorch.org/whl/cu126")
    parser.add_argument("--wheelhouse", type=Path, help="Use only local dependency wheels, without network")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT/"build/release/clean_install.json")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        metadata = json.loads(archive.read("quantlibtorch/_native/build_info.json"))
    work = Path(tempfile.mkdtemp(prefix="quantlibtorch-release-"))
    if work.is_relative_to(ROOT):
        raise RuntimeError("Clean installation must be outside the checkout")
    environment = work/"venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
    python = environment/("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    def run(command):
        subprocess.run([str(python), "-I", *command], cwd=work, env=env, check=True)
    sources = (["--no-index", "--find-links", str(args.wheelhouse.resolve())] if args.wheelhouse
               else ["--index-url", args.torch_index])
    run(["-m", "pip", "install", *sources, "torch=="+metadata["torch_version"]])
    run(["-m", "pip", "install", "--no-deps", "--no-index", str(wheel)])
    smoke = work/"wheel_smoke.py"
    shutil.copyfile(ROOT/"scripts/wheel_smoke.py", smoke)
    report = work/"smoke.json"
    run([str(smoke), "--output", str(report), *(["--require-cuda"] if args.require_cuda else [])])
    run(["-m", "pip", "check"])
    result = {"wheel": str(wheel), "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
              "clean_environment": str(environment), "system_site_packages": False,
              "smoke": json.loads(report.read_text(encoding="utf-8"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print("Validation evidence:", args.output)


if __name__ == "__main__":
    main()
