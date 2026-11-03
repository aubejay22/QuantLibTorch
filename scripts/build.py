"""Configure and build the native library against this Python's installed torch."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

from native_metadata import build_metadata

ROOT = Path(__file__).resolve().parents[1]
native_source_digest = runpy.run_path(str(ROOT / "_build_support.py"))["native_source_digest"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build" / "native")
    parser.add_argument("--boost-include", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--no-math-patch", action="store_true", help="Reproduce the expected upstream compilation failure")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    import torch
    if str(torch.__version__).split("+")[0] != "2.10.0":
        parser.error("QuantLibTorch requires the pinned torch==2.10.0 build")
    source_digest = native_source_digest(ROOT)
    # Fetch missing pinned dependencies and validate existing dependency bytes.
    subprocess.run([sys.executable, str(ROOT / "scripts" / "bootstrap.py")], check=True)
    command = ["cmake", "-S", str(ROOT), "-B", str(args.build_dir),
               f"-DPython3_EXECUTABLE={sys.executable}",
               f"-DQLT_COMPILE_JOBS={args.jobs}",
               f"-DQLT_PATCH_MATH={'OFF' if args.no_math_patch else 'ON'}"]
    if sys.platform == "win32":
        command += ["-G", "Visual Studio 17 2022", "-A", "x64"]
    else:
        command += ["-DCMAKE_BUILD_TYPE=Release"]
    if args.boost_include:
        command += [f"-DQLT_BOOST_INCLUDE_DIR={args.boost_include.resolve()}"]
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(args.build_dir), "--config", "Release",
                    "--parallel", str(args.jobs)], check=True)
    if native_source_digest(ROOT) != source_digest:
        raise SystemExit("Native sources changed during the build; rerun scripts/build.py before packaging.")
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    native = ROOT / "python" / "quantlibtorch" / "_native"
    library = native / ("quantlibtorch_native" + suffix)
    metadata = build_metadata()
    # Query a fresh process: a previously imported DLL must not satisfy this
    # check. Never stamp an old binary with the active environment's identity.
    probe = "import torch; torch.ops.load_library(" + repr(str(library)) + "); print(torch.ops.quantlibtorch.build_info())"
    compiled = json.loads(subprocess.check_output([sys.executable, "-I", "-c", probe], text=True))
    if compiled != metadata:
        raise SystemExit("Compiled native provenance disagrees with this build; use a clean build directory.")
    metadata["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (native / "build_info.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
