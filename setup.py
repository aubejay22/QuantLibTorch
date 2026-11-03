"""Package an explicitly built, ABI-matched native library.

Native compilation stays in scripts/build.py so it uses the caller's PyTorch,
compiler and pinned QuantLib checkout instead of a hidden isolated build.
"""
from pathlib import Path
import hashlib
import json
import runpy
import sys
import sysconfig

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from setuptools.errors import SetupError


ROOT = Path(__file__).resolve().parent
native_source_digest = runpy.run_path(str(ROOT / "_build_support.py"))["native_source_digest"]


def validate_native(root=ROOT):
    """Reject absent, stale or incompatible native artifacts before packaging."""
    native = root / "python" / "quantlibtorch" / "_native"
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    library = native / ("quantlibtorch_native" + suffix)
    stamp = native / "build_info.json"
    action = "Run python scripts/build.py with this Python and PyTorch, then retry with --no-build-isolation."
    if not library.is_file() or not stamp.is_file():
        raise SetupError("QuantLibTorch requires its native library and build metadata. " + action)
    try:
        import torch
    except ImportError as exc:
        raise SetupError("Install torch==2.10.0 before building the wheel. " + action) from exc
    try:
        metadata = json.loads(stamp.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise SetupError("Invalid native build metadata. " + action) from exc
    if not isinstance(metadata, dict):
        raise SetupError("Invalid native build metadata: expected an object. " + action)
    expected = {
        "torch_version": torch.__version__,
        "python_tag": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "quantlib_commit": "06edddbdb85f006c43a449b5fc846a00c67477f3",
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "native_source_sha256": native_source_digest(root),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SetupError(f"Native build metadata mismatch for {key}. " + action)
    if str(torch.__version__).split("+")[0] != "2.10.0":
        raise SetupError("QuantLibTorch supports the pinned torch==2.10.0 build only.")


class NativeBuildPy(build_py):
    def run(self):
        # Editable setup may precede compilation; importing/pricing still needs
        # the explicit native build. Distributed wheels must be self-contained.
        if not self.editable_mode:
            validate_native()
        super().run()


class NativeDistribution(Distribution):
    # Conservative interpreter + platform tags; never advertise a pure-Python
    # wheel for a LibTorch shared library. No setuptools extension is compiled.
    def has_ext_modules(self):
        return True


class WheelOnlySdist(sdist):
    def run(self):
        raise SetupError("QuantLibTorch is wheel-first: source distributions are disabled. "
                         "Developers must clone the repository and follow docs/install.md; "
                         "build a wheel after python scripts/build.py.")


def native_dependencies():
    """Wheel metadata requires the exact Torch variant used by the binary."""
    stamp = ROOT / "python/quantlibtorch/_native/build_info.json"
    if stamp.is_file():
        metadata = json.loads(stamp.read_text(encoding="utf-8"))
        return ["torch==" + metadata["torch_version"]]
    # Editable source setup can precede the explicit native build. Wheels
    # cannot take this fallback: NativeBuildPy rejects a missing native stamp.
    return ["torch==2.10.0"]


if __name__ == "__main__":
    setup(distclass=NativeDistribution, install_requires=native_dependencies(),
          cmdclass={"build_py": NativeBuildPy, "sdist": WheelOnlySdist})
