"""Two differentiation strategies sharing QuantLib's native Thomas solver."""
from pathlib import Path
import json
import hashlib
import sys
import sysconfig

import torch
from torch.autograd.function import once_differentiable


def _load_native() -> None:
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    library = Path(__file__).parent / "_native" / ("quantlibtorch_native" + suffix)
    if not library.exists():
        raise ImportError("QuantLibTorch native library missing. From the source checkout, run "
                          "python scripts/build.py, then reinstall; see docs/install.md.")
    action = ("Install the wheel matching your Python and exact PyTorch variant. "
              "Source developers: run python scripts/build.py and reinstall; see docs/install.md.")
    stamp = library.with_name("build_info.json")
    if not stamp.is_file():
        raise ImportError("QuantLibTorch native build metadata is missing. " + action)
    try:
        metadata = json.loads(stamp.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ImportError("QuantLibTorch native build metadata is unreadable. " + action) from exc
    if not isinstance(metadata, dict):
        raise ImportError("QuantLibTorch native build metadata must be an object. " + action)
    for key, expected in {
        "torch_version": str(torch.__version__),
        "python_tag": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "quantlib_commit": "06edddbdb85f006c43a449b5fc846a00c67477f3",
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }.items():
        if metadata.get(key) != expected:
            raise ImportError(f"QuantLibTorch native build is incompatible ({key}: "
                              f"built {metadata.get(key)!r}, running {expected!r}). " + action)
    if not isinstance(metadata.get("native_source_sha256"), str) or len(metadata["native_source_sha256"]) != 64:
        raise ImportError("QuantLibTorch native source fingerprint is missing or invalid. " + action)
    try:
        torch.ops.load_library(str(library))
    except (OSError, RuntimeError) as exc:
        raise ImportError("QuantLibTorch native library could not load. " + action) from exc
    try:
        compiled = json.loads(torch.ops.quantlibtorch.build_info())
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise ImportError("QuantLibTorch native compiled provenance is missing. " + action) from exc
    expected_compiled = {key: value for key, value in metadata.items() if key != "library_sha256"}
    if compiled != expected_compiled:
        raise ImportError("QuantLibTorch native binary and build metadata disagree. " + action)



_load_native()


class _DiscreteAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lower, diagonal, upper, rhs):
        # Function.forward runs without autograd recording. Both strategies use
        # the same compiled QuantLib implementation, including its pivot checks.
        x = torch.ops.quantlibtorch.solve(lower, diagonal, upper, rhs)
        ctx.save_for_backward(lower, diagonal, upper, x)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        lower, diagonal, upper, x = ctx.saved_tensors
        lam = torch.ops.quantlibtorch.solve(upper, diagonal, lower, grad_output)
        return -lam[1:] * x[:-1], -lam * x, -lam[:-1] * x[1:], lam


def tridiagonal_solve(lower, diagonal, upper, rhs, *, mode="full"):
    """Solve A x = rhs for a CPU float64 tridiagonal matrix, n >= 2.

    Diagonal lengths are (n-1, n, n-1); rhs is length n. Inputs may be
    noncontiguous. The no-pivot QuantLib solver requires nonzero elimination
    pivots, so invertibility alone is insufficient; diagonally dominant PDE
    matrices are the intended use. No batching or automatic dtype conversion.

    ``full`` records ATen operations inside the upstream algorithm.
    ``adjoint`` records a custom first-order VJP using A.T lambda = grad_output,
    without recording elimination. Higher derivatives require ``full``.
    """
    if mode == "full":
        return torch.ops.quantlibtorch.solve(lower, diagonal, upper, rhs)
    if mode == "adjoint":
        return _DiscreteAdjoint.apply(lower, diagonal, upper, rhs)
    raise ValueError("mode must be 'full' or 'adjoint'")
