# Install and build

## Release wheel

The first release targets **Windows x86-64, CPython 3.13 and PyTorch
2.10.0+cu126**. This CUDA-enabled PyTorch build can execute CPU pricing without
an NVIDIA GPU. CUDA financial execution requires a compatible NVIDIA GPU/driver
and uses only the supported packed APIs. A separate CPU-only PyTorch variant,
other Python versions, Linux wheels and macOS are outside the first matrix.
Consult [release support and evidence](release.md) for the exact validated
artifact; local execution is not proof of remote CI success.

Create an environment with Python 3.13, then use its interpreter:

```text
python -m pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-deps path/to/quantlibtorch-0.3.0-cp313-cp313-win_amd64.whl
python -c "import torch, quantlibtorch; print(torch.__version__)"
```

The wheel contains QuantLibTorch's native library and build metadata. It uses
PyTorch's installed libraries; it does not bundle PyTorch or require users to
compile QuantLib. Windows needs the compatible Microsoft Visual C++ runtime.
No CUDA toolkit is required for this package's tensor-based CUDA execution.

Distribution is **wheel-first**. Public-index publication is authorized but awaits the account and validation
gates recorded in [release.md](release.md). Do not assume `pip install quantlibtorch` downloads the release
until a verified index publication has occurred. Source distributions are not
published because an unbuilt archive is not directly pip-installable; use the
explicit checkout development procedure below.

## Compatibility checks

Use the exact full PyTorch version, including `+cu126`, recorded in the wheel.
Package and native metadata are checked for the Python ABI, platform, PyTorch
build and C++ ABI. Native/source fingerprints protect wheel creation from stale
build outputs. An incompatible binary must be rebuilt or replaced with a wheel
for the correct environment; editing metadata to bypass checks is unsupported.

| Symptom | Action |
| --- | --- |
| Wheel is unsupported on this platform | Check CPython 3.13, Windows x86-64 and wheel tag; use the matching environment |
| PyTorch/native ABI version mismatch | Install the exact PyTorch build required by the artifact; rebuild development binaries after changing PyTorch |
| Native library cannot load | Check PyTorch import, the Visual C++ runtime and wheel installation; retain the original loader error when reporting |
| CUDA is unavailable | Use CPU execution or verify the GPU, driver and CUDA-enabled PyTorch; scalar financial engines are CPU-only |
| Tensor dtype/device error | Use float64 and the engine's documented device; financial curve scalars remain on CPU even for CUDA batches |
| CUDA directional solve rejects a system | Revisit numerical configuration or use a separately constructed CPU plan; no automatic fallback occurs |

## Source development

A checkout is for maintainers and users intentionally compiling the library.
QuantLib 1.40 is pinned to `06edddbdb85f006c43a449b5fc846a00c67477f3`.
The stock `QuantLib==1.40` Python package is only an independent test oracle;
it is never linked into QuantLibTorch.

Use Git, CMake 3.24+, Boost headers and a C++17 compiler compatible with the
installed PyTorch. Windows uses Visual Studio 2022 C++ Build Tools. Linux source
development requires a compatible compiler and C++ ABI; Ubuntu commonly supplies
`build-essential cmake git libboost-dev`. Linux is experimental until its full
build, test and clean-install validation is recorded.

With PyTorch installed in the chosen environment:

```text
python scripts/bootstrap.py
python scripts/build.py --jobs 2
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install pytest QuantLib==1.40
python -m pytest -q
```

Windows checkout commands may use `.venv/Scripts/python.exe`; Linux environments
use `.venv/bin/python`. `bootstrap.py` downloads the pinned QuantLib sources on
first use and rejects a modified/wrong checkout; `build.py` verifies it again.
Pass `--boost-include PATH` when needed, with `boost/version.hpp` below PATH.
Rebuild after changing Python/PyTorch. No native build runs implicitly during
package installation. Editable installation can precede compilation, but
import requires the native library.

## Build a candidate artifact

After a successful native build and tests:

```text
python -m pip wheel --no-build-isolation --no-deps --no-index -w build/release .
```

Build tools must already be installed. Wheel creation rejects missing/stale
binaries or incompatible metadata. Test the resulting wheel in a fresh
environment outside the checkout before distributing it. The
[release procedure](release.md) provides the reproducible verification and
TestPyPI and production validation steps. Linux wheels must not be relabeled manylinux
without dependency-policy verification and clean-host testing.

## Notebooks

Install the tutorial dependencies in the same environment as the library:

```text
python -m pip install matplotlib nbformat nbclient ipykernel
```

To open notebooks interactively, add JupyterLab or use a notebook-capable editor
and select that environment's kernel. Automatic execution of `examples/` uses:

```text
python scripts/validate_examples.py
```

See [the user guide](user-guide.md) for the learning sequence and public API.
