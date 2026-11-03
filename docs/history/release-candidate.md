# Release candidate 0.3.0rc1

This is a controlled distribution candidate. No artifact has been uploaded to
TestPyPI or PyPI. The project license needs owner approval before distribution.

## Reviewed local artifact

`build/release/quantlibtorch-0.3.0rc1-cp313-cp313-win_amd64.whl`
(1,387,733 bytes) has SHA256
`b164746150f6a141c3afea74e3ab650887889f7a45e7281b302dcbdacd7fefb5`.
Two packaging runs from the same native binary produced this identical hash.
Strict Twine checks and the final installed-wheel CPU/CUDA smoke both pass.
The [packaging record](../benchmarks/release_packaging.json) contains dependencies,
native hashes, financial outputs and exact environment information; the
[regression record](../benchmarks/release_regression.json) records the full suite.
All seven notebooks and eleven bounded example runs pass in the external wheel
environment, including CUDA, with [executable evidence](../benchmarks/release_examples.json).
The wheel is a local controlled artifact, not a public download. Any later
license/metadata change requires rebuilding and revalidating it.

## Supported candidate configuration

| Component | Candidate scope | Evidence |
| --- | --- | --- |
| OS / architecture | Windows x86-64 | Windows 11 (10.0.26200), local native build and external clean venv |
| Python | CPython 3.13 (local Anaconda 3.13.5) | `cp313-cp313-win_amd64` wheel; other minors are excluded by metadata |
| PyTorch | **2.10.0+cu126**, exact variant | Wheel dependency and loader both require this version |
| CPU | float64 supported APIs | Full repository regression suite and installed-wheel smoke |
| CUDA | Compatible packed APIs, float64 | Local NVIDIA RTX 4060 Laptop; CPU/CUDA price and VJP parity |
| Linux | Experimental source build only | CI recipe exists; no remote CI execution or portable Linux wheel claimed |
| Other variants | CPU-only Torch, other CUDA builds, macOS, ARM, Python 3.12/3.14 | Not supported by this candidate |

The CUDA-enabled PyTorch wheel can execute CPU workloads without a GPU. CUDA
execution needs a compatible NVIDIA driver; a CUDA toolkit is not needed to
install this wheel. Numerical API restrictions remain in [capabilities.md](../capabilities.md).
Candidate validation reports live in `docs/benchmarks/release_*.json`.
The integrated native regression passed **2,229 tests**, with one intentional
skip, in the development venv (which inherits system packages). Installed-wheel
smoke tests use a separate venv without system packages, on the same host.
Neither result is a separate clean Windows machine or an official-python.org
interpreter test.
The DLL depends on PyTorch DLLs and the Microsoft VC/UCRT runtime; the matching
Microsoft Visual C++ x64 runtime must be installed. No Conda-specific or stock
QuantLib DLL appears in its direct dependency table.
The first candidate was built using Visual Studio 2022, Boost 1.82 headers and
pinned QuantLib 1.40 commit `06edddbdb85f006c43a449b5fc846a00c67477f3`.

## Compatibility and the previous metadata discrepancy

At this audit, both development interpreters imported the existing Phase 26
binary successfully: Python 3.13.5, PyTorch 2.10.0+cu126, CXX11 ABI true. The
historical mismatch was not reproducible with the current environment. Older
metadata recorded the active interpreter after a build without independently
proving that it described the compiled binary. Missing metadata was also
silently accepted by the loader.

The candidate closes those gaps without relaxing checks. CMake embeds the
Torch version (including CUDA suffix), Python tag, platform, C++ ABI, QuantLib
commit and source fingerprint in the binary. The build script queries a fresh
process before writing a matching SHA256 stamp. The loader requires the stamp,
checks the binary hash and runtime ABI, then compares compiled provenance with
the stamp. Packaging rejects stale native sources. Wheel metadata pins the
exact Torch variant, including `+cu126`; a CPU build would declare `+cpu`.
Never edit `build_info.json` to bypass a mismatch: install the matching wheel or
rebuild using the intended interpreter.

## Reproduce and validate the wheel

Use a clean CPython 3.13 environment and install the exact build tools:

```powershell
python -m pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install setuptools==72.1.0 wheel==0.45.1 build==1.2.2.post1 twine==6.1.0 pytest==8.3.4 QuantLib==1.40
python scripts/build.py --boost-include C:/path/to/boost_1_82_0
python -m pytest
$env:SOURCE_DATE_EPOCH = '1793577600'
python -m build --wheel --no-isolation --outdir build/release
python -m twine check --strict build/release/*.whl
python scripts/validate_wheel.py build/release/quantlibtorch-0.3.0rc1-cp313-cp313-win_amd64.whl --require-cuda
```

`validate_wheel.py` creates a new venv in the system temporary directory,
without system site packages. It installs Torch and the wheel, copies the
smoke script outside the checkout, runs isolated Python, checks the package
path, and runs `pip check`. It exercises analytic Greeks, European/American/
Bermudan FD pricing, full/adjoint price and gradient agreement, Heston, shout,
and CPU/CUDA asset-by-contract prices and gradients. `--wheelhouse DIR`
supports completely offline validation once dependency wheels are downloaded.
The environment is retained for inspection and its path is recorded in the
JSON report. Omit `--require-cuda` only for an explicitly CPU-only smoke run.

`SOURCE_DATE_EPOCH` makes repeated packaging of the same compiled binary
deterministic. This does not establish byte-identical native compiler output
across hosts. Dependency versions, compiler, source fingerprint, build log,
wheel hash and runtime checks provide the reproducible build recipe.

## Wheel-first policy

Public distribution consists only of validated wheels. Both PEP 517 source
archive builds and `setup.py sdist` fail with an actionable wheel-first error.
The earlier source archives omitted the required native binary and could not
be installed directly; they are historical artifacts, not release candidates.
Developers can clone the repository and follow [install.md](../install.md).
No compiler or QuantLib source checkout is required by wheel users. LibTorch
is supplied by the declared PyTorch dependency, not bundled a second time.
QuantLib is compiled from pinned sources with TorchReal; no stock double
QuantLib library is linked. Third-party notices are included under the wheel's
distribution metadata. The project license is still pending approval.

## Linux portability work still required

The Linux source job tests Ubuntu 24.04 / CPython 3.13 / Torch 2.10.0+cpu; it
has not run remotely in this release preparation. Linux RPATH now resolves
PyTorch libraries relative to the installed package, instead of embedding a
developer's absolute Torch path. That is necessary, but does not prove a
manylinux wheel is portable.

A Linux release needs a pinned [manylinux build image](https://github.com/pypa/manylinux),
actual execution of the full suite, `auditwheel show` and dependency/symbol
inspection, and clean installation on the oldest claimed distribution. Review
LibTorch's external libraries and glibc/C++ symbol baseline before selecting a
manylinux tag or attempting repair. Do not relabel an Ubuntu wheel as manylinux
or blindly bundle another copy of PyTorch's shared libraries. No Linux wheel
is included in the current candidate.

## TestPyPI rehearsal and approval

Read-only JSON requests for `quantlibtorch` on both PyPI and TestPyPI returned
HTTP 404 on 2026-11-02. This suggests the name was unregistered at that time;
it neither reserves the name nor guarantees that registration is permitted.
There is currently no Git remote configured, so remote CI and publisher
configuration cannot be validated from this checkout.

The manual `.github/workflows/release-candidate.yml` builds and tests a Windows
wheel. Publication defaults to **false**; there is no production publishing
job. Action commits and Boost archive SHA256 are pinned. The publishing job
has only OIDC token permission and receives only wheel artifacts, following
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

Before any upload, the owner must:

1. Approve the project's license; add that license and metadata, rebuild and
   revalidate the candidate. Set repository variable `LICENSE_APPROVED=true`
   only after that legal choice is finalized.
2. Choose/configure the repository remote without changing its visibility.
   Run the workflow with publication disabled and review the actual CI results.
   Configure real project/documentation URLs and verify registry README link
   rendering before public release; no URLs are invented for this local checkout.
3. Create a protected GitHub environment named `testpypi` with a required owner
   reviewer and protected release branch. Configure the matching TestPyPI
   pending Trusted Publisher (repository owner/name, workflow
   `release-candidate.yml`, environment `testpypi`). These are separate from
   production PyPI settings; no long-lived upload token is required.
4. Explicitly approve a TestPyPI upload. Dispatch the workflow with
   `publish_testpypi=true`. Before approving its protected environment, download
   **that run's exact wheel**, run `validate_wheel.py --require-cuda` on the
   CUDA validation host, and compare its SHA256 with the candidate artifact.
   Hosted Windows runners validate CPU execution only. Earlier local GPU tests
   do not validate a newly rebuilt hosted artifact.
5. Approve the environment only after reviewing that exact artifact's tests,
   CUDA evidence, hash, notices and version. Uploaded versions cannot be reused.

After an approved upload, verify index delivery in a new environment:

```powershell
python -m venv C:/path/outside/checkout/rehearsal
C:/path/outside/checkout/rehearsal/Scripts/python -m pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
C:/path/outside/checkout/rehearsal/Scripts/python -m pip download --no-deps --only-binary=:all: --index-url https://test.pypi.org/simple quantlibtorch==0.3.0rc1 -d build/testpypi-download
python scripts/validate_wheel.py build/testpypi-download/quantlibtorch-0.3.0rc1-cp313-cp313-win_amd64.whl --require-cuda --output build/release/testpypi-install.json
```

Compare the downloaded wheel's hash with the approved upload artifact. Use
TestPyPI only for the package and the official Torch index for dependencies;
avoid an extra mixed index. Production PyPI, repository visibility changes and
the final public release remain separate actions requiring explicit approval.
