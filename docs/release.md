# QuantLibTorch 0.3.0 release record

The owner has authorized public GitHub, BSD-3-Clause for eligible original code,
TestPyPI and the first production PyPI release. Publication is **not complete**.
Authentication and validation gates below must pass before any production upload.
The preserved release-candidate baseline is `1002f78`; its measured results and
artifact hashes remain in [the candidate record](history/release-candidate.md).

## Repository and licensing

The connected GitHub account is `aubejay22` (user ID `149632864`). The intended
repository is `aubejay22/QuantLibTorch`; a public URL is only verified after
creation and an unauthenticated read. No unrelated account or organization may
be substituted. Original project code uses [BSD-3-Clause](../LICENSE).
[Licensing](licensing.md) records upstream exceptions and included notices.

The original local history contains personal author email and workstation paths.
Keep it intact locally. The public export starts a separate history, preserves
numerical source and notebooks, and sanitizes personal metadata in reports and
archives. Never push the original local refs to the public repository. The
[security audit](release-audit.md) records the export procedure and its limits.

## Supported configuration

| Component | Release scope |
| --- | --- |
| OS / architecture | Windows x86-64; local Windows 11 validation |
| Python | CPython 3.13; local Anaconda 3.13.5, hosted interpreter validation required |
| PyTorch / LibTorch | Exactly `2.10.0+cu126`, including CUDA variant and checked C++ ABI |
| CPU | Documented float64 APIs; full regression and external wheel smoke |
| CUDA | Documented packed float64 APIs; exact uploaded wheel must pass local GPU validation |
| Build | Visual Studio 2022, Boost 1.82 headers, pinned QuantLib 1.40 |
| Linux | CI experiment; no portable Linux binary support until its checks pass |
| Other variants | Other Python/Torch variants, macOS and ARM are unsupported |

PyTorch is installed separately from its official cu126 wheel index. The
CUDA-enabled build supports CPU execution without a GPU. GPU execution needs a
compatible NVIDIA driver. Users need the Microsoft Visual C++ x64 runtime, but
no compiler, CUDA toolkit or QuantLib source checkout for wheel installation.
No stock-double QuantLib library is linked into the TorchReal extension.

Only validated wheels are published. Source-distribution builds deliberately
fail with an actionable message; source development follows [install.md](install.md).
Compiled build provenance, native SHA256, Python/platform tags, exact Torch
variant and C++ ABI are checked at load time. Never edit metadata to bypass them.

## Account configuration

GitHub CLI must authenticate as `aubejay22` with permission to create the personal
repository, push workflows, configure Actions environments and publish releases.
The connector identifies the account but does not expose repository creation,
environment administration or workflow dispatch. No credentials belong in source.

Create pending GitHub Trusted Publishers through each account's Publishing page:

| Field | TestPyPI | Production PyPI |
| --- | --- | --- |
| Account page | https://test.pypi.org/manage/account/publishing/ | https://pypi.org/manage/account/publishing/ |
| Project name | `quantlibtorch` | `quantlibtorch` |
| Owner | `aubejay22` | `aubejay22` |
| Repository | `QuantLibTorch` | `QuantLibTorch` |
| Workflow filename | `release.yml` | `release.yml` |
| GitHub environment | `testpypi` | `pypi` |

These are separate registry accounts/configurations. Configure GitHub environment
branch/tag restrictions to the trusted release branch and `v0.3.0` respectively.
If an environment requires manual review, its reviewer must approve the actual
run. The owner's general publication authorization does not bypass that control.
Pending publishers do not reserve a name. Read-only JSON checks for `quantlibtorch`
on both registries returned 404 on 2026-11-02.

## Build and promotion gates

1. Audit the sanitized public source and history, create the personal repository,
   push reviewed source and verify public access. Preserve the original history
   locally without rewriting or publishing it.
2. Run `release-candidate.yml` at an identified clean public revision. Its Windows
   job builds, runs the full test suite, builds a wheel, checks metadata/notices
   and performs an independent installed-wheel CPU smoke. Run Linux CI as an
   experiment; unsupported Linux does not block the Windows release.
3. Download that run's wheel and manifest. Validate the exact wheel on the CUDA
   host using `scripts/validate_wheel.py --require-cuda`; retain its SHA256 and
   report. A local rebuild's result cannot stand in for the hosted artifact.
4. Dispatch `release.yml` for the TestPyPI stage with the successful candidate
   run ID and locally validated wheel SHA256. The workflow uploads using OIDC,
   downloads from TestPyPI, checks the hash, and tests a fresh installation.
   Download the actual index wheel for local CUDA validation as well.
5. After all preceding gates pass, create `v0.3.0` at the candidate's exact
   public revision. Dispatch the production stage on that tag with candidate and
   successful rehearsal run IDs. The workflow validates the TestPyPI artifact
   again, publishes the GitHub release and promotes identical bytes to PyPI.
6. Verify the public project pages, download from production PyPI into a fresh
   environment, test CPU/CUDA pricing and gradients, and record the final public
   revision, workflow URLs, wheel hashes and evidence. Only then mark complete.

The release workflow is manual and restricted to trusted refs. Untrusted pull
requests receive no publishing credentials. Build jobs have read-only permissions;
OIDC permission is confined to the upload jobs, and GitHub release write permission
to its release job. Actions and dependency inputs are pinned. Production artifacts
are downloaded and hash-checked, never rebuilt during promotion.

## Local reproduction

Follow [source development](install.md#source-development), then:

```powershell
python scripts/build.py --boost-include C:/path/to/boost_1_82_0
python -m pytest
$env:SOURCE_DATE_EPOCH = '1793577600'
python -m build --wheel --no-isolation --outdir build/release-final
python -m twine check --strict build/release-final/*.whl
python scripts/validate_wheel.py build/release-final/quantlibtorch-0.3.0-cp313-cp313-win_amd64.whl --require-cuda
```

`validate_wheel.py` creates a venv outside the checkout without system packages,
removes Python path overrides, copies its smoke script out of the source tree,
and verifies installed package location, dependencies, version, analytic Greeks,
European/American/Bermudan FD, Heston, shout, full/adjoint agreement and packed
CPU/CUDA price/VJP parity. `--wheelhouse DIR` allows offline dependencies.

The seven existing notebooks and eleven bounded example runs are retained.
`validate_examples.py` exercises them using the independent wheel environment.
Native path remapping prevents developer home paths in shipped assertion strings.
Deterministic wheel packaging from the same binary does not establish identical
compiler output across hosts. Exact-artifact hashes remain authoritative.

## Current publication status

No GitHub release, TestPyPI upload or production PyPI upload is yet verified.
The native rebuild and regression, source/history audit, final wheel privacy
check, external installation and actual remote CI are release gates. Account
login and pending Trusted Publisher registration require owner interaction.
The release remains blocked from publication until these requirements pass;
local release preparation alone is not success.
