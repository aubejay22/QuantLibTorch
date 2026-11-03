# QuantLibTorch 0.3.0 release record

**QuantLibTorch 0.3.0 is published.** The
[GitHub release](https://github.com/aubejay22/QuantLibTorch/releases/tag/v0.3.0),
[production PyPI wheel](https://pypi.org/project/quantlibtorch/0.3.0/), and
[TestPyPI rehearsal](https://test.pypi.org/project/quantlibtorch/0.3.0/) contain
the same validated artifact. Actual TestPyPI and production PyPI installations
pass on hosted Windows and in independent external environments on the CPU/CUDA
host. All publication, installation, pricing and gradient gates passed.

The [official public repository](https://github.com/aubejay22/QuantLibTorch) is
owned by the verified personal account `aubejay22`. Unauthenticated reads verify
the repository, BSD-3-Clause license, documentation and existing notebooks.
Original project code uses BSD-3-Clause; upstream code retains its own terms.
[Licensing](licensing.md) and [the security audit](release-audit.md) explain the
seven included notices and retained algorithm attribution.

## Exact validated release artifact

| Field | Value |
| --- | --- |
| Public source revision | `291a09ebe024a47bf4d917f0f57fbbddf7bde5ba` |
| Successful Windows candidate | [Actions run 41574726134](https://github.com/aubejay22/QuantLibTorch/actions/runs/41574726134) |
| TestPyPI publication and installation | [Actions run 41695264905](https://github.com/aubejay22/QuantLibTorch/actions/runs/41695264905) |
| Production publication and installation | [Actions run 41696532378](https://github.com/aubejay22/QuantLibTorch/actions/runs/41696532378) |
| Wheel | `quantlibtorch-0.3.0-cp313-cp313-win_amd64.whl` |
| Bytes | 1,395,361 |
| SHA256 | `b0a1c5dd652c08c755440313083ed4f5ef05607a0dcc860081a4de476d3d6ded` |
| Local preserved download | `build/hosted-candidate-41574726134/` |

The candidate, actual GitHub release download, TestPyPI and production PyPI
wheel digests match. Both registries expose only this Windows wheel; no source
distribution or Linux wheel was uploaded. The annotated `v0.3.0` tag resolves
to the exact source revision above. Documentation commits after publication do
not change that tag or artifact. [Publication evidence](benchmarks/release_publication.json)
records the public URLs, asset digests, package metadata and source binding.

Original local history remains preserved privately. Public history is maintained
in the separate `build/public-release` checkout; never push the original local
repository. The earlier evidence branch contains documentation only.

## Compatibility and measured validation

| Configuration | Evidence |
| --- | --- |
| Windows Server 2022, official CPython 3.13.17 | Hosted native build; 2,171 tests passed, 98 skipped; independent wheel install |
| Windows 11, Anaconda CPython 3.13.5 | Corrected local native build; 2,268 tests passed, one intentional skip, 23 existing warnings in 1,092.43 s |
| Exact PyTorch/LibTorch | `2.10.0+cu126`, checked C++ ABI and Python/platform/native provenance |
| Hosted native compiler | Visual Studio 2022, MSVC 19.44.35229.0, Windows SDK 10.0.26100.0 |
| CPU | Documented float64 APIs; hosted full suite and independent installed pricing/gradients |
| CUDA | Exact hosted wheel passes packed float64 price/VJP checks on NVIDIA RTX 4060 Laptop |
| Teaching materials | Hosted wheel passes all seven notebooks and ten CPU scripts; prior CUDA-enabled example validation also retained |
| Linux | Isolated Ubuntu 24.04 CPU experiment passes; no portable Linux release wheel claimed |

The hosted skips are 97 unavailable-CUDA cases and the existing CPU Thomas
strict-dominance exception. All CUDA cases run in the corrected local native
suite. The exact hosted wheel separately passes GPU installation and financial
smoke checks. See [hosted Windows evidence](benchmarks/release_hosted_windows.json),
[hosted GPU evidence](benchmarks/release_hosted_cuda.json),
[artifact audit](benchmarks/release_hosted_audit.json),
[remote promotion preflight](benchmarks/release_hosted_preflight.json), and
[corrected local regression](benchmarks/release_json_regression.json).

From the independently installed production PyPI wheel, the README example produces:

| Quantity | Result |
| --- | --- |
| Option price | 8.827321225352126 |
| Spot delta | 0.573495979027775 |
| Volatility vega | 38.71517541592269 |

Heston, shout, European/American/Bermudan exercise, full/adjoint agreement and
packed CPU/CUDA prices and gradients also pass. Prices and gradients retain the
financial and differentiation restrictions in [capabilities.md](capabilities.md).

Wheel users need the matching PyTorch build and compatible Microsoft Visual C++
x64 runtime, but no compiler, QuantLib checkout or CUDA toolkit. CUDA execution
requires a compatible NVIDIA GPU/driver. A CUDA-enabled Torch wheel can execute
CPU workloads without a GPU. Other Python/Torch variants, macOS and ARM are not
validated. Source distributions are disabled; use the explicit source-build
procedure in [install.md](install.md). No stock-double QuantLib library is linked.

## Trusted Publishing

The owner manually verified both saved registrations. Their values match the
live GitHub configuration, and both registries accepted the OIDC publication:

| Field | TestPyPI | Production PyPI |
| --- | --- | --- |
| Account page | https://test.pypi.org/manage/account/publishing/ | https://pypi.org/manage/account/publishing/ |
| Project name | `quantlibtorch` | `quantlibtorch` |
| Owner | `aubejay22` | `aubejay22` |
| Repository | `QuantLibTorch` | `QuantLibTorch` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment | `testpypi` | `pypi` |

GitHub authentication is already available through the owner's existing Git
credential. The GitHub environments are configured separately: `testpypi`
permits the `master` branch, and `pypi` permits only the `v0.3.0` tag. Actions
default to read-only repository permissions. `RELEASE_AUDIT_PASSED=true` enables
the audited release workflow; it does not replace artifact or registry checks.
No permanent PyPI credentials are stored in the repository or workflow.

The connected tools did not read the private account pages. The saved field
comparison uses the values supplied by the owner, with successful OIDC exchanges
as independent operational evidence. [Publisher verification](benchmarks/release_publisher_verification.json)
records that distinction.

## Publication and installed-package evidence

The TestPyPI workflow reviewed the successful candidate run, source revision,
native provenance, seven notices and CUDA-tested wheel digest before upload.
Its Windows job downloaded the actual indexed artifact and installed it outside
the checkout. A separate GPU-host check used uncached `pip download` and direct
`pip install` from TestPyPI in another fresh environment, recording the served
URL and hash. CPU/CUDA financial checks and `pip check` passed.

Only then was `v0.3.0` created at the validated source revision. The production
workflow required the successful TestPyPI run at the same revision, repeated its
indexed-install check, published the GitHub release, and uploaded identical bytes
to production PyPI. Its actual production-download installation passed on the
hosted Windows runner. A second fresh external environment used uncached pip
download and direct installation from production PyPI; the served URL and hash
matched the candidate. Import, version reporting, analytic pricing, Heston,
shout, exercise styles, full/adjoint agreement, packed CPU/CUDA prices and
gradients, and dependency consistency all passed. The GPU host used the NVIDIA
GeForce RTX 4060 Laptop GPU. No source checkout was needed by either install.

- [Hosted TestPyPI installation](benchmarks/release_testpypi_hosted.json)
- [Independent TestPyPI CPU/CUDA installation](benchmarks/release_testpypi_install.json)
- [Hosted production PyPI installation](benchmarks/release_production_hosted.json)
- [Independent production PyPI CPU/CUDA installation](benchmarks/release_production_install.json)
- [Public releases, metadata and artifact hashes](benchmarks/release_publication.json)

The workflows cannot publish from pull requests. Publishing jobs have only
OIDC permission; the GitHub release job has repository write permission. Action
revisions and build dependencies are pinned. GitHub Actions artifacts have a
30-day retention period; evidence and manifests are preserved separately. The
published wheel and manifest are also attached to the GitHub release. Future
releases require new version-specific tags, environment policies and validated
artifacts; this workflow deliberately permits only the initial 0.3.0 release.

## Dependency and Linux findings

Clean-runner CI exposed an implicit nlohmann/json dependency previously supplied
by Conda. The corrected bootstrap pins version 3.11.2 and verifies SHA256
`665fa14b8af3837966949e8eb0052d583e2ac105d3438baba9951785512cf921`.
CMake checks those bytes and explicitly includes the verified header. Its MIT
notice and the Apache-2.0 notice for an excluded pre-C++14 compatibility fallback
are retained. The earlier local `396250...` wheel is superseded and must not be
published. The corrected local wheel is documented in
[release_json_packaging.json](benchmarks/release_json_packaging.json); production
promotion uses the separately validated hosted artifact above.

The isolated [Linux experiment](https://github.com/aubejay22/QuantLibTorch/actions/runs/41575865212)
at `f565a2f64d1ecd010cf4b0a8d6babdf8dd290da7` qualifies four generated friend
declarations and adds unchanged upstream `ql/discretizedasset.cpp` to the source
closure. It passes compilation/import, 2,171 tests with 98 skips, packaging,
dependency inspection and external installed pricing/gradients on Ubuntu 24.04,
CPython 3.13.17 and Torch 2.10.0+cpu. This branch remains unmerged.

Auditwheel reports a raw `linux_x86_64` wheel with a manylinux_2_34 symbol
constraint. A manylinux build/repair and oldest-distribution installation have
not been validated, and Linux CUDA has not run. These findings do not establish
a portable Linux PyPI artifact. The initial production scope remains Windows.
[Complete Linux evidence](benchmarks/release_linux_ci.json) records successful
and failed jobs; an overall green experimental workflow alone is not proof.

Cross-host native bit reproducibility is not claimed. The hosted wheel's source
fingerprint differs from the local build only because of Windows checkout line
endings; it was independently reconstructed from the exact Git revision. Compiler
output and wheel hashes are validated per artifact, rather than assumed equal.
The original 0.3.0rc1 record remains in
[history/release-candidate.md](history/release-candidate.md).
