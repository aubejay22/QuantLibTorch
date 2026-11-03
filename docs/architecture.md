# Current architecture

QuantLibTorch retains **Approach A**: selected original QuantLib 1.40 financial
engines and numerical infrastructure compiled with `QL_REAL = TorchReal` and
connected to LibTorch. The immutable upstream dependency is commit
`06edddbdb85f006c43a449b5fc846a00c67477f3`. A stock-double QuantLib library is
never linked into the native extension; the separate Python QuantLib package
is an independent test oracle in isolated processes.

This document describes maintained contracts. Detailed phase measurements and
design decisions through Phase 26 are preserved in
[the historical architecture archive](history/architecture.md). The
[capability matrix](capabilities.md) is the financial support reference.

## Layers and provenance

1. Python functions and immutable instrument/market/engine configurations validate
   inputs and construct fresh PyTorch graphs. `PricingLayer` registers optional
   neural modules; it delegates to the same public pricing functions.
2. Native bridges construct original QuantLib financial objects, engines,
   curves, payoffs and events. `cpp/bridge.cpp` registers the torch operators.
3. `TorchReal` and numerical adapters preserve active scalar dependencies.
   Tensor arrays, structured stencils and guarded solve adapters reduce dispatch
   overhead while retaining independent arithmetic/reference paths.
4. Packed latent pricing obtains derivative bases, payoff/event maps, rollback
   intervals and interpolation bases from original QuantLib objects. Generated
   ADI stages preserve the upstream step bodies with an adapted array type.
5. LibTorch supplies autograd and CPU/CUDA tensor operations. Scalar engines
   remain CPU; device-resident packed execution is an explicit separate API.

`scripts/prepare_quantlib.py` emits reproducible compatibility overlays;
`scripts/prepare_packed.py` emits packed adapters. The original checkout stays
clean. Compatibility patches, fixed-mesh hooks and explicit unsupported Heston
process methods are separate from numerical changes. The source closure is
listed in `cmake/quantlib_sources.cmake`.

## Differentiation contracts

Full AD differentiates the executed arithmetic and supported branches. Its
graph-connected backward formulas preserve tested higher and mixed derivatives.
Native fast paths retain guarded tensor fallbacks where necessary. A direct
price and a derivative of a price must use the same fixed financial/numerical
configuration for meaningful comparisons.

Equation adjoints apply VJPs of discrete solves, stencils and rollback stages.
For `A u = b`, the backward solves `A.T lambda = grad_u` and propagates
`lambda.T (db - dA u)`. They include coefficient dependencies and financial
events; they do not differentiate a separately discretized continuous adjoint.
These interfaces are first order and explicitly reject unsupported higher or
forward derivative requests.

No active coefficient is silently detached. Cached geometry, payoff/event maps,
grid bases and configuration are passive; active fields and market graphs are
supplied afresh. Multi-asset checkpoint boundaries deliberately rematerialize
first-order segments and return VJPs to the original inputs. Storage depends on
checkpoint interval and workload; it is not constant memory.

Exercise and interpolation can be nonsmooth. Exercise retains continuation at
equality, matching the upstream strict comparison. Gradients hold meshes, dates,
event schedules and other passive configuration fixed. Automatic mesher
derivatives are not supported. Heston adjoints reject damping because the
corresponding Krylov backward is unsupported; full AD supports tested damping.

## Financial and device behavior

Latent equity pricing uses state `(log S, Y)` with one or two latent coordinates.
The neural model supplies latent drift and joint diffusion loadings. The stock
drift is imposed as `r - q - 0.5 * covariance[0, 0]`; covariance is `L @ L.T`.
Rates, discounting, boundary rows and financial events retain their defined
QuantLib semantics. The general diffusion API instead accepts all drift
components and terminal data explicitly.

`LatentBatch` shares an operator across compatible contracts. `MultiAssetBatch`
groups independent assets with compatible shapes/times/schemes/events while
retaining separate coefficients, curves and readout. Asset batching does not
create a correlated multi-asset payoff; use `fd_basket` for its supported
two-asset basket contracts. Packed exercise supports one/two latent factors;
cash dividends require one. Damping/custom boundary callbacks are unsupported.

CPU/CUDA packed paths use float64. CUDA directional solves require strict forward
row diagonal dominance and checked residuals; unsupported systems raise. No
silent device fallback occurs. `MultiAssetBatch` evaluates financial splines on
device; `LatentBatch` has a graph-connected CPU spline readout. Financial curve
construction uses CPU scalar tensors, then connected transfers supply forwards
to CUDA. Full-mode graphs and first-order checkpointed graphs have different
memory costs.

## Validation and release preparation

The inherited native baseline passed 2,224 tests and one intentional skip:
[Phase 26 evidence](benchmarks/library_phase26_validation.json). Independent
stock prices, fixed-mesh parameter differences, dense/manufactured numerical
references, full/adjoint agreement, higher derivatives, graph freshness,
event ordering, solver failures and CPU/CUDA checks remain regression gates.
Current release validation and binary compatibility evidence are recorded in
[release documentation](release.md), separately from baseline measurements.

Documentation cleanup moves 5,690 lines of architecture history, 609 lines of
historical context and the 74-line old roadmap into `docs/history/`. Their
relative links are adjusted; numerical sources, tests, source snapshots and
benchmark evidence are preserved. The
[cleanup manifest](benchmarks/release_docs_cleanup.json) records original and
archive hashes. The [development guide](development.md) explains retained
diagnostic code and evidence dependencies.

The recorded Phase 24 European 20-asset × 30-contract workload took 0.7593 s on
CUDA versus 4.065 s on batched CPU for price/loss/VJP. The Phase 25 four-asset,
six-contract event workload favored CPU. These historical measured settings
and raw reports are in the archive; they are not release-wide speed claims.
Calibration error does not certify PDE discretization accuracy.

### Release candidate regression

The rebuilt Windows CPython 3.13 / PyTorch 2.10.0+cu126 candidate passes
**2,229 tests, with one intentional skip and 23 warnings in 703.48 seconds**.
The skip is the existing CPU Thomas strict-dominance exception; warnings concern
existing PyTorch tracing/profiling APIs. CUDA was available on the RTX 4060
Laptop GPU. All 2,224 prior tests remain, with five additional packaging cases.
[The regression record](benchmarks/release_regression.json) binds the result to
the native library and test-log hashes. Native changes only add compiled build
provenance and distribution configuration; numerical algorithms are unchanged.

The final wheel passes CPU/CUDA financial pricing and backward smoke checks in
an external venv with system packages disabled. All seven notebooks and eleven
bounded script configurations also pass in that environment without MKL/OMP
overrides. The inverse-volatility tutorial recovers the synthetic constant
volatility 0.24 and fits its two-parameter local-volatility example; the neural
example propagates gradients to drift and diffusion network outputs. These are
teaching experiments, not uniqueness or real-world dynamics claims. See
[example evidence](benchmarks/release_examples.json) and
[wheel evidence](benchmarks/release_packaging.json). Repeated packaging of the
same native binary is byte-identical; cross-host native bit reproducibility and
remote CI remain unverified.

The owner has authorized BSD-3-Clause, public GitHub, TestPyPI and production
PyPI publication. Remote authentication, publisher configuration and actual
index-artifact validation remain release gates. Linux binary portability and extra
Python/PyTorch variants are outside the first release claim. Further numerical
engines are not part of this release preparation.


### Authorized stable release

The official [public repository](https://github.com/aubejay22/QuantLibTorch)
belongs to the authenticated personal account `aubejay22`. Original local history
remains intact and private. A separate sanitized public history retains numerical
source, source archives, documentation and notebooks, replacing private report
paths and author metadata. Original code uses BSD-3-Clause; third-party code keeps
its own licenses and attribution. No QuantLibAAD/XAD code was introduced.

Real clean-runner CI exposed a header dependency omitted from the official Torch
wheels: local Conda had supplied nlohmann/json 3.11.2 implicitly. The build now
pins and verifies that header, includes it explicitly before ambient directories,
and checks its digest again at CMake configuration. The build and bootstrap
scripts are covered by the native source fingerprint. Seven license/notice files
include JSON's MIT attribution and the Apache-2.0 notice for a pre-C++14 fallback
excluded by this C++17 build. This is a build/provenance correction without any
numerical algorithm change. The earlier local wheel is superseded.

The corrected local Windows build passes **2,268 tests**, one intentional skip
and 23 existing warnings in **1,092.43 seconds**. Actual hosted Windows Server
2022 / official CPython 3.13.17 / MSVC 19.44.35229.0 validation passes **2,171
tests**, with 97 CUDA-unavailable skips and the same intentional skip. Its wheel
also passes seven notebooks, ten CPU example runs and an outside-checkout
installation. The exact downloaded hosted wheel then passes fresh independent
CPU/CUDA pricing and gradients on the RTX 4060 Laptop host. No stock QuantLib
library appears in its native imports.

The release candidate is public revision
`291a09ebe024a47bf4d917f0f57fbbddf7bde5ba`, built by
[Actions run 41574726134](https://github.com/aubejay22/QuantLibTorch/actions/runs/41574726134).
Its wheel SHA256 is
`b0a1c5dd652c08c755440313083ed4f5ef05607a0dcc860081a4de476d3d6ded`.
Runtime support remains Windows x86-64, CPython 3.13 and exactly Torch
2.10.0+cu126. CUDA execution is limited to the documented packed float64 APIs.
The hosted source fingerprint was independently reconstructed from Git with
Windows line endings. Different compiler output across hosts is not assumed
bit-identical. The native/private-path audit, notices, RECORD hashes and exact
artifact/source binding all pass.

An isolated Linux branch qualifies four generated friend declarations for GCC
namespace lookup and adds unchanged pinned `ql/discretizedasset.cpp` to the
source closure. [Its actual CI](https://github.com/aubejay22/QuantLibTorch/actions/runs/41575865212)
passes build/import, 2,171 tests with 98 skips, wheel inspection and a new
installed-package price/gradient check on Ubuntu 24.04 / CPython 3.13.17 /
Torch 2.10.0+cpu. These compatibility changes remain outside the Windows release
revision. The raw Linux wheel's symbol constraint is manylinux_2_34, but a
manylinux build/repair, oldest-distribution install and Linux CUDA have not been
validated. Linux portability therefore remains outside the initial PyPI claim.

The owner has authorized TestPyPI and production uploads, but confirmed that
Trusted Publisher account registration cannot be completed now. Publishing is
blocked on that external requirement. The next steps are registration, actual
TestPyPI upload/download validation, the exact `v0.3.0` tag and GitHub release,
then production upload/download validation. No registry upload or release tag
has been created. Public master remains at the tested revision so the guarded
promotion workflow can consume its exact artifact; evidence docs are prepared
separately for publication afterward.

See [the release record](release.md), [hosted build evidence](benchmarks/release_hosted_windows.json),
[exact hosted GPU evidence](benchmarks/release_hosted_cuda.json),
[artifact audit](benchmarks/release_hosted_audit.json),
[local regression](benchmarks/release_json_regression.json), and
[Linux evidence](benchmarks/release_linux_ci.json) for measurements and limits.
