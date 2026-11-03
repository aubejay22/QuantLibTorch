# QuantLibTorch — current project context

## Current mission: finish the authorized public release

Prepare a maintainable, documented Python package with executable examples and
validated binary wheels. Preserve existing numerical functionality. New pricing
engines, unrelated numerical research and architecture pivots are outside this
mission. The pre-cleanup checkpoint is `6770549`, based on validated Phase 26
evidence checkpoint `b128a95` and implementation `c80448a`.

The stable release target is **0.3.0**, based on preserved candidate checkpoint
`1002f78` (**0.3.0rc1**). The public repository is
https://github.com/aubejay22/QuantLibTorch. Its tested release revision is
`291a09ebe024a47bf4d917f0f57fbbddf7bde5ba`, with successful hosted candidate
run `41574726134`. Keep public master at this revision until the TestPyPI stage
finishes; the promotion workflow requires that exact revision. The separate
public checkout is `build/public-release`; never push the original local history.
Final evidence documentation is staged on `docs/release-0.3-evidence` instead. The initial target is Windows
x86-64, CPython 3.13, PyTorch 2.10.0+cu126, with CPU and supported CUDA execution.
Consult [release support and validation](docs/release.md) before making any
compatibility claim. Linux development/CI preparation is not a validated Linux
binary release. The package uses a wheel-first distribution policy.

The owner explicitly authorized BSD-3-Clause for eligible original code, public
GitHub under the authenticated personal account `aubejay22`, TestPyPI, and the
first production PyPI release. These decisions do not need reconfirmation.
Authentication, required platform approvals, security, licensing, TestPyPI and
exact-artifact validation remain mandatory gates. Original local history must
remain intact; sanitize personal metadata in a separate public export.

## Engineering invariants

- Approach A remains the priority: pinned QuantLib 1.40 at
  `06edddbdb85f006c43a449b5fc846a00c67477f3`, compiled with `QL_REAL = TorchReal`
  and linked to LibTorch. Never link a stock-double QuantLib into this library.
- Preserve upstream algorithm provenance. Generated compatibility overlays are
  reproducible and separate from numerical adapters. Do not copy QuantLibAAD/XAD
  implementation without license review.
- Keep active tensors graph-connected and out of passive geometry caches. Avoid
  in-place scalar-adapter tensor updates. Repeated pricing creates fresh graphs.
- Preserve full AD, first-order equation adjoints, latent neural models,
  calibration, CPU batching, CUDA and independent numerical references.
- Run `python -m pytest` after native builds. CPU correctness precedes optimization;
  GPU advantage requires a measured workload. Record findings and limitations in
  [architecture](docs/architecture.md).
- Discuss major architecture pivots. Reversible work within this release mission
  is authorized. Use reviewed, tested Git checkpoints for substantive milestones.

## Current library

The public API includes analytic European pricing, vanilla local-volatility FD,
original Heston and shout engines, a two-asset basket engine, joint-state latent
PDEs, compatible contract/independent asset batching and calibration. Reusable
financial configurations compose with a `torch.nn.Module` pricing layer.
Supported prices and sensitivities are listed in the
[capability matrix](docs/capabilities.md); do not infer full QuantLib coverage.

Scalar financial engines use CPU float64. Packed latent APIs support CPU/CUDA
float64 subject to checked solver-domain restrictions. Full mode supports tested
higher derivatives; equation adjoints are first order. Fixed meshes, dates and
event schedules remain passive. Heston/shout have explicit prepared geometry.

The corrected native build passes **2,268 tests with one intentional skip**.
Hosted Windows passes **2,171 tests with 98 skips**, seven notebooks, ten CPU
examples and an independent installation. The exact hosted wheel passes a fresh
CPU/CUDA installation and gradients on the RTX 4060 host. Its SHA256 is
`b0a1c5dd652c08c755440313083ed4f5ef05607a0dcc860081a4de476d3d6ded`.
The hidden Conda JSON header dependency discovered by actual CI is now explicitly
pinned, checksum-verified, and covered by seven retained license/notice files.
The earlier local `396250...` wheel is superseded and must not be published.

The owner confirmed that pending Trusted Publishers cannot be registered now.
No TestPyPI/PyPI upload, GitHub release or `v0.3.0` tag has been created. Account
registration is the external blocker, not general release authorization. Once
configured, perform actual TestPyPI download/install checks, tag the tested
public revision, publish GitHub and PyPI, then validate a production-index
installation. The complete goal remains unfinished until those gates pass.

Linux CPU build/tests/installation passed on isolated branch `ci/linux-compat`
(commit `f565a2f64d1ecd010cf4b0a8d6babdf8dd290da7`); it is unmerged and not a
portable Linux release. No Linux CUDA or manylinux portability claim is made.

## Navigation

- [README](README.md): installation and a first price/gradient.
- [User guide](docs/user-guide.md): models, financial inputs, AD modes and batching.
- [Architecture](docs/architecture.md): maintained implementation contracts.
- [Development](docs/development.md): source map, tests and audit decisions.
- [Release roadmap](docs/roadmap.md): remaining acceptance gates.
- [Historical project context](docs/history/project-context.md),
  [phase reports](docs/history/architecture.md) and
  [old roadmap](docs/history/roadmap.md): preserved provenance through Phase 26.

Historical reports describe their checkpoints. Their old restrictions and
proposed engine expansions do not override this current release mission.
