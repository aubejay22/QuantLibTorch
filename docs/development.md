# Development and repository map

Read [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) and
[architecture.md](architecture.md) before changing direction.
[AGENTS.md](../AGENTS.md) remains the engineering workflow. The current task is
package quality and controlled distribution; original algorithms and independent
references are regression requirements.

| Location | Responsibility |
| --- | --- |
| `python/quantlibtorch/` | Public functions, validated financial configuration, native loader and PyTorch composition |
| `cpp/`, `cpp/include/quantlibtorch/` | TorchReal adapter, native bridges, autograd boundaries and numerical kernels |
| `cmake/quantlib_sources.cmake` | Explicit pinned upstream dependency closure |
| `scripts/bootstrap.py`, `scripts/build.py` | Reproducible dependency acquisition and native builds |
| `scripts/prepare_quantlib.py`, `scripts/prepare_packed.py` | Compatibility overlays, mesh hooks and generated packed ADI adapters |
| `tests/` | Numerical, financial, derivative, CPU/CUDA, API and packaging regressions |
| `examples/` | Executable public-API tutorials and focused scripts |
| `docs/` | Current user/developer/release documentation |
| `docs/history/` | Archived phase narratives, context and feature roadmap |
| `docs/benchmarks/` | Measured evidence, preservation manifests and source snapshots |
| `build/`, `third_party/`, `.venv/` | Ignored generated build outputs, pinned source checkout and local environment |

## Build and checks

Follow [install.md](install.md#source-development), then run `python -m pytest`
after every native build. Use targeted tests while implementing a change and
the complete suite before accepting an integrated candidate. Run notebooks with
`python scripts/validate_examples.py`; use the isolated wheel verification in
[release.md](release.md) for distribution checks.

Stock QuantLib references run in separate Python processes. Do not load a
stock-double QuantLib library into the TorchReal build. Finite differences must
hold geometry/configuration fixed when verifying fixed-mesh derivatives.
Preserve tests of unsupported inputs and solver domains as well as successful
prices. A CUDA availability skip is not evidence that CUDA was exercised.

## Cleanup audit decisions

The audit starts from clean checkpoint `6770549` and reviews the source tree,
recent implementation/evidence commits, CMake closure, Python exports, tests,
script references and documentation links. The previous entry documents mixed
current support with thousands of lines of superseded phase status. The cleanup
archives those narratives and retains concise current contracts:

| Previous content | New location / treatment |
| --- | --- |
| 5,690-line `docs/architecture.md` phase narrative | [history/architecture.md](history/architecture.md); current architecture now focuses on implementation invariants |
| 609-line `PROJECT_CONTEXT.md` research/history narrative | [history/project-context.md](history/project-context.md); current context now records the release mission and engineering constraints |
| 74-line Phase 25/26 feature roadmap | [history/roadmap.md](history/roadmap.md); current roadmap now defines release acceptance gates |
| Long README covering many individual engines | Concise installation, one executable price/gradient, tutorial navigation and capability links |
| Existing capability matrix | Retained and clarified; current device/derivative limits remain explicit |

Archive notices distinguish historical claims. Relative Markdown links are
relocated; benchmark files and source snapshots remain at their original paths.
[release_docs_cleanup.json](benchmarks/release_docs_cleanup.json) records original
and archive hashes. Earlier Git commits retain byte-exact originals.

An AST/identifier audit of all 124 existing Python files found no top-level
private function/class with only its definition as an identifier occurrence.
That is evidence against easy dead-code deletion, not a proof that every path
is reachable. No supported implementation is removed from that heuristic alone.

No numerical source or regression test was removed on the basis of age or lack
of a Python import. In particular:

- `cpp/*checks.cpp` and `cpp/pde_probe.cpp` are native validation entry points;
  CMake/bridge/test references establish their use.
- `tests/stock_*_reference.py` are isolated independent financial oracles.
  Similar-looking engine/adjoint/curvature suites test different contracts.
- `pricing.py` and its earlier implicit-Euler API remain exported and tested;
  their narrower scope does not make them dead code.
- Scalar, tensor and feature-disabled arithmetic paths provide independent
  comparisons and supported fallbacks. Preserve them when pruning experiments.
- Benchmark, diagnostic, preservation and summary scripts support retained
  reports. They are development workflows, not installed runtime modules.
- Historical source ZIPs and benchmark JSONs preserve algorithm provenance,
  rejected experiments and measured comparisons. Do not relocate them without
  checking manifests and scripts for path dependencies.
- Ignored build outputs and environments are not distributed source clutter.
  Some retain baseline binaries for reproducibility; do not remove them merely
  to make a checkout's disk usage smaller during release validation.

Third-party notice obligations and the pending own-code license decision are
recorded in [licensing.md](licensing.md). Preserve existing attribution and
review any new dependency before copying implementation code.

## Reproducing historical measurements

Start from a phase section in [history/architecture.md](history/architecture.md).
It links exact scripts, command manifests, source archives and measured native
fingerprints. Commands in historical prose are repository-relative. Recreating
an earlier report may require its recorded Git checkout/build; running a current
script on the current binary does not recreate the measured artifact.

Runtime controls are in `scripts/runtime_controls.py`; the library does not
silently change process priority or CPU policy. Synchronize CUDA timing and
report thread count, warmup, setup, loss/backward work and peak memory. A faster
discrete computation is not evidence of continuous-model pricing accuracy.
