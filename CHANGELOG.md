# Changelog

## 0.3.0rc1 — controlled distribution candidate

First release candidate; not yet published. Project license approval and an
explicit TestPyPI upload authorization remain separate release gates.

- Retain original QuantLib analytic, vanilla FD, Heston, shout, basket and latent
  pricing, full reverse AD, first-order discrete adjoints, and CPU/CUDA batches.
  The [capability matrix](docs/capabilities.md) defines the supported subset.
- Add tutorials for option Greeks and exercise, volatility inversion and local
  volatility calibration, neural latent risk-neutral models, and multi-asset
  CPU/CUDA pricing and gradients. Execute notebooks with a reproducible runner.
- Separate current user guides and installation instructions from archived
  development history; retain numerical tests, independent financial references,
  benchmark evidence and original source snapshots.
- Expose `quantlibtorch.__version__`, improve public kernel annotations, and
  report missing or incompatible native artifacts with actionable import errors.
- Bind native provenance to the compiled library. Check exact PyTorch variant,
  Python/platform/ABI, pinned QuantLib commit, source fingerprint and binary hash;
  reject missing metadata and relabelled stale binaries.
- Prepare wheels for a deliberately limited Windows x64 / CPython 3.13 /
  PyTorch 2.10.0+cu126 candidate. CPU operations use the same wheel. Linux CI is
  an unvalidated development target, not a claimed portable release platform.
- Disable source distributions explicitly: developer builds require a checkout,
  pinned dependencies and an explicit native build. End users install wheels.

See [release procedure and validation](docs/release.md) for artifacts, exact
tested environments, reproducibility limits and pending publication steps.

## 0.3.0 development baseline

The unreleased Phase 26 checkpoint `b128a95` records 2,224 passing tests and one
intentional skip. Its original Heston/shout integration, financial market VJPs,
calendar behavior and historical measurements are preserved in the
[architecture history](docs/history/architecture.md). Earlier phase numbers and
versions there describe development checkpoints, not public releases.
