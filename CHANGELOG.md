# Changelog

## 0.3.0 — authorized release, publication pending

- Apply BSD-3-Clause to original code and retain QuantLib-derived attribution
  plus QuantLib, Boost and PyTorch notices.
- Remove developer paths from native diagnostics through compiler prefix maps.
- Preserve the original local history and prepare a separate audited public
  snapshot with private report metadata sanitized.
- Add revision-bound Windows wheel validation, secure OIDC publication through
  separate TestPyPI/PyPI environments, and fresh index-download installation
  checks. Linux remains an experimental CI target.
- Preserve the existing financial engines, AD modes, CUDA APIs and notebooks.

Public availability and exact measured validation are recorded in
[the release record](docs/release.md). General owner publication authorization
is granted; authentication and artifact validation remain mandatory.

## 0.3.0rc1 — controlled distribution candidate

Historical first release candidate, preserved at `1002f78`. Its pending license
and publication decisions were subsequently authorized for 0.3.0.

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
