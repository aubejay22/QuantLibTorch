# Release milestones

The first public release, **0.3.0**, is published and validated. Preserve
the validated engines, full AD, discrete adjoints, neural coefficients and
CPU/CUDA behavior. Adding engines or conducting unrelated numerical research
is outside scope. The original Phase 25/26 feature plan is preserved in
[history/roadmap.md](history/roadmap.md).

| Milestone | Acceptance gate |
| --- | --- |
| Repository cleanup | Audit dependencies; preserve independent references and provenance; archive obsolete narrative; Git checkpoints before/after significant changes |
| User experience | Concise README, explicit financial/derivative capability matrix, coherent public API, useful errors and installation guidance |
| Executable teaching material | Pricing/Greeks, inverse volatility, neural latent SDE and multi-asset CPU/CUDA notebooks execute with meaningful assertions |
| Compatibility and licensing | Exact tested support matrix, strict native ABI/source metadata, reviewed third-party notices; owner approves own-code license |
| Binary distribution | Correct native wheel; clean install outside repository; import, financial pricing and gradients; unsupported source installation fails clearly |
| Release candidate | Consistent version, changelog, artifact hashes and secure TestPyPI rehearsal; full regression suite passes |

All initial release milestones above are complete for the documented Windows
configuration. [Release documentation](release.md) records the public GitHub
release, actual TestPyPI and production PyPI artifacts, native regression tests,
executed notebooks and independent index-installed CPU/CUDA checks. The initial
release source is fixed at `v0.3.0`; evidence-only documentation may advance.
Linux/manylinux and additional Python/PyTorch release wheels remain outside the
support matrix until their own portability and clean-install gates pass.

The owner has approved BSD-3-Clause, public GitHub, TestPyPI and production PyPI
publication. Saved publisher fields were compared with GitHub, both OIDC uploads
succeeded, and real production-index installations passed pricing and gradients.
Future versions require their own reviewed source, version-specific tag and
environment policies, compatibility validation and secure publication gates.
