# Release preparation roadmap

The current mission is a controlled first Python package release. Preserve
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

See [release documentation](release.md) for measured status, artifacts and exact
remaining external actions. A workflow file does not establish successful
remote CI validation. Linux/manylinux and additional Python/PyTorch builds
remain experimental until their own clean-install runs pass.

The owner has approved BSD-3-Clause, public GitHub, TestPyPI and production PyPI
publication. Account authentication and required platform approvals remain gates;
they do not block independent engineering work. Completion requires a real
production-index installation with passing pricing and gradients.
