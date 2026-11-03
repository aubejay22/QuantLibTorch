# QuantLibTorch 0.3.0

First stable release of QuantLibTorch: selected original QuantLib 1.40 engines
compiled with a PyTorch-backed scalar, exposing graph-connected prices and
sensitivities through Python.

Available on [PyPI](https://pypi.org/project/quantlibtorch/0.3.0/) and in the
[GitHub release](https://github.com/aubejay22/QuantLibTorch/releases/tag/v0.3.0).

- Analytic European pricing and supported local-volatility finite-difference
  European, American and Bermudan pricing.
- Original Heston and shout engines, financial curves, a two-asset basket engine,
  latent neural diffusion models and compatible asset/contract batching.
- Full automatic differentiation with tested higher derivatives and first-order
  discrete equation adjoints; supported packed CPU/CUDA execution in float64.
- Seven educational notebooks covering pricing, Greeks, inverse volatility,
  neural SDEs, financial inputs, Heston, shout and batching.
- BSD-3-Clause for original code, with upstream QuantLib attribution and
  QuantLib, Boost, PyTorch, nlohmann/json and applicable Abseil notices included.

Binary support is limited to Windows x86-64, CPython 3.13 and exactly
PyTorch 2.10.0+cu126. Install that Torch build first from its official cu126
index, then install `quantlibtorch==0.3.0`. Wheel users need no QuantLib source
checkout or compiler. The Microsoft Visual C++ x64 runtime is required.

Scalar engines use CPU; CUDA is available only through the documented packed
APIs. Adjoint mode is first order. Meshes, schedules and dates remain passive.
The full QuantLib Python API is not exposed. Linux wheels, other Python/Torch
variants and portable cross-host native bit reproducibility are not claimed.

See the [release record](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/release.md)
for artifact hashes, installed price/gradient results and the supported
configuration, and the [capability matrix](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/capabilities.md)
for financial and differentiation restrictions.

Validation: 2,268 local native tests passed; hosted Windows passed 2,171 tests
and all seven notebooks. Actual TestPyPI and production PyPI installations passed
pricing and gradients in fresh external environments, including the supported
CPU/CUDA paths. The installed example gives price `8.827321225352126`, delta
`0.573495979027775` and vega `38.71517541592269`.

Release source: `291a09ebe024a47bf4d917f0f57fbbddf7bde5ba`.
Wheel SHA256: `b0a1c5dd652c08c755440313083ed4f5ef05607a0dcc860081a4de476d3d6ded`.
