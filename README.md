# QuantLibTorch

Differentiable option pricing with QuantLib and PyTorch. Price supported equity
options, compute Greeks, and fit volatility functions or neural diffusion models
using ordinary PyTorch losses and optimizers.

QuantLibTorch compiles selected original QuantLib 1.40 engines and numerical
operators with a PyTorch-backed scalar. It preserves their financial conventions
while connecting supported inputs to autograd. It is a focused extension;
the complete QuantLib Python API is not exposed.

## Install

**QuantLibTorch 0.3.0** is available on [PyPI](https://pypi.org/project/quantlibtorch/0.3.0/)
and in the [GitHub release](https://github.com/aubejay22/QuantLibTorch/releases/tag/v0.3.0).
The supported wheel targets Windows x86-64, CPython 3.13 and
**PyTorch 2.10.0+cu126**. This PyTorch build supports CPU execution; supported
batch APIs also use CUDA on a compatible NVIDIA GPU. A separate CPU-only PyTorch
build and Linux binary wheels are outside the first release matrix.

For the published 0.3.0 wheel, create a Python 3.13 environment and run:

```text
python -m pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install quantlibtorch==0.3.0
```

Users installing a wheel need no compiler or QuantLib source checkout. See
[installation](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/install.md)
for runtime requirements, source development and troubleshooting, and the
[release record](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/release.md)
for artifact hashes and validation evidence.

## Price and differentiate

```python
from datetime import date
import torch
from quantlibtorch import analytic_european

spot = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
vol = torch.tensor(0.20, dtype=torch.float64, requires_grad=True)
price = analytic_european(
    vol, spot=spot, strike=100.0, rate=0.03, dividend_yield=0.01,
    valuation_date=date(2025, 1, 1), maturity_date=date(2026, 1, 1),
)
delta, vega = torch.autograd.grad(price, (spot, vol))
print(price.item(), delta.item(), vega.item())
```

Prices are tensors. Vega above is per unit volatility: multiply by `0.01` for a
one-percentage-point change. All financial paths use `torch.float64`.

## Choose a workflow

| Task | Start here |
| --- | --- |
| European, American and Bermudan pricing; Greeks | [Pricing notebook](https://github.com/aubejay22/QuantLibTorch/blob/master/examples/option_pricing_greeks.ipynb) |
| Recover volatility from synthetic option prices | [Inverse volatility notebook](https://github.com/aubejay22/QuantLibTorch/blob/master/examples/inverse_volatility_calibration.ipynb) |
| Train neural risk-neutral drift/diffusion with latent factors | [Neural SDE notebook](https://github.com/aubejay22/QuantLibTorch/blob/master/examples/neural_latent_sde.ipynb) |
| Price and differentiate assets × contracts on CPU/CUDA | [Batching notebook](https://github.com/aubejay22/QuantLibTorch/blob/master/examples/multi_asset_gpu.ipynb) |
| Original Heston, shout engines and financial curves | [Heston](https://github.com/aubejay22/QuantLibTorch/blob/master/examples/heston_pricing.ipynb), [shout](https://github.com/aubejay22/QuantLibTorch/blob/master/examples/shout_pricing.ipynb), [financial inputs](https://github.com/aubejay22/QuantLibTorch/blob/master/examples/financial_inputs.ipynb) |
| Reusable instruments, markets, models and `PricingLayer` | [User guide](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/user-guide.md) |
| Exact instruments, derivatives, schemes and device restrictions | [Capability matrix](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/capabilities.md) |

Use `mode="full"` for supported higher derivatives; use `mode="adjoint"` for
first-order discrete equation VJPs in PDE engines. Both differentiate the
implemented discretization. Neural coefficient tensors retain their graphs.
`MultiAssetBatch` offers checkpointed adjoints; small workloads may run faster
on CPU. See the guide before selecting grids or interpreting a calibration fit.

## Development and provenance

Start with [build instructions](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/install.md#source-development), the
[repository map](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/development.md) and [current architecture](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/architecture.md).
Run `python -m pytest` after native builds. Independent financial references,
gradient tests, CPU/CUDA coverage and historical benchmark artifacts are retained.
The [historical archive](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/history/architecture.md) records earlier measured
results separately from current support claims.

Original project code is licensed under [BSD-3-Clause](https://github.com/aubejay22/QuantLibTorch/blob/master/LICENSE). QuantLib-derived
code retains its upstream license and attribution. QuantLib's notices are retained in
[LICENSES/QuantLib.txt](https://github.com/aubejay22/QuantLibTorch/blob/master/LICENSES/QuantLib.txt); see the
[licensing review](https://github.com/aubejay22/QuantLibTorch/blob/master/docs/licensing.md) for dependency notices and provenance.
