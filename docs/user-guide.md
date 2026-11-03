# User guide

QuantLibTorch prices supported equity derivatives using original QuantLib
engines and numerical infrastructure, with tensor inputs connected to PyTorch.
It is useful when the result is part of a loss: recovering volatility, fitting
a parametric model or training a neural network that supplies PDE coefficients.
It does not replace all of QuantLib's Python bindings. The
[capability matrix](capabilities.md) states exactly which inputs differentiate.

## Prices, Greeks and differentiation modes

Start with the [pricing and Greeks notebook](../examples/option_pricing_greeks.ipynb).
The analytic European path supports spot Delta, volatility Vega, rates, yields,
strike, dated discount factors and cash amounts on CPU float64. For a second
derivative, retain the first derivative's graph:

```python
import torch
from quantlibtorch import analytic_european

spot = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
volatility = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
price = analytic_european(volatility, spot=spot)
delta, = torch.autograd.grad(price, spot, create_graph=True)
gamma, = torch.autograd.grad(delta, spot)
```

Volatility `0.2` means 20% annualized volatility. Rates and continuous yields
are annualized continuously compounded fractions. An autograd Vega is price
change per unit volatility; divide by 100 for a one-percentage-point change.
An autograd derivative with respect to log spot is `S * Delta`.

| Mode | Use | Contract |
| --- | --- | --- |
| `full` | Higher reverse derivatives, gamma, Hessian-vector products | Differentiates executed supported arithmetic; graph memory may be large |
| `adjoint` | First-order price/loss gradients through PDE engines | Discrete equation VJPs; rejects higher/forward derivative requests |

Both modes target the same numerical price under matching settings. An adjoint
propagates coefficient sensitivities through the discrete equations and back to
the neural model; it does not approximate a neural gradient by bumping parameters.
The analytic engine uses full AD only. `MultiAssetBatch` exposes checkpointed
adjoints to trade recomputation for saved trajectory memory.

Exercise projections and interpolation are piecewise differentiable. Finite
differences across a branch change need not match the selected local derivative.
Mesh spacing, domains and time steps affect prices and sensitivities; check
refinement separately from optimizer convergence.

## Instruments, markets and reusable layers

Use functions for one-off calls, or fixed configurations for repeated pricing:

```python
from datetime import date
import torch
from quantlibtorch import (
    Exercise, VanillaOption, EquityMarket, FdVanillaEngine, PricingLayer,
)

option = VanillaOption(
    strike=100.0, maturity_date=date(2026, 1, 1), option_type="put",
    exercise=Exercise.bermudan([date(2025, 7, 1), date(2026, 1, 1)]),
)
market = EquityMarket(
    valuation_date=date(2025, 1, 1), spot=100.0,
    rate=0.03, dividend_yield=0.01,
)
layer = PricingLayer(option, market, FdVanillaEngine(
    time_steps=40, grid_points=81, mode="adjoint",
))
volatility = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
price = layer(volatility)
vega, = torch.autograd.grad(price, volatility)
```

`Exercise.european()` allows exercise at expiry. `Exercise.american()` starts at
valuation. Bermudan dates must increase, follow valuation and include maturity.
Dates are `datetime.date` objects. Calendar adjustment is explicit using
`adjust_date` or `advance_date`; engines do not silently shift event schedules.
Actual/365 Fixed is the default. Heston/shout also accept Actual/360.

`EquityMarket` holds fixed default numbers. Supply differentiable market tensors
as fresh call overrides on engines that support them, for example
`analytic_layer(volatility, spot=spot_tensor, rate=rate_tensor)`. The vanilla FD
engine differentiates **only its local-volatility tensor**; passing a spot
tensor there does not enable Delta. Choose the supported engine/input pair from
the matrix. Financial scalar inputs are CPU float64, including market overrides
for CUDA batch plans.

A dated curve is a sequence of `(date, discount_factor)` pairs, starting at
valuation with a fixed `1.0`, covering maturity and using increasing dates.
QuantLib applies log-linear discount interpolation. A supplied curve replaces
its flat rate; active shadowed rates raise. Cash schedules are `(date, amount)`
pairs. Analytic escrowed dividends and FD spot jumps are distinct financial
models; choose the convention deliberately.

`PricingLayer` is a `torch.nn.Module`. With `model=network`, positional inputs go
through that network; its tensor or tuple of tensors goes to the engine.
Parameters register normally with `parameters()` and `state_dict()`. Reconstruct
financial and numerical configuration separately when loading a state dict.
Every forward call creates a fresh graph; no implicit device/dtype conversion
occurs. Use `.double()` on neural models and recreate fields after each optimizer
step instead of reusing an old autograd graph.

## Volatility inversion and calibration

The [inverse volatility notebook](../examples/inverse_volatility_calibration.ipynb)
starts from synthetic prices and optimizes unknown volatility parameters.
Implied-volatility inversion recovers the constant Black volatility associated
with a particular price and financial contract. Fitting one shared constant
across contracts is a restricted calibration. Fitting a function
`sigma(t, S; theta)` is local-volatility calibration, a different inverse problem.
Several parameterizations can fit the same finite set of quotes.

`fd_vanilla` accepts either a scalar volatility or a surface of shape
`[len(times), len(log_spots)]`. `times` are years since valuation; `log_spots`
are natural logarithms of absolute spot levels. Samples use bilinear
interpolation with flat extrapolation. The samples can be outputs of a network
and retain its graph. `mesh_volatility` controls the passive automatic mesher;
it does not replace the supplied local volatility.

For larger latent models, `OptionQuote` and `LatentCalibration` group a fixed
quote objective. `loss()` builds a usual graph. `backward()` accumulates
first-order gradients contract by contract (or maturity group with
`execution="batched"`) and returns detached values for logging. Zero parameter
gradients before each update. Its streaming method requires deterministic model
callbacks and a separable quote loss; apply nonseparable regularizers separately.

## Neural SDEs and latent factors

The [neural latent SDE notebook](../examples/neural_latent_sde.ipynb) trains an
actual PyTorch neural network through the pricing PDE. With state
`Z = (log S, Y_1, ..., Y_m)`, the model supplies latent drift `b_Y` and joint
diffusion loadings `L`. The covariance is `C = L L.T`; rectangular noise loading
matrices are allowed. The risk-neutral log-stock drift is imposed as
`r - q - C[0,0]/2`, and the PDE discounts at `r`.

For a terminal payoff `g`, the interior equation is

\[
\partial_t V + b \cdot \nabla V
+ \tfrac12\sum_{ij} C_{ij}\partial_{ij}V - rV = 0,
\qquad V(T, Z) = g(\exp(Z_0)).
\]

The latent variable can represent hidden volatility or another model state;
the first diffusion row gives relative stock noise, and remaining rows give
absolute latent noise. A quote-only synthetic experiment fits a risk-neutral
model under Q. It does not identify physical dynamics under P or unique hidden
factors. Bounded smooth network outputs provide useful constraints; sampled
arrays alone do not establish well-posed SDE dynamics or appropriate boundaries.

`state_grid(axes)` supplies `[N, 1+m]` mesh coordinates with axis zero varying
fastest. Use it to sample the network without a Python callback inside rollback.
One time row means time-constant fields; otherwise fixed sample times cover the
whole maturity horizon. The solver interpolates **covariance**, drift and killing
between time samples, rather than interpolating noise loadings. Keep times
absolute from valuation across contracts of different maturities.

| Input/output | Single or contract batch | Asset batch |
| --- | --- | --- |
| Latent drift | `[T, N, m]` | `[A, T, N, m]` |
| Diffusion loadings | `[T, N, 1+m, W]` | `[A, T, N, 1+m, W]` |
| Initial state `(log S, Y)` | `[1+m]` | `[A, 1+m]` |
| Price | scalar / `[contracts]` | `[A, contracts]` |

`T` here counts coefficient samples, not rollback steps; `N` is the product of
axis lengths, and `W` counts noise factors. One/two latent factors are supported
by financial pricing. `diffusion_rollback` is the lower-level general equation
interface; it accepts all drift components, terminal values and optional killing.

## CPU, CUDA and batching

Follow the [multi-asset notebook](../examples/multi_asset_gpu.ipynb) for a complete
CPU/CUDA price-and-gradient comparison. `LatentBatch` reuses one operator over
compatible strikes and call/put contracts. `MultiAssetBatch` batches independent
assets with separate model fields and returns assets × contracts. It does not
price a correlated basket; `fd_basket` provides its supported two-asset min/max
payoffs on CPU.

Contracts within an asset share dates, numerical grid, coefficients, market,
scheme, steps and event schedule. Across assets, incompatible numerical shapes,
times or schedules form separate groups; asset order is preserved. Compatible
assets may have different grid coordinates, curves and strikes. Contract counts
must agree; create separate plans for ragged portfolios. Inspect `plan.groups`
to see execution groups. Per-asset lists of tensors accommodate different node
shapes across groups.

Move neural models and fields to the selected device explicitly and construct
the plan for that device. `MultiAssetBatch.price(..., markets=[...])` takes fresh
financial overrides for each asset. These scalar market tensors stay on CPU;
graph-connected transfers supply interval forwards to the device. See
[financial inputs](../examples/financial_inputs.ipynb) for executable examples.

Small batches often favor CPU because GPU kernel dispatch and synchronization
costs dominate. Larger compatible portfolios can benefit from CUDA. Measure
complete forward/loss/backward time after warmup, synchronize CUDA around timing,
and compare against an explicitly configured CPU thread count. Separate plan
construction from repeated execution and include it when relevant to your use.

All financial paths require float64. CUDA's checked directional solver rejects
systems outside its dominance/residual contract. Packed pricing does not support
damping or arbitrary boundary callbacks; one/two latent factors support exercise,
and cash dividends require one factor. There is no general `torch.compile` or
`torch.func` contract for financial engines. Full support details and remaining
QuantLib coverage gaps are in [capabilities](capabilities.md).

## Public API reference

Each public function/class has a docstring accessible through `help()`.
The implementation files are small entry points to their precise signatures:

| Area | Public entries and source |
| --- | --- |
| Financial configuration | `Exercise`, `VanillaOption`, `EquityMarket`, `PricingEngine`, `PricingLayer`, five engine configurations in [api.py](../python/quantlibtorch/api.py) |
| Analytic / vanilla FD | [analytic_european](../python/quantlibtorch/analytic.py), [fd_vanilla](../python/quantlibtorch/fd.py) |
| Original model engines | `HestonModel`, `HestonMesh`, `heston_mesh`, `fd_heston` in [heston.py](../python/quantlibtorch/heston.py); `ShoutMesh`, `shout_mesh`, `fd_shout` in [shout.py](../python/quantlibtorch/shout.py) |
| Basket / latent PDE | [fd_basket](../python/quantlibtorch/fd_basket.py); `state_grid`, `fd_latent`, `diffusion_rollback` in [latent.py](../python/quantlibtorch/latent.py) |
| Batching / inverse problems | `LatentBatch`, `diffusion_rollback_batch` in [batch.py](../python/quantlibtorch/batch.py); [MultiAssetBatch](../python/quantlibtorch/multi_asset.py); `OptionQuote`, `LatentCalibration` in [calibration.py](../python/quantlibtorch/calibration.py) |
| Conventions | `adjust_date`, `advance_date`, `year_fraction` in [conventions.py](../python/quantlibtorch/conventions.py) |
| Numerical primitives | `tridiagonal_solve` in [solver.py](../python/quantlibtorch/solver.py); `tridiagonal_apply`, `band_axpyb`, `tridiagonal_solve_pcr` in [kernels.py](../python/quantlibtorch/kernels.py) |
| Preserved prototype API | `european_call`, `log_moneyness_grid` in [pricing.py](../python/quantlibtorch/pricing.py); use the engine APIs above for full financial conventions |
