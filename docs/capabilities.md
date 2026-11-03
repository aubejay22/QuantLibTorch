# Supported financial and differentiation capabilities

All financial paths use pinned QuantLib 1.40 code compiled with `QL_REAL =
TorchReal`. This is a selected, tested dependency closure, not whole-library
automatic differentiation. Existing functional APIs remain available. Start
with the [user guide](user-guide.md) for input shapes and worked workflows.
Device capabilities below require a compatible installed binary; see the
[release matrix](release.md) for the tested operating system/Python/PyTorch build.

| API / original implementation | Instruments and events | Differentiable inputs | AD | Devices |
| --- | --- | --- | --- | --- |
| `analytic_european` / AnalyticEuropeanEngine and AnalyticDividendEuropeanEngine | European call/put; continuous yield; analytic escrowed cash dividends | Scalar volatility, spot, strike, flat rate/yield, noninitial discount factors, cash amounts | Full, including higher reverse derivatives | CPU float64 scalar |
| `fd_vanilla` / FdBlackScholesVanillaEngine | European, American, Bermudan call/put; continuous yield; spot or escrowed cash dividends; dated curves | Sampled local volatility | Full or first-order equation adjoints | CPU float64 scalar contract |
| `fd_heston` / FdHestonVanillaEngine, HestonModel, HestonProcess | European, American, Bermudan call/put; spot cash dividends; dated curves; explicit original or custom mesh | Initial variance, mean reversion, long-run variance, volatility of volatility, correlation, spot, rate/yield, noninitial dated discount factors | Full; first-order equation adjoints without damping | CPU float64 scalar contract |
| `fd_shout` / FdBlackScholesShoutEngine | One-shout call/put; American or Bermudan shout opportunities; European maturity-only; original escrowed cash dividends | Constant Black volatility, spot, rate/yield, noninitial dated discount factors | Full or first-order equation adjoints | CPU float64 scalar contract |
| `fd_basket` / Fd2dBlackScholesVanillaEngine | Two-asset min/max call/put; European, American, Bermudan; continuous yields, dated curves | Two local-volatility surfaces and correlation | Full or first-order equation adjoints | CPU float64 scalar contract |
| `fd_latent` / original spatial operators and Fdm2DimSolver/FdmNdimSolver | Equity call/put with one/two latent factors; European, American, Bermudan; continuous yield, dated curves | Joint-state drift/diffusion fields, initial state, rate/yield and dated discount factors | Full or first-order equation adjoints | CPU float64 scalar contract |
| `LatentBatch` / packed upstream ADI stages and financial events | Compatible common-expiry calls/puts; European, American, Bermudan; spot cash dividends with one latent factor | Shared drift/diffusion fields, initial state, per-call rate/yield and dated discount factors | Full or first-order equation adjoints | CPU/CUDA float64; connected CPU spline readout |
| `MultiAssetBatch` / independent asset operators, same packed stages/events | Asset × compatible contract groups; same events as LatentBatch | Per-asset drift/diffusion, initial state, per-call rates/discount factors and shared neural parameters | Full or checkpointed first-order equation adjoints | CPU/CUDA float64; device spline readout |
| `diffusion_rollback` / `diffusion_rollback_batch` | General diffusion terminal-value equations on fixed tensor-product axes | Drift, noise loadings, killing, terminal values | Full or first-order equation adjoints | CPU float64; single/shared-operator multiple RHS |

## Reusable Python and PyTorch configuration

`VanillaOption`, `Exercise`, and `EquityMarket` hold fixed financial data.
`AnalyticEuropeanEngine`, `FdVanillaEngine`, `FdLatentEngine`, `FdHestonEngine`
and `FdShoutEngine` hold reusable
engine settings. `PricingLayer` is a `torch.nn.Module`; it can register a neural
model and pass its tensor output(s) to an engine. Every call builds a fresh graph.
Differentiable market inputs are supplied as call overrides on the analytic,
Heston, shout and latent engines. `HestonModel` registers five trainable model
parameters (or fixed buffers); constrained custom modules can return the same
five tensors. A `PricingEngine` protocol allows additional engine adapters without
changing the layer. The optional model's parameters appear in `parameters()`
and `state_dict()`; reconstruct financial configuration separately on reload.

For equity latent pricing, differentiable drift fields describe the latent
coordinates. Equity log-price drift is imposed from rates and covariance to
retain the risk-neutral convention. General diffusion rollback accepts all
drift components explicitly.

The existing basket functions, batch plans and calibration APIs remain public;
they are not forced into the single-equity configuration interface.

## Numerical and financial conventions

Heston exposes Douglas, Craig-Sneyd, modified Craig-Sneyd and Hundsdorfer.
Full AD allows original implicit-Euler damping; adjoint damping is rejected
because its Krylov iteration lacks a supported equation adjoint. Shout exposes
all six 1D schemes and damping. Both retain original engine boundary behavior.
`heston_mesh` and `shout_mesh` prepare original automatic meshes from explicit
passive reference numbers. Reuse their immutable objects for pricing and
parameter bumps. Custom axes are allowed, retaining original boundary rows.
Gradients hold geometry fixed; no automatic mesher derivative is claimed.

- `fd_vanilla`: Douglas, Crank–Nicolson, implicit Euler, Craig–Sneyd,
  modified Craig–Sneyd, Hundsdorfer. Original damping behavior is retained.
- Basket, latent and packed APIs: Douglas, Craig–Sneyd, modified Craig–Sneyd,
  Hundsdorfer. Packed plans and latent pricing require zero damping.
- Dates use fixed unadjusted schedules. American exercise starts at valuation.
  Bermudan dates increase and include maturity. Heston/shout curves, process
  time and volatility use selectable Actual/365 Fixed (default) or Actual/360;
  other pricing APIs retain Actual/365 Fixed. `adjust_date`/`advance_date`
  explicitly prepare dates with original NullCalendar, WeekendsOnly or TARGET
  and following, modified following, preceding, modified preceding or
  unadjusted rules. `year_fraction` exposes the two supported day counters.
- Dated curves use original QuantLib log-linear discount interpolation. In
  active pricing their first discount must be a fixed one. A curve overrides
  its flat rate; an active overridden rate raises an error.
- Analytic cash dividends use the original discounted-spot approximation;
  FD spot dividends use the upstream jump/interpolation handler. These are
  distinct financial models. Neither is silently substituted for the other.
- Packed cash jumps use passive maps extracted from the original
  `FdmDividendHandler`. Dividends precede exercise at coincident events. The
  original scheduler supplies exact event callbacks and the maturity projection.
  Exercise keeps continuation at equality, matching upstream strict comparison.
- Numerical meshes, dates, exercise rules, strikes outside the analytic API,
  markets in `fd_vanilla`/`fd_basket` and fixed dividend amounts are not
  differentiable. Full AD follows selected branches; exercise boundaries and
  interpolation changes can be nonsmooth. Adjoint modes are first order only.

## Batch compatibility and device behavior

Within `LatentBatch`, contracts share mesh, dates, model fields, initial state,
curves, scheme, steps and event schedule; strikes and call/put types may differ.
`from_contracts` rejects incompatible overrides. Across `MultiAssetBatch` assets,
different grid shapes, coefficient times or event schedules form separate
groups. Compatible assets can have different coordinates, curves, strikes and
cash amounts; contract counts must agree. Output order matches input order.

`fd_latent` additionally accepts graph-connected rate/yield and dated discount
factors. `LatentBatch.price(..., rate=..., discount_curve=...)` accepts a fresh
market graph per call. `MultiAssetBatch.price(..., markets=[{...}, ...])` accepts
one override dictionary per asset. Missing entries use passive plan defaults;
`discount_curve=None` clears a configured curve. Active curves and flat rates
use original QuantLib interpolation and interval forwards on CPU float64;
connected transfers move these forwards to CUDA. Checkpointed backward returns
their VJPs to the input curve nodes. Plans never cache active market graphs.
These financial inputs support higher derivatives in full mode and first-order
VJPs in adjoint mode, alongside the model/initial-state inputs in the table.

One or two latent states support early exercise. Cash dividends currently
require one latent state; the upstream handler's higher-dimensional layout is
not generalized here. Fixed dividend maps use quadratic storage in stock-axis
length. Their construction is setup work; repeated prices reuse them.

CUDA preserves the same events, boundaries and schemes and requires strictly
row-diagonally-dominant directional forward systems and a checked scaled solve
residual. Unsupported equations raise. There is no automatic device fallback.
Full AD retains its graph; checkpointing is exposed only for first-order
`MultiAssetBatch` adjoints. Checkpoint replay includes the financial events.

## Remaining QuantLib coverage gaps

Interest-rate/credit/FX-specific instruments, bonds, swaps, callable products,
trees, Monte Carlo engines, quanto adjustments, further day counters and
calendars, arbitrary boundary callbacks, broader mesh/solver controls, and most
QuantLib model classes remain unexposed. No differentiability claim applies to
an engine merely because its sources compile. Fixed-mesh rate/curve
sensitivities in `fd_vanilla`/`fd_basket` and differentiable batch dividend
amounts remain next work. Heston Monte Carlo evolution, transition densities
and variance sampling explicitly fail in this PDE dependency build; they are
not exposed as Python model methods. Heston/shout CUDA and batching, Heston
adjoint damping, and Heston quanto/leverage functions remain unexposed.

Existing CUDA/CPU benchmark evidence is retained in the
[historical architecture](history/architecture.md). Current implementation
invariants are in [architecture.md](architecture.md). The first candidate uses
an exact Windows/Python/PyTorch matrix; Linux binary portability and additional
variants are not claimed. See [installation](install.md) and
[release validation](release.md).
