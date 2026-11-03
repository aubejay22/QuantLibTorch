"""Joint-state diffusion fields on QuantLib tensor-product meshes."""
from __future__ import annotations

from datetime import date
import math

import torch

from .fd import _axis, _integer, _serial
from .pricing import _finite_number
from .financial import _market


_SCHEMES = {"douglas": 0, "hundsdorfer": 1, "modified_craig_sneyd": 2, "craig_sneyd": 3}


def _tensor(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64 or value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense CPU float64 tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _axes(axes, minimum=2):
    if isinstance(axes, torch.Tensor):
        raise TypeError("axes must be fixed sequences, not tensors")
    result = []
    for axis in axes:
        if isinstance(axis, torch.Tensor):
            raise TypeError("axes must be fixed sequences, not tensors")
        values = list(axis)
        if len(values) < minimum:
            raise ValueError(f"each axis requires at least {minimum} nodes")
        result.append(_axis("axis", values, len(values), 0.0))
    if not result:
        raise ValueError("axes must not be empty")
    return result


def state_grid(axes) -> torch.Tensor:
    """Return fixed coordinates [N,d], with coordinate zero varying fastest.

    axes contains a sequence for each physical coordinate. A flat state can be
    reshaped to tuple(len(axis) for axis in reversed(axes)). In equity pricing,
    axis zero is log(spot); remaining coordinates are latent states.
    """
    axes = _axes(axes)
    grids = torch.meshgrid(*(torch.tensor(a, dtype=torch.float64) for a in reversed(axes)), indexing="ij")
    return torch.stack(tuple(reversed(grids)), dim=-1).reshape(-1, len(axes))


def _fields(drift, diffusion, times, axes, horizon, latent):
    _tensor("drift", drift)
    _tensor("diffusion", diffusion)
    n, d = math.prod(map(len, axes)), len(axes)
    if drift.ndim != 3 or drift.shape[1:] != (n, d-int(latent)) or drift.shape[0] < 1:
        raise ValueError("drift must have shape [time, nodes, latent directions]" if latent else
                         "drift must have shape [time, nodes, directions]")
    nt = drift.shape[0]
    if diffusion.ndim != 4 or diffusion.shape[:3] != (nt, n, d) or diffusion.shape[3] < 1:
        raise ValueError("diffusion must have shape [time, nodes, directions, noise factors]")
    times = _axis("times", times, nt, 0.0)
    if times[0] < 0 or (nt > 1 and (times[0] != 0 or times[-1] < horizon)):
        raise ValueError("times must be nonnegative and cover [0, horizon] for time-varying fields")
    return times


def _config(time_steps, scheme, mode, kernels):
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme must be one of {tuple(_SCHEMES)}")
    if mode not in ("full", "adjoint"):
        raise ValueError("mode must be full or adjoint")
    if kernels not in ("auto", "scalar"):
        raise ValueError("kernels must be auto or scalar")
    return _integer("time_steps", time_steps, 1), _SCHEMES[scheme], mode == "adjoint", 4 if kernels == "auto" else 0


def diffusion_rollback(drift: torch.Tensor, diffusion: torch.Tensor, terminal: torch.Tensor,
                       *, axes, times=None, killing=None, horizon=1.0, time_steps=30,
                       scheme="hundsdorfer", mode="full", kernels="auto") -> torch.Tensor:
    """Solve the backward diffusion PDE with QuantLib's original ADI schemes.

    drift is [T,N,d], diffusion is a noise loading L of shape [T,N,d,w],
    terminal is [N], and optional killing is [T,N]. N is the product of the
    axis lengths; axis zero varies fastest (see state_grid). All tensor inputs
    are CPU float64 and retain autograd connections, including strided views.
    The generator is b_i D_i + .5 C_ii D_ii + sum(i<j) C_ij D_ij - killing,
    where C=L L^T. Rectangular and rank-deficient factors are supported.

    One time row means constant in time. Otherwise times must cover [0,horizon].
    At each QuantLib setTime interval midpoint, drift, covariance and killing
    are linearly interpolated. This interpolates C, NOT L; covariance remains
    PSD between samples. Coefficients are supplied at every mesh node, so no
    spatial interpolation or Python callback occurs inside the rollback.

    Uses QuantLib's one-sided first derivatives, zero pure second derivatives
    at domain edges, and original mixed stencils including corners. There are
    no prescribed boundary values or damping. Domain adequacy and ADI stability
    need problem-specific refinement checks; PSD alone does not ensure them.
    Full mode supports reverse AD and higher derivatives; adjoint mode uses
    first-order equation VJPs. Fixed axes/times are not differentiable. This is
    unbatched CPU execution, without a torch.func/compile contract.
    """
    axes = _axes(axes)
    horizon = _finite_number("horizon", horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    times = _fields(drift, diffusion, times, axes, horizon, False)
    n = math.prod(map(len, axes))
    if _tensor("terminal", terminal).shape != (n,):
        raise ValueError("terminal must have shape [nodes]")
    if killing is None:
        killing = drift.new_zeros(drift.shape[:2])
    if _tensor("killing", killing).shape != drift.shape[:2]:
        raise ValueError("killing must have shape [time, nodes]")
    return torch.ops.quantlibtorch._coefficient_rollback(
        drift, diffusion, killing, terminal, axes, times, horizon,
        *_config(time_steps, scheme, mode, kernels))


def fd_latent(latent_drift: torch.Tensor, diffusion: torch.Tensor, initial_state: torch.Tensor,
              *, axes, times=None, strike=100.0, valuation_date=date(2025, 1, 1),
              maturity_date=date(2026, 1, 1), rate=0.03, dividend_yield=0.01,
              discount_curve=None, dividend_curve=None, exercise="european",
              exercise_dates=(), option_type="call", time_steps=30,
              scheme="hundsdorfer", mode="full", kernels="auto") -> torch.Tensor:
    """Price a vanilla option with learned latent dynamics using QuantLib.

    Coordinates are Z=(log S,Y_1,...,Y_m). latent_drift is [T,N,m] and
    diffusion is [T,N,1+m,w]. The first diffusion row contains relative stock
    noise loadings, remaining rows absolute latent loadings. C=L L^T ensures
    PSD, including all stock/latent and latent/latent covariances. The stock
    drift is imposed: b_logS = r-q-.5*C_00; discounting is r. Rates are interval
    forward rates from QuantLib's fixed-reference Actual/365 Fixed curves.
    Dates and optional dated log-linear discount curves follow fd_vanilla.

    initial_state is a differentiable [1+m] tensor containing log(spot) and
    latent starting values, strictly inside the fixed axes. One or two latent states
    are supported by the existing Fdm2DimSolver / FdmNdimSolver<3> interpolation
    adapters; diffusion_rollback itself is not restricted to these dimensions.
    These solvers also supply payoff averaging, exercise and theta snapshots.

    The stock-edge drift follows QuantLib Heston's linear-in-spot closure,
    removing the Ito variance correction at the two stock edges. Other edges
    use the stencils documented by diffusion_rollback; no general latent-domain
    boundary classification is inferred. Cash dividends, damping and custom
    boundary callbacks are not exposed. Exercise and interpolation derivatives
    are piecewise. Rates, yields and noninitial dated discount factors may be
    CPU float64 scalar tensors; original QuantLib curve interpolation retains
    their graphs in both modes. Strike, curve dates, axes and times are fixed.

    Time interpolation, differentiation and execution contracts are those of
    diffusion_rollback. Arbitrary arrays guarantee neither a well-posed SDE nor
    a true discounted stock martingale. Bounded, sufficiently regular neural
    coefficients provide a useful restricted model class; validate the model
    and numerical domain separately.
    """
    axes = _axes(axes, minimum=4)
    if len(axes) not in (2, 3):
        raise ValueError("fd_latent supports one or two latent states")
    reference, maturity = _serial("valuation_date", valuation_date), _serial("maturity_date", maturity_date)
    if maturity <= reference:
        raise ValueError("maturity_date must follow valuation_date")
    horizon = (maturity-reference)/365.0
    times = _fields(latent_drift, diffusion, times, axes, horizon, True)
    if _tensor("initial_state", initial_state).shape != (len(axes),):
        raise ValueError("initial_state must have shape [directions]")
    if any(not a[0] < float(x) < a[-1] for a, x in zip(axes, initial_state.detach())):
        raise ValueError("initial_state must lie strictly inside all fixed axes")
    strike = _finite_number("strike", strike)
    if strike <= 0:
        raise ValueError("strike must be positive")
    market = _market(rate, dividend_yield, discount_curve, dividend_curve, reference, maturity)
    if exercise not in ("european", "american", "bermudan") or option_type not in ("call", "put"):
        raise ValueError("invalid exercise or option_type")
    ex_dates = [_serial("exercise_dates", d) for d in exercise_dates]
    if exercise == "bermudan":
        if (not ex_dates or ex_dates[0] <= reference or ex_dates[-1] != maturity
                or any(b <= a for a, b in zip(ex_dates, ex_dates[1:]))):
            raise ValueError("Bermudan dates must increase, follow valuation and end at maturity")
    elif ex_dates:
        raise ValueError("exercise_dates require Bermudan exercise")
    return torch.ops.quantlibtorch._latent_price(
        latent_drift, diffusion, initial_state, axes, times, strike, reference, maturity,
        *market,
        {"european": 0, "american": 1, "bermudan": 2}[exercise], ex_dates,
        1 if option_type == "call" else -1, *_config(time_steps, scheme, mode, kernels))
