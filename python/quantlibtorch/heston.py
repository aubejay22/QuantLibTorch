"""Original QuantLib Heston engine on explicitly prepared, fixed geometry."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Sequence

import torch
from torch import nn

from .analytic import _scalar
from .financial import _market
from .conventions import _choice, _DAY_COUNTERS
from .fd import _axis, _curve, _integer, _serial
from .pricing import _finite_number


_SCHEMES = {"douglas": 0, "craig_sneyd": 1, "modified_craig_sneyd": 2,
            "hundsdorfer": 3}


@dataclass(frozen=True)
class HestonMesh:
    """Fixed log-spot and variance axes, normally made by :func:`heston_mesh`.

    Custom axes use the same original Heston operator and boundary rows.
    Variance starts at zero; meshes and interpolation interval choices are not
    differentiable. Keep the same object when checking parameter differences.
    Domain and resolution adequacy remain the caller's responsibility.
    """

    log_spots: Sequence[float]
    variances: Sequence[float]

    def __post_init__(self) -> None:
        for name in ("log_spots", "variances"):
            axis = getattr(self, name)
            if isinstance(axis, torch.Tensor):
                raise TypeError(f"{name} must be fixed numbers, not a tensor")
            axis = tuple(axis)
            if len(axis) < 5:
                raise ValueError(f"{name} requires at least five nodes")
            object.__setattr__(self, name, tuple(_axis(name, axis, len(axis), 0.)))
        if self.variances[0] != 0.:
            raise ValueError("Heston variance mesh must start at zero")


def _settings(*, valuation_date, maturity_date, option_type, exercise,
              exercise_dates, cash_dividends, strike, spot, time_steps, x_grid,
              v_grid, damping_steps, scheme, mode, kernels, scheme_choices=None,
              day_counter="actual365fixed"):
    reference = _serial("valuation_date", valuation_date)
    maturity = _serial("maturity_date", maturity_date)
    if maturity <= reference:
        raise ValueError("maturity_date must follow valuation_date")
    config = dict(valuation_date=reference, maturity_date=maturity)
    config["day_counter"] = _choice("day_counter", day_counter, _DAY_COUNTERS)
    for key, value, choices in (
        ("option_type", option_type, {"call": 1, "put": -1}),
        ("exercise", exercise, {"european": 0, "american": 1, "bermudan": 2}),
        ("scheme", scheme, _SCHEMES if scheme_choices is None else scheme_choices),
        ("adjoint", mode, {"full": 0, "adjoint": 1}),
        ("kernels", kernels, {"scalar": 0, "auto": 4}),
    ):
        if value not in choices:
            raise ValueError(f"{'mode' if key == 'adjoint' else key} must be one of {tuple(choices)}")
        config[key] = choices[value]
    for name, value, minimum in (("time_steps", time_steps, 1), ("x_grid", x_grid, 5),
                                 ("v_grid", v_grid, 5), ("damping_steps", damping_steps, 0)):
        config[name] = _integer(name, value, minimum)
    strike = _finite_number("strike", strike)
    if strike <= 0:
        raise ValueError("strike must be positive")
    ex_dates = [_serial("exercise_dates", d) for d in exercise_dates]
    if exercise == "bermudan":
        if not ex_dates or ex_dates[0] <= reference or ex_dates[-1] != maturity:
            raise ValueError("Bermudan exercise_dates must follow valuation and end at maturity")
        if any(b <= a for a, b in zip(ex_dates, ex_dates[1:])):
            raise ValueError("exercise_dates must be strictly increasing")
    elif ex_dates:
        raise ValueError("exercise_dates apply only to Bermudan exercise")
    cash = list(cash_dividends)
    cash_dates = [_serial("cash_dividends", d) for d, _ in cash]
    amounts = [_finite_number("cash_dividends", a) for _, a in cash]
    if any(d <= reference or d > maturity for d in cash_dates):
        raise ValueError("cash dividend dates must follow valuation and not exceed maturity")
    if any(b <= a for a, b in zip(cash_dates, cash_dates[1:])):
        raise ValueError("cash dividend dates must be strictly increasing")
    if any(a < 0 for a in amounts):
        raise ValueError("cash dividends must be nonnegative")
    return config, strike, cash_dates, amounts, ex_dates


def _parameters(v0, kappa, theta, sigma, rho, spot, rate, dividend_yield):
    names = ("spot", "v0", "kappa", "theta", "sigma", "rho", "rate", "dividend_yield")
    values = [_scalar(n, v) for n, v in zip(names,
              (spot, v0, kappa, theta, sigma, rho, rate, dividend_yield))]
    if any(v.item() <= 0 for v in values[:5]):
        raise ValueError("spot, v0, kappa, theta and sigma must be positive")
    if not -1. < values[5].item() < 1.:
        raise ValueError("rho must lie strictly between -1 and 1")
    return values


def heston_mesh(
    v0: float = .04, kappa: float = 1.5, theta: float = .04,
    sigma: float = .3, rho: float = -.7, *, spot: float = 100.,
    strike: float = 100., rate: float = .03, dividend_yield: float = .01,
    valuation_date: date = date(2025, 1, 1), maturity_date: date = date(2026, 1, 1),
    discount_curve=None, dividend_curve=None, cash_dividends=(),
    option_type: str = "call", exercise: str = "european", exercise_dates=(),
    time_steps: int = 30, x_grid: int = 51, v_grid: int = 21,
    damping_steps: int = 0, scheme: str = "hundsdorfer", day_counter: str = "actual365fixed",
) -> HestonMesh:
    """Prepare the original QuantLib automatic Heston mesh from fixed numbers.

    This uses the original variance mesher and equity mesher, including their
    parameter and dividend dependencies. Pass numerical reference parameters
    explicitly: tensors are rejected rather than detached. Reuse the returned
    mesh for training and parameter-bump checks. Preparation does not price or
    differentiate the model.
    """
    passive = [_finite_number(n, v) for n, v in zip(
        ("v0", "kappa", "theta", "sigma", "rho", "spot", "rate", "dividend_yield"),
        (v0, kappa, theta, sigma, rho, spot, rate, dividend_yield))]
    values = _parameters(*passive)
    config, strike, cash_dates, amounts, ex_dates = _settings(
        valuation_date=valuation_date, maturity_date=maturity_date,
        option_type=option_type, exercise=exercise, exercise_dates=exercise_dates,
        cash_dividends=cash_dividends, strike=strike, spot=spot,
        time_steps=time_steps, x_grid=x_grid, v_grid=v_grid,
        damping_steps=damping_steps, scheme=scheme, mode="full", kernels="auto",
        day_counter=day_counter)
    reference, maturity = config["valuation_date"], config["maturity_date"]
    rd, rv = _curve("discount_curve", discount_curve, reference, maturity)
    qd, qv = _curve("dividend_curve", dividend_curve, reference, maturity)
    x, v = torch.ops.quantlibtorch.heston_mesh(
        [p.item() for p in values], strike, config, rd, rv, qd, qv,
        cash_dates, amounts, ex_dates)
    return HestonMesh(x.tolist(), v.tolist())


def fd_heston(
    v0: float | torch.Tensor, kappa: float | torch.Tensor,
    theta: float | torch.Tensor, sigma: float | torch.Tensor,
    rho: float | torch.Tensor, *, mesh: HestonMesh,
    spot: float | torch.Tensor = 100., strike: float = 100.,
    rate: float | torch.Tensor = .03, dividend_yield: float | torch.Tensor = .01,
    valuation_date: date = date(2025, 1, 1), maturity_date: date = date(2026, 1, 1),
    discount_curve=None, dividend_curve=None, cash_dividends=(),
    option_type: str = "call", exercise: str = "european", exercise_dates=(),
    time_steps: int = 30, damping_steps: int = 0, scheme: str = "hundsdorfer",
    mode: str = "full", kernels: str = "auto", day_counter: str = "actual365fixed",
) -> torch.Tensor:
    """Price with the original FdHestonVanillaEngine on a fixed Heston mesh.

    The five model parameters, spot, flat rate/yield and noninitial dated
    discount factors retain their PyTorch graphs. Full AD supports higher
    reverse derivatives; equation adjoints are first order and require zero
    damping. Full mode supports original implicit-Euler damping. Both use original
    QuantLib Heston boundary rows, exercise projections and spot-dividend jumps.
    ``day_counter`` selects actual365fixed or actual360 consistently for curves
    and process time. Inputs are CPU float64 scalar tensors or fixed numbers; no Feller condition
    is imposed. Geometry, dates, strike and cash amounts are fixed configuration.

    Prepare ``mesh`` separately with :func:`heston_mesh`. These sensitivities
    hold geometry fixed and do not include changes to automatic mesher choices.
    Prices match automatic-engine prices at the mesh's preparation parameters
    with identical financial data, dates and numerical settings.
    Reuse geometry for finite differences and check domain adequacy when moving
    parameters. This scalar engine does not support CUDA or contract batching.
    """
    if not isinstance(mesh, HestonMesh):
        raise TypeError("mesh must be a HestonMesh; prepare it with heston_mesh")
    values = _parameters(v0, kappa, theta, sigma, rho, spot, rate, dividend_yield)
    if not mesh.log_spots[0] < math.log(values[0].item()) < mesh.log_spots[-1]:
        raise ValueError("spot must lie inside the Heston mesh")
    if not mesh.variances[0] < values[1].item() < mesh.variances[-1]:
        raise ValueError("v0 must lie inside the Heston variance mesh")
    config, strike, cash_dates, amounts, ex_dates = _settings(
        valuation_date=valuation_date, maturity_date=maturity_date,
        option_type=option_type, exercise=exercise, exercise_dates=exercise_dates,
        cash_dividends=cash_dividends, strike=strike, spot=values[0].item(),
        time_steps=time_steps, x_grid=len(mesh.log_spots), v_grid=len(mesh.variances),
        damping_steps=damping_steps, scheme=scheme, mode=mode, kernels=kernels,
        day_counter=day_counter)
    reference, maturity = config["valuation_date"], config["maturity_date"]
    _, _, rd, rv, qd, qv = _market(values[6], values[7], discount_curve, dividend_curve,
                                  reference, maturity)
    return torch.ops.quantlibtorch.fd_heston(
        torch.stack(values), strike, config, rd, rv, qd, qv,
        cash_dates, amounts, ex_dates, list(mesh.log_spots), list(mesh.variances))


class HestonModel(nn.Module):
    """Reusable five-parameter PyTorch Heston model for ``PricingLayer``.

    Initial values are fixed numbers. With ``trainable=True`` (the default),
    values are ordinary unconstrained leaf Parameters; the pricing engine
    validates their financial domain on every call. For constrained training,
    supply your own module returning positive v0/kappa/theta/sigma and bounded
    rho, or pass transformed tensors directly to :func:`fd_heston`.
    """

    def __init__(self, v0: float = .04, kappa: float = 1.5, theta: float = .04,
                 sigma: float = .3, rho: float = -.7, *, trainable: bool = True):
        super().__init__()
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a bool")
        names = ("v0", "kappa", "theta", "sigma", "rho")
        values = [_finite_number(n, v) for n, v in zip(names, (v0, kappa, theta, sigma, rho))]
        _parameters(*values, 100., .03, .01)
        for name, value in zip(names, values):
            tensor = torch.tensor(value, dtype=torch.float64)
            if trainable:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

    def forward(self) -> tuple[torch.Tensor, ...]:
        return self.v0, self.kappa, self.theta, self.sigma, self.rho
