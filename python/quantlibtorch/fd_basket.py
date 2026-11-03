"""Differentiable two-asset pricing through QuantLib's actual 2D basket engine."""
from __future__ import annotations

from datetime import date
import math

import torch

from .fd import _axis, _curve, _integer, _serial
from .pricing import _finite_number


def _surface(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise ValueError(f"{name} must use CPU float64")
    if value.layout != torch.strided or value.ndim not in (0, 2):
        raise ValueError(f"{name} must be a dense scalar or two-dimensional tensor")
    result = value.reshape(1, 1) if value.ndim == 0 else value
    if 0 in result.shape:
        raise ValueError(f"{name} dimensions must be nonempty")
    if not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
        raise ValueError(f"{name} must contain finite, strictly positive values")
    return result


def fd_basket(
    local_vol1: torch.Tensor,
    local_vol2: torch.Tensor,
    correlation: torch.Tensor,
    *,
    times1=None,
    log_spots1=None,
    times2=None,
    log_spots2=None,
    spot1: float = 100.0,
    spot2: float = 95.0,
    strike: float = 100.0,
    valuation_date: date = date(2025, 1, 1),
    maturity_date: date = date(2026, 1, 1),
    rate: float = 0.03,
    dividend_yield1: float = 0.01,
    dividend_yield2: float = 0.015,
    discount_curve=None,
    dividend_curve1=None,
    dividend_curve2=None,
    basket: str = "min",
    exercise: str = "european",
    exercise_dates=(),
    option_type: str = "call",
    time_steps: int = 30,
    grid_points=(31, 25),
    damping_steps: int = 0,
    scheme: str = "hundsdorfer",
    mesh_volatility1: float = 0.2,
    mesh_volatility2: float = 0.25,
    mode: str = "full",
    kernels: str = "auto",
) -> torch.Tensor:
    """Return a scalar NPV from QuantLib's Fd2dBlackScholesVanillaEngine.

    The payoff is a vanilla call/put on the minimum or maximum of two assets.
    Each local volatility is a positive CPU float64 scalar or a two-dimensional
    time/log-spot surface for its own asset. Coordinates are fixed Python
    sequences, with the interpolation/extrapolation conventions of fd_vanilla.
    Correlation is a CPU float64 scalar tensor strictly between -1 and 1.
    These three tensor inputs retain their original coefficient dependencies.

    Both assets share the risk-free discount curve. Each dividend curve uses
    dated discount factors and log-linear interpolation with Actual/365 Fixed;
    otherwise the corresponding flat rate is used. Financial inputs, dates,
    sampling axes, and independent Black mesh volatilities are fixed values.
    The mesh volatilities control only QuantLib's two automatic meshers.

    Douglas, Craig-Sneyd, modified Craig-Sneyd, and Hundsdorfer use QuantLib's
    actual multidimensional stages, mixed operator and directional solves.
    Full mode differentiates their arithmetic; adjoint mode composes implicit
    first-order directional solve VJPs with the remaining operator/event graph.
    Higher derivatives require full mode and stable nonsmooth branches.

    grid_points is (x_nodes, y_nodes). Damping must be zero: multidimensional
    implicit Euler/Krylov equation adjoints are not supported by this API.
    Cash dividends and custom boundary callbacks are not exposed by this
    engine adapter. Exercise and bicubic interpolation follow QuantLib, with
    piecewise sensitivities at exercise/limiter switches. This API is CPU-only,
    unbatched, and does not promise torch.compile or function-transform support.
    """
    surface1 = _surface("local_vol1", local_vol1)
    surface2 = _surface("local_vol2", local_vol2)
    if not isinstance(correlation, torch.Tensor):
        raise TypeError("correlation must be a torch.Tensor")
    if (correlation.device.type != "cpu" or correlation.dtype != torch.float64
            or correlation.layout != torch.strided or correlation.ndim != 0):
        raise ValueError("correlation must be a CPU float64 scalar tensor")
    if not bool(torch.isfinite(correlation)) or not bool(correlation.abs() < 1):
        raise ValueError("correlation must be finite and strictly between -1 and 1")
    market = {name: _finite_number(name, value) for name, value in (
        ("spot1", spot1), ("spot2", spot2), ("strike", strike), ("rate", rate),
        ("dividend_yield1", dividend_yield1), ("dividend_yield2", dividend_yield2),
        ("mesh_volatility1", mesh_volatility1), ("mesh_volatility2", mesh_volatility2))}
    if any(market[key] <= 0 for key in (
            "spot1", "spot2", "strike", "mesh_volatility1", "mesh_volatility2")):
        raise ValueError("spots, strike and mesh volatilities must be strictly positive")
    t1 = _axis("times1", times1, surface1.shape[0], 0.0)
    x1 = _axis("log_spots1", log_spots1, surface1.shape[1], math.log(market["spot1"]))
    t2 = _axis("times2", times2, surface2.shape[0], 0.0)
    x2 = _axis("log_spots2", log_spots2, surface2.shape[1], math.log(market["spot2"]))
    if t1[0] < 0 or t2[0] < 0:
        raise ValueError("times must be nonnegative")
    reference = _serial("valuation_date", valuation_date)
    maturity = _serial("maturity_date", maturity_date)
    if maturity <= reference:
        raise ValueError("maturity_date must follow valuation_date")
    if isinstance(grid_points, torch.Tensor):
        raise TypeError("grid_points must be a fixed pair of integer node counts")
    try:
        nx, ny = grid_points
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_points must contain exactly two node counts") from exc
    choices = {
        "basket": ({"min": 0, "max": 1}, basket),
        "exercise": ({"european": 0, "american": 1, "bermudan": 2}, exercise),
        "option_type": ({"call": 1, "put": -1}, option_type),
        "scheme": ({"douglas": 0, "craig_sneyd": 1, "modified_craig_sneyd": 2,
                    "hundsdorfer": 3}, scheme),
        "adjoint": ({"full": 0, "adjoint": 1}, mode),
        "kernels": ({"scalar": 0, "auto": 4}, kernels),
    }
    config = {}
    for name, (mapping, value) in choices.items():
        if value not in mapping:
            raise ValueError(f"{'mode' if name == 'adjoint' else name} must be one of {tuple(mapping)}")
        config[name] = mapping[value]
    damping = _integer("damping_steps", damping_steps, 0)
    if damping != 0:
        raise ValueError("damping_steps must be zero; multidimensional implicit Euler/Krylov adjoints are unsupported")
    config.update(time_steps=_integer("time_steps", time_steps, 1),
                  x_grid=_integer("x grid_points", nx, 5), y_grid=_integer("y grid_points", ny, 5),
                  damping_steps=damping, valuation_date=reference, maturity_date=maturity)
    risk_dates, risk_discounts = _curve("discount_curve", discount_curve, reference, maturity)
    yield_dates1, yield_discounts1 = _curve("dividend_curve1", dividend_curve1, reference, maturity)
    yield_dates2, yield_discounts2 = _curve("dividend_curve2", dividend_curve2, reference, maturity)
    ex_dates = [_serial("exercise_dates", d) for d in exercise_dates]
    if exercise == "bermudan":
        if not ex_dates or ex_dates[0] <= reference or ex_dates[-1] != maturity:
            raise ValueError("Bermudan exercise_dates must follow valuation and end at maturity")
        if any(b <= a for a, b in zip(ex_dates, ex_dates[1:])):
            raise ValueError("exercise_dates must be strictly increasing")
    elif ex_dates:
        raise ValueError("exercise_dates apply only to Bermudan exercise")
    return torch.ops.quantlibtorch.fd_basket(
        surface1, surface2, correlation, t1, x1, t2, x2, market, config,
        risk_dates, risk_discounts, yield_dates1, yield_discounts1,
        yield_dates2, yield_discounts2, ex_dates,
    )
