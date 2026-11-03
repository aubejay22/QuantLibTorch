"""Differentiable QuantLib FdBlackScholesVanillaEngine pricing.

Financial conventions, mesh construction, payoff averaging, rollback, exercise,
cash dividend jumps and final interpolation are executed by QuantLib itself.
Only the sampled local-volatility surface is a differentiable API input.
"""
from __future__ import annotations

from datetime import date
import math
import numbers

import torch

from .pricing import _finite_number
from . import solver as _solver  # loads the shared native dispatcher library


def _serial(name, value):
    if type(value) is not date:
        raise TypeError(f"{name} must be a datetime.date")
    if not date(1901, 1, 1) <= value <= date(2199, 12, 31):
        raise ValueError(f"{name} is outside QuantLib's date range")
    return (value - date(1899, 12, 31)).days + 1


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _axis(name, values, count, default):
    if values is None:
        if count != 1:
            raise ValueError(f"{name} is required for a nonconstant surface dimension")
        return [default]
    if isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a fixed sequence of numbers, not a tensor")
    axis = [_finite_number(name, v) for v in values]
    if len(axis) != count:
        raise ValueError(f"{name} must match its surface dimension")
    if any(b <= a for a, b in zip(axis, axis[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _curve(name, values, reference, maturity):
    if values is None:
        return [], []
    pairs = list(values)
    if len(pairs) < 2:
        raise ValueError(f"{name} requires at least two dated discount factors")
    dates = [_serial(name, d) for d, _ in pairs]
    discounts = [_finite_number(name, v) for _, v in pairs]
    if dates[0] != reference or dates[-1] < maturity:
        raise ValueError(f"{name} must start at valuation_date and cover maturity_date")
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError(f"{name} dates must be strictly increasing")
    if discounts[0] != 1.0 or any(v <= 0 for v in discounts):
        raise ValueError(f"{name} discount factors must be positive and start at one")
    return dates, discounts


def fd_vanilla(
    local_vol: torch.Tensor,
    *,
    times=None,
    log_spots=None,
    spot: float = 100.0,
    strike: float = 100.0,
    valuation_date: date = date(2025, 1, 1),
    maturity_date: date = date(2026, 1, 1),
    rate: float = 0.03,
    dividend_yield: float = 0.01,
    discount_curve=None,
    dividend_curve=None,
    cash_dividends=(),
    cash_dividend_model: str = "spot",
    exercise: str = "european",
    exercise_dates=(),
    option_type: str = "call",
    time_steps: int = 50,
    grid_points: int = 100,
    damping_steps: int = 2,
    scheme: str = "douglas",
    mesh_volatility: float = 0.2,
    mode: str = "full",
    kernels: str = "auto",
) -> torch.Tensor:
    """Return a scalar NPV from QuantLib's actual finite-difference engine.

    ``local_vol`` is a positive CPU float64 scalar or a tensor with shape
    ``(len(times), len(log_spots))``. Coordinates are fixed Python sequences:
    times are Actual/365 Fixed years after valuation and log_spots are natural
    logarithms of absolute spot levels. Samples are bilinearly interpolated in
    time/log-spot, with flat extrapolation. Singleton axes are constant.

    ``discount_curve`` and ``dividend_curve`` optionally contain ``(date, df)``
    pairs starting at ``(valuation_date, 1)`` and covering maturity. QuantLib
    applies log-linear discount interpolation with Actual/365 Fixed. Otherwise
    ``rate`` and ``dividend_yield`` specify continuously compounded flat curves.
    ``cash_dividends`` contains ``(date, amount)`` pairs for the spot dividend
    model by default; ``cash_dividend_model='escrowed'`` selects QuantLib's
    escrowed-dividend adjustment. Dates must increase, follow valuation, and
    not exceed maturity.

    ``exercise`` is european, american (from valuation), or bermudan. Bermudan
    ``exercise_dates`` must increase and end at maturity. ``option_type`` is
    call or put. Schemes are douglas, crank_nicolson, implicit_euler,
    craig_sneyd, modified_craig_sneyd and hundsdorfer.
    All financial inputs, dates, axes, and mesh settings are fixed, not tensors.

    ``mesh_volatility`` controls QuantLib's automatic Black-Scholes mesher only;
    it does not replace the supplied local volatility. Keep it fixed for
    coefficient sensitivities. Check mesh/domain convergence for your surface.

    ``full`` differentiates the selected numerical arithmetic. ``adjoint`` uses
    the same forward engine, with first-order VJPs for supported time steps,
    local-volatility assembly, band application, and solves. Native CPU
    implementations inside these boundaries preserve the pinned QuantLib
    arithmetic and pivot checks; their backward returns every coefficient
    dependency. Supported one-dimensional steps avoid passive operator copies.
    Unsupported step/operator
    configurations retain the original path and narrower solve VJP.
    Jumps, exercise and interpolation remain connected to autograd.
    Exercise and spline limiter branches give piecewise derivatives. Adjoint
    mode rejects create_graph=True; it is not a full-trajectory adjoint.
    ``kernels='auto'`` performs coefficient sampling, array arithmetic, spline
    expressions, band application and assembly on whole vectors. In full AD,
    sufficiently large, well-scaled, strictly diagonally dominant systems use
    fused CPU Thomas arithmetic, with a residual check. Its elementary reverse
    recurrence and graph-connected intermediate outputs support higher
    derivatives. Forward duals and unsupported native cases retain tensor
    parallel cyclic reduction (PCR); systems outside the guarded domain retain
    the upstream solve. These paths change floating-point evaluation order,
    preserve the discretization, and support higher derivatives.
    ``vectorized`` retains Thomas for every full-AD solve. ``pcr`` explicitly
    selects checked PCR for splitting solves and requires full mode; it has a
    stricter pivot requirement than Thomas and can reject matrices Thomas solves.
    Unchanged array runs keep their graph-connected tensor backing. ``scalar``
    retains upstream sampling/array arithmetic; ``apply`` vectorizes application
    only for controlled ablations. Adjoint mode preserves the upstream scaled
    solve recurrence and uses compact transpose solves for first-order VJPs.
    Spatial samples are reused within one price without detaching their graph.
    Under no_grad, the native adjoint boundaries also accelerate full-mode
    inference unless a forward-mode tangent is present.
    This API is CPU-only, unbatched, and does not promise torch.compile support.
    """
    if not isinstance(local_vol, torch.Tensor):
        raise TypeError("local_vol must be a torch.Tensor")
    if local_vol.device.type != "cpu" or local_vol.dtype != torch.float64:
        raise ValueError("local_vol must use CPU float64")
    if local_vol.layout != torch.strided or local_vol.ndim not in (0, 2):
        raise ValueError("local_vol must be a dense scalar or two-dimensional tensor")
    surface = local_vol.reshape(1, 1) if local_vol.ndim == 0 else local_vol
    if 0 in surface.shape:
        raise ValueError("local_vol surface dimensions must be nonempty")
    if not bool(torch.isfinite(surface).all()) or not bool((surface > 0).all()):
        raise ValueError("local_vol must contain finite, strictly positive values")
    market = {name: _finite_number(name, value) for name, value in (
        ("spot", spot), ("strike", strike), ("rate", rate),
        ("dividend_yield", dividend_yield), ("mesh_volatility", mesh_volatility))}
    if any(market[key] <= 0 for key in ("spot", "strike", "mesh_volatility")):
        raise ValueError("spot, strike, and mesh_volatility must be strictly positive")
    ts = _axis("times", times, surface.shape[0], 0.0)
    xs = _axis("log_spots", log_spots, surface.shape[1], math.log(market["spot"]))
    if ts[0] < 0:
        raise ValueError("times must be nonnegative")
    reference = _serial("valuation_date", valuation_date)
    maturity = _serial("maturity_date", maturity_date)
    if maturity <= reference:
        raise ValueError("maturity_date must follow valuation_date")
    choices = {
        "exercise": ({"european": 0, "american": 1, "bermudan": 2}, exercise),
        "option_type": ({"call": 1, "put": -1}, option_type),
        "scheme": ({"douglas": 0, "crank_nicolson": 1, "implicit_euler": 2,
                    "craig_sneyd": 3, "modified_craig_sneyd": 4, "hundsdorfer": 5}, scheme),
        "cash_dividend_model": ({"spot": 0, "escrowed": 1}, cash_dividend_model),
        "adjoint": ({"full": 0, "adjoint": 1}, mode),
        "kernels": ({"scalar": 0, "apply": 1, "vectorized": 2, "pcr": 3, "auto": 4}, kernels),
    }
    config = {}
    for key, (mapping, value) in choices.items():
        if value not in mapping:
            raise ValueError(f"{'mode' if key == 'adjoint' else key} must be one of {tuple(mapping)}")
        config[key] = mapping[value]
    if kernels == "pcr" and mode != "full":
        raise ValueError("PCR pricing supports mode='full' only")
    config.update(
        time_steps=_integer("time_steps", time_steps, 1),
        grid_points=_integer("grid_points", grid_points, 5),
        damping_steps=_integer("damping_steps", damping_steps, 0),
        valuation_date=reference, maturity_date=maturity,
    )
    risk_dates, risk_discounts = _curve("discount_curve", discount_curve, reference, maturity)
    yield_dates, yield_discounts = _curve("dividend_curve", dividend_curve, reference, maturity)
    ex_dates = [_serial("exercise_dates", d) for d in exercise_dates]
    if exercise == "bermudan":
        if not ex_dates or ex_dates[0] <= reference or ex_dates[-1] != maturity:
            raise ValueError("Bermudan exercise_dates must follow valuation and end at maturity")
        if any(b <= a for a, b in zip(ex_dates, ex_dates[1:])):
            raise ValueError("exercise_dates must be strictly increasing")
    elif ex_dates:
        raise ValueError("exercise_dates apply only to Bermudan exercise")
    cash_pairs = list(cash_dividends)
    cash_dates = [_serial("cash_dividends", d) for d, _ in cash_pairs]
    cash_amounts = [_finite_number("cash_dividends", a) for _, a in cash_pairs]
    if any(d <= reference or d > maturity for d in cash_dates):
        raise ValueError("cash dividend dates must follow valuation and not exceed maturity")
    if any(b <= a for a, b in zip(cash_dates, cash_dates[1:])):
        raise ValueError("cash dividend dates must be strictly increasing")
    if any(a < 0 for a in cash_amounts) or sum(cash_amounts) >= market["spot"]:
        raise ValueError("cash dividends must be nonnegative and total less than spot")
    return torch.ops.quantlibtorch.fd_vanilla(
        surface, ts, xs, market, config, risk_dates, risk_discounts,
        yield_dates, yield_discounts, cash_dates, cash_amounts, ex_dates,
    )
