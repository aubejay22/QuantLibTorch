"""Original QuantLib Black-Scholes shout pricing with fixed numerical geometry."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import torch

from .analytic import _scalar
from .financial import _market
from .fd import _axis, _curve
from .heston import _settings
from .pricing import _finite_number


_SCHEMES = {"douglas": 0, "crank_nicolson": 1, "implicit_euler": 2,
            "craig_sneyd": 3, "modified_craig_sneyd": 4, "hundsdorfer": 5}


@dataclass(frozen=True)
class ShoutMesh:
    """Explicit fixed log-spot axis for the original shout engine.

    Normally prepared by :func:`shout_mesh` using the original equity mesher.
    With cash dividends the coordinate represents escrow-adjusted spot.
    """

    log_spots: Sequence[float]

    def __post_init__(self) -> None:
        if isinstance(self.log_spots, torch.Tensor):
            raise TypeError("log_spots must be fixed numbers, not a tensor")
        xs = tuple(self.log_spots)
        if len(xs) < 5:
            raise ValueError("log_spots requires at least five nodes")
        object.__setattr__(self, "log_spots", tuple(_axis("log_spots", xs, len(xs), 0.)))


def shout_mesh(
    volatility: float = .2, *, spot: float = 100., strike: float = 100.,
    rate: float = .03, dividend_yield: float = .01,
    valuation_date: date = date(2025, 1, 1), maturity_date: date = date(2026, 1, 1),
    discount_curve=None, dividend_curve=None, cash_dividends=(),
    option_type: str = "call", exercise: str = "american", exercise_dates=(),
    time_steps: int = 50, x_grid: int = 51, damping_steps: int = 0,
    scheme: str = "douglas", day_counter: str = "actual365fixed",
) -> ShoutMesh:
    """Prepare QuantLib's original shout equity mesher from passive numbers.

    The original escrowed-dividend adjustment and mesh conventions are used.
    Tensor inputs are rejected. Reuse this mesh while changing active financial
    parameters so sensitivities exclude automatic geometry changes.
    """
    values = [_finite_number(n, v) for n, v in zip(
        ("spot", "volatility", "rate", "dividend_yield"),
        (spot, volatility, rate, dividend_yield))]
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError("spot and volatility must be positive")
    config, strike, cd, ca, ed = _settings(
        valuation_date=valuation_date, maturity_date=maturity_date, option_type=option_type,
        exercise=exercise, exercise_dates=exercise_dates, cash_dividends=cash_dividends,
        strike=strike, spot=spot, time_steps=time_steps, x_grid=x_grid, v_grid=5,
        damping_steps=damping_steps, scheme=scheme, mode="full", kernels="auto",
        scheme_choices=_SCHEMES, day_counter=day_counter)
    ref, end = config["valuation_date"], config["maturity_date"]
    rd, rv = _curve("discount_curve", discount_curve, ref, end)
    qd, qv = _curve("dividend_curve", dividend_curve, ref, end)
    xs = torch.ops.quantlibtorch.shout_mesh(values, strike, config, rd, rv, qd, qv, cd, ca, ed)
    return ShoutMesh(xs.tolist())


def fd_shout(
    volatility: torch.Tensor, *, mesh: ShoutMesh,
    spot: float | torch.Tensor = 100., strike: float = 100.,
    rate: float | torch.Tensor = .03, dividend_yield: float | torch.Tensor = .01,
    valuation_date: date = date(2025, 1, 1), maturity_date: date = date(2026, 1, 1),
    discount_curve=None, dividend_curve=None, cash_dividends=(),
    option_type: str = "call", exercise: str = "american", exercise_dates=(),
    time_steps: int = 50, damping_steps: int = 0, scheme: str = "douglas",
    mode: str = "full", kernels: str = "auto", day_counter: str = "actual365fixed",
) -> torch.Tensor:
    """Price a one-shout option with QuantLib FdBlackScholesShoutEngine.

    Exercise dates represent opportunities to lock intrinsic value while
    retaining the original shout payoff at maturity. American permits shouting
    throughout the interval; Bermudan restricts it to the supplied dates;
    European permits only maturity and reduces to the vanilla payoff.

    Constant Black volatility, spot, flat rate/yield and dated discount factors
    retain their graphs on CPU float64. Full AD supports higher derivatives on
    selected branches; equation adjoints are first order. Strike, dates, cash
    amounts and the explicit mesh are fixed. Cash dividends use the original
    engine's escrowed model, including its time-dependent shout value. The
    actual365fixed or actual360 day counter applies to curves and volatility. Six
    original schemes and damping are supported; CUDA/batching are not exposed.
    """
    if not isinstance(volatility, torch.Tensor):
        raise TypeError("volatility must be a torch.Tensor")
    if not isinstance(mesh, ShoutMesh):
        raise TypeError("mesh must be a ShoutMesh; prepare it with shout_mesh")
    values = [_scalar(n, v) for n, v in zip(
        ("spot", "volatility", "rate", "dividend_yield"),
        (spot, volatility, rate, dividend_yield))]
    if values[0].item() <= 0 or values[1].item() <= 0:
        raise ValueError("spot and volatility must be positive")
    config, strike, cd, ca, ed = _settings(
        valuation_date=valuation_date, maturity_date=maturity_date, option_type=option_type,
        exercise=exercise, exercise_dates=exercise_dates, cash_dividends=cash_dividends,
        strike=strike, spot=values[0].item(), time_steps=time_steps,
        x_grid=len(mesh.log_spots), v_grid=5, damping_steps=damping_steps,
        scheme=scheme, mode=mode, kernels=kernels, scheme_choices=_SCHEMES,
        day_counter=day_counter)
    ref, end = config["valuation_date"], config["maturity_date"]
    _, _, rd, rv, qd, qv = _market(values[2], values[3], discount_curve, dividend_curve, ref, end)
    return torch.ops.quantlibtorch.fd_shout(torch.stack(values), strike, config,
                                           rd, rv, qd, qv, cd, ca, ed, list(mesh.log_spots))
