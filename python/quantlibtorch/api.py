"""Reusable financial configuration and PyTorch pricing modules.

Configuration objects contain fixed Python data. Supply differentiable tensors
on each call, either directly or from a registered ``torch.nn.Module``. Engines
delegate to the same public pricing functions; they do not introduce a second
numerical implementation or change the functions' derivative contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Protocol, Sequence

import torch
from torch import nn

from .fd import _axis, _curve, _integer, _serial, fd_vanilla
from .latent import _axes, fd_latent
from .heston import HestonMesh, fd_heston
from .shout import ShoutMesh, fd_shout
from .conventions import _choice, _DAY_COUNTERS
from .pricing import _finite_number

DateValues = Sequence[tuple[date, float]]


def _fixed_axis(name: str, values: Sequence[float] | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a fixed sequence, not a tensor")
    values = tuple(values)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(_axis(name, values, len(values), 0.0))


@dataclass(frozen=True)
class Exercise:
    """An exercise rule; Bermudan dates must include the instrument's maturity.

    American exercise starts at the market valuation date. Dates are discrete
    configuration and are never differentiable.
    """

    style: Literal["european", "american", "bermudan"] = "european"
    dates: tuple[date, ...] = ()

    def __post_init__(self) -> None:
        if self.style not in ("european", "american", "bermudan"):
            raise ValueError("exercise style must be european, american, or bermudan")
        dates = tuple(self.dates)
        serials = [_serial("exercise dates", d) for d in dates]
        if self.style == "bermudan":
            if not serials or any(b <= a for a, b in zip(serials, serials[1:])):
                raise ValueError("Bermudan exercise dates must be nonempty and strictly increasing")
        elif dates:
            raise ValueError("exercise dates apply only to Bermudan exercise")
        object.__setattr__(self, "dates", dates)

    @classmethod
    def european(cls) -> Exercise:
        return cls("european")

    @classmethod
    def american(cls) -> Exercise:
        return cls("american")

    @classmethod
    def bermudan(cls, dates: Sequence[date]) -> Exercise:
        return cls("bermudan", tuple(dates))


@dataclass(frozen=True)
class VanillaOption:
    """Fixed strike, maturity, call/put type, and exercise convention.

    For engines supporting strike derivatives, override ``strike`` with a
    tensor when calling the pricing layer.
    """

    strike: float
    maturity_date: date
    option_type: Literal["call", "put"] = "call"
    exercise: Exercise = field(default_factory=Exercise)

    def __post_init__(self) -> None:
        strike = _finite_number("strike", self.strike)
        if strike <= 0:
            raise ValueError("strike must be positive")
        _serial("maturity_date", self.maturity_date)
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be call or put")
        if not isinstance(self.exercise, Exercise):
            raise TypeError("exercise must be an Exercise object")
        if self.exercise.style == "bermudan" and self.exercise.dates[-1] != self.maturity_date:
            raise ValueError("Bermudan exercise dates must end at maturity_date")
        object.__setattr__(self, "strike", strike)


@dataclass(frozen=True)
class EquityMarket:
    """Reusable passive equity market using Actual/365 Fixed conventions.

    Curves contain dated discount factors with log-linear interpolation. Cash
    dividends contain dated amounts. Sequences are copied to immutable tuples.
    Tensor inputs belong in ``PricingLayer.forward`` overrides, so market
    configuration never retains a parameter graph between optimizer steps.
    """

    spot: float = 100.0
    valuation_date: date = date(2025, 1, 1)
    rate: float = 0.03
    dividend_yield: float = 0.01
    discount_curve: DateValues | None = None
    dividend_curve: DateValues | None = None
    cash_dividends: DateValues = ()

    def __post_init__(self) -> None:
        reference = _serial("valuation_date", self.valuation_date)
        for name in ("spot", "rate", "dividend_yield"):
            object.__setattr__(self, name, _finite_number(name, getattr(self, name)))
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        for name in ("discount_curve", "dividend_curve"):
            values = getattr(self, name)
            if values is not None:
                values = tuple((d, _finite_number(name, v)) for d, v in values)
                _curve(name, values, reference, reference + 1)
                object.__setattr__(self, name, values)
        cash = tuple((d, _finite_number("cash_dividends", v)) for d, v in self.cash_dividends)
        serials = [_serial("cash_dividends", d) for d, _ in cash]
        if any(d <= reference for d in serials) or any(b <= a for a, b in zip(serials, serials[1:])):
            raise ValueError("cash dividend dates must increase and follow valuation_date")
        if any(v < 0 for _, v in cash):
            raise ValueError("cash dividends must be nonnegative")
        object.__setattr__(self, "cash_dividends", cash)


def _arguments(instrument: VanillaOption, market: EquityMarket,
               overrides: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    if not isinstance(instrument, VanillaOption) or not isinstance(market, EquityMarket):
        raise TypeError("instrument and market must be VanillaOption and EquityMarket objects")
    unknown = overrides.keys() - allowed
    if unknown:
        raise TypeError(f"unsupported pricing overrides: {', '.join(sorted(unknown))}")
    if instrument.maturity_date <= market.valuation_date:
        raise ValueError("maturity_date must follow valuation_date")
    if instrument.exercise.dates and instrument.exercise.dates[0] <= market.valuation_date:
        raise ValueError("exercise dates must follow valuation_date")
    result = dict(
        spot=market.spot, strike=instrument.strike,
        valuation_date=market.valuation_date, maturity_date=instrument.maturity_date,
        rate=market.rate, dividend_yield=market.dividend_yield,
        discount_curve=market.discount_curve, dividend_curve=market.dividend_curve,
        cash_dividends=market.cash_dividends, option_type=instrument.option_type,
        exercise=instrument.exercise.style, exercise_dates=instrument.exercise.dates,
    )
    result.update(overrides)
    return result


_MARKET_OVERRIDES = {"spot", "strike", "rate", "dividend_yield", "discount_curve",
                     "dividend_curve", "cash_dividends"}


class PricingEngine(Protocol):
    """Structural interface for engines accepted by :class:`PricingLayer`."""

    def price(self, instrument: VanillaOption, market: EquityMarket,
              *inputs: torch.Tensor, **overrides: Any) -> torch.Tensor: ...


@dataclass(frozen=True)
class AnalyticEuropeanEngine:
    """QuantLib analytic European pricing with CPU float64 full AD.

    Supply a scalar volatility tensor. Tensor market overrides support the
    differentiable inputs of ``analytic_european``. A nonempty cash schedule
    selects QuantLib's analytic dividend engine and its escrowed approximation.
    Early exercise is rejected. There is no equation-adjoint mode.
    """

    def price(self, instrument: VanillaOption, market: EquityMarket,
              volatility: torch.Tensor, **overrides: Any) -> torch.Tensor:
        from .analytic import analytic_european

        kwargs = _arguments(instrument, market, overrides, _MARKET_OVERRIDES)
        if kwargs.pop("exercise") != "european":
            raise ValueError("AnalyticEuropeanEngine supports European exercise only")
        kwargs.pop("exercise_dates")
        return analytic_european(volatility, **kwargs)


@dataclass(frozen=True)
class FdVanillaEngine:
    """Configuration for QuantLib's 1D Black-Scholes finite-difference engine.

    ``price`` takes a scalar or sampled local-volatility tensor. Financial
    overrides remain fixed values, as required by ``fd_vanilla``. Full AD
    supports higher derivatives; equation adjoints support first derivatives.
    """

    time_steps: int = 50
    grid_points: int = 100
    damping_steps: int = 2
    scheme: str = "douglas"
    mesh_volatility: float = 0.2
    cash_dividend_model: str = "spot"
    times: Sequence[float] | None = None
    log_spots: Sequence[float] | None = None
    mode: Literal["full", "adjoint"] = "full"
    kernels: str = "auto"

    def __post_init__(self) -> None:
        for name, minimum in (("time_steps", 1), ("grid_points", 5), ("damping_steps", 0)):
            object.__setattr__(self, name, _integer(name, getattr(self, name), minimum))
        if self.scheme not in ("douglas", "crank_nicolson", "implicit_euler", "craig_sneyd",
                               "modified_craig_sneyd", "hundsdorfer"):
            raise ValueError("unsupported finite-difference scheme")
        if self.cash_dividend_model not in ("spot", "escrowed"):
            raise ValueError("cash_dividend_model must be spot or escrowed")
        volatility = _finite_number("mesh_volatility", self.mesh_volatility)
        if volatility <= 0:
            raise ValueError("mesh_volatility must be positive")
        object.__setattr__(self, "mesh_volatility", volatility)
        _validate_mode(self.mode, self.kernels, ("auto", "scalar", "apply", "vectorized", "pcr"))
        if self.kernels == "pcr" and self.mode != "full":
            raise ValueError("PCR pricing supports mode='full' only")
        for name in ("times", "log_spots"):
            object.__setattr__(self, name, _fixed_axis(name, getattr(self, name)))
        if self.times is not None and self.times[0] < 0:
            raise ValueError("times must be nonnegative")

    def price(self, instrument: VanillaOption, market: EquityMarket,
              local_vol: torch.Tensor, **overrides: Any) -> torch.Tensor:
        kwargs = _arguments(instrument, market, overrides, _MARKET_OVERRIDES)
        return fd_vanilla(
            local_vol, **kwargs, times=self.times, log_spots=self.log_spots,
            time_steps=self.time_steps, grid_points=self.grid_points,
            damping_steps=self.damping_steps, scheme=self.scheme,
            mesh_volatility=self.mesh_volatility, cash_dividend_model=self.cash_dividend_model,
            mode=self.mode, kernels=self.kernels,
        )


def _validate_mode(mode: str, kernels: str, choices: tuple[str, ...]) -> None:
    if mode not in ("full", "adjoint"):
        raise ValueError("mode must be full or adjoint")
    if kernels not in choices:
        raise ValueError(f"kernels must be one of {choices}")


@dataclass(frozen=True)
class FdLatentEngine:
    """QuantLib 2D/3D latent pricing with fixed axes and coefficient times.

    Inputs are latent drift, diffusion loadings and initial log-spot/latent
    state, matching ``fd_latent``. The market's passive spot is unused because
    initial_state explicitly supplies log-spot. Rate/yield and dated discount
    factor overrides may be graph-connected tensors. No cash-dividend path is
    exposed by this scalar adapter.
    """

    axes: Sequence[Sequence[float]]
    times: Sequence[float] | None = None
    time_steps: int = 30
    scheme: str = "hundsdorfer"
    mode: Literal["full", "adjoint"] = "full"
    kernels: str = "auto"

    def __post_init__(self) -> None:
        axes = tuple(tuple(a) for a in _axes(self.axes, minimum=4))
        if len(axes) not in (2, 3):
            raise ValueError("FdLatentEngine supports one or two latent states")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "times", _fixed_axis("times", self.times))
        object.__setattr__(self, "time_steps", _integer("time_steps", self.time_steps, 1))
        if self.scheme not in ("douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer"):
            raise ValueError("unsupported latent finite-difference scheme")
        _validate_mode(self.mode, self.kernels, ("auto", "scalar"))

    def price(self, instrument: VanillaOption, market: EquityMarket,
              latent_drift: torch.Tensor, diffusion: torch.Tensor,
              initial_state: torch.Tensor, **overrides: Any) -> torch.Tensor:
        kwargs = _arguments(instrument, market, overrides, _MARKET_OVERRIDES - {"spot"})
        if kwargs.pop("cash_dividends"):
            raise ValueError("FdLatentEngine does not support cash dividends")
        kwargs.pop("spot")
        return fd_latent(
            latent_drift, diffusion, initial_state, **kwargs, axes=self.axes, times=self.times,
            time_steps=self.time_steps, scheme=self.scheme, mode=self.mode, kernels=self.kernels,
        )


@dataclass(frozen=True)
class FdHestonEngine:
    """Original QuantLib Heston engine with explicitly frozen geometry.

    Supply five scalar inputs (v0, kappa, theta, sigma, rho), or register a
    ``HestonModel`` on the pricing layer. Spot, rate/yield and dated curve
    overrides may be tensors. Strike and cash dividends remain fixed.
    """

    mesh: HestonMesh
    time_steps: int = 30
    damping_steps: int = 0
    scheme: str = "hundsdorfer"
    mode: Literal["full", "adjoint"] = "full"
    kernels: str = "auto"
    day_counter: str = "actual365fixed"

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, HestonMesh):
            raise TypeError("mesh must be a HestonMesh; prepare it with heston_mesh")
        object.__setattr__(self, "time_steps", _integer("time_steps", self.time_steps, 1))
        object.__setattr__(self, "damping_steps", _integer("damping_steps", self.damping_steps, 0))
        if self.scheme not in ("douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer"):
            raise ValueError("unsupported Heston finite-difference scheme")
        _validate_mode(self.mode, self.kernels, ("auto", "scalar"))
        _choice("day_counter", self.day_counter, _DAY_COUNTERS)

    def price(self, instrument: VanillaOption, market: EquityMarket,
              v0: torch.Tensor, kappa: torch.Tensor, theta: torch.Tensor,
              sigma: torch.Tensor, rho: torch.Tensor, **overrides: Any) -> torch.Tensor:
        kwargs = _arguments(instrument, market, overrides, _MARKET_OVERRIDES)
        return fd_heston(v0, kappa, theta, sigma, rho, **kwargs, mesh=self.mesh,
                         time_steps=self.time_steps, damping_steps=self.damping_steps,
                         scheme=self.scheme, mode=self.mode, kernels=self.kernels,
                         day_counter=self.day_counter)


@dataclass(frozen=True)
class FdShoutEngine:
    """Original one-shout engine with constant Black volatility.

    The instrument exercise rule specifies when shouting is allowed. Cash
    dividends use the original escrowed model. Financial tensor overrides
    support spot, rate/yield and dated discount factors on the explicit mesh.
    """

    mesh: ShoutMesh
    time_steps: int = 50
    damping_steps: int = 0
    scheme: str = "douglas"
    mode: Literal["full", "adjoint"] = "full"
    kernels: str = "auto"
    day_counter: str = "actual365fixed"

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, ShoutMesh):
            raise TypeError("mesh must be a ShoutMesh; prepare it with shout_mesh")
        object.__setattr__(self, "time_steps", _integer("time_steps", self.time_steps, 1))
        object.__setattr__(self, "damping_steps", _integer("damping_steps", self.damping_steps, 0))
        if self.scheme not in ("douglas", "crank_nicolson", "implicit_euler", "craig_sneyd",
                               "modified_craig_sneyd", "hundsdorfer"):
            raise ValueError("unsupported shout finite-difference scheme")
        _validate_mode(self.mode, self.kernels, ("auto", "scalar"))
        _choice("day_counter", self.day_counter, _DAY_COUNTERS)

    def price(self, instrument: VanillaOption, market: EquityMarket,
              volatility: torch.Tensor, **overrides: Any) -> torch.Tensor:
        kwargs = _arguments(instrument, market, overrides, _MARKET_OVERRIDES)
        return fd_shout(volatility, **kwargs, mesh=self.mesh, time_steps=self.time_steps,
                        damping_steps=self.damping_steps, scheme=self.scheme,
                        mode=self.mode, kernels=self.kernels, day_counter=self.day_counter)


class PricingLayer(nn.Module):
    """Compose an instrument, market and engine with ordinary PyTorch models.

    Without ``model``, positional inputs pass directly to ``engine.price``.
    With a registered ``nn.Module``, they go to that model first; its tensor or
    tuple of tensors supplies the engine inputs. Keyword arguments are market
    overrides and go directly to the engine. Every call constructs a fresh
    graph. Model parameters participate in ``parameters()`` and ``state_dict``;
    financial configuration is supplied separately when reconstructing a layer.

    No implicit dtype/device conversion occurs. The engine's supported devices,
    tensor shapes, gradient order and nonsmooth derivative rules still apply.
    A custom engine can implement the small :class:`PricingEngine` protocol.
    """

    def __init__(self, instrument: VanillaOption, market: EquityMarket,
                 engine: PricingEngine, *, model: nn.Module | None = None) -> None:
        super().__init__()
        _arguments(instrument, market, {}, set())
        if not callable(getattr(engine, "price", None)):
            raise TypeError("engine must provide a callable price method")
        if model is not None and not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.instrument = instrument
        self.market = market
        self.engine = engine
        self.model = model

    def forward(self, *inputs: torch.Tensor, **overrides: Any) -> torch.Tensor:
        if self.model is not None:
            result = self.model(*inputs)
            inputs = result if isinstance(result, tuple) else (result,)
        if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs):
            raise TypeError("engine inputs must be tensors; model must return a tensor or tuple of tensors")
        return self.engine.price(self.instrument, self.market, *inputs, **overrides)

    def extra_repr(self) -> str:
        return (f"engine={type(self.engine).__name__}, option_type={self.instrument.option_type}, "
                f"exercise={self.instrument.exercise.style}, strike={self.instrument.strike}")
