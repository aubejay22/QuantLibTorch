"""Original QuantLib analytic European engines with graph-connected market data."""
from __future__ import annotations

from datetime import date
import torch

from .fd import _serial
from .pricing import _finite_number


def _scalar(name, value):
    if not isinstance(value, torch.Tensor):
        return torch.tensor(_finite_number(name, value), dtype=torch.float64, device="cpu")
    if (value.device.type != "cpu" or value.dtype != torch.float64
            or value.layout != torch.strided or value.ndim != 0):
        raise ValueError(f"{name} must be a dense CPU float64 scalar tensor")
    if not bool(torch.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _discounts(name, points, reference, maturity):
    if points is None:
        return [], torch.empty(0, dtype=torch.float64, device="cpu")
    points = list(points)
    if len(points) < 2:
        raise ValueError(f"{name} requires at least two dated discount factors")
    dates = [_serial(name, d) for d, _ in points]
    values = [_scalar(name, v) for _, v in points]
    if dates[0] != reference or dates[-1] < maturity:
        raise ValueError(f"{name} must start at valuation_date and cover maturity_date")
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError(f"{name} dates must be strictly increasing")
    if values[0].requires_grad or values[0].item() != 1.0:
        raise ValueError(f"{name} first discount must be a fixed one")
    if any(v.item() <= 0 for v in values):
        raise ValueError(f"{name} discounts must be positive")
    return dates, torch.stack(values)


def analytic_european(
    volatility: torch.Tensor, *, spot: float | torch.Tensor = 100.0,
    strike: float | torch.Tensor = 100.0,
    rate: float | torch.Tensor = 0.03, dividend_yield: float | torch.Tensor = 0.01,
    valuation_date: date = date(2025, 1, 1), maturity_date: date = date(2026, 1, 1),
    discount_curve=None, dividend_curve=None, cash_dividends=(),
    option_type: str = "call", mode: str = "full",
) -> torch.Tensor:
    """Price with QuantLib AnalyticEuropeanEngine or AnalyticDividendEuropeanEngine.

    Volatility is a positive CPU float64 scalar tensor. Spot, strike, continuous
    flat rates, noninitial dated discount factors and cash dividend amounts may
    also be scalar tensors and retain their graphs. Full reverse AD and higher
    derivatives are supported. Dates and the first curve discount (one) are
    fixed. Dated curves use QuantLib log-linear interpolation and Actual/365
    Fixed; supplying a curve replaces its flat rate. Active shadowed flat rates
    raise instead of silently losing a requested sensitivity.

    Cash schedules select QuantLib's analytic escrowed-dividend approximation,
    which subtracts discounted dividends from spot. It is not the FD cash-jump
    model. Dates must increase, follow valuation and not exceed maturity;
    dividend amounts are nonnegative and adjusted spot must remain positive.
    This engine supports European exercise, CPU scalar execution and full AD
    only. There is no equation adjoint or automatic CUDA fallback.
    """
    if not isinstance(volatility, torch.Tensor):
        raise TypeError("volatility must be a torch.Tensor")
    if mode != "full":
        raise ValueError("analytic_european supports mode='full' only")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be call or put")
    reference = _serial("valuation_date", valuation_date)
    maturity = _serial("maturity_date", maturity_date)
    if maturity <= reference:
        raise ValueError("maturity_date must follow valuation_date")
    names = ("spot", "strike", "volatility", "rate", "dividend_yield")
    values = [_scalar(n, v) for n, v in zip(names, (spot, strike, volatility, rate, dividend_yield))]
    if any(v.item() <= 0 for v in values[:3]):
        raise ValueError("spot, strike and volatility must be positive")
    risk_dates, risk_dfs = _discounts("discount_curve", discount_curve, reference, maturity)
    yield_dates, yield_dfs = _discounts("dividend_curve", dividend_curve, reference, maturity)
    for n, v, ds in (("rate", values[3], risk_dates), ("dividend_yield", values[4], yield_dates)):
        if ds and v.requires_grad:
            raise ValueError(f"differentiable {n} is overridden by its dated curve")
    cash_pairs = list(cash_dividends)
    cash_dates = [_serial("cash_dividends", d) for d, _ in cash_pairs]
    cash_values = [_scalar("cash_dividends", v) for _, v in cash_pairs]
    if (any(d <= reference or d > maturity for d in cash_dates)
            or any(b <= a for a, b in zip(cash_dates, cash_dates[1:]))):
        raise ValueError("cash dividend dates must increase, follow valuation and not exceed maturity")
    if any(v.item() < 0 for v in cash_values):
        raise ValueError("cash dividends must be nonnegative")
    cash = torch.stack(cash_values) if cash_values else torch.empty(0, dtype=torch.float64, device="cpu")
    return torch.ops.quantlibtorch.analytic_european(
        torch.stack(values), reference, maturity, 1 if option_type == "call" else -1,
        risk_dates, risk_dfs, yield_dates, yield_dfs, cash_dates, cash)
