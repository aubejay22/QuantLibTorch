"""Per-call graph-connected markets for fixed-geometry financial solvers."""
import torch
from .analytic import _scalar, _discounts


def _market(rate, dividend_yield, discount_curve, dividend_curve, reference, maturity):
    rate = _scalar("rate", rate)
    dividend_yield = _scalar("dividend_yield", dividend_yield)
    curves = []
    for name, points in (("discount_curve", discount_curve), ("dividend_curve", dividend_curve)):
        points = None if points is None else list(points)
        if points and isinstance(points[0][1], torch.Tensor) and _tangent(points[0][1]):
            raise ValueError(f"{name} first discount must be a fixed one")
        curves.append(points)
    discount_curve, dividend_curve = curves
    risk_dates, risk_dfs = _discounts("discount_curve", discount_curve, reference, maturity)
    yield_dates, yield_dfs = _discounts("dividend_curve", dividend_curve, reference, maturity)
    for name, value, dates in (("rate", rate, risk_dates), ("dividend_yield", dividend_yield, yield_dates)):
        if dates and (value.requires_grad or _tangent(value)):
            raise ValueError(f"differentiable {name} is overridden by its dated curve")
    return rate, dividend_yield, risk_dates, risk_dfs, yield_dates, yield_dfs


def _tangent(value):
    return torch.autograd.forward_ad.unpack_dual(value).tangent is not None
