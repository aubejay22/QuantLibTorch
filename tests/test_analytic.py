from datetime import date
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from quantlibtorch import analytic_european

DT = torch.float64
REF, MID, END = date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)


def stock(requests):
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("stock_analytic_reference.py"))],
                       input=json.dumps(requests, default=lambda x: x.isoformat()),
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("cash", [False, True])
def test_price_all_market_greeks_and_gamma_match_independent_stock(kind, cash):
    x = torch.tensor([103., 99., .23, -.01, .02, 1.7], dtype=DT, requires_grad=True)
    def config(y):
        return dict(spot=y[0], strike=y[1], rate=y[3], dividend_yield=y[4], option_type=kind,
                    cash_dividends=[(MID, y[5])] if cash else [])
    price = analytic_european(x[2], **config(x))
    grad, = torch.autograd.grad(price, x, create_graph=True)
    requests = [dict(volatility=x[2].item(), **config(x.detach().tolist()))]
    bumps = [1e-3, 1e-3, 1e-5, 1e-5, 1e-5, 1e-4]
    for i, h in enumerate(bumps):
        for sign in (1, -1):
            y = x.detach().tolist()
            y[i] += sign*h
            requests.append(dict(volatility=y[2], **config(y)))
    expected = stock(requests)
    assert price.item() == pytest.approx(expected[0], abs=2e-12)
    for i, h in enumerate(bumps):
        fd = (expected[1+2*i] - expected[2+2*i])/(2*h)
        assert grad[i].item() == pytest.approx(fd, rel=3e-7, abs=2e-7)
    gamma, = torch.autograd.grad(grad[0], x)
    assert gamma[0].item() == pytest.approx((expected[1]-2*expected[0]+expected[2])/bumps[0]**2, rel=1e-4)


def test_dated_curves_cash_and_neural_parameter_graphs():
    x = torch.tensor([.98, .95, .997, .985, 1.8, -.4], dtype=DT, requires_grad=True)
    def price(y):
        return analytic_european(.1 + .2*y[5].sigmoid(), spot=102.,
            discount_curve=[(REF, 1.), (MID, y[0]), (END, y[1])],
            dividend_curve=[(REF, 1.), (MID, y[2]), (END, y[3])], cash_dividends=[(MID, y[4])])
    assert torch.autograd.gradcheck(price, (x,), eps=1e-5, atol=2e-6)
    assert torch.autograd.gradgradcheck(price, (x,), eps=1e-5, atol=2e-5)
    def request(y):
        return dict(volatility=(.1+.2*torch.tensor(y[5]).sigmoid()).item(), spot=102.,
                    discount_curve=[(REF, 1.), (MID, y[0]), (END, y[1])],
                    dividend_curve=[(REF, 1.), (MID, y[2]), (END, y[3])], cash_dividends=[(MID, y[4])])
    req = request(x.detach().tolist())
    req["volatility"] = (.1+.2*x[5].sigmoid()).item()
    assert price(x).item() == pytest.approx(stock([req])[0], abs=3e-12)


@pytest.mark.parametrize("kwargs,match", [
    ({"spot": torch.tensor(100.)}, "float64"),
    ({"spot": -1.}, "positive"),
    ({"rate": float("nan")}, "finite"),
    ({"mode": "adjoint"}, "full"),
    ({"option_type": "digital"}, "option_type"),
    ({"maturity_date": REF}, "follow"),
    ({"cash_dividends": [(REF, 1.)]}, "dates"),
    ({"cash_dividends": [(MID, -1.)]}, "nonnegative"),
    ({"discount_curve": [(REF, 1.), (END, 0.)]}, "positive"),
    ({"discount_curve": [(REF, torch.tensor(1., dtype=DT, requires_grad=True)), (END, .97)]}, "fixed one"),
    ({"discount_curve": [(REF, 1.), (END, .97)], "rate": torch.tensor(.03, dtype=DT, requires_grad=True)}, "overridden"),
])
def test_validation(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        analytic_european(torch.tensor(.2, dtype=DT), **kwargs)


def test_adjusted_spot_failure_restores_evaluation_date():
    before = torch.ops.quantlibtorch._evaluation_date()
    with pytest.raises(RuntimeError, match="underlying"):
        analytic_european(torch.tensor(.2, dtype=DT), cash_dividends=[(MID, 200.)])
    assert torch.ops.quantlibtorch._evaluation_date() == before
