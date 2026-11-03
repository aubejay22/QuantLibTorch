"""Ownership and global-state isolation of native QuantLib pricing calls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import math

import pytest
import torch

from quantlibtorch import fd_vanilla
from quantlibtorch.fd import _serial

GRID = dict(time_steps=2, grid_points=9, damping_steps=0)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_price_graph_survives_later_pricing_calls(mode):
    sigma = torch.tensor(.21, dtype=torch.float64, requires_grad=True)
    original = fd_vanilla(sigma, mode=mode, **GRID)
    # The native financial objects used by original have already been destroyed.
    # A later call must not invalidate its coefficient graph or saved adjoints.
    fd_vanilla(sigma, mode=mode, valuation_date=date(2030, 1, 1),
               maturity_date=date(2032, 1, 1), option_type="put", **GRID)
    repeated = fd_vanilla(sigma, mode=mode, **GRID)
    original_grad, = torch.autograd.grad(original, sigma)
    repeated_grad, = torch.autograd.grad(repeated, sigma)
    torch.testing.assert_close(original, repeated, rtol=0, atol=0)
    torch.testing.assert_close(original_grad, repeated_grad, rtol=0, atol=0)


def test_evaluation_date_restored_after_native_exception():
    initial_date = torch.ops.quantlibtorch._evaluation_date()
    # Keep QuantLib's raw null date unanchored: it means today's date dynamically.
    assert initial_date == 0
    surface = torch.full((1, 1), .2, dtype=torch.float64)
    reference = _serial("reference", date(2030, 1, 1))
    maturity = _serial("maturity", date(2031, 1, 1))
    market = dict(spot=100., strike=100., rate=.03, dividend_yield=.01, mesh_volatility=.2)
    config = dict(GRID, valuation_date=reference, maturity_date=maturity,
                  exercise=0, option_type=1, scheme=0, adjoint=1)
    # Bypass Python validation deliberately. Native curve construction rejects
    # this negative discount AFTER it sets its temporary evaluation date.
    with pytest.raises(RuntimeError, match="discount factors.*positive"):
        torch.ops.quantlibtorch.fd_vanilla(
            surface, [0.], [math.log(100.)], market, config,
            [reference, maturity], [1., -.5], [], [], [], [], [],
        )
    assert torch.ops.quantlibtorch._evaluation_date() == initial_date
    value = fd_vanilla(surface, mode="full", **GRID)
    assert torch.isfinite(value)
    assert torch.ops.quantlibtorch._evaluation_date() == initial_date


def test_concurrent_explicit_dates_match_sequential_pricing():
    initial_date = torch.ops.quantlibtorch._evaluation_date()

    def price(offset):
        start = date(2025, 1, 1) + timedelta(days=offset)
        sigma = torch.tensor(.21, dtype=torch.float64, requires_grad=True)
        value = fd_vanilla(sigma, valuation_date=start,
                           maturity_date=start + timedelta(days=365), mode="adjoint", **GRID)
        grad, = torch.autograd.grad(value, sigma)
        return value.item(), grad.item()

    expected = price(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(price, [0, 1700, 3400, 5100]))
    assert actual == [expected] * 4
    assert torch.ops.quantlibtorch._evaluation_date() == initial_date
