"""Actual QuantLib engine prices, financial behavior and coefficient VJPs."""
from datetime import date
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from quantlibtorch import fd_vanilla


REFERENCE = Path(__file__).with_name("stock_quantlib_reference.py")
SMALL = dict(time_steps=12, grid_points=31, damping_steps=2)
REF_DATE, EXPIRY = date(2025, 1, 1), date(2026, 1, 1)
MID = date(2025, 7, 1)


def stock(**request):
    if importlib.util.find_spec("QuantLib") is None:
        pytest.skip("install QuantLib==1.40 for the isolated stock pricing reference")
    output = subprocess.run(
        [sys.executable, str(REFERENCE)],
        input=json.dumps(request, default=lambda obj: obj.isoformat()),
        capture_output=True, text=True, check=True,
    )
    return json.loads(output.stdout)


MARKETS = [
    {},
    dict(option_type="put", spot=92.0, rate=-0.01),
    dict(exercise="american", option_type="put", spot=93.0, rate=0.05),
    dict(exercise="bermudan", option_type="put", spot=95.0,
         exercise_dates=(date(2025, 4, 1), MID, EXPIRY)),
    dict(cash_dividends=((date(2025, 4, 1), 1.3), (date(2025, 10, 1), 2.1))),
    dict(exercise="american", cash_dividends=((MID, 3.0),)),
    dict(discount_curve=((REF_DATE, 1.0), (MID, 0.987), (EXPIRY, 0.954)),
         dividend_curve=((REF_DATE, 1.0), (MID, 0.997), (EXPIRY, 0.984))),
]


@pytest.mark.parametrize("market", MARKETS)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_prices_and_parallel_vega_match_stock_engine(market, mode):
    # Black volatility controls the mesh and is intentionally held fixed as
    # local volatility changes in both implementations and the stock bump.
    sigma = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    actual = fd_vanilla(sigma, mode=mode, **SMALL, **market)
    actual_vega, = torch.autograd.grad(actual, sigma)
    expected = stock(volatility=0.23, bump=1e-5, **SMALL, **market)
    assert actual.item() == pytest.approx(expected["price"], abs=2e-10)
    assert actual_vega.item() == pytest.approx(expected["parallel_vol_shift_derivative"], rel=3e-6, abs=2e-6)


@pytest.mark.parametrize("scheme", ["crank_nicolson", "implicit_euler"])
def test_other_schemes_match_stock(scheme):
    sigma = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
    actual = fd_vanilla(sigma, scheme=scheme, mode="adjoint", **SMALL)
    vega, = torch.autograd.grad(actual, sigma)
    expected = stock(volatility=0.19, bump=1e-5, scheme=scheme, **SMALL)
    assert actual.item() == pytest.approx(expected["price"], abs=2e-10)
    assert vega.item() == pytest.approx(expected["parallel_vol_shift_derivative"], rel=3e-6, abs=2e-6)


def test_time_varying_surface_matches_independent_stock_surface():
    times = [0.0, 0.2, 0.63, 1.0]
    values = torch.tensor([[0.21], [0.18], [0.24], [0.23]], dtype=torch.float64, requires_grad=True)
    actual = fd_vanilla(values, times=times, mode="adjoint", **SMALL)
    gradient, = torch.autograd.grad(actual, values)
    expected = stock(times=times, time_vols=values[:, 0].tolist(), bump=1e-5, **SMALL)
    assert actual.item() == pytest.approx(expected["price"], abs=2e-10)
    assert gradient.sum().item() == pytest.approx(expected["parallel_vol_shift_derivative"], rel=3e-6, abs=2e-6)
    assert bool((gradient.abs() > 1e-5).all())


@pytest.mark.parametrize("exercise", ["european", "american", "bermudan"])
def test_surface_vjps_through_exercise_and_dividend_jumps(exercise):
    times = [0.0, 0.37, 1.0]
    xs = [math.log(s) for s in (40.0, 70.0, 100.0, 140.0, 250.0)]
    surface = (0.21 + 0.025 * torch.sin(torch.arange(15, dtype=torch.float64))).reshape(3, 5)
    surface.requires_grad_()
    initial = surface.detach().clone()
    market = dict(times=times, log_spots=xs, exercise=exercise, option_type="put", spot=96.0,
                  cash_dividends=((MID, 1.7),))
    if exercise == "bermudan":
        # Exercise and dividend on the same date must preserve upstream ordering.
        market["exercise_dates"] = (MID, EXPIRY)
    full = fd_vanilla(surface, mode="full", **SMALL, **market)
    adjoint = fd_vanilla(surface, mode="adjoint", **SMALL, **market)
    full_grad, = torch.autograd.grad(full, surface)
    adj_grad, = torch.autograd.grad(adjoint, surface)
    torch.testing.assert_close(full, adjoint, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(full_grad, adj_grad, rtol=2e-9, atol=1e-9)
    assert bool((adj_grad.abs().sum(dim=1) > 1e-6).all())
    direction = torch.cos(torch.arange(15, dtype=torch.float64)).reshape_as(surface)
    h = 1e-5
    with torch.no_grad():
        fd = (fd_vanilla(surface + h*direction, mode="adjoint", **SMALL, **market)
              - fd_vanilla(surface - h*direction, mode="adjoint", **SMALL, **market)) / (2*h)
    torch.testing.assert_close((adj_grad * direction).sum(), fd, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(surface, initial, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_neural_parameter_gradient_through_real_engine(mode):
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1)).double()
    parameters = tuple(model.parameters())
    with torch.no_grad():
        for i, p in enumerate(parameters):
            p.copy_(torch.linspace(-0.2 + 0.03*i, 0.3, p.numel()).reshape_as(p))
    times = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float64)
    x = torch.tensor([-0.7, 0.0, 0.7], dtype=torch.float64)
    t, space = torch.meshgrid(times, x, indexing="ij")
    inputs = torch.stack((t, space), dim=-1)

    def loss():
        field = 0.1 + 0.3 * torch.sigmoid(model(inputs).squeeze(-1))
        price = fd_vanilla(field, times=times.tolist(), log_spots=(x + math.log(100.0)).tolist(),
                           mode=mode, **SMALL)
        return (price - 9.0).square()

    gradients = torch.autograd.grad(loss(), parameters)
    directions = tuple(torch.sin(torch.arange(p.numel(), dtype=torch.float64) + i + 1).reshape_as(p)
                       for i, p in enumerate(parameters))
    analytic = sum((g*d).sum() for g, d in zip(gradients, directions))
    original = tuple(p.detach().clone() for p in parameters)
    h = 1e-5
    with torch.no_grad():
        for p, base, d in zip(parameters, original, directions):
            p.copy_(base+h*d)
        plus = loss()
        for p, base, d in zip(parameters, original, directions):
            p.copy_(base-h*d)
        minus = loss()
        for p, base in zip(parameters, original):
            p.copy_(base)
    assert abs(analytic.item()) > 1e-4
    torch.testing.assert_close(analytic, (plus-minus)/(2*h), rtol=1e-5, atol=2e-6)


def test_adjoint_explicitly_rejects_higher_derivatives():
    sigma = torch.tensor(0.21, dtype=torch.float64, requires_grad=True)
    price = fd_vanilla(sigma, mode="adjoint", time_steps=3, grid_points=11)
    with pytest.raises(RuntimeError, match="first.derivatives|first.order|create_graph"):
        torch.autograd.grad(price, sigma, create_graph=True)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_surface_finite_difference_gradcheck(mode):
    surface = torch.tensor([[0.19, 0.22], [0.23, 0.21]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda v: fd_vanilla(v, times=[0.0, 1.0], log_spots=[3.8, 5.4],
                             spot=103.0, time_steps=3, grid_points=9, damping_steps=1, mode=mode),
        (surface,),
    )


def test_full_second_derivatives():
    sigma = torch.tensor(0.237, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda v: fd_vanilla(v, spot=103.0, time_steps=3, grid_points=9, damping_steps=1, mode="full"),
        (sigma,),
    )


def test_actual_engine_converges_to_analytic_european_price_and_vega():
    normal = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    d1 = (0.03 - 0.01 + 0.5 * 0.2**2) / 0.2
    exact = 100.0 * math.exp(-0.01) * normal(d1) - 100.0 * math.exp(-0.03) * normal(d1 - 0.2)
    exact_vega = 100.0 * math.exp(-0.01 - 0.5*d1*d1) / math.sqrt(2.0*math.pi)
    errors = []
    for steps, points in ((10, 21), (60, 121)):
        sigma = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        price = fd_vanilla(sigma, mode="adjoint", time_steps=steps, grid_points=points)
        vega, = torch.autograd.grad(price, sigma)
        errors.append((abs(price.item() - exact), abs(vega.item() - exact_vega)))
    assert errors[1][0] < 0.004
    assert errors[1][1] < 0.02
    assert errors[1][0] < errors[0][0] / 10
    assert errors[1][1] < errors[0][1] / 10


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_noncontiguous_surface_is_preserved(mode):
    base = torch.linspace(0.18, 0.24, 18, dtype=torch.float64).reshape(3, 6).requires_grad_()
    surface = base[:, ::2]
    assert not surface.is_contiguous()
    args = dict(times=[0.0, 0.4, 1.0], log_spots=[3.7, 4.6, 5.5], **SMALL)
    a = fd_vanilla(surface, mode=mode, **args)
    b = fd_vanilla(surface.contiguous(), mode=mode, **args)
    ga, = torch.autograd.grad(a, base)
    gb, = torch.autograd.grad(b, base)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(ga, gb)
    assert bool((ga[:, 1::2] == 0).all())


@pytest.mark.parametrize("vol, kwargs, error", [
    (torch.tensor(0.2), {}, "CPU float64"),
    (torch.ones(2, dtype=torch.float64), {}, "scalar or two-dimensional"),
    (torch.empty(0, 2, dtype=torch.float64), {}, "nonempty"),
    (torch.tensor(0.0, dtype=torch.float64), {}, "strictly positive"),
    (torch.tensor(float("nan"), dtype=torch.float64), {}, "finite"),
    (torch.full((2, 2), 0.2, dtype=torch.float64), {}, "times is required"),
    (torch.full((2, 1), 0.2, dtype=torch.float64), {"times": [0.4, 0.3]}, "strictly increasing"),
    (torch.tensor(0.2, dtype=torch.float64), {"spot": torch.tensor(100.)}, "fixed real number"),
    (torch.tensor(0.2, dtype=torch.float64), {"times": torch.tensor([0.0])}, "fixed sequence"),
    (torch.tensor(0.2, dtype=torch.float64), {"discount_curve": [(REF_DATE, 1.), (EXPIRY, torch.tensor(.97))]}, "fixed real number"),
    (torch.tensor(0.2, dtype=torch.float64), {"exercise": "bermudan"}, "end at maturity"),
    (torch.tensor(0.2, dtype=torch.float64), {"cash_dividends": [(REF_DATE, 1.)]}, "follow valuation"),
    (torch.tensor(0.2, dtype=torch.float64), {"cash_dividends": [(MID, 100.)]}, "less than spot"),
    (torch.tensor(0.2, dtype=torch.float64), {"mode": "bad"}, "mode must"),
    (torch.tensor(0.2, dtype=torch.float64), {"grid_points": 3}, "at least 5"),
    (torch.tensor(0.2, dtype=torch.float64), {"maturity_date": REF_DATE}, "must follow"),
])
def test_invalid_fd_inputs_are_rejected(vol, kwargs, error):
    with pytest.raises((ValueError, TypeError), match=error):
        fd_vanilla(vol, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_engine_rejects_cuda():
    with pytest.raises(ValueError, match="CPU float64"):
        fd_vanilla(torch.tensor(0.2, device="cuda", dtype=torch.float64))
