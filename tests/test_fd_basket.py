"""Real QuantLib 2D financial integration and coefficient sensitivities."""
from datetime import date
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from quantlibtorch import fd_basket, fd_vanilla
from quantlibtorch.fd import _serial


DTYPE = torch.float64
REFERENCE = Path(__file__).with_name("stock_basket_reference.py")
SMALL = dict(time_steps=4, grid_points=(11, 9))
TINY = dict(time_steps=2, grid_points=(7, 5))
START, MID, END = date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)
SCHEMES = ["douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer"]


def stock(**request):
    if importlib.util.find_spec("QuantLib") is None:
        pytest.skip("install QuantLib==1.40 for the isolated stock reference")
    output = subprocess.run([sys.executable, str(REFERENCE)],
                            input=json.dumps(request, default=lambda x: x.isoformat()),
                            capture_output=True, text=True, check=True)
    return json.loads(output.stdout)


def scalars():
    return tuple(torch.tensor(v, dtype=DTYPE, requires_grad=True) for v in (.22, .28, .35))


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_basket_prices_and_all_coefficient_vjps_match_stock(scheme, mode):
    values = scalars()
    actual = fd_basket(*values, scheme=scheme, mode=mode, **SMALL)
    gradients = torch.autograd.grad(actual, values)
    expected = stock(volatility1=.22, volatility2=.28, correlation=.35,
                     scheme=scheme, bump=1e-5, **SMALL)
    assert actual.item() == pytest.approx(expected["price"], abs=3e-10)
    for gradient, reference in zip(gradients, expected["coefficient_derivatives"]):
        assert gradient.item() == pytest.approx(reference, rel=5e-6, abs=3e-6)
    assert all(abs(g.item()) > 1e-4 for g in gradients)


MARKETS = [
    dict(basket="max", option_type="put", spot1=92., spot2=98., rate=-.01),
    dict(exercise="american", option_type="put", rate=.06),
    dict(basket="max", exercise="bermudan", option_type="put",
         exercise_dates=(MID, END), rate=.05),
    dict(discount_curve=((START, 1.), (MID, .981), (END, .951)),
         dividend_curve1=((START, 1.), (MID, .997), (END, .984)),
         dividend_curve2=((START, 1.), (MID, .988), (END, .974))),
]


@pytest.mark.parametrize("market", MARKETS)
def test_basket_financial_conventions_and_exercise_match_stock(market):
    values = scalars()
    actual = fd_basket(*values, mode="adjoint", **SMALL, **market)
    gradients = torch.autograd.grad(actual, values)
    expected = stock(volatility1=.22, volatility2=.28, correlation=.35,
                     bump=1e-5, **SMALL, **market)
    assert actual.item() == pytest.approx(expected["price"], abs=3e-10)
    for gradient, reference in zip(gradients, expected["coefficient_derivatives"]):
        assert gradient.item() == pytest.approx(reference, rel=1e-5, abs=4e-6)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_two_time_surfaces_match_independent_stock_local_vol_handles(mode):
    times1, times2 = [0., .31, 1.], [0., .57, 1.]
    first = torch.tensor([[.21], [.25], [.23]], dtype=DTYPE, requires_grad=True)
    second = torch.tensor([[.29], [.24], [.27]], dtype=DTYPE, requires_grad=True)
    rho = torch.tensor(-.25, dtype=DTYPE, requires_grad=True)
    actual = fd_basket(first, second, rho, times1=times1, times2=times2, mode=mode, **SMALL)
    gradients = torch.autograd.grad(actual, (first, second, rho))
    expected = stock(time_vols1=first[:, 0].tolist(), times1=times1,
                     time_vols2=second[:, 0].tolist(), times2=times2,
                     correlation=rho.item(), bump=1e-5, **SMALL)
    assert actual.item() == pytest.approx(expected["price"], abs=4e-10)
    for gradient, reference in zip(gradients, expected["coefficient_derivatives"]):
        assert gradient.sum().item() == pytest.approx(reference, rel=6e-6, abs=4e-6)
    assert all(bool((g.abs() > 1e-6).all()) for g in gradients)


@pytest.mark.parametrize("scheme", ["douglas", "modified_craig_sneyd", "hundsdorfer"])
def test_basket_spatial_surface_vjps_match_full_and_directional_differences(scheme):
    bases = tuple((.23 + .02 * torch.sin(torch.arange(12, dtype=DTYPE) + i)).reshape(2, 6)
                  .requires_grad_() for i in (0., 1.))
    surfaces = tuple(base[:, ::2] for base in bases)
    rho = torch.tensor(.31, dtype=DTYPE, requires_grad=True)
    before = tuple(value.detach().clone() for value in (*bases, rho))
    args = dict(times1=[0., 1.], times2=[0., 1.],
                log_spots1=[math.log(v) for v in (40., 100., 250.)],
                log_spots2=[math.log(v) for v in (35., 95., 270.)],
                scheme=scheme, **SMALL)
    full = fd_basket(*surfaces, rho, mode="full", **args)
    adjoint = fd_basket(*surfaces, rho, mode="adjoint", **args)
    gf = torch.autograd.grad(full, (*bases, rho))
    ga = torch.autograd.grad(adjoint, (*bases, rho))
    torch.testing.assert_close(adjoint, full, rtol=2e-12, atol=2e-12)
    for actual, expected in zip(ga, gf):
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-9)
    directions = tuple(torch.cos(torch.arange(v.numel(), dtype=DTYPE)).reshape_as(v)
                       for v in surfaces) + (torch.tensor(.4, dtype=DTYPE),)
    directional = sum((g[:, ::2] * d).sum() for g, d in zip(ga[:2], directions[:2])) + ga[2] * directions[2]
    h = 1e-5
    with torch.no_grad():
        plus = fd_basket(*(v + h*d for v, d in zip((*surfaces, rho), directions)), mode="adjoint", **args)
        minus = fd_basket(*(v - h*d for v, d in zip((*surfaces, rho), directions)), mode="adjoint", **args)
    torch.testing.assert_close(directional, (plus-minus)/(2*h), rtol=2e-5, atol=3e-6)
    for value, original in zip((*bases, rho), before):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_basket_neural_coefficient_loss_parameter_vjp(mode):
    parameters = torch.linspace(-.25, .3, 9, dtype=DTYPE, requires_grad=True)
    coordinates = torch.tensor([[0., -.6, 1.], [0., .6, 1.],
                                [1., -.6, 1.], [1., .6, 1.]], dtype=DTYPE)
    args = dict(times1=[0., 1.], times2=[0., 1.],
                log_spots1=[math.log(50.), math.log(200.)],
                log_spots2=[math.log(45.), math.log(190.)], mode=mode, **TINY)

    def loss(weights):
        first = (.1 + .3*torch.sigmoid(coordinates @ weights[:3])).reshape(2, 2)
        second = (.12 + .3*torch.sigmoid(coordinates @ weights[3:6])).reshape(2, 2)
        correlation = .7*torch.tanh(weights[6:].sum())
        value = fd_basket(first, second, correlation, **args)
        return (value-4.).square() + .01*weights.square().sum()

    gradient, = torch.autograd.grad(loss(parameters), parameters)
    direction = torch.sin(torch.arange(9, dtype=DTYPE)+1.)
    with torch.no_grad():
        difference = (loss(parameters+1e-5*direction)-loss(parameters-1e-5*direction))/(2e-5)
    assert (gradient * direction).sum().abs().item() > 1e-5
    torch.testing.assert_close((gradient*direction).sum(), difference, rtol=2e-5, atol=3e-6)


def test_basket_adjoint_first_order_contract():
    values = scalars()
    output = fd_basket(*values, mode="adjoint", **TINY)
    with pytest.raises(RuntimeError, match="first derivatives only"):
        torch.autograd.grad(output, values, create_graph=True)


def test_basket_full_hessian_vector_against_gradient_differences():
    coefficients = torch.tensor([.22, .28, .35], dtype=DTYPE, requires_grad=True)
    direction = torch.tensor([.3, -.2, .4], dtype=DTYPE)

    def price(values):
        return fd_basket(values[0], values[1], values[2], mode="full", **TINY)

    gradient, = torch.autograd.grad(price(coefficients), coefficients, create_graph=True)
    hvp, = torch.autograd.grad((gradient*direction).sum(), coefficients)
    h = 2e-5
    plus = (coefficients.detach()+h*direction).requires_grad_()
    minus = (coefficients.detach()-h*direction).requires_grad_()
    gp, = torch.autograd.grad(price(plus), plus)
    gm, = torch.autograd.grad(price(minus), minus)
    torch.testing.assert_close(hvp, (gp-gm)/(2*h), rtol=3e-5, atol=5e-6)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_basket_graph_survives_other_engine_calls_and_restores_date(mode):
    values = scalars()
    initial_date = torch.ops.quantlibtorch._evaluation_date()
    original = fd_basket(*values, mode=mode, **TINY)
    fd_vanilla(values[0], mode=mode, time_steps=2, grid_points=7,
               valuation_date=date(2030, 1, 1), maturity_date=date(2031, 1, 1))
    repeat = fd_basket(*values, mode=mode, **TINY)
    go = torch.autograd.grad(original, values)
    gr = torch.autograd.grad(repeat, values)
    torch.testing.assert_close(original, repeat, rtol=0, atol=0)
    for a, b in zip(go, gr):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.ops.quantlibtorch._evaluation_date() == initial_date


def test_basket_native_failure_restores_evaluation_date():
    values = scalars()
    initial_date = torch.ops.quantlibtorch._evaluation_date()
    reference, maturity = _serial("reference", START), _serial("maturity", END)
    market = dict(spot1=100., spot2=95., strike=100., rate=.03,
                  dividend_yield1=.01, dividend_yield2=.015,
                  mesh_volatility1=.2, mesh_volatility2=.25)
    config = dict(time_steps=2, x_grid=7, y_grid=5, damping_steps=0,
                  valuation_date=reference, maturity_date=maturity,
                  option_type=1, exercise=0, scheme=3, adjoint=1, basket=0, kernels=4)
    with pytest.raises(RuntimeError, match="discount factors.*positive"):
        torch.ops.quantlibtorch.fd_basket(
            values[0].reshape(1, 1), values[1].reshape(1, 1), values[2],
            [0.], [math.log(100.)], [0.], [math.log(95.)], market, config,
            [reference, maturity], [1., -.5], [], [], [], [], [])
    assert torch.ops.quantlibtorch._evaluation_date() == initial_date


@pytest.mark.parametrize("changes,error", [
    (dict(damping_steps=1), ValueError),
    (dict(kernels="pcr"), ValueError),
    (dict(scheme="implicit_euler"), ValueError),
    (dict(grid_points=(4, 7)), ValueError),
    (dict(grid_points=(7,)), ValueError),
    (dict(spot2=torch.tensor(95.)), TypeError),
    (dict(times1=torch.tensor([0.])), TypeError),
    (dict(exercise="bermudan"), ValueError),
    (dict(exercise_dates=[END]), ValueError),
    (dict(mesh_volatility2=-.2), ValueError),
])
def test_basket_rejects_unsupported_configuration(changes, error):
    with pytest.raises(error):
        fd_basket(*scalars(), **changes)


@pytest.mark.parametrize("index,replacement,error", [
    (0, .2, TypeError),
    (0, torch.tensor(.2, dtype=torch.float32), ValueError),
    (1, torch.ones(3, dtype=DTYPE), ValueError),
    (1, torch.tensor(-.2, dtype=DTYPE), ValueError),
    (2, .3, TypeError),
    (2, torch.tensor([.3], dtype=DTYPE), ValueError),
    (2, torch.tensor(1., dtype=DTYPE), ValueError),
    (2, torch.tensor(float("nan"), dtype=DTYPE), ValueError),
])
def test_basket_rejects_invalid_coefficient_tensors(index, replacement, error):
    values = list(scalars())
    values[index] = replacement
    with pytest.raises(error):
        fd_basket(*values)
