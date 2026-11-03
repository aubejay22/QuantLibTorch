"""Original shout engine financial prices and fixed-geometry AD contracts."""
from datetime import date
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from quantlibtorch import (EquityMarket, Exercise, FdShoutEngine, PricingLayer,
                          ShoutMesh, VanillaOption, fd_shout, shout_mesh)


DTYPE = torch.float64
START, MID, END = date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)
SMALL = dict(time_steps=12, x_grid=31)
SCHEMES = ("douglas", "crank_nicolson", "implicit_euler", "craig_sneyd",
           "modified_craig_sneyd", "hundsdorfer")
REFERENCE = Path(__file__).with_name("stock_shout_reference.py")


def stock(**request):
    if importlib.util.find_spec("QuantLib") is None:
        pytest.skip("install QuantLib==1.40 for the isolated stock reference")
    output = subprocess.run([sys.executable, str(REFERENCE)],
                            input=json.dumps(request, default=lambda value: value.isoformat()),
                            capture_output=True, text=True, check=True)
    return json.loads(output.stdout)


def setup(**settings):
    config = SMALL | settings
    mesh = shout_mesh(.2, **config)
    config.pop("x_grid", None)
    return mesh, config


def scalar(value):
    return torch.tensor(value, dtype=DTYPE, requires_grad=True)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_original_shout_price_and_fixed_mesh_vega(scheme, mode):
    mesh, config = setup(scheme=scheme)
    vol = scalar(.2)
    actual = fd_shout(vol, mesh=mesh, mode=mode, **config)
    gradient, = torch.autograd.grad(actual, vol)
    expected = stock(**SMALL, scheme=scheme)
    assert actual.item() == pytest.approx(expected["price"], abs=5e-10)
    h = 1e-5
    with torch.no_grad():
        plus = fd_shout(vol+h, mesh=mesh, mode=mode, **config)
        minus = fd_shout(vol-h, mesh=mesh, mode=mode, **config)
    torch.testing.assert_close(gradient, (plus-minus)/(2*h), rtol=2e-6, atol=2e-6)


MARKETS = [
    dict(exercise="european", option_type="call"),
    dict(exercise="american", option_type="put", rate=-.01),
    dict(exercise="bermudan", exercise_dates=(MID, END), option_type="put"),
    dict(exercise="american", cash_dividends=((MID, 1.25),)),
    dict(exercise="bermudan", exercise_dates=(MID, END), option_type="put",
         cash_dividends=((MID, .8), (END, .5))),
    dict(discount_curve=((START, 1.), (MID, .982), (END, .95)),
         dividend_curve=((START, 1.), (MID, .996), (END, .983))),
]


@pytest.mark.parametrize("market", MARKETS)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_shout_original_financial_conventions(market, mode):
    mesh, config = setup(**market)
    vol = scalar(.2)
    actual = fd_shout(vol, mesh=mesh, mode=mode, **config)
    gradient, = torch.autograd.grad(actual, vol)
    expected = stock(**SMALL, **market)
    assert actual.item() == pytest.approx(expected["price"], abs=5e-10)
    h = 1e-5
    with torch.no_grad():
        plus = fd_shout(vol+h, mesh=mesh, mode=mode, **config)
        minus = fd_shout(vol-h, mesh=mesh, mode=mode, **config)
    torch.testing.assert_close(gradient, (plus-minus)/(2*h), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_shout_damping_and_market_vjps(mode):
    mesh, config = setup(damping_steps=2, cash_dividends=((MID, 1.25),))
    values = tuple(scalar(value) for value in (.2, 100., .03, .01))

    def price(values, mode=mode, kernels="auto"):
        return fd_shout(values[0], spot=values[1], rate=values[2], dividend_yield=values[3],
                         mesh=mesh, mode=mode, kernels=kernels, **config)

    actual = price(values)
    gradients = torch.autograd.grad(actual, values)
    expected = stock(**SMALL, damping_steps=2, cash_dividends=((MID, 1.25),))
    assert actual.item() == pytest.approx(expected["price"], abs=5e-10)
    scalar_price = price(values, "full", "scalar")
    scalar_gradients = torch.autograd.grad(scalar_price, values)
    torch.testing.assert_close(actual, scalar_price, rtol=1e-11, atol=1e-11)
    for i, (gradient, scalar_gradient) in enumerate(zip(gradients, scalar_gradients)):
        torch.testing.assert_close(gradient, scalar_gradient, rtol=1e-8, atol=1e-8)
        h = 1e-5
        with torch.no_grad():
            plus = price(tuple(value + (h if j == i else 0.) for j, value in enumerate(values)))
            minus = price(tuple(value - (h if j == i else 0.) for j, value in enumerate(values)))
        torch.testing.assert_close(gradient, (plus-minus)/(2*h), rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_shout_dated_curve_nodes_remain_connected(mode):
    risk_base, yield_base = (.982, .95), (.996, .983)
    mesh, config = setup(discount_curve=((START, 1.), (MID, risk_base[0]), (END, risk_base[1])),
                         dividend_curve=((START, 1.), (MID, yield_base[0]), (END, yield_base[1])),
                         cash_dividends=((MID, .8),))
    nodes = torch.tensor((*risk_base, *yield_base), dtype=DTYPE, requires_grad=True)
    vol = scalar(.2)

    def price(nodes):
        settings = config | dict(discount_curve=((START, 1.), (MID, nodes[0]), (END, nodes[1])),
                                 dividend_curve=((START, 1.), (MID, nodes[2]), (END, nodes[3])))
        return fd_shout(vol, mesh=mesh, mode=mode, **settings)

    value = price(nodes)
    gradients, = torch.autograd.grad(value, nodes)
    h = 1e-6
    with torch.no_grad():
        for index in range(4):
            direction = torch.eye(4, dtype=DTYPE)[index]
            difference = (price(nodes+h*direction)-price(nodes-h*direction))/(2*h)
            torch.testing.assert_close(gradients[index], difference, rtol=2e-5, atol=3e-6)
    assert bool((gradients.abs() > 1e-4).all())


def test_shout_full_higher_derivatives_and_adjoint_boundary():
    mesh, config = setup(time_steps=4, x_grid=15)
    vol = scalar(.2)
    price = fd_shout(vol, mesh=mesh, mode="full", **config)
    first, = torch.autograd.grad(price, vol, create_graph=True)
    second, = torch.autograd.grad(first, vol)
    h = 1e-5
    slopes = []
    for shifted in (.2+h, .2-h):
        point = scalar(shifted)
        value = fd_shout(point, mesh=mesh, mode="full", **config)
        slopes.append(torch.autograd.grad(value, point)[0])
    torch.testing.assert_close(second, (slopes[0]-slopes[1])/(2*h), rtol=2e-5, atol=3e-5)
    adjoint = fd_shout(vol, mesh=mesh, mode="adjoint", **config)
    with pytest.raises(RuntimeError, match="first|higher|create_graph"):
        torch.autograd.grad(adjoint, vol, create_graph=True)


def test_shout_pricing_layer_fresh_graphs_and_market_override():
    mesh, config = setup(time_steps=4, x_grid=15)
    option = VanillaOption(100., END, exercise=Exercise.american())
    layer = PricingLayer(option, EquityMarket(), FdShoutEngine(mesh=mesh, time_steps=4, mode="adjoint"))
    vol, rate = scalar(.2), scalar(.04)
    for _ in range(2):
        actual = layer(vol, rate=rate)
        expected = fd_shout(vol, mesh=mesh, mode="adjoint", rate=rate, **config)
        torch.testing.assert_close(actual, expected)
        gradients = torch.autograd.grad(actual, (vol, rate))
        assert all(torch.isfinite(value) for value in gradients)


def test_shout_validates_tensor_domain_and_passive_mesh():
    with pytest.raises((TypeError, ValueError)):
        shout_mesh(scalar(.2))
    with pytest.raises((TypeError, ValueError)):
        ShoutMesh([1., 2., 2., 3., 4.])
    mesh, config = setup(time_steps=2, x_grid=9)
    for vol in (scalar(-.2), torch.tensor(.2, dtype=torch.float32)):
        with pytest.raises((ValueError, RuntimeError)):
            fd_shout(vol, mesh=mesh, **config)


def test_shout_active_flat_rate_shadowed_by_curve_raises():
    mesh, config = setup(time_steps=2, x_grid=9)
    with pytest.raises((ValueError, RuntimeError), match="rate|curve"):
        fd_shout(scalar(.2), mesh=mesh, rate=scalar(.03),
                  discount_curve=((START, 1.), (END, .95)), **config)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_shout_actual360_prices_and_volatility_sensitivities(mode):
    settings = dict(day_counter="actual360", option_type="put", exercise="bermudan",
                    exercise_dates=(MID, END), cash_dividends=((MID, .8),),
                    discount_curve=((START, 1.), (MID, .98), (END, .95)))
    mesh, config = setup(**settings)
    vol = scalar(.2)
    actual = fd_shout(vol, mesh=mesh, mode=mode, **config)
    gradient, = torch.autograd.grad(actual, vol)
    expected = stock(**SMALL, **settings)
    assert actual.item() == pytest.approx(expected["price"], abs=5e-10)
    h = 1e-5
    with torch.no_grad():
        plus = fd_shout(vol+h, mesh=mesh, mode=mode, **config)
        minus = fd_shout(vol-h, mesh=mesh, mode=mode, **config)
    torch.testing.assert_close(gradient, (plus-minus)/(2*h), rtol=2e-6, atol=2e-6)


def test_shout_layer_propagates_actual360():
    mesh, config = setup(day_counter="actual360", time_steps=4, x_grid=15)
    layer = PricingLayer(VanillaOption(100., END, exercise=Exercise.american()), EquityMarket(),
        FdShoutEngine(mesh=mesh, time_steps=4, day_counter="actual360", mode="adjoint"))
    vol = scalar(.2)
    actual = layer(vol)
    expected = fd_shout(vol, mesh=mesh, mode="adjoint", **config)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
