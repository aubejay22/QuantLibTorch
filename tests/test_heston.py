"""Original Heston financial engine, independent fixed-mesh VJPs and API use."""
from datetime import date
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from quantlibtorch import (
    EquityMarket, Exercise, FdHestonEngine, HestonMesh, HestonModel,
    PricingLayer, VanillaOption, fd_heston, heston_mesh,
)


DTYPE = torch.float64
START, MID, END = date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)
NAMES = ("v0", "kappa", "theta", "sigma", "rho")
BASE = dict(zip(NAMES, (.045, 1.5, .04, .3, -.6)))
SMALL = dict(time_steps=8, x_grid=15, v_grid=9)
SCHEMES = ("douglas", "craig_sneyd", "modified_craig_sneyd", "hundsdorfer")
REFERENCE = Path(__file__).with_name("stock_heston_reference.py")


def stock(**request):
    if importlib.util.find_spec("QuantLib") is None:
        pytest.skip("install QuantLib==1.40 for the isolated stock reference")
    output = subprocess.run([sys.executable, str(REFERENCE)],
                            input=json.dumps(request, default=lambda value: value.isoformat()),
                            capture_output=True, text=True, check=True)
    return json.loads(output.stdout)


def parameters():
    return tuple(torch.tensor(value, dtype=DTYPE, requires_grad=True) for value in BASE.values())


def setup(**settings):
    config = SMALL | settings
    mesh = heston_mesh(**BASE, **config)
    config.pop("x_grid", None)
    config.pop("v_grid", None)
    return mesh, config


def check_gradients(actual, expected, names=NAMES):
    for value, name in zip(actual, names):
        assert value.item() == pytest.approx(expected["gradients"][name], rel=2e-5, abs=5e-6)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_original_heston_engine_price_and_all_parameters(scheme, mode):
    mesh, config = setup(scheme=scheme)
    values = parameters()
    actual = fd_heston(*values, mesh=mesh, mode=mode, **config)
    gradients = torch.autograd.grad(actual, values)
    expected = stock(**BASE, **SMALL, scheme=scheme, bump=1e-5)
    assert actual.item() == pytest.approx(expected["price"], abs=2e-9)
    assert expected["fixed_price"] == pytest.approx(expected["price"], abs=1e-12)
    for axis, reference in zip((mesh.log_spots, mesh.variances), expected["axes"]):
        torch.testing.assert_close(torch.tensor(axis, dtype=DTYPE),
                                   torch.tensor(reference, dtype=DTYPE), rtol=1e-11, atol=1e-11)
    check_gradients(gradients, expected)
    assert all(abs(value.item()) > 1e-4 for value in gradients)


MARKETS = [
    dict(option_type="put", exercise="american", rate=.06),
    dict(option_type="put", exercise="bermudan", exercise_dates=(MID, END),
         cash_dividends=((MID, 1.25),)),
    dict(option_type="put", exercise="american", cash_dividends=((MID, .8), (END, .5))),
    dict(option_type="call", discount_curve=((START, 1.), (MID, .982), (END, .95)),
         dividend_curve=((START, 1.), (MID, .996), (END, .983))),
]


@pytest.mark.parametrize("market", MARKETS)
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_heston_exercise_dividends_and_curves(market, mode):
    mesh, config = setup(**market)
    values = parameters()
    actual = fd_heston(*values, mesh=mesh, mode=mode, **config)
    gradients = torch.autograd.grad(actual, values)
    expected = stock(**BASE, **SMALL, **market, bump=1e-5)
    assert actual.item() == pytest.approx(expected["price"], abs=3e-9)
    check_gradients(gradients, expected)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_heston_spot_flat_rate_and_yield_sensitivities(mode):
    mesh, config = setup(option_type="put", exercise="american")
    values = parameters()
    market = tuple(torch.tensor(value, dtype=DTYPE, requires_grad=True) for value in (100., .03, .01))
    actual = fd_heston(*values, mesh=mesh, mode=mode, **config,
                       **dict(zip(("spot", "rate", "dividend_yield"), market)))
    gradients = torch.autograd.grad(actual, (*values, *market))
    expected = stock(**BASE, **SMALL, option_type="put", exercise="american", bump=1e-5)
    assert actual.item() == pytest.approx(expected["price"], abs=3e-9)
    check_gradients(gradients, expected, (*NAMES, "spot", "rate", "dividend_yield"))


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_heston_dated_curve_node_sensitivities(mode):
    discount = ((START, 1.), (MID, .98), (END, .95))
    dividend = ((START, 1.), (MID, .995), (END, .985))
    mesh, config = setup(discount_curve=discount, dividend_curve=dividend)
    risk = torch.tensor([.98, .95], dtype=DTYPE, requires_grad=True)
    div = torch.tensor([.995, .985], dtype=DTYPE, requires_grad=True)
    config["discount_curve"] = ((START, 1.), (MID, risk[0]), (END, risk[1]))
    config["dividend_curve"] = ((START, 1.), (MID, div[0]), (END, div[1]))
    actual = fd_heston(*parameters(), mesh=mesh, mode=mode, **config)
    gradients = torch.autograd.grad(actual, (risk, div))
    names = [f"{curve}:{i}" for curve in ("discount_curve", "dividend_curve") for i in (1, 2)]
    expected = stock(**BASE, **SMALL, discount_curve=discount, dividend_curve=dividend,
                     bump=1e-6, gradient_names=names)
    check_gradients(torch.cat(gradients), expected, names)


def test_full_heston_directional_curvature_matches_gradient_difference():
    mesh, config = setup(time_steps=3, x_grid=9, v_grid=5)
    values = torch.tensor(list(BASE.values()), dtype=DTYPE, requires_grad=True)
    direction = torch.tensor([.01, .2, .01, .05, .1], dtype=DTYPE)

    def gradient(point, create_graph=False):
        price = fd_heston(*point.unbind(), mesh=mesh, mode="full", **config)
        return torch.autograd.grad(price, point, create_graph=create_graph)[0]

    first = gradient(values, True)
    second = torch.autograd.grad((first * direction).sum(), values)[0]
    h = 2e-4
    plus = (values.detach() + h*direction).requires_grad_()
    minus = (values.detach() - h*direction).requires_grad_()
    expected = (gradient(plus) - gradient(minus))/(2*h)
    torch.testing.assert_close(second, expected, rtol=2e-4, atol=2e-5)


def test_heston_adjoint_rejects_higher_derivatives():
    mesh, config = setup(time_steps=2, x_grid=9, v_grid=5)
    values = parameters()
    price = fd_heston(*values, mesh=mesh, mode="adjoint", **config)
    with pytest.raises(RuntimeError, match="first.order|first derivatives|create_graph|higher"):
        torch.autograd.grad(price, values, create_graph=True)


def test_heston_original_damping_in_full_mode():
    mesh, config = setup(damping_steps=2)
    values = parameters()
    price = fd_heston(*values, mesh=mesh, mode="full", **config)
    gradients = torch.autograd.grad(price, values)
    expected = stock(**BASE, **SMALL, damping_steps=2, bump=1e-5)
    assert price.item() == pytest.approx(expected["price"], abs=3e-8)
    check_gradients(gradients, expected)


def test_heston_adjoint_damping_rejected_explicitly():
    mesh, config = setup(damping_steps=2)
    with pytest.raises((ValueError, RuntimeError), match="damping|Krylov"):
        fd_heston(*parameters(), mesh=mesh, mode="adjoint", **config)


def test_reusable_heston_model_layer_and_fresh_graphs():
    mesh, config = setup(time_steps=3, x_grid=9, v_grid=5)
    model = HestonModel(**BASE)
    layer = PricingLayer(VanillaOption(100., END), EquityMarket(),
                         FdHestonEngine(mesh=mesh, time_steps=3, mode="adjoint"), model=model)
    assert len(list(layer.parameters())) > 0
    for _ in range(2):
        price = layer()
        direct = fd_heston(*model(), mesh=mesh, mode="adjoint", **config)
        torch.testing.assert_close(price, direct)
        gradients = torch.autograd.grad(price, tuple(layer.parameters()))
        assert all(torch.isfinite(value).all() for value in gradients)
    assert layer.state_dict()


@pytest.mark.parametrize("name,value", [("v0", -.01), ("kappa", -1.), ("theta", -.01),
                                        ("sigma", -.2), ("rho", 1.1)])
def test_heston_rejects_invalid_model_parameters(name, value):
    mesh, config = setup()
    values = BASE | {name: value}
    with pytest.raises((ValueError, RuntimeError)):
        fd_heston(**values, mesh=mesh, **config)


def test_heston_mesh_requires_passive_geometry():
    with pytest.raises((TypeError, ValueError)):
        heston_mesh(**(BASE | {"v0": torch.tensor(.04, dtype=DTYPE, requires_grad=True)}), **SMALL)
    with pytest.raises((TypeError, ValueError)):
        HestonMesh([1., 2., 2.], [0., .1, .2])


def test_heston_active_rate_shadowed_by_curve_raises():
    mesh, config = setup()
    with pytest.raises((ValueError, RuntimeError), match="rate|curve"):
        fd_heston(**BASE, mesh=mesh, **config,
                   rate=torch.tensor(.03, dtype=DTYPE, requires_grad=True),
                   discount_curve=((START, 1.), (END, .95)))


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_heston_custom_nonuniform_mesh_preserves_original_operator(mode):
    mesh = HestonMesh([3.5, 3.9, 4.2, 4.45, 4.6, 4.8, 5.1, 5.5, 5.9],
                      [0., .012, .03, .065, .13, .25])
    values = parameters()
    actual = fd_heston(*values, mesh=mesh, time_steps=6, mode=mode)
    gradients = torch.autograd.grad(actual, values)
    expected = stock(**BASE, axes=[mesh.log_spots, mesh.variances], time_steps=6, bump=1e-5)
    assert actual.item() == pytest.approx(expected["fixed_price"], abs=2e-9)
    check_gradients(gradients, expected)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_heston_actual360_prices_and_parameter_sensitivities(mode):
    settings = dict(day_counter="actual360", option_type="put", exercise="bermudan",
                    exercise_dates=(MID, END), cash_dividends=((MID, .8),),
                    discount_curve=((START, 1.), (MID, .98), (END, .95)))
    mesh, config = setup(**settings)
    values = parameters()
    actual = fd_heston(*values, mesh=mesh, mode=mode, **config)
    gradients = torch.autograd.grad(actual, values)
    expected = stock(**BASE, **SMALL, **settings, bump=1e-5)
    assert actual.item() == pytest.approx(expected["price"], abs=3e-9)
    check_gradients(gradients, expected)


@pytest.mark.parametrize("regime", [
    dict(v0=.025, kappa=.6, theta=.03, sigma=.55, rho=-.8),
    dict(v0=.04, kappa=2., theta=.05, sigma=.2, rho=.5),
], ids=["violated_feller", "positive_correlation"])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_heston_parameter_regimes_preserve_original_boundaries(regime, mode):
    mesh = heston_mesh(**regime, **SMALL)
    values = tuple(torch.tensor(regime[name], dtype=DTYPE, requires_grad=True) for name in NAMES)
    actual = fd_heston(*values, mesh=mesh, mode=mode, time_steps=SMALL["time_steps"])
    gradients = torch.autograd.grad(actual, values)
    expected = stock(**regime, **SMALL, bump=1e-5)
    assert actual.item() == pytest.approx(expected["price"], abs=3e-9)
    check_gradients(gradients, expected)


def test_heston_layer_propagates_actual360():
    mesh, config = setup(day_counter="actual360", time_steps=3, x_grid=9, v_grid=5)
    model = HestonModel(**BASE)
    layer = PricingLayer(VanillaOption(100., END), EquityMarket(),
        FdHestonEngine(mesh=mesh, time_steps=3, day_counter="actual360", mode="adjoint"), model=model)
    actual = layer()
    expected = fd_heston(*model(), mesh=mesh, mode="adjoint", **config)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
