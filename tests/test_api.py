"""Public object API composition, graph lifetime, and unsupported contracts."""
from dataclasses import FrozenInstanceError
from datetime import date
import math

import pytest
import torch

from quantlibtorch.api import (
    AnalyticEuropeanEngine, EquityMarket, Exercise, FdLatentEngine,
    FdVanillaEngine, PricingLayer, VanillaOption,
)
from quantlibtorch import fd_latent, fd_vanilla, state_grid


DT = torch.float64
REF, MID, END = date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
@pytest.mark.parametrize("exercise", [Exercise.european(), Exercise.american(),
                                     Exercise.bermudan([MID, END])])
def test_vanilla_layer_retains_exercise_curves_dividends_and_gradients(mode, exercise):
    option = VanillaOption(103., END, "put", exercise)
    market = EquityMarket(spot=97., discount_curve=[(REF, 1.), (MID, .98), (END, .95)],
                          cash_dividends=[(MID, 1.2)])
    engine = FdVanillaEngine(time_steps=5, grid_points=15, damping_steps=1, mode=mode)
    layer = PricingLayer(option, market, engine)
    sigma = torch.tensor(.23, dtype=DT, requires_grad=True)
    actual = layer(sigma)
    expected = fd_vanilla(sigma, strike=103., spot=97., option_type="put",
                          exercise=exercise.style, exercise_dates=exercise.dates,
                          discount_curve=market.discount_curve, cash_dividends=market.cash_dividends,
                          time_steps=5, grid_points=15, damping_steps=1, mode=mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, sigma)[0],
                               torch.autograd.grad(expected, sigma)[0], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_registered_neural_model_supports_repeated_optimizer_steps_and_independent_bumps(mode):
    class Volatility(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw = torch.nn.Parameter(torch.tensor(-.4, dtype=DT))

        def forward(self):
            return .1 + .2 * torch.sigmoid(self.raw)

    layer = PricingLayer(VanillaOption(100., END), EquityMarket(),
                         FdVanillaEngine(time_steps=4, grid_points=13, damping_steps=0, mode=mode),
                         model=Volatility())
    assert dict(layer.named_parameters()).keys() == {"model.raw"}
    assert layer.state_dict().keys() == {"model.raw"}
    optimizer = torch.optim.SGD(layer.parameters(), lr=.002)
    for _ in range(2):
        optimizer.zero_grad()
        loss = (layer() - 9.).square()
        loss.backward()
        ad = layer.model.raw.grad.clone()
        raw = layer.model.raw.detach().clone()
        with torch.no_grad():
            layer.model.raw.copy_(raw + 1e-5)
            plus = (layer() - 9.).square()
            layer.model.raw.copy_(raw - 1e-5)
            minus = (layer() - 9.).square()
            layer.model.raw.copy_(raw)
        torch.testing.assert_close(ad, (plus-minus)/2e-5, rtol=1e-6, atol=1e-7)
        optimizer.step()


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_latent_engine_keeps_parameter_and_initial_state_connections(mode):
    axes = [[math.log(s) for s in (50., 80., 100., 130., 200.)], [-1., -.3, .2, 1.]]
    nodes = state_grid(axes)
    p = torch.tensor([.03, .21, math.log(101.), .05], dtype=DT, requires_grad=True)
    drift = (p[0] * nodes[:, 1])[None, :, None]
    factor = (p[1] * torch.eye(2, dtype=DT)).expand(1, len(nodes), 2, 2)
    engine = FdLatentEngine(axes, time_steps=3, mode=mode)
    layer = PricingLayer(VanillaOption(100., END, "put", Exercise.american()), EquityMarket(), engine)
    actual = layer(drift, factor, p[2:])
    expected = fd_latent(drift, factor, p[2:], axes=axes, time_steps=3, mode=mode,
                         option_type="put", exercise="american")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, p, retain_graph=True)[0],
                               torch.autograd.grad(expected, p)[0], rtol=0, atol=0)


def test_analytic_layer_tensors_and_independent_black_delta():
    sigma = torch.tensor(.2, dtype=DT, requires_grad=True)
    spot = torch.tensor(100., dtype=DT, requires_grad=True)
    rate = torch.tensor(.03, dtype=DT, requires_grad=True)
    layer = PricingLayer(VanillaOption(100., END), EquityMarket(), AnalyticEuropeanEngine())
    price = layer(sigma, spot=spot, rate=rate)
    delta, rho, vega = torch.autograd.grad(price, (spot, rate, sigma), create_graph=True)
    d1 = (.03-.01+.2**2/2)/.2
    n1 = .5 * (1+math.erf(d1/math.sqrt(2)))
    n2 = .5 * (1+math.erf((d1-.2)/math.sqrt(2)))
    assert delta.item() == pytest.approx(math.exp(-.01)*n1, rel=1e-12)
    assert rho.item() == pytest.approx(100*math.exp(-.03)*n2, rel=1e-12)
    assert vega.item() == pytest.approx(100*math.exp(-.01-d1*d1/2)/math.sqrt(2*math.pi), rel=1e-12)
    gamma, = torch.autograd.grad(delta, spot)
    assert gamma.item() == pytest.approx(math.exp(-.01-d1*d1/2)/(.2*100*math.sqrt(2*math.pi)), rel=1e-12)


def test_market_defers_engine_specific_dividend_admissibility():
    # Discounted cash, not its undiscounted sum, controls admissibility in the
    # analytic dividend engine. The generic market must preserve that contract.
    market = EquityMarket(spot=100., rate=.5, cash_dividends=[(MID, 101.)])
    option = VanillaOption(100., END)
    sigma = torch.tensor(.2, dtype=DT)
    price = PricingLayer(option, market, AnalyticEuropeanEngine())(sigma)
    assert torch.isfinite(price) and price > 0
    with pytest.raises(ValueError, match="total less than spot"):
        PricingLayer(option, market, FdVanillaEngine())(sigma)


def test_configuration_snapshots_mutable_sequences():
    curve = [[REF, 1.], [END, .97]]
    dates = [MID, END]
    times = [0., 1.]
    market = EquityMarket(discount_curve=curve)
    exercise = Exercise.bermudan(dates)
    engine = FdVanillaEngine(times=times)
    curve[1][1] = .5
    dates.clear()
    times[1] = 3.
    assert market.discount_curve == ((REF, 1.), (END, .97))
    assert exercise.dates == (MID, END)
    assert engine.times == (0., 1.)
    with pytest.raises(FrozenInstanceError):
        market.spot = 99.


@pytest.mark.parametrize("make", [
    lambda: EquityMarket(spot=torch.tensor(100., requires_grad=True)),
    lambda: EquityMarket(discount_curve=[(REF, 1.), (END, torch.tensor(.97))]),
    lambda: VanillaOption(torch.tensor(100.), END),
    lambda: FdVanillaEngine(times=torch.tensor([0., 1.])),
    lambda: FdLatentEngine([torch.arange(4.), [0., 1., 2., 3.]]),
])
def test_configuration_rejects_tensor_values_instead_of_detaching(make):
    with pytest.raises(TypeError, match="tensor"):
        make()


def test_unsupported_derivatives_and_features_fail_explicitly():
    sigma = torch.tensor(.2, dtype=DT, requires_grad=True)
    market = EquityMarket()
    fd = PricingLayer(VanillaOption(100., END), market, FdVanillaEngine(time_steps=3, grid_points=11))
    with pytest.raises(TypeError, match="spot.*tensor"):
        fd(sigma, spot=torch.tensor(100., dtype=DT, requires_grad=True))
    with pytest.raises(TypeError, match="unsupported pricing overrides: typo"):
        fd(sigma, typo=1)
    analytic = PricingLayer(VanillaOption(100., END, exercise=Exercise.american()), market,
                            AnalyticEuropeanEngine())
    with pytest.raises(ValueError, match="European exercise only"):
        analytic(sigma)
    with pytest.raises(TypeError, match="engine inputs must be tensors"):
        fd(.2)
    axes = [[1., 2., 3., 4.], [-1., 0., 1., 2.]]
    latent = PricingLayer(VanillaOption(100., END), EquityMarket(cash_dividends=[(MID, 1.)]),
                          FdLatentEngine(axes))
    with pytest.raises(ValueError, match="does not support cash dividends"):
        latent(sigma, sigma, sigma)
    with pytest.raises(TypeError, match="unsupported pricing overrides: spot"):
        latent(sigma, sigma, sigma, spot=100.)


@pytest.mark.parametrize("make", [
    lambda: Exercise("invalid"),
    lambda: Exercise.bermudan([]),
    lambda: Exercise.bermudan([END, MID]),
    lambda: VanillaOption(100., END, exercise=Exercise.bermudan([MID])),
    lambda: EquityMarket(cash_dividends=[(MID, -1.)]),
    lambda: FdVanillaEngine(time_steps=0),
    lambda: FdVanillaEngine(scheme="invalid"),
    lambda: FdVanillaEngine(cash_dividend_model="invalid"),
    lambda: FdVanillaEngine(mode="adjoint", kernels="pcr"),
    lambda: PricingLayer(VanillaOption(100., REF), EquityMarket(), AnalyticEuropeanEngine()),
])
def test_invalid_configuration_is_reported_before_native_dispatch(make):
    with pytest.raises(ValueError):
        make()


def test_custom_engine_protocol_and_tuple_model_outputs():
    class TwoInputs(torch.nn.Module):
        def forward(self, value):
            return value.square(), value.exp()

    class CustomEngine:
        def price(self, instrument, market, first, second, **overrides):
            return first + second

    layer = PricingLayer(VanillaOption(100., END), EquityMarket(), CustomEngine(), model=TwoInputs())
    x = torch.tensor(.4, dtype=DT, requires_grad=True)
    actual = layer(x)
    grad, = torch.autograd.grad(actual, x)
    torch.testing.assert_close(grad, 2*x+x.exp())
