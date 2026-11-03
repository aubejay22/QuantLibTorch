"""Numerical and neural-VJP checks for the QuantLib-backed PDE experiment."""

import math

import pytest
import torch

from quantlibtorch.pricing import european_call, log_moneyness_grid


def black_scholes_call(spot, strike, rate, dividend, maturity, sigma):
    root_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * sigma**2) * maturity) / (
        sigma * root_t
    )
    d2 = d1 - sigma * root_t
    normal = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return spot * math.exp(-dividend * maturity) * normal(d1) - strike * math.exp(
        -rate * maturity
    ) * normal(d2)


def test_constant_vol_converges_to_black_scholes():
    parameters = dict(spot=100.0, strike=100.0, rate=0.03, dividend=0.01, maturity=1.0)
    reference = black_scholes_call(**parameters, sigma=0.2)
    prices = []
    for steps, nodes in ((20, 39), (120, 159)):
        sigma = torch.tensor(0.2, dtype=torch.float64, requires_grad=(nodes == 159))
        price = european_call(
            sigma.expand(steps, nodes),
            **parameters,
            log_moneyness_min=-1.2,
            log_moneyness_max=1.2,
            mode="adjoint",
        )
        prices.append(price.item())
        if sigma.requires_grad:
            vega = torch.autograd.grad(price, sigma)[0].item()
    coarse_error, fine_error = (abs(price - reference) for price in prices)
    assert fine_error < 0.04
    assert fine_error < 0.2 * coarse_error
    d1 = (0.03 - 0.01 + 0.5 * 0.2**2) / 0.2
    analytic_vega = 100.0 * math.exp(-0.01 - 0.5 * d1**2) / math.sqrt(2 * math.pi)
    assert vega == pytest.approx(analytic_vega, rel=0.003)


def test_full_and_adjoint_match_price_and_coefficient_vjp():
    x = log_moneyness_grid(17, log_moneyness_min=-0.6, log_moneyness_max=0.6)
    t = torch.arange(5, dtype=torch.float64).reshape(-1, 1) / 5
    coefficients = (0.19 + 0.035 * torch.sin(3 * x) + 0.025 * t).requires_grad_()
    options = dict(spot=103.0, log_moneyness_min=-0.6, log_moneyness_max=0.6)
    full = european_call(coefficients, mode="full", **options)
    adjoint = european_call(coefficients, mode="adjoint", **options)
    full_grad = torch.autograd.grad(full, coefficients)[0]
    adjoint_grad = torch.autograd.grad(adjoint, coefficients)[0]
    torch.testing.assert_close(full, adjoint, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(full_grad, adjoint_grad, rtol=1e-9, atol=1e-10)
    assert torch.isfinite(full_grad).all()
    assert (full_grad.abs().sum(dim=1) > 1e-6).all()
    # A nonuniform field direction exercises every time slice and spatial row.
    direction = torch.cos(torch.arange(coefficients.numel(), dtype=torch.float64)).reshape_as(
        coefficients
    )
    epsilon = 1e-5
    with torch.no_grad():
        finite_difference = (
            european_call(coefficients + epsilon * direction, mode="adjoint", **options)
            - european_call(coefficients - epsilon * direction, mode="adjoint", **options)
        ) / (2 * epsilon)
    torch.testing.assert_close(
        (adjoint_grad * direction).sum(), finite_difference, rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_neural_network_parameter_vjp_matches_directional_finite_difference(mode):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1), torch.nn.Sigmoid()
    ).double()
    parameters = tuple(model.parameters())
    with torch.no_grad():
        for i, parameter in enumerate(parameters):
            parameter.copy_(
                torch.linspace(-0.3 + 0.05 * i, 0.4, parameter.numel()).reshape_as(parameter)
            )
    times = torch.arange(4, dtype=torch.float64) / 4
    space = log_moneyness_grid(13, log_moneyness_min=-0.6, log_moneyness_max=0.6)
    t, x = torch.meshgrid(times, space, indexing="ij")
    inputs = torch.stack((t, x), dim=-1)

    def price():
        local_vol = 0.1 + 0.3 * model(inputs).squeeze(-1)
        return european_call(
            local_vol,
            mode=mode,
            log_moneyness_min=-0.6,
            log_moneyness_max=0.6,
        )

    # A nonlinear pricing loss verifies composition beyond an isolated scalar Greek.
    loss = (price() - 9.0).square()
    gradients = torch.autograd.grad(loss, parameters)
    directions = tuple(
        torch.sin(torch.arange(p.numel(), dtype=torch.float64) + i + 1).reshape_as(p)
        for i, p in enumerate(parameters)
    )
    analytic = sum((g * d).sum() for g, d in zip(gradients, directions))
    baseline = tuple(p.detach().clone() for p in parameters)
    epsilon = 1e-5
    with torch.no_grad():
        for p, initial, d in zip(parameters, baseline, directions):
            p.copy_(initial + epsilon * d)
        loss_plus = (price() - 9.0).square()
        for p, initial, d in zip(parameters, baseline, directions):
            p.copy_(initial - epsilon * d)
        loss_minus = (price() - 9.0).square()
        for p, initial in zip(parameters, baseline):
            p.copy_(initial)
    finite_difference = (loss_plus - loss_minus) / (2 * epsilon)
    assert abs(analytic.item()) > 1e-4
    torch.testing.assert_close(analytic, finite_difference, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "coefficients, error",
    [
        (torch.ones(2, 4), "CPU float64"),
        (torch.ones(4, dtype=torch.float64), "shape"),
        (torch.ones(0, 4, dtype=torch.float64), "at least"),
        (torch.ones(2, 1, dtype=torch.float64), "at least"),
        (torch.zeros(2, 4, dtype=torch.float64), "strictly positive"),
        (torch.full((2, 4), float("nan"), dtype=torch.float64), "finite"),
    ],
)
def test_rejects_invalid_coefficients(coefficients, error):
    with pytest.raises(ValueError, match=error):
        european_call(coefficients)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"spot": 0}, "strictly positive"),
        ({"maturity": 0}, "strictly positive"),
        ({"rate": float("inf")}, "finite"),
        ({"log_moneyness_min": 1.0, "log_moneyness_max": -1.0}, "less than"),
        ({"spot": 1000}, "inside"),
        ({"mode": "invalid"}, "mode"),
    ],
)
def test_rejects_invalid_financial_inputs(kwargs, error):
    with pytest.raises(ValueError, match=error):
        european_call(torch.full((2, 4), 0.2, dtype=torch.float64), **kwargs)


def test_differentiable_financial_tensor_is_not_silently_detached():
    with pytest.raises(TypeError, match="fixed real number"):
        european_call(
            torch.full((2, 4), 0.2, dtype=torch.float64),
            spot=torch.tensor(100.0, requires_grad=True),
        )


def test_drift_dominated_centered_mesh_is_rejected():
    # Without this guard the centered scheme produces a negative call price.
    with pytest.raises(ValueError, match="not monotone"):
        european_call(torch.full((20, 31), 0.001, dtype=torch.float64), rate=.01, dividend=.1)


def test_large_negative_discount_time_step_is_rejected():
    with pytest.raises(ValueError, match="increase time steps"):
        european_call(torch.full((1, 31), 0.2, dtype=torch.float64), rate=-2.)
