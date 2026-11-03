"""Financial validation with an isolated original QuantLib Heston operator."""
from datetime import date
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from quantlibtorch import fd_latent, state_grid


DT = torch.float64
AXES = [[math.log(s) for s in [40., 60., 80., 100., 120., 150., 200.]],
        [0., .02, .05, .1, .2]]
PARAMS = [1.5, .04, .3, -.6, math.log(100.), .045]


def heston_fields(parameters, axes=AXES):
    kappa, theta, sigma, rho = parameters[:4]
    v = state_grid(axes)[:, 1]
    root = v.sqrt()
    zeros = torch.zeros_like(root)
    drift = (kappa*(theta-v))[None, :, None]
    factor = torch.stack((torch.stack((root, zeros), -1),
                          torch.stack((sigma*rho*root, sigma*(1-rho.square()).sqrt()*root), -1)), -2)[None]
    return drift, factor


def stock(request):
    completed = subprocess.run([sys.executable, str(Path(__file__).with_name("stock_latent_reference.py"))],
                               input=json.dumps(request), text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


@pytest.mark.parametrize("scheme", ["douglas", "hundsdorfer", "modified_craig_sneyd", "craig_sneyd"])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_heston_price_parameters_and_initial_state_match_original_quantlib(scheme, mode):
    p = torch.tensor(PARAMS, dtype=DT, requires_grad=True)
    b, l = heston_fields(p)
    result = fd_latent(b, l, p[4:], axes=AXES, time_steps=5, scheme=scheme, mode=mode)
    grad, = torch.autograd.grad(result, p)
    oracle = stock(dict(axes=AXES, parameters=PARAMS, time_steps=5, scheme=scheme, bump=1e-5))
    torch.testing.assert_close(result, torch.tensor(oracle["price"], dtype=DT), rtol=2e-12, atol=3e-12)
    torch.testing.assert_close(grad, torch.tensor(oracle["gradients"], dtype=DT), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("exercise", ["european", "american", "bermudan"])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_dated_curves_exercise_and_state_restore(exercise, mode):
    risk = [(date(2025, 1, 1), 1.), (date(2025, 7, 1), .975), (date(2026, 1, 1), .94)]
    div = [(date(2025, 1, 1), 1.), (date(2025, 5, 1), .996), (date(2026, 1, 1), .98)]
    dates = [date(2025, 5, 1), date(2025, 9, 1), date(2026, 1, 1)] if exercise == "bermudan" else []
    p = torch.tensor(PARAMS, dtype=DT, requires_grad=True)
    b, l = heston_fields(p)
    before = torch.ops.quantlibtorch._evaluation_date()
    result = fd_latent(b, l, p[4:], axes=AXES, time_steps=5, mode=mode, exercise=exercise,
                       option_type="put", discount_curve=risk, dividend_curve=div, exercise_dates=dates)
    assert torch.ops.quantlibtorch._evaluation_date() == before
    oracle = stock(dict(axes=AXES, parameters=PARAMS, exercise=exercise, option_type="put", time_steps=5,
                         discount_curve=[(str(d), v) for d, v in risk], dividend_curve=[(str(d), v) for d, v in div],
                         exercise_dates=list(map(str, dates)), bump=1e-5))
    grad, = torch.autograd.grad(result, p)
    torch.testing.assert_close(result, torch.tensor(oracle["price"], dtype=DT), rtol=2e-12, atol=5e-12)
    torch.testing.assert_close(grad, torch.tensor(oracle["gradients"], dtype=DT), rtol=4e-5, atol=4e-6)


def joint_problem(m=1):
    axes = [[math.log(s) for s in [50., 75., 95., 110., 140., 200.]]] + [[-1., -.3, .2, 1.]]*m
    x = state_grid(axes)
    features = torch.stack([torch.cat((torch.full((len(x), 1), t, dtype=DT), x), -1) for t in [0., .4, 1.]])
    features = torch.cat((features[..., :1], features[..., 1:2]-math.log(100.), features[..., 2:]), -1)
    torch.manual_seed(123+m)
    net = torch.nn.Sequential(torch.nn.Linear(m+2, 4), torch.nn.Tanh(), torch.nn.Linear(4, m+(m+1)**2)).double()
    initial = torch.tensor([math.log(101.)]+[.05]*m, dtype=DT, requires_grad=True)
    return axes, features, net, initial


def neural_fields(net, features, m, params=None):
    raw = net(features) if params is None else torch.func.functional_call(net, params, features)
    drift = .08*torch.tanh(raw[..., :m])
    factor = .15*torch.eye(m+1, dtype=DT)+.08*torch.tanh(raw[..., m:].reshape(*raw.shape[:-1], m+1, m+1))
    return drift, factor


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_joint_neural_parameters_and_initial_latents_with_central_differences(m, mode):
    axes, features, net, initial = joint_problem(m)
    params = dict(net.named_parameters())
    directions = {k: .1*torch.cos(torch.arange(v.numel(), dtype=DT)).reshape_as(v) for k, v in params.items()}
    def loss(p, z):
        b, l = neural_fields(net, features, m, p)
        value = fd_latent(b, l, z, axes=axes, times=[0., .4, 1.], time_steps=3, mode=mode)
        return (value-8.).square()
    value = loss(params, initial)
    grads = torch.autograd.grad(value, (*params.values(), initial))
    assert all(torch.isfinite(g).all() for g in grads)
    # Latent drift, diffusion network and initial latent coordinates all affect price.
    assert grads[-2][:m].abs().sum() > 1e-7
    assert grads[-2][m:].abs().sum() > 1e-7
    assert grads[-1][1:].abs().min() > 1e-7
    h = 1e-5
    with torch.no_grad():
        bump = (loss({k: v+h*directions[k] for k, v in params.items()}, initial)-
                loss({k: v-h*directions[k] for k, v in params.items()}, initial))/(2*h)
        zdir = torch.tensor([.01]+[.3]*m, dtype=DT)
        zbump = (loss(params, initial+h*zdir)-loss(params, initial-h*zdir))/(2*h)
    projection = sum((g*v).sum() for g, v in zip(grads[:-1], directions.values()))
    torch.testing.assert_close(projection, bump, rtol=4e-5, atol=3e-6)
    torch.testing.assert_close((grads[-1]*zdir).sum(), zbump, rtol=4e-5, atol=3e-6)


def test_full_latent_curvature_and_mode_equality():
    axes, features, net, initial = joint_problem(1)
    values, gradients = [], []
    for mode in ["full", "adjoint"]:
        b, l = neural_fields(net, features, 1)
        value = fd_latent(b, l, initial, axes=axes, times=[0., .4, 1.], time_steps=3, mode=mode)
        gradient = torch.autograd.grad(value, initial, create_graph=mode == "full")[0]
        values.append(value.detach())
        gradients.append(gradient.detach())
        if mode == "full":
            hessian_row = torch.autograd.grad(gradient[1], initial)[0]
            assert torch.isfinite(hessian_row).all() and hessian_row.abs().sum() > 0
    torch.testing.assert_close(values[0], values[1], rtol=3e-12, atol=3e-12)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize("bad", ["outside", "endpoint", "shape", "axes", "times", "nan", "dates", "curve", "exercise", "type"])
def test_financial_input_contracts(bad):
    p = torch.tensor(PARAMS, dtype=DT)
    b, l = heston_fields(p)
    z = p[4:]
    kw = dict(axes=AXES, time_steps=2)
    if bad == "outside": z = z+10
    elif bad == "endpoint": z = torch.tensor([AXES[0][-1], PARAMS[-1]], dtype=DT)
    elif bad == "shape": b = b[..., :0]
    elif bad == "axes": kw["axes"] = [AXES[0], [0., .04, .2]]
    elif bad == "times": kw["times"] = [-1.]
    elif bad == "nan": z = z*float("nan")
    elif bad == "dates": kw["maturity_date"] = date(2024, 1, 1)
    elif bad == "curve": kw["discount_curve"] = [(date(2025, 1, 1), 1.), (date(2026, 1, 1), -1.)]
    elif bad == "exercise": kw["exercise"] = "bermudan"
    elif bad == "type": kw["option_type"] = "straddle"
    before = torch.ops.quantlibtorch._evaluation_date()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        fd_latent(b, l, z, **kw)
    assert torch.ops.quantlibtorch._evaluation_date() == before
