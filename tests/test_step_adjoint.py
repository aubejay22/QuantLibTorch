"""Discrete step VJPs against dense equations and the original schemes."""
from datetime import date
import math

import pytest
import torch

from quantlibtorch import fd_vanilla
from test_operator_adjoint import matrix as reflected_matrix

DTYPE = torch.float64


def dense_step(bands, state, dt, theta, kind):
    n = state.numel()
    compact = torch.diag(bands[1])+torch.diag(bands[0, 1:], -1)+torch.diag(bands[2, :-1], 1)
    beta = 0. if kind == 2 else theta*dt
    alpha = dt-beta if kind == 0 else (theta*dt if kind == 2 else 0.)
    return torch.linalg.solve(torch.eye(n, dtype=DTYPE)-beta*compact,
                              state+alpha*(reflected_matrix(bands) @ state))


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('theta', [0., .5, 1.])
@pytest.mark.parametrize('n', [2, 9])
def test_step_all_bands_and_state_vjp_with_reflected_endpoints(kind, theta, n):
    # Nonzero outer bands test the different endpoint rules of apply and solve.
    base = (torch.sin(torch.arange(6*n, dtype=DTYPE)).reshape(3, 2*n).requires_grad_(),
            torch.cos(torch.arange(2*n, dtype=DTYPE)).requires_grad_())
    bands, state = base[0][:, ::2], base[1][::2]
    before = [t.detach().clone() for t in base]
    actual = torch.ops.quantlibtorch._step_probe(bands, state, .04, theta, kind, True)
    original = torch.ops.quantlibtorch._step_probe(bands, state, .04, theta, kind, False)
    expected = dense_step(bands, state, .04, theta, kind)
    for value in (actual, original):
        torch.testing.assert_close(value, expected, rtol=2e-12, atol=2e-12)
    weights = torch.linspace(-.4, .8, 2*n, dtype=DTYPE)[::2]
    references = torch.autograd.grad(expected, base, weights)
    for output in (actual, original):
        for gradient, reference in zip(torch.autograd.grad(output, base, weights), references):
            torch.testing.assert_close(gradient, reference, rtol=2e-11, atol=2e-12)
    for value, initial in zip(base, before):
        torch.testing.assert_close(value, initial, rtol=0, atol=0)


@pytest.mark.parametrize('kind', [0, 1, 2])
def test_step_negative_views_differences_and_first_order_contract(kind):
    raw = (torch.cos(torch.arange(9, dtype=DTYPE)).reshape(3, 3).requires_grad_(),
           torch.tensor([.2, -.4, .1], dtype=DTYPE, requires_grad=True))
    bands, state = (torch._neg_view(t) for t in raw)
    function = lambda b, u: torch.ops.quantlibtorch._step_probe(b, u, .07, .6, kind, True)
    actual = function(bands, state)
    expected = dense_step(bands, state, .07, .6, kind)
    weights = torch._neg_view(torch.tensor([.3, -.7, .2], dtype=DTYPE))
    for gradient, reference in zip(torch.autograd.grad(actual, raw, weights),
                                   torch.autograd.grad(expected, raw, weights)):
        torch.testing.assert_close(gradient, reference, rtol=2e-11, atol=2e-12)
    assert torch.autograd.gradcheck(function, (bands, state))
    with pytest.raises(RuntimeError, match='first derivatives only'):
        torch.autograd.grad(function(bands, state).sum(), raw, create_graph=True)


@pytest.mark.parametrize('scheme', ['douglas', 'implicit_euler', 'crank_nicolson'])
@pytest.mark.parametrize('exercise', ['european', 'american', 'bermudan'])
def test_step_engine_nonuniform_curves_coincident_events_vjp(scheme, exercise):
    field = (.21+.02*torch.sin(torch.arange(15, dtype=DTYPE))).reshape(3, 5).requires_grad_()
    start, mid, end = date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)
    args = dict(times=[0., .37, 1.], log_spots=[math.log(s) for s in (40., 70., 100., 150., 250.)],
                valuation_date=start, maturity_date=end, exercise=exercise, option_type='put', spot=96.,
                cash_dividends=[(mid, 1.7)], discount_curve=[(start, 1.), (mid, .98), (end, .965)],
                dividend_curve=[(start, 1.), (mid, .997), (end, .985)],
                time_steps=6, grid_points=21, scheme=scheme)
    if exercise == 'bermudan': args['exercise_dates'] = [mid, end]
    value = fd_vanilla(field, mode='adjoint', **args)
    reference = fd_vanilla(field, mode='full', kernels='vectorized', **args)
    gradient, = torch.autograd.grad(value, field)
    expected, = torch.autograd.grad(reference, field)
    torch.testing.assert_close(value, reference, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(gradient, expected, rtol=2e-9, atol=1e-9)
    direction = torch.cos(torch.arange(15, dtype=DTYPE)).reshape_as(field)
    with torch.no_grad():
        difference = (fd_vanilla(field+1e-5*direction, mode='adjoint', **args)
                      -fd_vanilla(field-1e-5*direction, mode='adjoint', **args))/(2e-5)
    torch.testing.assert_close((gradient*direction).sum(), difference, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('scheme', ['douglas', 'implicit_euler', 'crank_nicolson'])
def test_step_dispatch_activates_only_for_the_supported_adjoint_path(scheme):
    field = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    for mode, kernels, expected in [('adjoint', 'auto', True), ('full', 'auto', False),
                                    ('adjoint', 'scalar', False)]:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            value = fd_vanilla(field, time_steps=3, grid_points=11, scheme=scheme, mode=mode, kernels=kernels)
            torch.autograd.grad(value, field)
        counts = {item.key: item.count for item in profile.key_averages()}
        assert (counts.get('qlt::step_adjoint_forward', 0) > 0) == expected
        assert (counts.get('qlt::step_adjoint_backward', 0) > 0) == expected


@pytest.mark.parametrize('kind', [0, 1, 2, 3])
def test_step_fallback_preserves_boundary_callbacks_active_settings_and_subclasses(kind):
    vol = torch.tensor([[.21], [.27]], dtype=DTYPE, requires_grad=True)
    state = torch.cos(torch.arange(9, dtype=DTYPE)).requires_grad_()
    settings = torch.tensor([.04, .6, .8], dtype=DTYPE, requires_grad=(kind == 2))
    inputs = (vol, state, settings) if kind == 2 else (vol, state)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = torch.ops.quantlibtorch._guarded_step_probe(vol, state, settings, kind, True)
        ga = torch.autograd.grad(actual, inputs, state.detach())
    counts = {event.key: event.count for event in profile.key_averages()}
    assert counts.get('qlt::step_adjoint_forward', 0) == (1 if kind == 0 else 0)
    expected = torch.ops.quantlibtorch._guarded_step_probe(vol, state, settings, kind, False)
    ge = torch.autograd.grad(expected, inputs, state.detach())
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    for gradient, reference in zip(ga, ge):
        torch.testing.assert_close(gradient, reference, rtol=2e-10, atol=2e-11)
    if kind == 1:
        assert actual[0].item() == .125
    if kind == 2:
        assert torch.autograd.gradcheck(
            lambda v, u, s: torch.ops.quantlibtorch._guarded_step_probe(v, u, s, kind, True),
            (vol, state, settings))
