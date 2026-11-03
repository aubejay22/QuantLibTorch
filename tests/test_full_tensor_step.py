"""Full-mode tensor step composition against original schemes and dense AD."""
from contextlib import nullcontext
from datetime import date
import math
import os

import pytest
import torch
from torch.autograd import forward_ad

from quantlibtorch import fd_vanilla
from test_backward_duals import compare_incoming_duals
from test_step_adjoint import dense_step

DTYPE = torch.float64


def inputs(n=9, scale=1.):
    return ((.4*scale*torch.sin(torch.arange(3*n, dtype=DTYPE))).reshape(3, n).requires_grad_(),
            torch.cos(torch.arange(n, dtype=DTYPE)).requires_grad_())


def step(bands, state, dt=.04, theta=.6, kind=0, tensor=True, mode=4):
    return torch.ops.quantlibtorch._step_probe(bands, state, dt, theta, kind, False, True, mode, tensor)


def enabled():
    return os.environ.get('QLT_FULL_TENSOR_STEP', '1') == '1' and os.environ.get('QLT_OPERATOR_ARITHMETIC', '1') == '1'


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('theta', [-.3, 0., .5, 1., 1.3])
@pytest.mark.parametrize('n', [8, 11])
def test_full_step_values_and_all_vjps_against_original_and_dense(kind, theta, n):
    xs = inputs(n)
    actual = step(*xs, theta=theta, kind=kind)
    previous = step(*xs, theta=theta, kind=kind, tensor=False)
    original = step(*xs, theta=theta, kind=kind, tensor=False, mode=0)
    dense = dense_step(*xs, .04, theta, kind)
    # The composition keeps forward arithmetic and the selected solver intact.
    assert torch.equal(actual.view(torch.int64), previous.view(torch.int64))
    for expected in (original, dense):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    weights = torch.sin(torch.arange(n, dtype=DTYPE))
    ga = torch.autograd.grad(actual, xs, weights)
    for expected in (previous, original, dense):
        torch.testing.assert_close(ga, torch.autograd.grad(expected, xs, weights), rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative', 'shared'])
def test_full_step_views_aliases_third_derivatives_and_no_mutation(kind, view):
    leaves = inputs(18 if view == 'strided' else 9)
    xs = leaves
    if view == 'strided': xs = (leaves[0][:, ::2], leaves[1][::2])
    elif view == 'negative': xs = tuple(torch._neg_view(t) for t in leaves)
    elif view == 'shared':
        leaves = (torch.linspace(.01, .1, 9, dtype=DTYPE, requires_grad=True),)
        xs = (leaves[0].expand(3, 9), leaves[0])
    before, versions = [t.clone() for t in leaves], [t._version for t in leaves]
    actual, reference = step(*xs, kind=kind), dense_step(*xs, .04, .6, kind)
    la, le = actual.sin().sum(), reference.sin().sum()
    for order in range(3):
        ga = torch.autograd.grad(la, leaves, create_graph=order < 2)
        ge = torch.autograd.grad(le, leaves, create_graph=order < 2)
        torch.testing.assert_close(ga, ge, rtol=2e-9, atol=2e-10)
        la = sum((g*torch.cos(t)).sum() for g, t in zip(ga, leaves))
        le = sum((g*torch.cos(t)).sum() for g, t in zip(ge, leaves))
    for t, saved, version in zip(leaves, before, versions):
        torch.testing.assert_close(t, saved, rtol=0, atol=0)
        assert t._version == version


@pytest.mark.parametrize('kind', [0, 1, 2])
def test_full_step_first_and_second_differences(kind):
    f = lambda b, r: step(b, r, kind=kind)
    assert torch.autograd.gradcheck(f, inputs(), fast_mode=True)
    assert torch.autograd.gradgradcheck(f, inputs(), fast_mode=True)


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('no_grad', [False, True])
def test_full_step_forward_duals_retain_original_dispatch(kind, no_grad):
    xs = inputs()
    directions = tuple(.2*torch.sin(t.detach()+.3) for t in xs)
    primal, tangent = torch.autograd.functional.jvp(lambda *args: dense_step(*args, .04, .6, kind), xs, directions)
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            duals = tuple(forward_ad.make_dual(t, d) for t, d in zip(xs, directions))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                actual, derivative = forward_ad.unpack_dual(step(*duals, kind=kind))
    assert not any(e.key == 'qlt::full_tensor_step' for e in profile.key_averages())
    torch.testing.assert_close(actual, primal, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(derivative, tangent, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('create_graph', [False, True])
def test_full_step_incoming_dual_weights(kind, create_graph):
    xs = inputs()
    compare_incoming_duals(xs, step(*xs, kind=kind), dense_step(*xs, .04, .6, kind), create_graph=create_graph)


@pytest.mark.parametrize('transform', ['grad', 'jvp'])
def test_full_step_function_transforms_keep_existing_tensor_path(transform):
    xs = inputs()
    if transform == 'grad':
        run = lambda f: torch.func.grad(lambda *args: f(*args).square().sum(), argnums=(0, 1))(*xs)
    else:
        run = lambda f: torch.func.jvp(f, xs, tuple(torch.ones_like(t)*.2 for t in xs))
    torch.testing.assert_close(run(step), run(lambda *args: dense_step(*args, .04, .6, 0)), rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize('scale', [1e-150, 1e150])
def test_full_step_extreme_coefficient_scales(scale):
    xs = inputs(scale=scale)
    actual = step(*xs, dt=.04/scale)
    original = step(*xs, dt=.04/scale, tensor=False, mode=0)
    torch.testing.assert_close(actual, original, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), xs),
                               torch.autograd.grad(original.square().sum(), xs), rtol=2e-10, atol=1e-160)


def test_douglas_reuses_one_application_with_the_same_solve_checks():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        step(*inputs())
    counts = {e.key: e.count for e in profile.key_averages()}
    assert counts.get('qlt::full_tensor_step', 0) == int(enabled())
    prepared = (enabled() and os.environ.get('QLT_STEP_PREPARATION', '1') == '1'
                and os.environ.get('QLT_ARITHMETIC_NATIVE_FORWARD', '1') == '1')
    assert counts.get('qlt::step_preparation_forward', 0) == int(prepared)
    assert counts.get('qlt::apply_arithmetic_forward', 0) == (0 if prepared else (1 if enabled() else 2))
    assert counts.get('qlt::auto_domain_native', 0) == 1
    assert counts.get('qlt::auto_residual_native', 0) == 1


@pytest.mark.parametrize('pattern', ['small', 'nondominant', 'tiny_diagonal'])
def test_rejected_full_step_keeps_original_values_vjps_and_input_state(pattern):
    if pattern == 'small': xs = inputs(7)
    else:
        bands = torch.zeros(3, 9, dtype=DTYPE)
        if pattern == 'nondominant': bands[0] = 1.6; bands[2] = 1.6
        else: bands[1, 0] = 1.-1e-14
        xs = bands.requires_grad_(), torch.cos(torch.arange(9, dtype=DTYPE)).requires_grad_()
    before = tuple(t.clone() for t in xs)
    actual, reference = step(*xs, dt=1., theta=1.), step(*xs, dt=1., theta=1., tensor=False)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), xs), torch.autograd.grad(reference.sum(), xs), rtol=0, atol=0)
    torch.testing.assert_close(xs, before, rtol=0, atol=0)


@pytest.mark.parametrize('kind', [0, 1, 2, 3])
def test_full_step_guards_preserve_boundaries_active_settings_and_subclasses(kind):
    vol = torch.tensor([[.21], [.27]], dtype=DTYPE, requires_grad=True)
    state = torch.cos(torch.arange(9, dtype=DTYPE)).requires_grad_()
    settings = torch.tensor([.04, .6, .8], dtype=DTYPE, requires_grad=kind == 2)
    xs = (vol, state, settings) if kind == 2 else (vol, state)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = torch.ops.quantlibtorch._guarded_step_probe(vol, state, settings, kind, False, 4)
    assert any(e.key == 'qlt::full_tensor_step' for e in profile.key_averages()) == (enabled() and kind == 0)
    reference = torch.ops.quantlibtorch._guarded_step_probe(vol, state, settings, kind, False, 2)
    torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(torch.autograd.grad(actual.sin().sum(), xs),
                               torch.autograd.grad(reference.sin().sum(), xs), rtol=2e-10, atol=2e-11)
    if kind == 1: assert actual[0].item() == .125


@pytest.mark.parametrize('scheme', ['douglas', 'crank_nicolson', 'implicit_euler'])
@pytest.mark.parametrize('exercise', ['european', 'american', 'bermudan'])
def test_financial_full_step_curves_coincident_events_and_curvature(scheme, exercise):
    field = torch.tensor([[.21, .24], [.26, .22]], dtype=DTYPE, requires_grad=True)
    start, mid, end = date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1)
    options = dict(times=[0., .7], log_spots=[math.log(70.), math.log(145.)], time_steps=3, grid_points=11,
        scheme=scheme, exercise=exercise, option_type='put', spot=96., valuation_date=start, maturity_date=end,
        cash_dividends=[(mid, 1.7)], discount_curve=[(start, 1.), (mid, .98), (end, .965)],
        dividend_curve=[(start, 1.), (mid, .997), (end, .985)])
    if exercise == 'bermudan': options['exercise_dates'] = [mid, end]
    actual, reference = fd_vanilla(field, **options), fd_vanilla(field, kernels='scalar', **options)
    torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-12)
    ga, = torch.autograd.grad(actual, field, create_graph=True)
    ge, = torch.autograd.grad(reference, field, create_graph=True)
    torch.testing.assert_close(ga, ge, rtol=2e-9, atol=2e-9)
    direction = torch.cos(field)
    torch.testing.assert_close(torch.autograd.grad(ga, field, direction), torch.autograd.grad(ge, field, direction),
                               rtol=2e-8, atol=2e-8)
