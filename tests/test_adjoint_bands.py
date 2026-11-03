"""Original matrix inputs preserve the existing first-order step contract."""
from contextlib import nullcontext
import math
import os

import pytest
import torch

from quantlibtorch import fd_vanilla
from test_step_adjoint import dense_step
from test_step_arithmetic import inputs

DTYPE = torch.float64


def enabled():
    return (os.environ.get('QLT_ADJOINT_ASSEMBLED_BANDS', '1') == '1'
            and os.environ.get('QLT_STEP_NATIVE_ARITHMETIC', '1') == '1')


def stored():
    return enabled() and all(os.environ.get(flag, '1') == '1'
        for flag in ('QLT_ASSEMBLY_ADJOINT', 'QLT_NATIVE_ADJOINT'))


def consumed():
    return stored() and os.environ.get('QLT_STEP_ADJOINT', '1') == '1'


def counts(profile):
    return {event.key: event.count for event in profile.key_averages()}


def step(bands, state, dt=.04, theta=.6, kind=0, assembled=True, native=True):
    return torch.ops.quantlibtorch._step_probe(bands, state, dt, theta, kind,
                                              True, native, 0, False, assembled)


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('theta', [-.3, 0., .5, 1., 1.3])
@pytest.mark.parametrize('n', [2, 9])
def test_original_forward_bits_and_all_band_state_vjps(kind, theta, n):
    xs = inputs(n)
    before, versions = tuple(t.clone() for t in xs), tuple(t._version for t in xs)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = step(*xs, theta=theta, kind=kind)
    original = step(*xs, theta=theta, kind=kind, assembled=False, native=False)
    assert torch.equal(actual.view(torch.int64), original.view(torch.int64))
    assert bool(counts(profile).get('qlt::assembled_step_adjoint_forward', 0)) == enabled()
    weights = torch.sin(torch.arange(n, dtype=DTYPE))
    torch.testing.assert_close(torch.autograd.grad(actual, xs, weights),
                               torch.autograd.grad(original, xs, weights), rtol=2e-12, atol=1e-160)
    torch.testing.assert_close(xs, before, rtol=0, atol=0)
    assert tuple(t._version for t in xs) == versions


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('view', ['strided', 'negative', 'shared', 'expanded'])
def test_dense_reference_vjps_for_views_and_aliases(kind, view):
    leaves = inputs(18 if view == 'strided' else 9)
    xs = leaves
    if view == 'strided': xs = (leaves[0][:, ::2], leaves[1][::2])
    elif view == 'negative': xs = tuple(torch._neg_view(t) for t in leaves)
    elif view == 'shared':
        leaves = (torch.linspace(.01, .1, 9, dtype=DTYPE, requires_grad=True),)
        xs = (leaves[0].expand(3, 9), leaves[0])
    else:
        leaves = (torch.linspace(.01, .1, 4, dtype=DTYPE, requires_grad=True),)
        xs = (leaves[0][:3, None].expand(3, 9), leaves[0][3:].expand(9))
    before, versions = tuple(t.clone() for t in leaves), tuple(t._version for t in leaves)
    actual, expected = step(*xs, kind=kind), dense_step(*xs, .04, .6, kind)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    weights = torch._neg_view(torch.cos(torch.arange(actual.numel(), dtype=DTYPE)))
    torch.testing.assert_close(torch.autograd.grad(actual, leaves, weights),
                               torch.autograd.grad(expected, leaves, weights), rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(leaves, before, rtol=0, atol=0)
    assert tuple(t._version for t in leaves) == versions


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('scale', [1e-150, 1., 1e150])
def test_extreme_coefficient_and_time_step_scales(kind, scale):
    xs = inputs(9, scale)
    actual = step(*xs, dt=.04/scale, kind=kind)
    expected = step(*xs, dt=.04/scale, kind=kind, native=False)
    assert torch.equal(actual.view(torch.int64), expected.view(torch.int64))
    weights = torch.cos(xs[1])
    torch.testing.assert_close(torch.autograd.grad(actual, xs, weights),
                               torch.autograd.grad(expected, xs, weights), rtol=2e-12, atol=1e-160)


@pytest.mark.parametrize('kind', [0, 1, 2])
def test_independent_differences_and_first_order_only(kind):
    xs = inputs(5)
    assert torch.autograd.gradcheck(lambda b, s: step(b, s, kind=kind), xs, fast_mode=True)
    with pytest.raises(RuntimeError, match='first derivatives only'):
        torch.autograd.grad(step(*xs, kind=kind).sum(), xs, create_graph=True)


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('pattern', ['strided', 'negative', 'expanded'])
def test_incoming_vjp_views(kind, pattern):
    xs = inputs(9)
    weights = torch.sin(torch.arange(18, dtype=DTYPE))[::2]
    if pattern == 'negative': weights = torch._neg_view(weights)
    elif pattern == 'expanded': weights = torch.tensor(.3, dtype=DTYPE).expand(9)
    actual, expected = step(*xs, kind=kind), dense_step(*xs, .04, .6, kind)
    torch.testing.assert_close(torch.autograd.grad(actual, xs, weights),
                               torch.autograd.grad(expected, xs, weights), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('bad', ['inverse_zero', 'first_denominator_zero', 'later_pivot_zero', 'nonfinite_state'])
def test_forward_failures_keep_original_pivot_and_finite_checks(bad):
    bands, state = torch.zeros(3, 2, dtype=DTYPE), torch.ones(2, dtype=DTYPE)
    if bad == 'inverse_zero': bands[1, 0] = float('inf')
    elif bad == 'first_denominator_zero': bands[1, 0] = 1.
    elif bad == 'later_pivot_zero': bands[1, 1] = 1.
    else: state[0] = float('nan')
    message = 'division by zero' if bad in ('inverse_zero', 'later_pivot_zero') else 'non-finite step solution'
    for native in (False, True):
        with pytest.raises(RuntimeError, match=message):
            step(bands, state, dt=1., theta=1., kind=1, native=native)


def test_tiny_nonzero_transpose_pivot_still_rejects_after_finite_forward():
    bands = torch.tensor([[0., 90.90909090909092], [0., 90.90909090909089],
                          [2.0185873175002846e-14, 0.]], dtype=DTYPE, requires_grad=True)
    state = torch.ones(2, dtype=DTYPE, requires_grad=True)
    for native in (False, True):
        value = step(bands, state, dt=1., theta=.011, kind=1, native=native)
        assert torch.isfinite(value).all()
        with pytest.raises(RuntimeError, match='division by zero'):
            torch.autograd.grad(value.sum(), (bands, state))


@pytest.mark.parametrize('kind', [0, 1, 2])
def test_nonfinite_incoming_vjp_is_rejected(kind):
    xs = inputs()
    for native in (False, True):
        value = step(*xs, kind=kind, native=native)
        with pytest.raises(RuntimeError, match='non-finite step adjoint'):
            torch.autograd.grad(value, xs, torch.full_like(value, float('inf')))


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('active', [0, 1])
def test_partial_input_activity(kind, active):
    xs = tuple(t.detach().requires_grad_(i == active) for i, t in enumerate(inputs()))
    actual, expected = step(*xs, kind=kind), dense_step(*xs, .04, .6, kind)
    torch.testing.assert_close(torch.autograd.grad(actual.sin().sum(), xs[active]),
                               torch.autograd.grad(expected.sin().sum(), xs[active]), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('context', [torch.no_grad, torch.inference_mode])
def test_graph_disabled_preparation(kind, context):
    with context():
        xs = inputs()
        actual = step(*xs, kind=kind)
        expected = step(*xs, kind=kind, native=False)
        assert torch.equal(actual.view(torch.int64), expected.view(torch.int64))
        assert not actual.requires_grad


@pytest.mark.parametrize('transition', [0, 1, 2], ids=['exception', 'other_kernel', 'no_grad'])
def test_rebinding_copy_invalidation_and_reversed_backward_after_owner_destruction(transition):
    leaves = (torch.tensor(.21, dtype=DTYPE, requires_grad=True),
              torch.tensor(.27, dtype=DTYPE, requires_grad=True),
              torch.linspace(.5, 1.3, 11, dtype=DTYPE, requires_grad=True))
    before, versions = tuple(t.clone() for t in leaves), tuple(t._version for t in leaves)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = torch.ops.quantlibtorch._assembled_owner_sequence(*leaves, True, transition, True)
    expected = torch.ops.quantlibtorch._assembled_owner_sequence(*leaves, False, transition, True)
    events = counts(profile)
    extra = int(transition == 2)  # a fresh no-grad matrix is valid for inference
    assert events.get('qlt::assembled_step_adjoint_forward', 0) == ((3+extra) if consumed() else 0)
    assert events.get('qlt::assembled_bands_store', 0) == ((2+extra) if stored() else 0)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(actual[2], actual[0], rtol=0, atol=0)
    for i in (3, 2, 0, 1):
        ga = torch.autograd.grad(actual[i].sin().sum(), leaves, allow_unused=True, retain_graph=True)
        ge = torch.autograd.grad(expected[i].sin().sum(), leaves, allow_unused=True, retain_graph=True)
        ga = tuple(torch.zeros_like(t) if g is None else g for g, t in zip(ga, leaves))
        ge = tuple(torch.zeros_like(t) if g is None else g for g, t in zip(ge, leaves))
        torch.testing.assert_close(ga, ge, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(leaves, before, rtol=0, atol=0)
    assert tuple(t._version for t in leaves) == versions


@pytest.mark.parametrize('mode', ['full', 'adjoint', 'no_grad', 'inference_mode'])
def test_financial_dispatch_and_fresh_graphs(mode):
    recording = mode in ('full', 'adjoint')
    context = nullcontext if recording else (torch.no_grad if mode == 'no_grad' else torch.inference_mode)
    for nodes, shift in ((9, 0.), (17, .02), (9, 0.)):
        raw = (torch.tensor([[.21, .24], [.23, .25]], dtype=DTYPE)+shift).requires_grad_()
        settings = dict(times=[0., .7], log_spots=[math.log(70.), math.log(145.)],
                        time_steps=3, grid_points=nodes, mode=mode if recording else 'full')
        with context():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                actual = fd_vanilla(raw, **settings)
            expected = fd_vanilla(raw, kernels='scalar', **settings)
        events = counts(profile)
        assert bool(events.get('qlt::assembled_step_adjoint_forward', 0)) == (consumed() and mode != 'full')
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
        if recording:
            ga, = torch.autograd.grad(actual, raw, create_graph=mode == 'full')
            ge, = torch.autograd.grad(expected, raw, create_graph=mode == 'full')
            torch.testing.assert_close(ga, ge, rtol=2e-9, atol=2e-9)
            if mode == 'full':
                torch.testing.assert_close(torch.autograd.grad(ga, raw, torch.cos(raw)),
                                           torch.autograd.grad(ge, raw, torch.cos(raw)), rtol=2e-8, atol=2e-8)
        else:
            assert not actual.requires_grad


def test_pricing_exception_restores_state_before_a_fresh_adjoint_price():
    previous = torch.ops.quantlibtorch._evaluation_date()
    with pytest.raises(RuntimeError):
        fd_vanilla(torch.tensor(1e300, dtype=DTYPE, requires_grad=True), time_steps=3, grid_points=11, mode='adjoint')
    assert torch.ops.quantlibtorch._evaluation_date() == previous
    sigma = torch.tensor(.23, dtype=DTYPE, requires_grad=True)
    actual = fd_vanilla(sigma, time_steps=3, grid_points=11, mode='adjoint')
    expected = fd_vanilla(sigma, time_steps=3, grid_points=11, mode='full', kernels='scalar')
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(torch.autograd.grad(actual, sigma), torch.autograd.grad(expected, sigma),
                               rtol=2e-9, atol=2e-9)
