"""Native step arithmetic against forced original QuantLib first-order paths."""
import os

import pytest
import torch

import quantlibtorch

DTYPE = torch.float64


def inputs(n=7, scale=1.):
    bands = (scale*.4*torch.sin(torch.arange(3*n, dtype=DTYPE))).reshape(3, n).requires_grad_()
    state = torch.cos(torch.arange(n, dtype=DTYPE)).requires_grad_()
    return bands, state


def step(bands, state, dt, theta, kind, native=True):
    return torch.ops.quantlibtorch._step_probe(bands, state, dt, theta, kind, True, native)


def compare(xs, dt, theta, kind):
    initial, versions = [t.detach().clone() for t in xs], [t._version for t in xs]
    actual = step(*xs, dt, theta, kind)
    expected = step(*xs, dt, theta, kind, native=False)
    assert torch.equal(actual.view(torch.int64), expected.view(torch.int64))
    weights = torch.sin(torch.arange(actual.numel(), dtype=DTYPE))
    torch.testing.assert_close(torch.autograd.grad(actual, xs, weights),
                               torch.autograd.grad(expected, xs, weights), rtol=2e-12, atol=1e-160)
    for value, before, version in zip(xs, initial, versions):
        torch.testing.assert_close(value, before, rtol=0, atol=0)
        assert value._version == version


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('theta', [-.3, 0., .5, 1., 1.3])
@pytest.mark.parametrize('n', [2, 7])
def test_native_step_matches_original_forward_bits_and_vjps(kind, theta, n):
    compare(inputs(n), .04, theta, kind)


@pytest.mark.parametrize('kind', [0, 1, 2])
@pytest.mark.parametrize('scale', [1e-150, 1., 1e150])
def test_native_step_extreme_coefficient_and_step_scales(kind, scale):
    compare(inputs(9, scale), .04/scale, .6, kind)


@pytest.mark.parametrize('kind', [0, 1, 2])
def test_native_step_expanded_and_shared_inputs(kind):
    base = torch.tensor(.13, dtype=DTYPE, requires_grad=True)
    bands, state = base.expand(3, 5), base.expand(5)
    actual, expected = step(bands, state, .03, .7, kind), step(bands, state, .03, .7, kind, False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), base),
                               torch.autograd.grad(expected.square().sum(), base), rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize('bad', ['inverse_zero', 'first_denominator_zero', 'later_pivot_zero', 'nonfinite_state'])
def test_native_step_preserves_original_forward_failures(bad):
    bands = torch.zeros(3, 2, dtype=DTYPE)
    state = torch.ones(2, dtype=DTYPE)
    if bad == 'inverse_zero': bands[1, 0] = float('inf')
    elif bad == 'first_denominator_zero': bands[1, 0] = 1.
    elif bad == 'later_pivot_zero': bands[1, 1] = 1.
    else: state[0] = float('nan')
    message = 'division by zero' if bad in ('inverse_zero', 'later_pivot_zero') else 'non-finite step solution'
    for native in (False, True):
        with pytest.raises(RuntimeError, match=message):
            step(bands, state, 1., 1., 1, native)


@pytest.mark.parametrize('kind', [0, 1, 2])
def test_native_step_preserves_nonfinite_incoming_vjp_rejection(kind):
    xs = inputs()
    for native in (False, True):
        value = step(*xs, .04, .6, kind, native)
        with pytest.raises(RuntimeError, match='non-finite step adjoint'):
            torch.autograd.grad(value, xs, torch.full_like(value, float('inf')))


def test_native_step_dispatch_and_first_order_contract():
    xs = inputs()
    enabled = os.environ.get('QLT_STEP_NATIVE_ARITHMETIC', '1') == '1'
    for native in (False, True):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            value = step(*xs, .04, .6, 0, native)
            torch.autograd.grad(value.sum(), xs)
        names = {e.key for e in prof.key_averages()}
        for scope in ('qlt::step_arithmetic_native_forward', 'qlt::step_arithmetic_native_backward'):
            assert (scope in names) == (enabled and native)
    with pytest.raises(RuntimeError, match='first derivatives only'):
        torch.autograd.grad(step(*xs, .04, .6, 0).sum(), xs, create_graph=True)


def test_native_step_keeps_tiny_nonzero_transpose_pivot_rejection():
    # The differently ordered scaled forward recurrence has a finite pivot
    # (about 1.11e-16). The compact transpose recurrence has a nonzero pivot
    # about 2.47e-32, inside QuantLib's close(pivot, 0) threshold.
    bands = torch.tensor([[0., 90.90909090909092], [0., 90.90909090909089],
                          [2.0185873175002846e-14, 0.]], dtype=DTYPE, requires_grad=True)
    state = torch.ones(2, dtype=DTYPE, requires_grad=True)
    for native in (False, True):
        value = step(bands, state, 1., .011, 1, native)
        assert torch.isfinite(value).all()
        with pytest.raises(RuntimeError, match='division by zero'):
            torch.autograd.grad(value.sum(), (bands, state))
