"""Value semantics and independent VJPs for persistent QuantLib array views."""
from contextlib import nullcontext
import operator

import pytest
import torch

import quantlibtorch  # register native verification operations


def arithmetic_reference(a, b, c):
    rows = []
    for op in (operator.add, operator.sub, operator.mul, operator.truediv):
        rows.extend([op(a, b)]*4 + [op(a, c)]*2 + [op(c, a)]*2
                    + [op(a, b), op(a, c), op(a, a)])
    return torch.stack(rows)


@pytest.mark.parametrize("mode", [0, 2])
def test_all_array_overloads_values_vjps_and_no_mutation(mode):
    a = torch.linspace(.3, 1.2, 14, dtype=torch.float64)[::2].requires_grad_()
    b = torch.linspace(1.1, 1.9, 7, dtype=torch.float64).requires_grad_()
    c = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    original = [x.detach().clone() for x in (a, b, c)]
    actual = torch.ops.quantlibtorch._array_arithmetic(a, b, c, mode)
    expected = arithmetic_reference(a, b, c)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    weights = torch.cos(torch.arange(actual.numel(), dtype=torch.float64)).reshape_as(actual)
    actual_grad = torch.autograd.grad(actual, (a, b, c), weights)
    expected_grad = torch.autograd.grad(expected, (a, b, c), weights)
    for result, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(result, reference, rtol=1e-13, atol=1e-13)
    for result, reference in zip((a, b, c), original):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_array_arithmetic_first_and_second_differences():
    args = tuple(x.requires_grad_() for x in (
        torch.tensor([.4, .9], dtype=torch.float64),
        torch.tensor([.7, 1.2], dtype=torch.float64),
        torch.tensor(.8, dtype=torch.float64)))
    function = lambda *xs: torch.ops.quantlibtorch._array_arithmetic(*xs, 2)
    assert torch.autograd.gradcheck(function, args)
    assert torch.autograd.gradgradcheck(function, args)


def test_packed_array_copy_rebind_subrange_and_reorder():
    a = torch.linspace(.3, 1.4, 12, dtype=torch.float64)[::2].requires_grad_()
    b = torch.linspace(.8, 1.1, 6, dtype=torch.float64).requires_grad_()
    before = a.detach().clone()
    actual = torch.ops.quantlibtorch._array_views(a, b)
    expected = [a, torch.cat((a[:1], a[1:2]+b[1:2], a[2:])), a[1:-1],
                torch.cat((a[-1:], a[1:-1], a[:1]))]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    weights = [torch.arange(x.numel(), dtype=torch.float64).sin() for x in actual]
    g = torch.autograd.grad(actual, (a, b), weights)
    ref = torch.autograd.grad(expected, (a, b), weights)
    for result, reference in zip(g, ref):
        torch.testing.assert_close(result, reference, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(a, before, rtol=0, atol=0)


@pytest.mark.parametrize("unpack_no_grad,pack_no_grad", [(False, False), (True, False), (False, True), (True, True)])
def test_packing_respects_autograd_boundaries(unpack_no_grad, pack_no_grad):
    a = torch.arange(1., 5., dtype=torch.float64, requires_grad=True)
    actual = torch.ops.quantlibtorch._array_grad_boundary(a, unpack_no_grad, pack_no_grad)
    with torch.no_grad() if unpack_no_grad else nullcontext():
        views = a.unbind()
    with torch.no_grad() if pack_no_grad else nullcontext():
        expected = torch.stack(views)
    torch.testing.assert_close(actual, expected)
    assert actual.requires_grad == expected.requires_grad
    if actual.requires_grad:
        g, = torch.autograd.grad(actual.sum(), a, allow_unused=True)
        ref, = torch.autograd.grad(expected.sum(), a, allow_unused=True)
        assert (g is None) == (ref is None)
        if g is not None:
            torch.testing.assert_close(g, ref)


def test_packed_expanded_views_accumulate_gradient():
    x = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    value = torch.ops.quantlibtorch._array_roundtrip(x.expand(9))
    weights = torch.linspace(-1., 2., 9, dtype=torch.float64)
    g, = torch.autograd.grad(value, x, weights)
    torch.testing.assert_close(g, weights.sum())


@pytest.mark.parametrize("no_grad", [False, True])
def test_full_engine_forward_mode_with_vector_arrays(no_grad):
    # no_grad disables reverse recording, not forward-mode tangents. Inference
    # must not select the custom first-order solve when a dual tensor is passed.
    from quantlibtorch import fd_vanilla
    from torch.autograd import forward_ad
    sigma = torch.tensor([[.19, .23], [.21, .22]], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([[.3, -.8], [.5, .2]], dtype=torch.float64)
    kwargs = dict(times=[0., 1.], log_spots=[4., 5.], time_steps=3, grid_points=9, mode="full")
    value = fd_vanilla(sigma, **kwargs)
    gradient, = torch.autograd.grad(value, sigma)
    with torch.no_grad() if no_grad else nullcontext():
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(sigma, direction)
            actual, tangent = forward_ad.unpack_dual(fd_vanilla(dual, **kwargs))
    torch.testing.assert_close(actual, value, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(tangent, (gradient*direction).sum(), rtol=2e-10, atol=1e-10)
