"""Requested assembly VJPs retain active operands needed by later derivatives."""
import pytest
import torch

from test_diffusion_arithmetic import arguments, assembly, equation
from test_backward_duals import compare_incoming_duals

DTYPE = torch.float64


def inputs(mask, views=False):
    strided = views and mask % 3 == 1
    leaves = tuple(t.detach().requires_grad_(bool(mask & (1 << i)))
                   for i, t in enumerate(arguments(8 if strided else 4)))
    if strided:
        xs = (leaves[0][::2], leaves[1][:, ::2], leaves[2][:, ::2], *leaves[3:])
    elif views and mask % 3 == 2:
        xs = tuple(torch._neg_view(t) for t in leaves)
    else:
        xs = leaves
    return leaves, xs


def compare(ga, ge, leaves):
    ga = tuple(torch.zeros_like(t) if g is None else g for g, t in zip(ga, leaves))
    ge = tuple(torch.zeros_like(t) if g is None else g for g, t in zip(ge, leaves))
    torch.testing.assert_close(ga, ge, rtol=2e-10, atol=2e-11)
    return ga, ge


@pytest.mark.parametrize('mask', range(1, 32))
def test_every_activity_subset_through_third_order_preserves_inputs(mask):
    leaves, xs = inputs(mask, views=True)
    before, versions = tuple(t.clone() for t in leaves), tuple(t._version for t in leaves)
    weights = torch.cos(torch.arange(12, dtype=DTYPE)).reshape(3, 4).requires_grad_()
    active = tuple(t for t in leaves if t.requires_grad)+(weights,)
    directions = tuple(torch.cos(t.detach()+.2) for t in active)
    actual = assembly(*xs, alias=mask % 3)
    expected = equation(*xs)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    la, le = (actual.sin()*weights).sum(), (expected.sin()*weights).sum()
    for order in range(3):
        ga = torch.autograd.grad(la, active, create_graph=order < 2, allow_unused=True)
        ge = torch.autograd.grad(le, active, create_graph=order < 2, allow_unused=True)
        ga, ge = compare(ga, ge, active)
        if order < 2:
            la = sum((g*d).sum() for g, d in zip(ga, directions))
            le = sum((g*d).sum() for g, d in zip(ge, directions))
    torch.testing.assert_close(leaves, before, rtol=0, atol=0)
    assert tuple(t._version for t in leaves) == versions


@pytest.mark.parametrize('first', range(5))
def test_one_requested_vjp_retains_cross_derivatives_to_other_active_inputs(first):
    xs = arguments(4)
    weights = torch.sin(torch.arange(12, dtype=DTYPE)).reshape(3, 4).requires_grad_()
    actual, expected = assembly(*xs), equation(*xs)
    ga, = torch.autograd.grad((actual.sin()*weights).sum(), xs[first], create_graph=True)
    ge, = torch.autograd.grad((expected.sin()*weights).sum(), xs[first], create_graph=True)
    torch.testing.assert_close(ga, ge, rtol=2e-12, atol=2e-13)
    direction = torch.cos(xs[first].detach())
    active = (*xs, weights)
    la, le = (ga*direction).sum(), (ge*direction).sum()
    for order in range(2):
        ga = torch.autograd.grad(la, active, create_graph=order == 0, allow_unused=True)
        ge = torch.autograd.grad(le, active, create_graph=order == 0, allow_unused=True)
        ga, ge = compare(ga, ge, active)
        if order == 0:
            la = sum((g*torch.cos(t.detach())).sum() for g, t in zip(ga, active))
            le = sum((g*torch.cos(t.detach())).sum() for g, t in zip(ge, active))


@pytest.mark.parametrize('mask', range(1, 32))
@pytest.mark.parametrize('create_graph', [False, True])
def test_every_activity_subset_preserves_incoming_dual_vjp_weights(mask, create_graph):
    leaves, xs = inputs(mask)
    compare_incoming_duals(tuple(t for t in leaves if t.requires_grad), assembly(*xs), equation(*xs),
                           create_graph=create_graph)


@pytest.mark.parametrize('mask', [3, 9, 24])
def test_partial_activity_independent_first_and_second_differences(mask):
    _, xs = inputs(mask)
    assert torch.autograd.gradcheck(assembly, xs, fast_mode=True)
    assert torch.autograd.gradgradcheck(assembly, xs, fast_mode=True)
