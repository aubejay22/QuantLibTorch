"""Native stencil VJP against a dense reflected-boundary operator."""
import os

import pytest
import torch

import quantlibtorch

DTYPE = torch.float64


def matrix(bands):
    n = bands.shape[1]
    rows = torch.arange(n)
    left = torch.cat((torch.tensor([1]), torch.arange(n-1)))
    right = torch.cat((torch.arange(1, n), torch.tensor([n-2])))
    result = torch.diag(bands[1])
    result = result.index_put((rows, left), bands[0], accumulate=True)
    return result.index_put((rows, right), bands[2], accumulate=True)


@pytest.mark.parametrize('n', [2, 11])
@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative'])
@pytest.mark.parametrize('mode', [1, 4])
def test_native_application_values_and_all_vjps(n, view, mode):
    generator = torch.Generator().manual_seed(991)
    width = 2*n if view == 'strided' else n
    bases = (torch.randn(3, width, dtype=DTYPE, generator=generator).requires_grad_(),
             torch.randn(width, dtype=DTYPE, generator=generator).requires_grad_())
    if view == 'strided': args = (bases[0][:, ::2], bases[1][::2])
    elif view == 'negative': args = tuple(torch._neg_view(t) for t in bases)
    else: args = bases
    before = [t.detach().clone() for t in bases]
    actual = torch.ops.quantlibtorch._hook_apply(*args, mode, True)
    expected = matrix(args[0]) @ args[1]
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    weights = torch._neg_view(torch.linspace(-.4, .7, n, dtype=DTYPE))
    for gradient, reference in zip(torch.autograd.grad(actual, bases, weights),
                                   torch.autograd.grad(expected, bases, weights)):
        torch.testing.assert_close(gradient, reference, rtol=2e-12, atol=2e-12)
    for value, original in zip(bases, before):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_native_application_alias_and_no_grad():
    bands = torch.cos(torch.arange(21, dtype=DTYPE)).reshape(3, 7).requires_grad_()
    rhs = bands[1]
    value = torch.ops.quantlibtorch._hook_apply(bands, rhs, 4, True)
    reference = matrix(bands) @ rhs
    weights = torch.linspace(-.3, .8, 7, dtype=DTYPE)
    ga, = torch.autograd.grad(value, bands, weights)
    ge, = torch.autograd.grad(reference, bands, weights)
    torch.testing.assert_close(ga, ge, rtol=2e-12, atol=2e-12)
    with torch.no_grad():
        inference = torch.ops.quantlibtorch._hook_apply(bands, rhs, 4, True)
    assert not inference.requires_grad
    torch.testing.assert_close(inference, reference)


def test_native_application_differences_and_first_order_contract():
    bands = torch.cos(torch.arange(9, dtype=DTYPE)).reshape(3, 3).requires_grad_()
    rhs = torch.tensor([.2, -.3, .4], dtype=DTYPE, requires_grad=True)
    function = lambda b, r: torch.ops.quantlibtorch._hook_apply(b, r, 4, True)
    assert torch.autograd.gradcheck(function, (bands, rhs))
    with pytest.raises(RuntimeError, match='first derivatives only'):
        torch.autograd.grad(function(bands, rhs).sum(), bands, create_graph=True)


def test_native_application_forward_dispatch():
    bands = torch.ones(3, 7, dtype=DTYPE, requires_grad=True)
    rhs = torch.linspace(.1, .5, 7, dtype=DTYPE, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result = torch.ops.quantlibtorch._hook_apply(bands, rhs, 4, True)
        torch.autograd.grad(result.sum(), (bands, rhs))
    counts = {item.key: item.count for item in profile.key_averages()}
    assert counts['qlt::apply_adjoint_forward'] == 1
    assert counts['qlt::apply_adjoint_backward'] == 1
    native = os.environ.get('QLT_ADJOINT_ARITHMETIC_FORWARD', '1') == '1'
    assert ('qlt::apply_arithmetic_native_forward' in counts) == native
    # The fallback re-enters the original apply; native arithmetic avoids it.
    assert counts['qlt::band_apply'] == (1 if native else 2)
