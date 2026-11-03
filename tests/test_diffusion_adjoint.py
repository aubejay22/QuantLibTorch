"""Native local-volatility assembly VJP against independent tensor equations."""
import pytest
import torch

import quantlibtorch

DTYPE = torch.float64


def equation(variance, dx, dxx, rate, yield_rate):
    discount = torch.stack((torch.zeros_like(rate), -rate, torch.zeros_like(rate)))[:, None]
    return (rate-yield_rate-.5*variance)*dx + .5*variance*dxx + discount


@pytest.mark.parametrize('alias', [0, 1, 2])
@pytest.mark.parametrize('view', ['contiguous', 'strided', 'negative'])
def test_assembly_values_all_vjps_and_input_preservation(alias, view):
    n = 7
    width = 2*n if view == 'strided' else n
    generator = torch.Generator().manual_seed(718)
    base = [torch.randn(width, dtype=DTYPE, generator=generator).requires_grad_(),
            torch.randn(3, width, dtype=DTYPE, generator=generator).requires_grad_(),
            torch.randn(3, width, dtype=DTYPE, generator=generator).requires_grad_(),
            torch.tensor(.03, dtype=DTYPE, requires_grad=True),
            torch.tensor(.01, dtype=DTYPE, requires_grad=True)]
    args = base
    if view == 'strided': args = [base[0][::2], base[1][:, ::2], base[2][:, ::2], *base[3:]]
    elif view == 'negative': args = [torch._neg_view(t) for t in base]
    before = [t.detach().clone() for t in base]
    actual = torch.ops.quantlibtorch._hook_diffusion(*args, True, alias)
    original = torch.ops.quantlibtorch._hook_diffusion(*args, False, alias, 0)
    expected = equation(*args)
    weights = torch._neg_view(torch.cos(torch.arange(3*n, dtype=DTYPE)).reshape(3, n))
    references = torch.autograd.grad(expected, base, weights)
    for output in (actual, original):
        torch.testing.assert_close(output, expected, rtol=2e-12, atol=2e-12)
        for gradient, reference in zip(torch.autograd.grad(output, base, weights), references):
            torch.testing.assert_close(gradient, reference, rtol=2e-12, atol=2e-12)
    for value, initial in zip(base, before):
        torch.testing.assert_close(value, initial, rtol=0, atol=0)


@pytest.mark.parametrize('active', [(0,), (1,), (2,), (3,), (4,), tuple(range(5))])
def test_assembly_partial_gradients_and_differences(active):
    args = [torch.tensor([.04, .06, .05], dtype=DTYPE),
            torch.cos(torch.arange(9, dtype=DTYPE)).reshape(3, 3),
            torch.sin(torch.arange(9, dtype=DTYPE)).reshape(3, 3),
            torch.tensor(.03, dtype=DTYPE), torch.tensor(.01, dtype=DTYPE)]
    for index in active: args[index].requires_grad_()
    function = lambda *inputs: torch.ops.quantlibtorch._hook_diffusion(*inputs, True, 0)
    assert torch.autograd.gradcheck(function, tuple(args))
    with torch.no_grad():
        value = function(*args)
    assert not value.requires_grad
    torch.testing.assert_close(value, equation(*args))
    with pytest.raises(RuntimeError, match='first derivatives only'):
        torch.autograd.grad(function(*args).sum(), [args[i] for i in active], create_graph=True)


def test_assembly_shared_inputs_accumulate_vjps():
    base = torch.cos(torch.arange(9, dtype=DTYPE)).reshape(3, 3).requires_grad_()
    args = (base[0], base, base, base[1, 0], base[2, 0])
    actual = torch.ops.quantlibtorch._hook_diffusion(*args, True, 1)
    expected = equation(*args)
    ga, = torch.autograd.grad(actual.sum(), base)
    ge, = torch.autograd.grad(expected.sum(), base)
    torch.testing.assert_close(ga, ge, rtol=2e-12, atol=2e-12)
