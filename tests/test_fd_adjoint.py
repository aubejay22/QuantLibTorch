"""Independent dense reference for the modern QuantLib band-solve hook."""
import pytest
import torch

import quantlibtorch  # register native operations


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("n", [2, 7])
def test_band_hook_all_coefficients_and_scale_vjp(adjoint, n):
    generator = torch.Generator().manual_seed(311)
    args = tuple(x.requires_grad_() for x in (
        torch.rand(n-1, dtype=torch.float64, generator=generator)*.2,
        torch.rand(n, dtype=torch.float64, generator=generator)+2.,
        -torch.rand(n-1, dtype=torch.float64, generator=generator)*.3,
        torch.randn(n, dtype=torch.float64, generator=generator),
        torch.tensor(.17, dtype=torch.float64),
        torch.tensor(1.4, dtype=torch.float64),
    ))
    lower, diagonal, upper, rhs, a, b = args
    matrix = a*(torch.diag(lower, -1)+torch.diag(diagonal)+torch.diag(upper, 1))
    matrix = matrix+b*torch.eye(n, dtype=torch.float64)
    actual = torch.ops.quantlibtorch._band_solve(*args, adjoint)
    expected = torch.linalg.solve(matrix, rhs)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
    weights = torch.linspace(-.4, 1.2, n, dtype=torch.float64)
    for result, reference in zip(torch.autograd.grad(actual, args, weights),
                                 torch.autograd.grad(expected, args, weights)):
        torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-13)


def test_band_adjoint_rejects_higher_derivatives():
    diagonal = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    off = torch.tensor([.1], dtype=torch.float64)
    scalar = torch.tensor(1., dtype=torch.float64)
    value = torch.ops.quantlibtorch._band_solve(off, diagonal, -off, diagonal*0+1, scalar, scalar, True)
    with pytest.raises(RuntimeError, match="first derivatives only"):
        torch.autograd.grad(value.sum(), diagonal, create_graph=True)


@pytest.mark.parametrize('hook', ['split', 'tridiagonal'])
@pytest.mark.parametrize('negative_index', [0, 1, 2, 3, 'all'])
def test_native_adjoint_lazy_negative_views_match_dense(hook, negative_index):
    n = 7
    raw = tuple(t.requires_grad_() for t in (
        torch.full((n-1,), -.1, dtype=torch.float64),
        -torch.linspace(2., 3., n, dtype=torch.float64),
        torch.full((n-1,), .2, dtype=torch.float64),
        -torch.cos(torch.arange(n, dtype=torch.float64))))
    args = tuple(torch._neg_view(t) if negative_index == 'all' or negative_index == i else t
                 for i, t in enumerate(raw))
    assert any(t.is_neg() for t in args)
    matrix = torch.diag(args[0], -1)+torch.diag(args[1])+torch.diag(args[2], 1)
    expected = torch.linalg.solve(matrix, args[3])
    if hook == 'split':
        actual = torch.ops.quantlibtorch._band_solve(*args, torch.tensor(1., dtype=torch.float64),
                                                   torch.tensor(0., dtype=torch.float64), True)
    else:
        actual = torch.ops.quantlibtorch._tridiagonal_hook(*args, True, False)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    weights = torch._neg_view(-torch.linspace(-.3, .7, n, dtype=torch.float64))
    for gradient, reference in zip(torch.autograd.grad(actual, raw, weights),
                                   torch.autograd.grad(expected, raw, weights)):
        torch.testing.assert_close(gradient, reference, rtol=2e-12, atol=2e-12)


def test_band_adjoint_strided_inputs_and_upstream_gradient():
    generator = torch.Generator().manual_seed(572)
    n = 11
    args = tuple(x.requires_grad_() for x in (
        (torch.rand(2*(n-1), generator=generator, dtype=torch.float64)*.1)[::2],
        (torch.rand(2*n, generator=generator, dtype=torch.float64)+2.)[::2],
        (torch.rand(2*(n-1), generator=generator, dtype=torch.float64)*-.2)[::2],
        torch.randn(2*n, generator=generator, dtype=torch.float64)[::2],
        torch.tensor(-.3, dtype=torch.float64),
        torch.tensor(1.7, dtype=torch.float64),
    ))
    assert all(not x.is_contiguous() for x in args[:4])
    originals = [x.clone() for x in args]
    lower, diagonal, upper, rhs, a, b = args
    matrix = a*(torch.diag(lower, -1)+torch.diag(diagonal)+torch.diag(upper, 1))
    matrix = matrix+b*torch.eye(n, dtype=torch.float64)
    actual = torch.ops.quantlibtorch._band_solve(*args, True)
    expected = torch.linalg.solve(matrix, rhs)
    weights = torch.randn(2*n, generator=generator, dtype=torch.float64)[::2]
    torch.testing.assert_close(actual, expected)
    for result, reference in zip(torch.autograd.grad(actual, args, weights),
                                 torch.autograd.grad(expected, args, weights)):
        torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-13)
    for x, original in zip(args, originals):
        torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_band_adjoint_finite_differences():
    args = tuple(x.requires_grad_() for x in (
        torch.tensor([.1, -.2], dtype=torch.float64),
        torch.tensor([2., 3., 2.5], dtype=torch.float64),
        torch.tensor([-.1, .3], dtype=torch.float64),
        torch.tensor([.5, -.8, 1.2], dtype=torch.float64),
        torch.tensor(.3, dtype=torch.float64),
        torch.tensor(1., dtype=torch.float64),
    ))
    assert torch.autograd.gradcheck(lambda *xs: torch.ops.quantlibtorch._band_solve(*xs, True), args)


@pytest.mark.parametrize("diagonal", [[0., 1.], [1., 1.]])
def test_band_adjoint_rejects_failed_pivots(diagonal):
    off = torch.tensor([1.], dtype=torch.float64)
    diag = torch.tensor(diagonal, dtype=torch.float64, requires_grad=True)
    one, zero = torch.tensor(1., dtype=torch.float64), torch.tensor(0., dtype=torch.float64)
    with pytest.raises(RuntimeError, match="division by zero|non-finite"):
        torch.ops.quantlibtorch._band_solve(off, diag, off, off.expand(2), one, zero, True)
