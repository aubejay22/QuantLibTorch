import pytest
import torch

from quantlibtorch import tridiagonal_solve


def inputs(n=6):
    generator = torch.Generator().manual_seed(129)
    return tuple(t.requires_grad_() for t in (
        torch.rand(n - 1, dtype=torch.float64, generator=generator) * 0.3,
        torch.rand(n, dtype=torch.float64, generator=generator) + 2.,
        -torch.rand(n - 1, dtype=torch.float64, generator=generator) * 0.4,
        torch.randn(n, dtype=torch.float64, generator=generator),
    ))


def dense_solve(lower, diagonal, upper, rhs):
    a = torch.diag(lower, -1) + torch.diag(diagonal) + torch.diag(upper, 1)
    return torch.linalg.solve(a, rhs)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
@pytest.mark.parametrize("n", [2, 6, 19])
def test_forward_residual_and_all_input_vjps(mode, n):
    args = inputs(n)
    actual = tridiagonal_solve(*args, mode=mode)
    expected = dense_solve(*args)
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    lower, diagonal, upper, rhs = args
    matrix = torch.diag(lower, -1) + torch.diag(diagonal) + torch.diag(upper, 1)
    torch.testing.assert_close(matrix @ actual, rhs, rtol=2e-14, atol=2e-14)
    weights = torch.linspace(-.7, 1.3, n, dtype=torch.float64)
    actual_vjp = torch.autograd.grad(actual, args, weights)
    expected_vjp = torch.autograd.grad(expected, args, weights)
    for a, e in zip(actual_vjp, expected_vjp):
        torch.testing.assert_close(a, e, rtol=3e-13, atol=3e-14)


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_finite_difference_gradcheck(mode):
    assert torch.autograd.gradcheck(lambda *args: tridiagonal_solve(*args, mode=mode), inputs(4))


def test_full_second_derivative():
    assert torch.autograd.gradgradcheck(tridiagonal_solve, inputs(3))


def test_adjoint_rejects_double_backward():
    args = inputs(3)
    x = tridiagonal_solve(*args, mode="adjoint")
    grad = torch.autograd.grad(x.square().sum(), args[1], create_graph=True)[0]
    with pytest.raises(RuntimeError):
        grad.sum().backward()


@pytest.mark.parametrize("mode", ["full", "adjoint"])
def test_noncontiguous_and_no_input_mutation(mode):
    original = inputs(7)
    args = tuple(torch.stack([x, x], dim=1)[:, 0] for x in original)
    saved = [x.clone() for x in args]
    x = tridiagonal_solve(*args, mode=mode)
    torch.autograd.grad(x.sum(), original)
    torch.testing.assert_close(x, dense_solve(*args))
    for before, after in zip(saved, args):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("bad,match", [
    ("dtype", "float64"), ("shape", "one-dimensional"),
    ("size", "at least 2"), ("length", "length"), ("nan", "finite"),
    ("pivot", "cannot be close to zero"), ("late_pivot", "division by zero"),
])
def test_invalid_input(bad, match):
    lower, diagonal, upper, rhs = (x.detach().clone() for x in inputs(3))
    if bad == "dtype":
        rhs = rhs.float()
    elif bad == "shape":
        rhs = rhs.unsqueeze(0)
    elif bad == "size":
        lower, diagonal, upper, rhs = lower[:0], diagonal[:1], upper[:0], rhs[:1]
    elif bad == "length":
        lower = lower[:1]
    elif bad == "nan":
        rhs[0] = float("nan")
    elif bad == "pivot":
        diagonal[0] = 0.
    elif bad == "late_pivot":
        lower.fill_(1.)
        diagonal.fill_(1.)
        upper.fill_(1.)
    with pytest.raises(RuntimeError, match=match):
        tridiagonal_solve(lower, diagonal, upper, rhs)


def test_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        tridiagonal_solve(*inputs(), mode="typo")


def test_sparse_input_is_rejected():
    lower, diagonal, upper, rhs = inputs()
    with pytest.raises(RuntimeError, match="dense strided"):
        tridiagonal_solve(lower, diagonal, upper, rhs.to_sparse())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_is_explicitly_rejected():
    with pytest.raises(RuntimeError, match="CPU"):
        tridiagonal_solve(*(x.cuda() for x in inputs()))
